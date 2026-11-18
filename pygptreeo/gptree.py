"""GPTree: Main tree structure for online regression.

This module implements the GPTree class, which provides the core logic for
dynamically growing a binary tree of Gaussian Process regressors. The tree
adapts to incoming data by splitting nodes and training local GP models.

The GPTree manages:
    - Dynamic tree growth based on data accumulation
    - Keeping the tree within a memory limit (sleeping and removed leaves)
    - Routing of data points to appropriate leaf nodes
    - Prediction aggregation across multiple leaf nodes
    - Training coordination for all GP models

This implementation is suitable for online learning scenarios where data
arrives sequentially and the model needs to adapt continuously.
"""

# Standard library imports
import logging
from collections import OrderedDict, deque
from copy import deepcopy
from typing import Callable, Optional, Type, Union

# Third-party imports
import joblib
import numpy as np
from sklearn.utils import resample
from tqdm import tqdm

# Local imports
from pygptreeo.default_gpr import Default_GPR
from pygptreeo.gpnode import GPNode, TARGET_COVERAGE
from pygptreeo.gp_interface import GPRegressorInterface
from pygptreeo.output_alignment import MIN_GRID, MIN_OUTPUTS
from pygptreeo.output_basis import OutputBasisLearner
from pygptreeo.global_mean import GlobalMeanLearner, make_global_mean

logger = logging.getLogger(__name__)

# The accounting of GPTree(max_memory_mb=...)
NODE_OVERHEAD_BYTES = 10000   # the objects of a leaf and of one inner node, besides their arrays
GPR_OVERHEAD_BYTES = 10000    # the objects of one GP of a leaf, besides its arrays
FITTED_SHARE = 0.5            # share of the leaves' memory that the GPs awake may take when leaves are removed

# The sigma of GPTree(widen_sigma_with_distance=True)
SIGMA_DISTANCE_START = 2.0    # sigma grows from this distance from a leaf's fitted points, in their standard deviations
SIGMA_DISTANCE_WINDOW = 200   # the growth is learned from the last predictions beyond it, this many
SIGMA_DISTANCE_MIN = 20       # and there is none before this many


class GPTree:
    """Implements the GPTree for dynamic learning and regression.

    The GPTree class provides the core logic for a dynamically growing tree
    structure where each node (a GPNode instance) contains a Gaussian Process
    Regressor (GPR). Its primary purpose is to learn a function on the fly from
    a stream of data points (x, y) by adaptively partitioning the input space.
    As data points are added, the tree grows by splitting nodes that become
    too full, and the GPRs within the nodes are updated to model the local
    data distribution.

    Attributes:
        root (GPNode): The root node of the GPTree.
        GPR (GPRegressorInterface): The base GPR configuration used as a
            template for the GPR in each new GPNode.
        Nbar (int): The maximum number of training points a GPNode can hold
            before it attempts to split.
        theta (float): A parameter that influences the size of the overlapping
            region between sibling GPNodes after a split. The overlap is
            calculated as theta * range_of_split_dimension.
        use_calibrated_sigma (bool): If True, the sigma (standard deviation)
            from GPNode predictions will be scaled using the node's calibrated
            sigma_scaler. This is intended to provide better uncertainty estimates.
        n_features (int): The number of features in the input data X.
            Automatically determined from the first data point.
        first_point (bool): A flag indicating if the next point to be processed
            is the first one. Used for initial setup like determining n_features.
        append_points_between_fits (bool): Whether a point a leaf receives
            between two of its fits is added to its fitted GPs at once.
        widen_sigma_with_distance (bool): Whether sigma grows with the distance
            of a query from the fitted points of its leaf.
        max_memory_mb (Optional[float]): The memory the tree stays within, in MB,
            or None for no limit.
        memory_mb (float): The tree's estimate of its own size, in MB.

    Methods:
        fit(X_train, y_train, show_progress=False, shuffle=True, forward_GPR_to_next_leaf=False):
            Constructs the tree from a batch of training data. Only for a tree
            without points.
        batch_update_tree(X, y, sigma):
            Adds a batch of data points to the tree, one after the other as
            `update_tree` does.
        update_tree(x, y, allow_training=True):
            Adds a single data point to the tree, potentially causing node
            splits and GPR retraining.
        apply_fit_result(result):
            Gives the result of a fit that `update_tree(defer_fits=True)`
            handed out, and `solve_fit_task` carried out, to its leaf.
        predict(X_test, mode='recursive', show_progress=False):
            Predicts target values for X_test using the ensemble of GPRs in
            the leaf nodes.
        save(path):
            Saves the trained GPTree object to a file.
    """
    def __init__(self,
                 GPR: Optional[GPRegressorInterface] = None,
                 Nbar: Optional[int] = 100,
                 theta: Optional[float] = 0.0001,
                 use_calibrated_sigma: Optional[bool] = True,
                 split_dimension_criteria: Optional[str] = 'min_lengthscale',
                 splitting_strategy: Optional[str] = 'gradual',
                 max_n_pred_leaves: Optional[int] = None,
                 aggregation: Optional[str] = "default",
                 n_outputs: Optional[int] = 1,
                 output_model: Optional[str] = 'independent',
                 output_shape: Optional[tuple] = None,
                 output_basis_components: Union[str, int, float] = 'noise',
                 output_basis_max_components: Optional[int] = None,
                 output_basis_min_points: int = 50,
                 output_basis_refit_every: Optional[int] = None,
                 output_basis_reservoir_size: int = 2000,
                 global_mean: Union[None, str, GlobalMeanLearner] = None,
                 global_mean_kwargs: Optional[dict] = None,
                 max_memory_mb: Optional[float] = None,
                 append_points_between_fits: bool = True,
                 widen_sigma_with_distance: bool = True,
                 **kwargs):
        """Initializes the GPTree.

        Args:
            GPR (Optional[GPRegressorInterface]): The Gaussian Process
                Regressor instance to be used as a template for nodes.
                Defaults to `Default_GPR()` (scikit-learn adapter).
            Nbar (Optional[int]): Maximum number of training points a node
                can hold before splitting. Defaults to 100.
            theta (Optional[float]): Parameter influencing the overlap region
                between sibling nodes. The overlap is calculated as
                theta * range_of_split_dimension. Defaults to 0.0001.
            use_calibrated_sigma (Optional[bool]): If True, enables sigma
                calibration in GPNode predictions. Defaults to True.
            split_dimension_criteria (Optional[str]): Method to select split
                dimension. Options: 'min_lengthscale' (the dimension with the
                smallest fitted ARD length scale, pooled over all of a leaf's GPs;
                falls back to 'max_spread' when the kernel has no per-dimension
                length scales), 'max_spread', 'max_variance', 'max_uncertainty',
                'random'. Defaults to 'min_lengthscale'.
            splitting_strategy (Optional[str]): Strategy for splitting nodes.
                'standard' (each child keeps its own half of the parent's points)
                or 'gradual' (each child also receives a copy of its sibling's
                points, dropped as its own arrive, so it starts with a full-size
                fit and stays continuous across the split). Defaults to 'gradual'.
            max_n_pred_leaves (Optional[int]): Maximum number of leaves to use
                for prediction. Defaults to None (use all).
            aggregation (Optional[str]): Method for aggregating predictions.
                'default'/'moe' or 'poe'. Defaults to 'default'.
            n_outputs (Optional[int]): Number of output dimensions. Defaults to 1 (single output).
            output_model (Optional[str]): How the outputs of a multi-output problem are
                modelled inside each leaf (ignored when n_outputs=1):
                'independent' (default): one GP per output with its own kernel
                hyperparameters. 'shared': one GP with a single shared kernel for all
                outputs (one kernel matrix per leaf instead of n_outputs); needs a
                backend whose ``supports_multitarget()`` is True (the scikit-learn
                adapter). 'pca': a tree-global linear basis of the output space is
                learned from a reservoir sample of the stream and each leaf models
                the ``k`` basis scores with ``k`` GPs; predictions are mapped back to
                output space with propagated uncertainties. Suited to strongly
                correlated outputs such as a function f(t; x) sampled on a t-grid.
                'aligned': as 'shared', for outputs that are a curve on a grid,
                given in the order of the grid. Each leaf also learns a warp of
                the grid that depends on the inputs and brings the features of
                its curves (a peak, an edge, the phase of an oscillation) to
                common positions, fits its GP to the aligned curves, and keeps
                the alignment if it lowers the leave-one-out error of its
                curves. Suited to curves with features whose position depends
                on the inputs; needs the scikit-learn adapter.
            output_shape (Optional[tuple]): For 'aligned': the shape ``(p1, p2)``
                of the grid, for outputs that are a function on a grid of two
                dimensions (a surface), flattened row by row
                (``values.ravel()``). The leaves then warp both axes of the
                grid. Defaults to a curve, ``(n_outputs,)``.
            output_basis_components ('noise', int or float): For 'pca': how many
                basis components to keep. 'noise' (default) keeps every component
                whose variance lies above what the per-point observation noise
                ``sigma`` alone would produce (the Marchenko-Pastur edge), so noisy
                data get few components and clean data get more. An int fixes the
                number; a float in (0, 1) keeps the smallest number of components
                explaining that fraction of the output variance (the dropped
                components then set an accuracy floor of about ``sqrt(1 - fraction)``
                times the output scale).
            output_basis_max_components (Optional[int]): For 'pca': hard cap on the
                number of components (and hence GPs per leaf). Defaults to None.
            output_basis_min_points (int): For 'pca': number of observed points
                before the first basis is fitted (leaves cannot train before that).
                Defaults to 50.
            output_basis_refit_every (Optional[int]): For 'pca': refit the basis every
                this many observed points; None (default) uses a doubling schedule.
                Leaves pick up a refitted basis at their next retrain.
            output_basis_reservoir_size (int): For 'pca': size of the reservoir
                sample of outputs the basis is fitted on. Defaults to 2000.
            global_mean (None, str or GlobalMeanLearner): Optional tree-wide global
                model whose *residual* the leaf GPs model. None (default): no
                global model. 'additive_gp': an ``AdditiveGPGlobalMean`` (a GP with
                a low-order additive + Matern kernel, fitted on a coverage reservoir
                of the stream and refit when the reservoir turns over). 'net': a
                ``NetGlobalMean`` (a feature network trained on the stream; needs
                PyTorch). Or any
                ``GlobalMeanLearner`` instance. The global model captures smooth,
                large-scale structure from all the data the tree has seen; the
                leaves capture the rest and, at their edges, revert to it rather
                than to a leaf constant. Each leaf remembers the snapshot it was
                fitted against and refits before predicting if a newer one exists
                (for 'net' with its kernel hyperparameters kept).
                The predicted sigma combines the residual GP's uncertainty with the
                learner's estimate of the global model's own error
                (``error_scale``) and remains the uncertainty of the estimate of
                the underlying function. See ``pygptreeo.global_mean``.
            global_mean_kwargs (Optional[dict]): Keyword arguments for the built-in
                learner when ``global_mean`` is given as a string (e.g.
                ``dict(reservoir_size=300, n_restarts_optimizer=1)``).
            max_memory_mb (Optional[float]): The memory the tree stays within,
                in MB (10^6 bytes), by its own estimate (``memory_mb``): the
                points and fitted GPs of its leaves, a fixed amount per leaf,
                and the global model and output basis at their full size. At
                the limit the tree first puts leaves to sleep, the one whose
                GPs were unused for the longest time first: the GPs release
                their Cholesky factors and compute them again when the leaf
                is next used, so no prediction changes. Once the GPs that are
                awake take less than half of the memory left for the leaves,
                the tree instead removes the leaf that has gone longest
                without a point, with its points and GPs, and the leaf's
                sibling takes over the parent's region. A prediction never
                removes a leaf. Leaves only sleep with a GP backend that can
                release and restore a fit (the scikit-learn adapter); with
                another one they are only removed. Defaults to None (no limit).
            append_points_between_fits (bool): If True (default), a point a
                leaf receives between two of its fits is added to its fitted
                GPs at once, with the hyperparameters and the scaling of the
                last fit kept (``GPNode.append_point``). If False, a point
                enters the GPs with the leaf's next fit. This is also the case
                with a GP backend that cannot append a point (only the
                scikit-learn adapter can).
            widen_sigma_with_distance (bool): If True (default), the
                calibrated sigma of a query that lies more than 2 standard
                deviations from the mean of its leaf's fitted points, along
                some input, is multiplied by 1 + slope * (distance - 2). The
                slope is one number for the tree, set so that sigma covers
                the errors of its latest predictions at such distances. This
                is for a leaf that predicts beyond its own points. The
                predictions themselves do not change. Without
                `use_calibrated_sigma` it has no effect.
            **kwargs: Additional keyword arguments passed to the constructor
                of the root `GPNode`. These can include parameters like
                `split_position_method`, `retrain_every_n_points`, and
                `use_standard_scaling` (bool, defaults to True).
        """

        # Use Default_GPR if no GPR is provided
        if GPR is None:
            GPR = Default_GPR()

        self.GPR = GPR
        self.splitting_strategy = splitting_strategy
        self.n_outputs = n_outputs

        if output_model not in ('independent', 'shared', 'pca', 'aligned'):
            raise ValueError(f"Unknown output_model '{output_model}'. Use 'independent', 'shared', 'pca' or 'aligned'.")
        if n_outputs == 1:
            output_model = 'independent'
        if output_model in ('shared', 'aligned'):
            supports = getattr(GPR, 'supports_multitarget', None)
            if supports is None or not supports():
                raise ValueError(f"output_model='{output_model}' needs a GP backend with supports_multitarget() == True "
                                 "(e.g. the scikit-learn adapter returned by Default_GPR)")
        if output_model == 'aligned' and n_outputs < MIN_OUTPUTS:
            raise ValueError(f"output_model='aligned' is for a curve on a grid of at least {MIN_OUTPUTS} points, "
                             f"not for {n_outputs} outputs")
        if output_shape is not None:
            output_shape = tuple(int(p) for p in output_shape)
            if len(output_shape) not in (1, 2) or int(np.prod(output_shape)) != n_outputs:
                raise ValueError(f"output_shape must be (n_outputs,) or (p1, p2) with p1 * p2 = n_outputs = {n_outputs}, not {output_shape}")
            if len(output_shape) == 2 and min(output_shape) < MIN_GRID:
                raise ValueError(f"output_shape (p1, p2) is for a grid of at least {MIN_GRID} points along each axis, not {output_shape}")
        self.output_shape = output_shape
        self.output_model = output_model
        # Tree-global output basis for 'pca' (shared by reference with every node)
        self.output_basis = None
        if output_model == 'pca':
            self.output_basis = OutputBasisLearner(
                n_outputs=n_outputs, n_components=output_basis_components,
                max_components=output_basis_max_components,
                min_points=output_basis_min_points, refit_every=output_basis_refit_every,
                reservoir_size=output_basis_reservoir_size)

        # Tree-wide global model (None by default; shared by reference with every node)
        self.global_mean = make_global_mean(global_mean, **(global_mean_kwargs or {}))

        self.append_points_between_fits = append_points_between_fits
        self.widen_sigma_with_distance = widen_sigma_with_distance
        # The last predictions beyond SIGMA_DISTANCE_START: (error in units of sigma, distance beyond), and the slope in use
        self._distance_errors = deque(maxlen=SIGMA_DISTANCE_WINDOW)
        self._sigma_slope = 0.0

        self.root = GPNode(0, my_GPR=GPR, Nbar=Nbar, split_dimension_criteria=split_dimension_criteria,
                          splitting_strategy=self.splitting_strategy, n_outputs=n_outputs,
                          output_model=output_model, output_shape=output_shape, output_basis=self.output_basis,
                          global_mean=self.global_mean,
                          **kwargs)  # Initialize root node of the GPTree

        self.theta = theta

        self.n_features = 0

        self.use_calibrated_sigma = use_calibrated_sigma

        self.max_n_pred_leaves = max_n_pred_leaves

        if max_memory_mb is not None and max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        self.max_memory_mb = max_memory_mb
        # Under a limit: the leaves with their size, least recently reached by a
        # point first, and those with fitted GPs that are awake, with the size of
        # what the GPs hold, least recently used first.
        self._leaves = OrderedDict()
        self._fitted = OrderedDict()
        self._leaf_bytes = 0
        self._fitted_bytes = 0
        self._fixed_bytes = None    # global model and output basis; known at the first point

        self.aggregation = aggregation

        # The fits that update_tree(defer_fits=True) handed out and whose results
        # are still to come: task id -> (the node the task was made for, the frame of the task)
        self._fit_tasks = {}
        self._n_fit_tasks = 0

        self.first_point = True


    def __getstate__(self):
        # The fits handed out are not saved: their results cannot reach a loaded tree
        return {**self.__dict__, '_fit_tasks': {}}


    def update_tree(self, x: np.ndarray, y: Union[float, np.ndarray], sigma: Union[float, np.ndarray],
                    allow_training=True, defer_fits=False):
        """Updates the tree structure and node GPRs with a new data point (x, y, sigma).

        This method implements a process similar to Algorithm 1 in the DLGP
        (Deep Locally-Weighted Gaussian Processes) article, with some modifications.
        It navigates from the root to find an appropriate leaf node for the new
        data point `x` based on the probabilistic splitting functions of intermediate
        nodes. The point is added to the chosen leaf's training data.

        If `allow_training` is True, the leaf node's GPR may be retrained based
        on its buffer conditions (e.g., `retrain_every_n_points`) or if it becomes
        full. If the node's capacity `Nbar` is reached, it splits into two
        child nodes, its data is partitioned, and its own GPR and data are
        typically deleted to save memory. With `max_memory_mb` set, the tree
        is brought back within the limit at the end.

        Args:
            x (np.ndarray): The new input data point (features), expected as a
                1D array or a 2D array with one row.
            y (float or np.ndarray): The corresponding target value(s) for the data point.
                For single output: float. For multi-output: array of shape (n_outputs,).
            sigma (float or np.ndarray): The uncertainty (standard deviation) for this point. Required.
                Can be scalar (same for all outputs) or array (per-output).
            allow_training (bool): If True (default), the GPR model in the
                selected leaf node can be retrained if its conditions
                (e.g., buffer full, node full) are met. If False, GPR retraining
                is suppressed during this update. This is useful, for example,
                during the initial `fit` method where tree construction is the
                priority before a final training pass on all leaves.
            defer_fits (bool): If True, a leaf that is due a fit is not fitted
                here. The fit is returned as a task for `solve_fit_task`,
                which can run in another process, and `apply_fit_result`
                takes its result. Until then the leaf keeps its
                hyperparameters and takes its points as appended points, and
                neither it nor the leaves it splits into are handed another
                task. A leaf that cannot hand its fit out is fitted here (see
                `GPNode.fit_task`). Defaults to False.

        Returns:
            With `defer_fits`, the fit task if the leaf became due one: a dict
            that can be pickled. None otherwise.
        """

        # The first input point is used to determine self.n_features
        if self.first_point:
            self.n_features = x.size
            self.root.init_data_set(self.n_features)
            if self.max_memory_mb is not None:
                self._start_accounting()
            self.first_point = False

        # The tree-global output basis ('pca') and the global model see every observation
        if self.output_basis is not None:
            self.output_basis.observe(np.asarray(y, dtype=float).reshape(-1), sigma)
        if self.global_mean is not None:
            self.global_mean.observe(x, y, sigma)

        # Find a leaf node for the new (x,y,sigma) point
        # - Start from the root node
        # - For each level, pick a branch according node.prob_func(x), until a leaf node is reached
        node = self.root
        while not node.is_leaf:
            node = node.children[int(np.random.binomial(1, node.prob_func(x)[0][0]))]
        # A sleeping leaf is woken before the point is stored, so that the
        # prediction registered below is one made without the point
        node.wake()

        # Check if this point should be merged with a nearby point (updates existing point in-place)
        if node.should_merge_point(x, y, sigma):
            # Point was merged with existing point, don't add as new point
            # Still register prediction performance and update sigma scaler
            self._register_prediction(node, x, y, allow_training)
            self._after_update(node)
            return

        # Check if this point should be rejected (if well-predicted by current GP)
        if node.should_reject_point(x, y):
            # Point is well-predicted, don't store it
            self._after_update(node)
            return

        # Add new point and register prediction performance. Without training, a node
        # that has no fitted GP stays at its prior, which says nothing about the fitted GP.
        node.store_point(x, y, sigma, remove_shared=True)
        self._register_prediction(node, x, y, allow_training)

        # Retrain GP? The node will decide based Nbar and/or its buffer of training points
        task = None
        if allow_training:
            if defer_fits:
                task = self._fit_task(node)
            # A node with a fit handed out waits for its result
            did_retrain = node.fit_task_id not in self._fit_tasks and node.fit_my_GPR()
            # Between two fits the point is added to the fitted GPs as they are
            if self.append_points_between_fits and not did_retrain:
                node.append_point(x, y, sigma)

        # If the node is full, generate child nodes
        if node.n_points >= node.Nbar:
            # Create child nodes. Each child node gets a copy of the current parent GP.
            node.generate_children(self.GPR, self.n_features)
            
            # Compute parameters for the probability function 
            node.compute_split_position_and_overlap(self.theta)

            # Now pass parent's split info to children
            node.children[0].parent_split_index = node.split_index
            node.children[0].parent_split_position = node.split_position
            node.children[1].parent_split_index = node.split_index
            node.children[1].parent_split_position = node.split_position

            if node.splitting_strategy == 'gradual':

                # First distribute data as usual between the two child nodes
                node.split_training_data()

                # Since we are doing gradual splitting, give
                # each child a copy of the other child's data
                order = node.split_coordinate(node.children[1].my_X_data).argsort()
                node.children[0].shared_X_data = node.children[1].my_X_data[order]
                node.children[0].shared_y_data = node.children[1].my_y_data[order]
                node.children[0].shared_sigma_data = node.children[1].my_sigma_data[order]
                node.children[0].n_shared_points = node.children[0].shared_X_data.shape[0]

                order = node.split_coordinate(node.children[0].my_X_data).argsort()[::-1]
                node.children[1].shared_X_data = node.children[0].my_X_data[order]
                node.children[1].shared_y_data = node.children[0].my_y_data[order]
                node.children[1].shared_sigma_data = node.children[0].my_sigma_data[order]
                node.children[1].n_shared_points = node.children[1].shared_X_data.shape[0]

            else:
                # Standard splitting
                node.split_training_data()

            # Retrain the child-node GPs?
            # for child in node.children:
            #     child.fit_my_GPR()

            # GP and training data of non-leaf nodes is not needed
            node.delete_data(delete_own_data=True, delete_shared_data=True)
            node.delete_my_GPR()

        self._after_update(node)
        return task


    def batch_update_tree(self, X: np.ndarray, y: np.ndarray, sigma: Union[float, np.ndarray],
                          defer_fits=False, show_progress=False) -> list:
        """Updates the tree with a batch of points, one after the other as `update_tree` does.

        Args:
            X (np.ndarray): The inputs, of shape (N, n_features).
            y (np.ndarray): The targets, of shape (N, n_outputs), or (N,) for a
                single output.
            sigma (float or np.ndarray): The uncertainties (standard deviations)
                of the targets: one number for all of them, or an array of
                shape (N,) or (N, n_outputs).
            defer_fits (bool): As for `update_tree`. Defaults to False.
            show_progress (bool): Display a progress bar. Defaults to False.

        Returns:
            The fit tasks handed out, as a list. It is empty without `defer_fits`.
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        sigma = np.array(sigma, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must have shape (N, n_features), not {X.shape}")
        if not self.first_point and X.shape[1] != self.n_features:
            raise ValueError(f"X has {X.shape[1]} features, and the points of the tree have {self.n_features}")
        N = X.shape[0]
        if y.ndim == 1:
            y = y[:, None]
        if y.shape != (N, self.n_outputs):
            raise ValueError(f"y must have one row of {self.n_outputs} target(s) for each of the {N} rows of X, not shape {y.shape}")
        if sigma.ndim == 1:
            sigma = sigma[:, None]
        if sigma.ndim == 2 and sigma.shape not in ((N, 1), (N, self.n_outputs)) or sigma.ndim > 2:
            raise ValueError(f"sigma must be a number or have shape ({N},) or ({N}, {self.n_outputs}), not {sigma.shape}")
        sigma = np.array(np.broadcast_to(sigma, (N, self.n_outputs)))

        tasks = []
        for i in tqdm(range(N), disable=not show_progress, desc="Updating the tree"):
            task = self.update_tree(X[i:i+1], y[i:i+1], sigma[i:i+1], defer_fits=defer_fits)
            if task is not None:
                tasks.append(task)
        return tasks


    def _fit_task(self, node: GPNode):
        """The fit that `node` is due as a task to hand out, or None if it is not
        due one, waits for the result of one, or has to be fitted in place."""
        if node.fit_task_id in self._fit_tasks or not node.fit_due():
            return None
        made = node.fit_task()
        if made is None:
            return None
        task, frame = made
        self._n_fit_tasks += 1
        task['id'] = node.fit_task_id = self._n_fit_tasks
        self._fit_tasks[task['id']] = (node, frame)
        return task


    def apply_fit_result(self, result: dict) -> bool:
        """Take the result of `solve_fit_task` for a task from `update_tree`.

        The leaf of the task is refitted to the points it holds now, with the
        hyperparameters found and the scaling of the task, without optimising.
        If the leaf has split since, the leaves under it are refitted. A
        result with ``hyperparameters=None``, for a fit that failed, only lets
        the leaf be handed a new task. Returns False if no leaf was refitted:
        the fit failed, the leaf was removed, the global model has a new
        version since the task was made, or the tree was saved and loaded
        since.
        """
        node, frame = self._fit_tasks.pop(result['id'], (None, None))
        if node is None or result['hyperparameters'] is None:
            return False
        # The leaves have refitted to a newer snapshot of the global model, or will with their next prediction
        if self.global_mean is not None and self.global_mean.current is not frame[1]:
            return False
        # A node that was removed from the tree has no parent, or lost its children
        leaves = [leaf for leaf in node.leaves if leaf.is_leaf and (leaf.parent is not None or leaf is self.root)]
        for leaf in leaves:
            leaf.apply_fit(result['hyperparameters'], frame)
            if self.max_memory_mb is not None:
                self._account(leaf)
        if self.max_memory_mb is not None:
            self._enforce_memory_limit(remove=False)
        return bool(leaves)


    @property
    def memory_mb(self):
        """The tree's estimate of its own size in MB (10^6 bytes): the arrays of
        its leaves' points and GPs, a fixed amount per leaf and per GP for their
        other objects, and the global model and output basis at their full size."""
        if self.first_point:
            return 0.0
        if self.max_memory_mb is not None:
            return self._memory() / 1e6
        return (self._fixed_size() + sum(self._leaf_size(leaf)[0] for leaf in self.root.leaves)) / 1e6


    def _fixed_size(self):
        """Bytes of the global model and the output basis once their samples are full."""
        size = 0
        if self.global_mean is not None:
            size += self.global_mean.max_nbytes(self.n_features, self.n_outputs)
        if self.output_basis is not None:
            size += self.output_basis.max_nbytes()
        return size


    @staticmethod
    def _leaf_size(leaf: GPNode):
        """Bytes of a leaf, and those of them that its GPs hold from their fits."""
        fitted = leaf.nbytes_fitted()
        return leaf.nbytes_points() + fitted + NODE_OVERHEAD_BYTES + GPR_OVERHEAD_BYTES * len(leaf.my_GPRs), fitted


    def _memory(self):
        return self._fixed_bytes + self._leaf_bytes


    def _start_accounting(self):
        self._fixed_bytes = self._fixed_size()
        if self._fixed_bytes == float('inf'):
            raise ValueError("max_memory_mb needs a global model of a fixed size, and this one keeps every "
                             "point of the stream: give it a reservoir_size and a store_size")
        if self._fixed_bytes >= 1e6 * self.max_memory_mb:
            raise ValueError(f"max_memory_mb={self.max_memory_mb} is less than the {self._fixed_bytes / 1e6:.3g} MB of the "
                             "global model and output basis alone: raise it, or reduce their reservoir_size")
        self._account(self.root, updated=True)


    def _account(self, leaf: GPNode, updated=False):
        """Record the current size of `leaf` and mark its GPs as the most recently
        used; with `updated`, also mark the leaf as the last one a point reached."""
        size, fitted = self._leaf_size(leaf)
        self._leaf_bytes += size - self._leaves.get(leaf, 0)
        self._leaves[leaf] = size
        if updated:
            self._leaves.move_to_end(leaf)
        self._fitted_bytes -= self._fitted.pop(leaf, 0)
        if fitted and not leaf.asleep:
            self._fitted[leaf] = fitted
            self._fitted_bytes += fitted


    def _forget(self, node: GPNode):
        """Take a node that is no longer a leaf of the tree out of the accounts."""
        self._leaf_bytes -= self._leaves.pop(node)
        self._fitted_bytes -= self._fitted.pop(node, 0)


    def _after_update(self, node: GPNode):
        """Account for the leaf a point went to, or the two it was split into, and
        bring the tree back within its memory limit."""
        if self.max_memory_mb is None:
            return
        if node.is_leaf:
            self._account(node, updated=True)
        else:
            self._forget(node)
            for child in node.children:
                self._account(child, updated=True)
        self._enforce_memory_limit(remove=True)


    def _enforce_memory_limit(self, remove: bool):
        """Put the leaves whose GPs were unused for the longest time to sleep until
        the tree is within its limit. With `remove`, once the GPs awake take less
        than `FITTED_SHARE` of the leaves' memory, remove the leaves that have
        gone longest without a point instead."""
        limit = 1e6 * self.max_memory_mb
        for_fitted = FITTED_SHARE * (limit - self._fixed_bytes)
        while self._memory() > limit:
            if self._fitted_bytes > for_fitted or not remove or len(self._leaves) == 1:
                # The least recently used leaf whose GPs can release part of their fit
                asleep = next((leaf for leaf in self._fitted if leaf.sleep()), None)
                if asleep is not None:
                    self._account(asleep)
                    logger.debug("Leaf %s put to sleep", asleep.name)
                    continue
            if not remove:
                return
            if len(self._leaves) == 1:
                raise ValueError(f"max_memory_mb={self.max_memory_mb} is too small for a single leaf")
            stalest = next(iter(self._leaves))
            self._forget(stalest)
            self._collapse_leaf(stalest)


    def _collapse_leaf(self, leaf: GPNode):
        """Remove `leaf` with its points and GP. Its sibling, a leaf or a subtree,
        takes the parent's place in the tree and so covers the parent's region."""
        parent = leaf.parent
        sibling = parent.right if leaf.is_left else parent.left
        grandparent = parent.parent
        logger.debug("Removed leaf %s", leaf.name)

        sibling.parent = grandparent
        sibling.is_left = parent.is_left
        sibling.parent_split_index = parent.parent_split_index
        sibling.parent_split_position = parent.parent_split_position
        if grandparent is None:
            self.root = sibling
        elif parent.is_left:
            grandparent.left = grandparent.children[0] = sibling
        else:
            grandparent.right = grandparent.children[1] = sibling

        # A node's name is its path from the root
        to_rename = [(sibling, parent.name)]
        while to_rename:
            node, name = to_rename.pop()
            node.name = name
            if not node.is_leaf:
                to_rename += [(node.left, name + "0"), (node.right, name + "1")]

        # Without these links the removed nodes are freed at once
        parent.left = parent.right = parent.children = None
        parent.parent = leaf.parent = None


    def fit(self, X_train: np.ndarray, y_train: np.ndarray, sigma_train: np.ndarray,
            show_progress: Optional[bool]=False, shuffle: Optional[bool]=True,
            forward_GPR_to_next_leaf: Optional[bool]=False):
        """
        Construct the binary tree by assigning a set of training samples to nodes and train the leaf-node GPs.

        Only for a tree without points; a tree that holds points takes more with
        `batch_update_tree` or `update_tree`.

        Arguments
        ----------
        X_train: np.ndarray
            The training data in feature space. Has shape=(N_train, n_features).

        y_train: np.ndarray
            The training data in target space. Has shape=(N_train, n_outputs) for multi-output,
            or (N_train, 1) for single output.

        sigma_train: np.ndarray
            Per-point uncertainties (standard deviations). Has shape=(N_train, n_outputs) for per-output
            uncertainties, or (N_train,) or (N_train, 1) for shared uncertainty. Required.

        show_progress: Optional[bool]=False
            Display a progress bar in the terminal using tqdm.

        shuffle: Optional[bool]=True
            Shuffle the training set to avoid an unbalanced tree.

        forward_GPR_to_next_leaf: Optional[bool]=False
            When training the leaf-node GPs, let the next leaf start from a copy of the trained GP
            from the previous leaf. Note: For multi-output, only the first GP is forwarded.

        The sigma calibration uses predictions of points made before they are added.
        Without a global model the leaf GPs are fitted only at the end, so there are
        none and the predicted sigma is the leaf GPs' own, also with
        use_calibrated_sigma. With a global model the leaves are fitted against each
        of its versions while the tree is built, and their predictions are used.
        """
        if not self.first_point:
            raise ValueError("fit builds a tree from scratch, and this tree holds points already. "
                             "Use batch_update_tree to add points to it, or fit a new GPTree.")
        self.n_features = X_train.shape[1]
        N = X_train.shape[0]
        self.root.init_data_set(self.n_features)

        # Ensure y_train has shape (N, n_outputs)
        if y_train.ndim == 1:
            y_train = y_train.reshape((-1, 1))

        # Ensure sigma_train has shape (N, n_outputs) or is broadcastable
        if sigma_train.ndim == 1:
            # Broadcast to all outputs
            sigma_train = sigma_train.reshape((-1, 1))
            if self.n_outputs > 1:
                sigma_train = np.tile(sigma_train, (1, self.n_outputs))
        elif sigma_train.shape[1] == 1 and self.n_outputs > 1:
            # Broadcast single column to all outputs
            sigma_train = np.tile(sigma_train, (1, self.n_outputs))

        if shuffle:
            X_train, y_train, sigma_train = resample(X_train, y_train, sigma_train, replace=False)

        # Construct the tree
        for x, y, sigma in tqdm(zip(X_train, y_train, sigma_train), total=N, disable=not show_progress, desc="Building binary tree"):
            x = x.reshape((1, x.shape[0]))
            y = y.reshape((1, self.n_outputs))  # Multi-output support
            sigma = sigma.reshape((1, self.n_outputs)) if hasattr(sigma, 'reshape') else np.array([sigma]).reshape((1, self.n_outputs))

            self.update_tree(x, y, sigma, allow_training=False)

        # Train all the leaves
        for i, leaf in tqdm(enumerate(self.root.leaves), total=len(self.root.leaves), disable=not show_progress, desc="Training"):
            leaf.fit_my_GPR(force_training=True)
            if self.max_memory_mb is not None:
                self._account(leaf)
                self._enforce_memory_limit(remove=False)
            # No prediction of a fitted GP was registered: sigma is the GP's own
            if leaf.residuals_list[0].shape[0] == 0:
                leaf.clear_pred_perf()
            if forward_GPR_to_next_leaf and i != len(self.root.leaves) - 1:
                # For multi-output, forward only the first GP
                self.root.leaves[i+1].my_GPRs[0] = deepcopy(leaf.my_GPRs[0])
                if self.n_outputs == 1:
                    self.root.leaves[i+1].my_GPR = self.root.leaves[i+1].my_GPRs[0]
        if self.max_memory_mb is not None:
            self._enforce_memory_limit(remove=True)   # sleeping leaves keep the training data of their GPs
                
                
    def _register_prediction(self, node: GPNode, x: np.ndarray, y, allow_training: bool):
        """Register how the node predicted the point before it was fitted to it: for
        the node's sigma scaler and, beyond SIGMA_DISTANCE_START, for the slope
        of sigma with the distance from the node's fitted points."""
        errors = node.register_pred_perf(x, y, only_if_fitted=not allow_training)
        if not self.use_calibrated_sigma:
            return
        if self.widen_sigma_with_distance and errors is not None:
            beyond = node.distance_from_fit(x)[0] - SIGMA_DISTANCE_START
            if beyond > 0.0:
                self._distance_errors.append((float(np.mean(errors)), beyond))
                if len(self._distance_errors) >= SIGMA_DISTANCE_MIN:
                    # the slope with which sigma covers these errors at their mean distance
                    error, distance = np.array(self._distance_errors).T
                    self._sigma_slope = max(0.0, (np.quantile(error, TARGET_COVERAGE) - 1.0) / distance.mean())
        node.update_sigma_scaler()


    def _predict_leaf(self, leaf: GPNode, x: np.ndarray):
        """The prediction of one leaf, with a sigma that grows with the distance from
        the leaf's fitted points. Under a memory limit this counts as a use of the
        leaf's GPs, and if the leaf had to be woken another one is put to sleep."""
        mu, sigma = leaf.predict(x, return_std=True, use_calibrated_sigma=self.use_calibrated_sigma)
        if self._sigma_slope > 0.0:
            beyond = np.maximum(leaf.distance_from_fit(x) - SIGMA_DISTANCE_START, 0.0)
            sigma = sigma * (1.0 + self._sigma_slope * beyond)[:, None]
        if self.max_memory_mb is not None and not self.first_point:
            self._account(leaf)
            self._enforce_memory_limit(remove=False)
        return mu, sigma


    def predict_recursive(self, X_test: np.ndarray, show_progress: Optional[bool]=False, return_leaf_names: Optional[bool]=False):
        """ 
        A predict function that uses a recursive function to collect contributing leaves.
        Should be the fastest alternative when N_train >> Nbar. 
        
        Arguments
        ---------

        X_test: np.ndarray
            The points in feature space where we'd like to predict the target function. Has shape=(N_test, n_features).

        show_progress: Optional[bool]=False
            Display a progress bar in the terminal using tqdm.

        return_leaf_names: Optional[bool]=False
            Also return a list with the names of the leaves contributing to the prediction. 

        Returns
        -------

        mean_DLGP: np.ndarray
            The posterior mean used to predict the target function. Has shape=(N_test, 1).
        
        std_DLGP: np.ndarray
            The posterior standard deviation used to quantify the uncertainty in the prediction. Has shape=(N_test, 1).
        
        leaf_names: Optional[list]
            A list with the names of the leaves contributing to the prediction. Only returned if return_leaf_names=True.
        """
        
        global sum_probs, collection_done
        def collect_leaves(x: np.ndarray, current_node: GPNode, current_prob: float):
            """Recursively traverses the tree to find relevant leaf nodes for prediction.

            This helper function is called by `predict_recursive` for a single test
            point `x`. It navigates down the tree from `current_node`, calculating
            the probability of taking each path. If a path's probability is non-zero,
            it continues traversal.

            When a leaf node is reached, it's added to the `leaves` list (in the
            outer scope of `predict_recursive`), and its accumulated probability
            (`current_prob`) is added to `pred_leaf_probs` (also in the outer scope).

            The traversal uses `sum_probs` and `collection_done` (global variables
            within `predict_recursive`'s scope) for early stopping: if the sum of
            probabilities of collected leaves reaches or exceeds 1.0, the search
            is considered complete.

            Args:
                x (np.ndarray): The single test point (1-row numpy array) for which
                    leaf nodes are being collected.
                current_node (GPNode): The GPNode from which to continue the traversal.
                current_prob (float): The accumulated probability of reaching the
                    `current_node` from the root.
            """

            global sum_probs, collection_done

            if collection_done or current_prob <= 0:
                return
            
            # Return if we have reached a leaf node
            if current_node.is_leaf:
                leaves.append(current_node)
                pred_leaf_probs.append(current_prob)

                sum_probs += current_prob
                if sum_probs >= 1:
                    collection_done = True                
                return

            # Ok, not a leaf node. Now, for both child nodes:
            # - compute the probability
            # - call this function again

            new_p = current_node.prob_func(x)[0,0]

            p0 = current_prob*(1 - new_p)
            if p0 > 0:
                collect_leaves(x, current_node.left, p0)

            p1 = current_prob*new_p
            if p1 > 0:
                collect_leaves(x, current_node.right, p1)

            # Done
            return
        

        mean_DLGP = np.zeros((X_test.shape[0], self.n_outputs))
        var_DLGP = np.zeros((X_test.shape[0], self.n_outputs))

        for i, x in tqdm(enumerate(X_test), total=X_test.shape[0], disable=not show_progress, desc="Predicting"):
            x = x.reshape((1, x.shape[0]))

            # Collect leaves
            sum_probs = 0
            collection_done = False

            leaves = []
            pred_leaf_probs = []
            collect_leaves(x, self.root, 1.0)

            # Limit the number of leaves used?
            if self.max_n_pred_leaves:
                pred_leaf_probs = np.array(pred_leaf_probs)
                ordering = np.argsort(pred_leaf_probs)[::-1]

                pred_leaf_probs = pred_leaf_probs[ordering]
                leaves = [leaves[idx] for idx in ordering]

                pred_leaf_probs = pred_leaf_probs[:self.max_n_pred_leaves]
                pred_leaf_probs = pred_leaf_probs / np.sum(pred_leaf_probs)
                pred_leaf_probs = list(pred_leaf_probs)

                leaves = leaves[:self.max_n_pred_leaves]

            # Compute joint prediciton
            # The default: mixture of experts, following the DLGP paper
            if self.aggregation == "default" or self.aggregation == "moe":

                # The spread of the leaves' means is summed apart from their
                # variances, which can be smaller than its rounding error
                second_moment = np.zeros(self.n_outputs)
                for leaf, ptilde in zip(leaves, pred_leaf_probs):

                    mu_leaf, sigma_leaf = self._predict_leaf(leaf, x)
                    # mu_leaf and sigma_leaf have shape (1, n_outputs)

                    mean_DLGP[i, :] += ptilde * mu_leaf[0, :]
                    var_DLGP[i, :] += ptilde * sigma_leaf[0, :]**2
                    second_moment += ptilde * mu_leaf[0, :]**2

                var_DLGP[i, :] += np.maximum(second_moment - mean_DLGP[i, :]**2, 0.0)

            # Generalized product of experts
            elif self.aggregation == "poe":

                # Collect individual predictions
                mus = []
                vars_ = []
                betas = pred_leaf_probs
                for leaf, ptilde in zip(leaves, pred_leaf_probs):
                    mu_leaf, sigma_leaf = self._predict_leaf(leaf, x)
                    # mu_leaf and sigma_leaf have shape (1, n_outputs)
                    mus.append(mu_leaf[0, :])
                    vars_.append(sigma_leaf[0, :]**2)
                mus = np.array(mus)  # Shape: (n_leaves, n_outputs)
                vars_ = np.array(vars_)  # Shape: (n_leaves, n_outputs)

                # Compute weighted precisions
                betas = np.array(betas)[:, None]  # shape (M, 1)
                precisions = betas / vars_        # shape (M, n_outputs)

                # Combined precision and variance
                total_precision = np.sum(precisions, axis=0)  # shape (n_outputs,)
                var_poe = 1.0 / total_precision                # shape (n_outputs,)

                # Combined mean
                weighted_means = np.sum(precisions * mus, axis=0)  # shape (n_outputs,)
                mu_poe = var_poe * weighted_means                  # shape (n_outputs,)

                # Store in result arrays
                mean_DLGP[i, :] = mu_poe
                var_DLGP[i, :] = var_poe

            # print(f"DEBUG: mean_DLGP: {mean_DLGP}  var_DLGP: {var_DLGP}")

        if return_leaf_names:
            leaf_names = [leaf.name for leaf in leaves]
            return mean_DLGP, np.sqrt(var_DLGP), leaf_names

        return mean_DLGP, np.sqrt(var_DLGP)


    def predict_loop(self, X_test: np.ndarray, show_progress: Optional[bool]=False):
        """ 
        A predict function that simply loops over all leaves.
        
        Arguments
        ---------

        X_test: np.ndarray
            The points in feature space where we'd like to predict the target function. Has shape=(N_test, n_features).

        show_progress: Optional[bool]=False
            Display a progress bar in the terminal using tqdm.

        Returns
        -------

        mean_DLGP: np.ndarray
            The posterior mean used to predict the target function. Has shape=(N_test, 1).
        
        std_DLGP: np.ndarray
            The posterior standard deviation used to quantify the uncertainty in the prediction. Has shape=(N_test, 1).
        """
        mean_DLGP = np.zeros((X_test.shape[0], self.n_outputs))
        var_DLGP = np.zeros((X_test.shape[0], self.n_outputs))
        # The spread of the leaves' means is summed apart from their variances,
        # which can be smaller than its rounding error
        second_moment = np.zeros((X_test.shape[0], self.n_outputs))

        for leaf in tqdm(self.root.leaves, disable=not show_progress, desc="Predicting"):

            ptilde = leaf.marg_prob(X_test)  # Shape: (n_test, 1)

            # A leaf predicts only at the points it can own, so the work per leaf is
            # its share of X_test and the total does not grow with the number of leaves.
            mask = ptilde[:, 0] > 0.0
            if not np.any(mask):
                continue

            mu_leaf, sigma_leaf = self._predict_leaf(leaf, X_test[mask])
            # mu_leaf and sigma_leaf have shape (n_masked, n_outputs)

            # Broadcast ptilde to match shape (n_masked, n_outputs)
            mean_DLGP[mask] += ptilde[mask] * mu_leaf
            var_DLGP[mask] += ptilde[mask] * sigma_leaf**2
            second_moment[mask] += ptilde[mask] * mu_leaf**2

        var_DLGP += np.maximum(second_moment - mean_DLGP**2, 0.0)

        return mean_DLGP, np.sqrt(var_DLGP)


    def predict_each(self, X_test):
        """ Get the prediction of each leaf node individually.  """
        res = []
        for leaf in self.root.leaves:
            ptilde = leaf.marg_prob(X_test)
            ptilde = ptilde.reshape(mean_DLGP.shape)

            mu_leaf, sigma_leaf = self._predict_leaf(leaf, X_test)
            res.append((mu_leaf, sigma_leaf, ptilde))
        return res
    

    def predict(self, X_test: np.ndarray, mode: Optional[str]='recursive', show_progress: Optional[bool]=False, return_leaf_names: Optional[bool]=False):
        """ Main predict function that calls a specific predict function according to the 'mode' argument.  """
        if mode == 'recursive':
            return self.predict_recursive(X_test, show_progress=show_progress, return_leaf_names=return_leaf_names)
        elif mode == 'loop':
            return self.predict_loop(X_test, show_progress=show_progress)
        elif mode == 'each':
            return self.predict_each(X_test, show_progress=show_progress)
        else:
            raise ValueError(f"Unknown mode argument: '{mode}'. The valid options are 'recursive', 'loop' or 'each'")


    def save(self, path: str):
        """Saves the trained GPTree object to a file using joblib.

        This method serializes the entire GPTree instance, including its
        structure (all GPNodes) and the trained GPR models within each node.

        Args:
            path (str): The file path where the GPTree object will be saved.
        """
        joblib.dump(self, path)


def solve_fit_task(task: dict) -> dict:
    """Carry out a fit task from `GPTree.update_tree(defer_fits=True)`: optimise
    the hyperparameters of its GPs. Needs no tree, so it can run in any process.
    `GPTree.apply_fit_result` takes the result, which can be pickled."""
    hyperparameters = []
    for gpr, y in zip(task['gprs'], task['y']):
        gpr.fit(task['X'], y)
        hyperparameters.append(gpr.get_hyperparameters())
    return {'id': task['id'], 'hyperparameters': hyperparameters}
