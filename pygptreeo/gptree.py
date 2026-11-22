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
import gc
import logging
import os
import pickle
from collections import OrderedDict, deque
from contextlib import contextmanager
from typing import Optional, Union

# Third-party imports
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
from pygptreeo.leaf_index import LeafIndex
from pygptreeo.neighbours import NeighbourSearch

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
    """A dynamically growing binary tree of local Gaussian process regressors,
    which learns a function from a stream of data points.

    Each leaf (a GPNode) holds the points of a region of the input space and
    the GPs fitted to them. A leaf that holds ``Nbar`` points is split in two.
    The options are described in ``__init__``.

    Attributes:
        root (GPNode): The root node of the tree.
        n_features (int): Number of inputs, taken from the first data point.
        first_point (bool): True until the first data point has arrived.
        memory_mb (float): The tree's estimate of its own size, in MB.
        n_leaves (int): Number of leaves of the tree.
        n_points_seen (int): Number of points passed to `update_tree` so far.
            The tree can hold fewer: a point may be merged, and leaves are
            removed under a memory limit.
        leaf_index (Optional[LeafIndex]): A balanced index over the leaves,
            through which the leaf of a point is found whatever the depth of
            the tree of cuts.
    """

    def __init__(self,
                 GPR: Optional[GPRegressorInterface] = None,
                 Nbar: Optional[int] = 100,
                 theta: Optional[float] = 0.0001,
                 use_calibrated_sigma: Optional[bool] = True,
                 split_dimension_criteria: Optional[str] = 'min_lengthscale',
                 splitting_strategy: Optional[str] = 'gradual',
                 max_n_pred_leaves: Optional[int] = None,
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
                 predict_from: str = 'leaf',
                 **kwargs):
        """Initializes the GPTree.

        Args:
            GPR (Optional[GPRegressorInterface]): The GP regressor used as a
                template for the GPs of the nodes. Defaults to `Default_GPR()`
                (the scikit-learn adapter).
            Nbar (Optional[int]): Number of points at which a leaf is split.
                Defaults to 100.
            theta (Optional[float]): Width of the overlap between the two
                children of a split, as a fraction of the range of the parent's
                points along the split. Defaults to 0.0001.
            use_calibrated_sigma (Optional[bool]): If True (default), the
                predicted sigma of a leaf is scaled by a factor learned from the
                leaf's latest prediction errors.
            split_dimension_criteria (Optional[str]): How the input along which
                a leaf is split is chosen. 'min_lengthscale' (default): the input
                with the smallest fitted length scale over the leaf's GPs, or as
                'max_spread' for a kernel without a length scale per input. Also
                'max_spread', 'max_variance', 'max_uncertainty' and 'random'.
            splitting_strategy (Optional[str]): 'standard': each child keeps its
                own half of the parent's points. 'gradual' (default): each child
                also receives a copy of its sibling's points, which it drops as
                its own arrive.
            max_n_pred_leaves (Optional[int]): Maximum number of leaves used for
                a prediction. Defaults to None (all).
            n_outputs (Optional[int]): Number of outputs. Defaults to 1.
            output_model (Optional[str]): How a leaf models several outputs.
                'independent' (default): one GP per output. 'shared': one GP
                with one kernel for all outputs. 'pca': one GP per component of
                a basis of the outputs that the tree learns by PCA. 'aligned':
                as 'shared', for outputs that are a curve on a grid, given in
                the order of the grid; a leaf aligns the features of its curves
                along the grid if that lowers its leave-one-out error. 'shared'
                and 'aligned' need the scikit-learn adapter. See the README and
                ``docs/design.md``.
            output_shape (Optional[tuple]): For 'aligned': the shape ``(p1, p2)``
                of the grid of outputs that are a function on a grid of two
                dimensions, flattened row by row (``values.ravel()``). Defaults
                to a curve, ``(n_outputs,)``.
            output_basis_components ('noise', int or float): For 'pca': the
                number of components kept. 'noise' (default): those above the
                level of the observation noise. An int: that many. A float in
                (0, 1): those that explain that fraction of the variance of the
                outputs. See ``OutputBasisLearner``.
            output_basis_max_components (Optional[int]): For 'pca': the largest
                number of components. Defaults to None (no limit).
            output_basis_min_points (int): For 'pca': number of points before
                the first basis is fitted. No leaf is fitted before that.
                Defaults to 50.
            output_basis_refit_every (Optional[int]): For 'pca': number of points
                between two fits of the basis. Defaults to None: a fit each time
                the number of points has doubled.
            output_basis_reservoir_size (int): For 'pca': size of the sample of
                outputs that the basis is fitted on. Defaults to 2000.
            global_mean (None, str or GlobalMeanLearner): A model of the whole
                input space whose residual the leaf GPs model. None (default):
                no such model. 'additive_gp': an ``AdditiveGPGlobalMean``.
                'net': a ``NetGlobalMean`` (needs PyTorch). Or a
                ``GlobalMeanLearner`` instance. See the README and
                ``docs/design.md``.
            global_mean_kwargs (Optional[dict]): Keyword arguments for the
                learner when ``global_mean`` is a string (e.g.
                ``dict(reservoir_size=300, n_restarts_optimizer=1)``).
            max_memory_mb (Optional[float]): The memory the tree stays within,
                in MB (10^6 bytes), by its own estimate (``memory_mb``). At the
                limit the tree puts leaves to sleep, and then removes leaves.
                See the README and ``docs/design.md``. Defaults to None (no
                limit).
            append_points_between_fits (bool): If True (default), a point that a
                leaf receives between two of its fits is added to its fitted GPs
                at once, with the hyperparameters of the last fit. If False, or
                with a GP backend that cannot append a point, it enters the GPs
                with the leaf's next fit.
            widen_sigma_with_distance (bool): If True (default), the calibrated
                sigma grows with the distance of a query from the points of its
                leaf's last fit, beyond 2 of their standard deviations (see
                ``docs/design.md``). The predictions do not change. Without
                `use_calibrated_sigma` it has no effect.
            predict_from (str): The points a prediction is conditioned on.
                'leaf' (default): those of the leaf that the query falls in.
                'neighbours': the ``Nbar`` stored points nearest to the query,
                with the kernel of the query's leaf; needs the scikit-learn
                adapter. See the README and ``docs/design.md``.
            **kwargs: Further settings of the nodes (see `NodeSettings`): for
                instance `split_position_method`, `retrain_every_n_points` and
                `use_standard_scaling`.
        """

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
            if not GPR.supports_multitarget():
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

        if predict_from not in ('leaf', 'neighbours'):
            raise ValueError(f"Unknown predict_from '{predict_from}'. Use 'leaf' or 'neighbours'.")
        self.predict_from = predict_from
        # Tree-wide nearest-neighbour search (None by default; shared by reference with every node)
        self.neighbours = None
        if predict_from == 'neighbours':
            if not GPR.supports_conditioning():
                raise ValueError("predict_from='neighbours' needs a GP backend with supports_conditioning() == True "
                                 "(e.g. the scikit-learn adapter returned by Default_GPR)")
            self.neighbours = NeighbourSearch(self, Nbar)

        self.root = GPNode(0, my_GPR=GPR, Nbar=Nbar, split_dimension_criteria=split_dimension_criteria,
                          splitting_strategy=self.splitting_strategy, n_outputs=n_outputs,
                          output_model=output_model, output_shape=output_shape, output_basis=self.output_basis,
                          global_mean=self.global_mean, neighbours=self.neighbours,
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

        # Balanced index over the leaves, built with the first point. None for
        # a tree whose cuts are not along the inputs.
        self.leaf_index = None
        # The fits that update_tree(defer_fits=True) handed out and whose results
        # are still to come: task id -> (the node the task was made for, the frame of the task)
        self._fit_tasks = {}
        self._n_fit_tasks = 0

        self.first_point = True
        self.n_points_seen = 0


    def update_tree(self, x: np.ndarray, y: Union[float, np.ndarray], sigma: Union[float, np.ndarray],
                    allow_training=True, defer_fits=False):
        """Adds a data point (x, y, sigma) to the tree.

        Follows Algorithm 1 of the DLGP article (Dividing Local Gaussian
        Processes, Lederer et al.), with some modifications. The point is stored
        in its leaf, the leaf's GPs are fitted again if a fit is due, and a leaf
        that then holds `Nbar` points is split in two. With `max_memory_mb` set,
        the tree is brought back within the limit at the end.

        Args:
            x (np.ndarray): The inputs of the point: a 1D array, or a 2D array
                with one row.
            y (float or np.ndarray): The target: a float, or an array of shape
                (n_outputs,) for several outputs.
            sigma (float or np.ndarray): The uncertainty (standard deviation) of
                the target: one number for all outputs, or one per output.
            allow_training (bool): If False, the leaf's GPs are not fitted here,
                as when `fit` builds the tree before it fits the leaves.
                Defaults to True.
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

        self.n_points_seen += 1
        x = np.asarray(x, dtype=float).reshape(1, -1)
        y = np.asarray(y, dtype=float).reshape(1, self.n_outputs)
        sigma = np.broadcast_to(np.asarray(sigma, dtype=float).reshape(1, -1), (1, self.n_outputs))

        # The first input point is used to determine self.n_features
        if self.first_point:
            self.n_features = x.size
            self.root.init_data_set(self.n_features)
            self.leaf_index = LeafIndex(self.n_features, blend=None if self.neighbours is None else self.neighbours.blend)
            self.leaf_index.add(self.root)
            if self.max_memory_mb is not None:
                self._start_accounting()
            self.first_point = False

        # The tree-global output basis ('pca') and the global model see every observation
        if self.output_basis is not None:
            self.output_basis.observe(y, sigma)
        if self.global_mean is not None:
            self.global_mean.observe(x, y, sigma)

        node = self._leaf_of(x)
        # A sleeping leaf is woken before the point is stored, so that the
        # prediction registered below is one made without the point
        node.wake()

        # A point close to a stored one is merged into it, after its prediction
        # was registered, and is not added as a new point
        nearest = node.merge_candidate(x)
        if nearest is not None:
            self._register_prediction(node, x, y, allow_training)
            node.merge_with_point(x, y, sigma, nearest)
            if self.leaf_index is not None:
                self.leaf_index.grow(node, x[0])
            self._after_update(node)
            return

        # Register the prediction performance, then add the new point: the point is
        # not yet among the stored points that a prediction may be conditioned on.
        # Without training, a node that has no fitted GP stays at its prior, which
        # says nothing about the fitted GP.
        self._register_prediction(node, x, y, allow_training)
        node.store_point(x, y, sigma)
        if self.leaf_index is not None:
            self.leaf_index.grow(node, x[0])

        # The node fits its GPs if a fit is due
        task = None
        if allow_training:
            if defer_fits:
                task = self._fit_task(node)
            # A node with a fit handed out waits for its result
            did_retrain = node.fit_task_id not in self._fit_tasks and node.fit_my_GPR()
            # Between two fits the point is added to the fitted GPs as they are
            if self.append_points_between_fits and not did_retrain:
                node.append_point(x, y, sigma)

        # A full node is split in two
        if node.n_points >= node.Nbar:
            node.split(self.theta)
            if self.leaf_index is not None:
                axis = node.cut_axis()
                if axis is None:
                    self.leaf_index = None      # the cells are no boxes
                else:
                    self.leaf_index.split(node, node.left, node.right, axis, node.split_position, 0.5 * node.overlap)

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
        X, y, sigma = self._batch(X, y, sigma)
        tasks = []
        for i in tqdm(range(X.shape[0]), disable=not show_progress, desc="Updating the tree"):
            task = self.update_tree(X[i:i+1], y[i:i+1], sigma[i:i+1], defer_fits=defer_fits)
            if task is not None:
                tasks.append(task)
        return tasks


    def _batch(self, X, y, sigma):
        """The points of a batch, in the forms that `batch_update_tree` takes, as
        arrays of shape (N, n_features), (N, n_outputs) and (N, n_outputs)."""
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
        return X, y, np.array(np.broadcast_to(sigma, (N, self.n_outputs)))


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
        self._let_leaves_sleep()
        if self.max_memory_mb is not None:
            self._enforce_memory_limit(remove=False)
        return bool(leaves)


    def __getstate__(self):
        """The nodes are saved as a list, with their links as positions in it.
        The fits handed out are not saved: their results cannot reach a loaded tree."""
        state = {**self.__dict__, '_fit_tasks': {}}
        nodes, links = [self.root], []
        for node in nodes:              # breadth first; the list grows while it is read
            if node.is_leaf:
                links.append(None)
            else:
                links.append(len(nodes))
                nodes += [node.left, node.right]
        state['_saved_nodes'], state['_saved_links'] = nodes, links
        return state


    def __setstate__(self, state):
        nodes, links = state.pop('_saved_nodes'), state.pop('_saved_links')
        self.__dict__.update(state)
        for node, first in zip(nodes, links):
            if first is not None:
                node.left, node.right = nodes[first], nodes[first + 1]
                node.children = [node.left, node.right]
                node.left.parent = node.right.parent = node


    def _leaf_of(self, x: np.ndarray):
        """The leaf that the point ``x`` (one row) is stored in. The index of the
        leaves gives it wherever the point belongs to one leaf alone. Within
        the overlap of two children the cuts are followed from the root, each
        branch taken with the probability of ``prob_func``."""
        if self.leaf_index is not None:
            leaf = self.leaf_index.locate(x[0])
            if leaf is not None:
                return leaf
        node = self.root
        while not node.is_leaf:
            node = node.children[int(np.random.binomial(1, node.prob_func(x)[0][0]))]
        return node


    def _prediction_leaves(self, x: np.ndarray):
        """The leaves of ``_leaves_of`` and their weights, of which no more than
        ``max_n_pred_leaves`` are kept, those with the largest weights."""
        leaves, weights = self._leaves_of(x)
        if self.max_n_pred_leaves:
            ordering = np.argsort(weights)[::-1][:self.max_n_pred_leaves]
            kept = np.array(weights)[ordering]
            leaves, weights = [leaves[idx] for idx in ordering], list(kept / kept.sum())
        return leaves, weights


    def _blends(self) -> bool:
        """Whether predictions pass from one leaf to the next within zones at the cuts."""
        return self.neighbours is not None and self.neighbours.blend is not None and self.leaf_index is not None


    def _leaves_of(self, x: np.ndarray):
        """The leaves that predict at the point ``x`` (one row) and their weights:
        one leaf where the point belongs to it alone, and within the overlap
        of two children the leaves on both sides, each with the product of
        the probabilities of ``prob_func`` along its branch. With zones at
        the cuts (``NEIGHBOUR_BLEND``) the leaves and weights are those of
        the zones."""
        if self.leaf_index is not None:
            if self._blends():
                leaves, weights = self.leaf_index.blend(x[0])
                if leaves:
                    return leaves, weights
            leaf = self.leaf_index.locate(x[0])
            if leaf is not None:
                return [leaf], [1.0]
        leaves, weights, total = [], [], 0.0
        stack = [(self.root, 1.0)]
        while stack and total < 1:
            node, weight = stack.pop()
            if node.is_leaf:
                leaves.append(node)
                weights.append(weight)
                total += weight
                continue
            p = node.prob_func(x)[0, 0]
            # the right child goes on the stack first, so the left one is visited first
            if weight * p > 0:
                stack.append((node.right, weight * p))
            if weight * (1 - p) > 0:
                stack.append((node.left, weight * (1 - p)))
        return leaves, weights


    @property
    def n_leaves(self):
        """Number of leaves of the tree."""
        if self.leaf_index is not None:
            return len(self.leaf_index)
        return len(self.root.leaves)


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
        if self.neighbours is not None and self.neighbours.awake is not None:
            self.neighbours.in_use.pop(node, None)


    def _after_update(self, node: GPNode):
        """Account for the leaf a point went to, or the two it was split into, let
        the leaves sleep that predictions from neighbours do not keep awake,
        and bring the tree back within its memory limit."""
        if self.max_memory_mb is not None:
            if node.is_leaf:
                self._account(node, updated=True)
            else:
                self._forget(node)
                for child in node.children:
                    self._account(child, updated=True)
        elif not node.is_leaf and self.neighbours is not None and self.neighbours.awake is not None:
            self.neighbours.in_use.pop(node, None)
        self._let_leaves_sleep()
        if self.max_memory_mb is not None:
            self._enforce_memory_limit(remove=True)


    def _let_leaves_sleep(self):
        """With predictions from neighbours, put the leaves to sleep whose GPs are
        not among the last ``NEIGHBOUR_AWAKE`` that were used. A leaf that
        predicts from its own points, or from its aligned curves, keeps its
        GPs awake."""
        search = self.neighbours
        if search is None or search.awake is None:
            return
        while len(search.in_use) > search.awake:
            leaf, _ = search.in_use.popitem(last=False)
            if leaf.is_leaf and not leaf.prefers_own and leaf.alignment is None and leaf.sleep() \
                    and self.max_memory_mb is not None and leaf in self._leaves:
                self._account(leaf)


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

        if self.leaf_index is not None:
            # the leaves along the removed cut extend over the removed leaf's cell
            axis = parent.cut_axis()
            along = [sibling]
            while along:
                node = along.pop()
                if node.is_leaf:
                    self.leaf_index.extend(node, leaf, axis, leaf.is_left)
                elif node.cut_axis() == axis:
                    along.append(node.left if leaf.is_left else node.right)
                else:
                    along += [node.left, node.right]
            self.leaf_index.remove(leaf)

        sibling.parent = grandparent
        sibling.is_left = parent.is_left
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
            show_progress: Optional[bool]=False, shuffle: Optional[bool]=True):
        """Build the tree from a set of data points and then fit each leaf once.

        Only for a tree without points; a tree that holds points takes more with
        `batch_update_tree` or `update_tree`.

        Args:
            X_train (np.ndarray): The inputs, of shape (N, n_features).
            y_train (np.ndarray): The targets, of shape (N, n_outputs), or (N,)
                for a single output.
            sigma_train (np.ndarray): The uncertainties (standard deviations) of
                the targets, of shape (N, n_outputs), or (N,) or (N, 1) for the
                same one for all outputs.
            show_progress (bool): Display a progress bar. Defaults to False.
            shuffle (bool): Shuffle the points first, to avoid an unbalanced
                tree. Defaults to True.

        The sigma calibration uses predictions of points made before they are added.
        Without a global model the leaf GPs are fitted only at the end, so there are
        none and the predicted sigma is the leaf GPs' own, also with
        use_calibrated_sigma. With a global model the leaves are fitted against each
        of its versions while the tree is built, and their predictions are used.
        """
        if not self.first_point:
            raise ValueError("fit builds a tree from scratch, and this tree holds points already. "
                             "Use batch_update_tree to add points to it, or fit a new GPTree.")
        X_train, y_train, sigma_train = self._batch(X_train, y_train, sigma_train)
        self.n_features = X_train.shape[1]
        self.root.init_data_set(self.n_features)

        if shuffle:
            X_train, y_train, sigma_train = resample(X_train, y_train, sigma_train, replace=False)

        for i in tqdm(range(X_train.shape[0]), disable=not show_progress, desc="Building binary tree"):
            self.update_tree(X_train[i:i+1], y_train[i:i+1], sigma_train[i:i+1], allow_training=False)

        for leaf in tqdm(self.root.leaves, disable=not show_progress, desc="Training"):
            leaf.fit_my_GPR(force_training=True)
            if self.max_memory_mb is not None:
                self._account(leaf)
            self._let_leaves_sleep()
            if self.max_memory_mb is not None:
                self._enforce_memory_limit(remove=False)
            # No prediction of a fitted GP was registered: sigma is the GP's own
            if leaf.residuals_list[0].shape[0] == 0:
                leaf.clear_pred_perf()
        if self.max_memory_mb is not None:
            self._enforce_memory_limit(remove=True)   # sleeping leaves keep the training data of their GPs
                
                
    def _register_prediction(self, node: GPNode, x: np.ndarray, y, allow_training: bool):
        """Register how the node predicted the point before it was fitted to it: for
        the node's sigma scaler and, beyond SIGMA_DISTANCE_START, for the slope
        of sigma with the distance from the node's fitted points."""
        prediction = None
        if self._blends() and not self.first_point and (allow_training or node._is_fitted()):
            # within the zones of several leaves the prediction to calibrate is theirs together
            leaves, weights = self._prediction_leaves(x)
            if len(leaves) > 1:
                mu, sigma = self._predict_blended(x, leaves, weights, calibrated=False)
                prediction = mu[None, :], sigma[None, :]
        errors = node.register_pred_perf(x, y, only_if_fitted=not allow_training, prediction=prediction)
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


    def _predict_blended(self, x: np.ndarray, leaves, weights, calibrated: bool = True):
        """The prediction at one query (one row) within the zones of several
        leaves: from its neighbours with the kernels of the leaves mixed, the
        mixture of the leaves' predictions where that cannot be done, or the
        two mixed where the neighbours fit the mixed kernel badly
        (``NeighbourSearch.share``). Returns the mean and sigma per output;
        without ``calibrated`` sigma is that of the GPs."""
        weights = np.asarray(weights, dtype=float)
        mixed = self.neighbours.predict_mixed(x, leaves, weights)
        share = 0.0 if mixed is None else mixed[2]
        if share < 1.0:
            parts = [self._predict_leaf(leaf, x, calibrated) for leaf in leaves]
            mu_leaves = sum(weight * mu_leaf[0] for weight, (mu_leaf, _) in zip(weights, parts))
            variance = sum(weight * (sigma_leaf[0] ** 2 + (mu_leaf[0] - mu_leaves) ** 2)
                           for weight, (mu_leaf, sigma_leaf) in zip(weights, parts))
            if mixed is None:
                return mu_leaves, np.sqrt(variance)
        mu, sigma = mixed[:2]
        if calibrated and self.use_calibrated_sigma:
            sigma = sigma * sum(weight * np.asarray(leaf.sigma_scalers) for weight, leaf in zip(weights, leaves))
            if self._sigma_slope > 0.0:
                beyond = [max(leaf.distance_from_fit(x)[0] - SIGMA_DISTANCE_START, 0.0) for leaf in leaves]
                sigma = sigma * (1.0 + self._sigma_slope * float(np.dot(weights, beyond)))
        if self.max_memory_mb is not None:
            for leaf in leaves:
                self._account(leaf)
            self._enforce_memory_limit(remove=False)
        if share < 1.0:
            mu, sigma = self.neighbours.mix(share, mu, sigma, mu_leaves, np.sqrt(variance))
        return mu, sigma


    def _predict_leaf(self, leaf: GPNode, x: np.ndarray, calibrated: bool = True):
        """The prediction of one leaf, with a sigma that grows with the distance from
        the leaf's fitted points. Under a memory limit this counts as a use of the
        leaf's GPs, and if the leaf had to be woken another one is put to sleep.
        Without ``calibrated`` sigma is that of the GPs."""
        mu, sigma = leaf.predict(x, use_calibrated_sigma=calibrated and self.use_calibrated_sigma)
        if calibrated and self._sigma_slope > 0.0:
            beyond = np.maximum(leaf.distance_from_fit(x) - SIGMA_DISTANCE_START, 0.0)
            sigma = sigma * (1.0 + self._sigma_slope * beyond)[:, None]
        if self.max_memory_mb is not None and not self.first_point:
            self._account(leaf)
            self._enforce_memory_limit(remove=False)
        return mu, sigma


    def predict_recursive(self, X_test: np.ndarray, show_progress: Optional[bool]=False):
        """Predict at the points ``X_test`` one at a time, each from the leaves it
        can belong to. The cost of a point does not grow with the number of leaves.

        Args:
            X_test (np.ndarray): The points, of shape (N_test, n_features).
            show_progress (bool): Display a progress bar. Defaults to False.

        Returns:
            The predicted mean and standard deviation, each of shape
            (N_test, n_outputs).
        """
        
        mean_DLGP = np.zeros((X_test.shape[0], self.n_outputs))
        var_DLGP = np.zeros((X_test.shape[0], self.n_outputs))

        for i, x in tqdm(enumerate(X_test), total=X_test.shape[0], disable=not show_progress, desc="Predicting"):
            x = x.reshape((1, x.shape[0]))
            self._let_leaves_sleep()

            leaves, pred_leaf_probs = self._prediction_leaves(x)

            if len(leaves) > 1 and self._blends():
                mu, sigma = self._predict_blended(x, leaves, pred_leaf_probs)
                mean_DLGP[i, :], var_DLGP[i, :] = mu, sigma ** 2
                continue

            # Mixture of experts, following the DLGP paper.
            # mu_leaf and sigma_leaf have shape (1, n_outputs)
            parts = [(ptilde,) + self._predict_leaf(leaf, x) for leaf, ptilde in zip(leaves, pred_leaf_probs)]
            for ptilde, mu_leaf, _ in parts:
                mean_DLGP[i, :] += ptilde * mu_leaf[0, :]
            # The variance of the mixture: that of each leaf, and the spread of
            # their means about the mixture's. In this form a small sigma
            # survives a large mean.
            for ptilde, mu_leaf, sigma_leaf in parts:
                var_DLGP[i, :] += ptilde * (sigma_leaf[0, :]**2 + (mu_leaf[0, :] - mean_DLGP[i, :])**2)

        self._let_leaves_sleep()
        return mean_DLGP, np.sqrt(var_DLGP)


    def predict_loop(self, X_test: np.ndarray, show_progress: Optional[bool]=False):
        """Predict at the points ``X_test`` leaf by leaf: each leaf predicts at once
        for all the points it can own. The cost grows with the number of leaves.

        Args:
            X_test (np.ndarray): The points, of shape (N_test, n_features).
            show_progress (bool): Display a progress bar. Defaults to False.

        Returns:
            The predicted mean and standard deviation, each of shape
            (N_test, n_outputs).
        """
        mean_DLGP = np.zeros((X_test.shape[0], self.n_outputs))
        var_DLGP = np.zeros((X_test.shape[0], self.n_outputs))

        parts = []
        for leaf in tqdm(self.root.leaves, disable=not show_progress, desc="Predicting"):

            ptilde = leaf.marg_prob(X_test)  # Shape: (n_test, 1)

            # A leaf predicts only at the points it can own, so the work per leaf is
            # its share of X_test and the total does not grow with the number of leaves.
            mask = ptilde[:, 0] > 0.0
            if not np.any(mask):
                continue

            mu_leaf, sigma_leaf = self._predict_leaf(leaf, X_test[mask])
            # mu_leaf and sigma_leaf have shape (n_masked, n_outputs)

            mean_DLGP[mask] += ptilde[mask] * mu_leaf
            parts.append((mask, ptilde[mask], mu_leaf, sigma_leaf))

        # The variance of the mixture, in the form in which a small sigma survives a large mean
        for mask, ptilde, mu_leaf, sigma_leaf in parts:
            var_DLGP[mask] += ptilde * (sigma_leaf**2 + (mu_leaf - mean_DLGP[mask])**2)

        return mean_DLGP, np.sqrt(var_DLGP)


    def predict(self, X_test: np.ndarray, mode: Optional[str]='recursive', show_progress: Optional[bool]=False):
        """Predict the mean and standard deviation at the points ``X_test``, of shape
        (N_test, n_features): with `predict_recursive` (``mode='recursive'``, the
        default) or `predict_loop` (``mode='loop'``)."""
        if mode == 'recursive':
            return self.predict_recursive(X_test, show_progress=show_progress)
        elif mode == 'loop':
            if self._blends():
                # the zones at the cuts are those of the leaf index, which 'recursive' follows
                return self.predict_recursive(X_test, show_progress=show_progress)
            result = self.predict_loop(X_test, show_progress=show_progress)
            self._let_leaves_sleep()
            return result
        else:
            raise ValueError(f"Unknown mode argument: '{mode}'. The valid options are 'recursive' or 'loop'")


    def save(self, path: str):
        """Save the tree to a file, which `GPTree.load` reads.

        The leaves are put to sleep first (see `GPNode.sleep`), so the file
        holds no Cholesky factors: a leaf computes its factor again when it
        is next used. The file is written under a temporary name and then
        renamed, so a process that reads `path` meanwhile gets the previous
        file.

        Args:
            path (str): The file path where the tree will be saved.
        """
        for leaf in self.root.leaves:
            if leaf.sleep() and self.max_memory_mb is not None:
                self._account(leaf)
        temporary = f"{path}.tmp"
        with _gc_paused(), open(temporary, 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)


    @staticmethod
    def load(path: str) -> 'GPTree':
        """Load a tree from a file that `save` wrote. Its leaves are asleep."""
        with _gc_paused(), open(path, 'rb') as file:
            return pickle.load(file)


    def release(self):
        """Break the links between the tree's nodes, so that its memory is freed
        as soon as the tree is dropped. The tree cannot be used afterwards.

        The nodes of a tree refer to each other in loops, so without this a
        dropped tree keeps its memory until Python's garbage collector finds
        it. For a process that replaces its tree by a newer one many times,
        as with `load`.
        """
        nodes = [self.root]
        for node in nodes:              # the list grows while it is read
            if not node.is_leaf:
                nodes += node.children
            node.parent = node.left = node.right = node.children = None
        if self.leaf_index is not None:
            self.leaf_index.release()
        if self.neighbours is not None:
            self.neighbours.tree = None
            self.neighbours.in_use.clear()
        self.root = self.leaf_index = None


@contextmanager
def _gc_paused():
    """The garbage collector is off within the block: it takes much of the time
    of pickling and unpickling the many objects of a large tree."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def solve_fit_task(task: dict) -> dict:
    """Carry out a fit task from `GPTree.update_tree(defer_fits=True)`: optimise
    the hyperparameters of its GPs. Needs no tree, so it can run in any process.
    `GPTree.apply_fit_result` takes the result, which can be pickled."""
    hyperparameters = []
    for gpr, y in zip(task['gprs'], task['y']):
        gpr.fit(task['X'], y)
        hyperparameters.append(gpr.get_hyperparameters())
    return {'id': task['id'], 'hyperparameters': hyperparameters}
