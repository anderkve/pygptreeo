"""GPNode: Individual node implementation for GPTree.

This module implements the GPNode class, which represents a single node in the
GPTree structure. Each node manages its own training data, Gaussian Process model,
and handles splitting logic when it becomes too full.

Key responsibilities:
    - Store and manage local training data
    - Train and maintain a local GP regressor
    - Implement probabilistic splitting functions
    - Handle node splitting into child nodes
    - Provide local predictions with calibrated uncertainty
"""

# Standard library imports
import logging
from collections import deque
from copy import deepcopy
from typing import Optional, Union

# Third-party imports
import numpy as np
from binarytree import Node
from sklearn.preprocessing import StandardScaler

# Local imports
from pygptreeo.gp_interface import GPRegressorInterface
from pygptreeo.output_alignment import output_warp
from pygptreeo.output_basis import OutputBasisLearner
from pygptreeo.global_mean import GlobalMeanLearner
from pygptreeo.neighbours import NeighbourSearch

logger = logging.getLogger(__name__)

# Module-level constants
DEFAULT_OVERLAP = 0.001  # Default initial overlap for node boundaries
DEFAULT_N_POINTS_PRED_PERF = 25  # Number of recent predictions tracked for calibration
DEFAULT_SIGMA_SCALER = 10.0  # Initial sigma scaling factor for uncertainty calibration
TARGET_COVERAGE = 0.68  # Target coverage for calibrated uncertainty (1 sigma)
ALIGNMENT_MARGIN = 1.5  # an alignment is kept if it lowers the leave-one-out error of a leaf's curves by this factor
ALIGNMENT_MAX_WAIT = 8  # after repeated rejections an alignment is tried at one fit in this many
SIGMA_PROFILE_FLOOR = 1e-6  # smallest sigma of an output of a shared-kernel GP, relative to the RMS over its outputs
N_SCORES = 50  # Recent points over which a leaf compares its own GPs with the neighbours' prediction
MIN_SCORES = 10  # No comparison with fewer points
OWN_MARGIN = 0.9  # A leaf predicts from its own points while their RMS error is below this times the neighbours'
NODE_LINKS = ('left', 'right', 'parent', 'children')  # Saved by the tree, not by the node


class GPNode(Node):
    """Represents a node within the GPTree structure.

    Each GPNode is responsible for a specific region of the input space.
    It holds the training data relevant to this region, manages its own
    Gaussian Process Regressors (my_GPRs) to model the data, handles the
    splitting process into child nodes when it becomes too full or complex,
    and makes local predictions within its domain.

    Attributes:
        my_GPRs (list[GPRegressorInterface]): The GPR instances of this node.
        Nbar (int): The maximum number of training points this node can hold
            before it attempts to split.
        my_X_data (numpy.ndarray): Training data features for this node.
        my_y_data (numpy.ndarray): Training data targets for this node.
        split_index (int): The feature index used for splitting this node.
        split_position (float): The value at which the split occurs along the
            `split_index` feature.
        children (list[GPNode]): A list containing the left and right child
            nodes after a split. None if it's a leaf node.
        is_leaf (bool): True if the node is a leaf node (has no children),
            False otherwise.
        parent (GPNode): The parent node in the tree. None for the root node.
        value (int): Inherited from binarytree.Node, stores n_points.
        name (str): A string identifier for the node (e.g., "0", "01", "010").
        split_position_method (str): Method used to determine split point.
        retrain_every_n_points (int): Frequency of GPR retraining.
        overlap (float): The size of the overlapping region with sibling nodes.
        asleep (bool): True while the node's fitted GPs are partly released to
            save memory (see `sleep`). The next prediction restores them.
    """

    def __init__(self, *args,
                 my_GPR: GPRegressorInterface,
                 Nbar: Optional[int] = 100,
                 split_position_method='median',
                 retrain_every_n_points=100,
                 name="0",
                 split_dimension_criteria='min_lengthscale',
                 splitting_strategy: Optional[str] = 'gradual',
                 use_standard_scaling: Optional[bool] = True,
                 use_hyperparameter_inheritance: Optional[bool] = False,
                 enable_point_merging: Optional[bool] = False,
                 merge_distance_threshold: Optional[float] = 0.01,
                 min_points_before_merging: Optional[int] = 10,
                 n_outputs: Optional[int] = 1,
                 output_model: Optional[str] = 'independent',
                 output_shape: Optional[tuple] = None,
                 output_basis: Optional[OutputBasisLearner] = None,
                 global_mean: Optional[GlobalMeanLearner] = None,
                 neighbours: Optional[NeighbourSearch] = None,
                 my_GPRs: Optional[list] = None):
        """Initializes a GPNode.

        Args:
            *args: Arguments passed to the `binarytree.Node` parent class constructor.
                Typically, the first argument is the initial `value` for the node,
                which corresponds to `n_points`.
            my_GPR (GPRegressorInterface): The Gaussian Process Regressor instance
                to be used by this node.
            Nbar (Optional[int]): The maximum number of training points this node
                can hold before it considers splitting. Defaults to 100.
            split_position_method (str): The method used to determine the split
                position when the node splits. Valid options include 'median',
                'mean', 'random', 'randomchoice'. Defaults to 'median'.
            retrain_every_n_points (int): Specifies how many new data points
                should be accumulated in the buffer before the node's GPR is
                retrained. Defaults to 1.
            name (str): A string identifier for the node, often representing its
                path from the root (e.g., "0", "01"). Defaults to "0".
            split_dimension_criteria (str): The method used to determine the
                split dimension. Valid options: 'max_spread' (split on dimension with
                largest range), 'max_variance' (split on dimension with highest variance),
                'max_uncertainty' (split on dimension where GP is most uncertain),
                'min_lengthscale' (split on dimension with the smallest fitted ARD
                length scale, i.e. where the GP says the function varies fastest;
                pooled over all of the node's GPs for multi-output; falls back to
                'max_spread' when no per-dimension length scales are available),
                'random' (random dimension). Defaults to 'min_lengthscale'.
            use_standard_scaling (Optional[bool]): If True, standardizes both X and y
                data before fitting the GP and inverse transforms predictions.
                Defaults to False. Cannot be used together with use_hyperparameter_inheritance.
            use_hyperparameter_inheritance (Optional[bool]): If True, child nodes inherit
                their parent's optimized kernel hyperparameters as starting values.
                Defaults to True. Cannot be used together with use_standard_scaling.
            enable_point_merging (Optional[bool]): If True, merges new points with nearby
                existing points via weighted averaging. Defaults to False.
            merge_distance_threshold (Optional[float]): Distance threshold below which points
                are merged. Measured as Euclidean distance in input space. Defaults to 0.01.
            min_points_before_merging (Optional[int]): Minimum number of points required
                before merging starts. Defaults to 10.
            n_outputs (Optional[int]): Number of output dimensions. Defaults to 1 (single output).
            output_model (Optional[str]): How several outputs are modelled (ignored for
                n_outputs=1). 'independent': one GP per output, each with its own kernel
                hyperparameters and per-output standardisation. 'shared': one GP with a
                single shared kernel for all outputs (2-D target) and a common output
                scale; needs a backend with ``supports_multitarget()``. 'pca': the
                outputs are projected on a tree-global linear basis (``output_basis``)
                and one GP is trained per basis score; predictions are mapped back to
                output space with propagated variances. 'aligned': as 'shared', for
                outputs that are a curve on a grid; the node also learns a warp of
                the grid that aligns the features of its curves, and keeps it if
                it lowers their leave-one-out error (see ``output_alignment``).
                Defaults to 'independent'.
            output_basis (Optional[OutputBasisLearner]): The tree-global output basis
                learner, shared by reference between all nodes. Required for 'pca'.
            global_mean (Optional[GlobalMeanLearner]): The tree-wide global model
                learner, shared by reference between all nodes, or None (default).
                When present, the node's GPs model the residual of the learner's
                current snapshot: the snapshot is subtracted from the raw targets at
                fit time, added back at predict time, and remembered per node. A
                node that is asked to predict while a newer snapshot exists refits
                against it first (see ``predict``).
            neighbours (Optional[NeighbourSearch]): The tree's search for the
                stored points nearest to a query, with which the node predicts
                from those points (see ``predict``), or None (default).
            my_GPRs (Optional[list]): Explicit list of GP regressors for this node
                (used when creating children: one clone per parent GP). If None, the
                list is built from ``my_GPR`` according to ``output_model``.
        """

        super().__init__(*args)

        # Validate that both use_standard_scaling and use_hyperparameter_inheritance are not enabled
        if use_standard_scaling and use_hyperparameter_inheritance:
            raise ValueError(
                "Cannot use both use_standard_scaling=True and use_hyperparameter_inheritance=True. "
                "Standard scaling changes the coordinate system between parent and child nodes, "
                "making hyperparameter inheritance incompatible. Please choose one or the other."
            )

        self.Nbar = Nbar
        self.n_outputs = n_outputs

        if output_model not in ('independent', 'shared', 'pca', 'aligned'):
            raise ValueError(f"Unknown output_model '{output_model}'. Use 'independent', 'shared', 'pca' or 'aligned'.")
        if n_outputs == 1:
            output_model = 'independent'  # the modes only differ for several outputs
        if output_model == 'pca' and output_basis is None:
            raise ValueError("output_model='pca' requires an OutputBasisLearner passed as output_basis")
        self.output_model = output_model
        self.output_shape = tuple(output_shape) if output_shape is not None else (n_outputs,)   # the grid of the outputs ('aligned')
        self.output_basis = output_basis   # tree-global, shared by reference across all nodes
        self._fitted_basis = None          # the OutputBasis this node's GPs were fitted with ('pca')
        self.z_scaler = None               # (shift, scale) per basis score, fitted per leaf ('pca')
        self.y_common_scaler = None        # (mu, scale) of the common output scaling ('shared', 'aligned')
        self.alignment = None              # the AlignedCurves the node's GP was fitted with, if any ('aligned')
        self._alignment_rejections = 0     # alignments rejected in a row, and the fits to pass before the next try
        self._alignment_wait = 0
        self.sigma_profile = None          # factor per output on the sigma of a shared-kernel GP ('shared', 'aligned')

        # Tree-wide global model (shared by reference; None when the tree has none) and
        # the snapshot of it that this node's GPs were fitted against.
        self.global_mean = global_mean
        self._fitted_global = None

        # The tree's neighbour search, the squared errors of the neighbours' and of
        # the node's own prediction at the last points it received, whether its own
        # GPs predict for now, and the variance of its last predictions
        self.neighbours = neighbours
        self._scores = deque(maxlen=N_SCORES)
        self.prefers_own = False
        self._variance_unit = None

        # Mean and standard deviation per input of the points of the last fit
        self.fit_mean = None
        self.fit_std = None

        # GP regressors: one per output ('independent'), one for all outputs
        # ('shared', 'aligned'), or one per basis score ('pca'; the list is sized
        # at the first fit, once the basis rank is known). Children receive their
        # own list via my_GPRs: one clone per parent GP.
        if my_GPRs is not None:
            self.my_GPRs = list(my_GPRs)
        elif n_outputs == 1 or output_model in ('shared', 'pca', 'aligned'):
            self.my_GPRs = [my_GPR]
        else:
            self.my_GPRs = [deepcopy(my_GPR) for _ in range(n_outputs)]

        self.parent = None
        self.children = None

        self.split_position_method = split_position_method
        self.retrain_every_n_points = retrain_every_n_points
        self.split_dimension_criteria = split_dimension_criteria
        self.splitting_strategy = splitting_strategy
        self.use_standard_scaling = use_standard_scaling
        self.use_hyperparameter_inheritance = use_hyperparameter_inheritance

        # Point merging parameters
        self.enable_point_merging = enable_point_merging
        self.merge_distance_threshold = merge_distance_threshold
        self.min_points_before_merging = min_points_before_merging

        # Number of merges per stored point
        self.merge_counts = None  # Will be initialized with data

        # Standard scalers for X and y (fitted during GP training), one y scaler per output
        self.X_scaler = None
        self.y_scalers = [None for _ in range(n_outputs)]

        self.is_left = None
        self.is_leaf = True
        
        self.n_points = 0
        self.n_points_since_retrain = 0
        self.asleep = False
        # The last fit task that GPTree.update_tree(defer_fits=True) handed out for
        # this node or, until it has one of its own, for the node it was split from
        self.fit_task_id = None

        self.my_X_data = None
        self.my_y_data = None
        self.n_features = None

        self.shared_X_data = None
        self.shared_y_data = None
        self.n_shared_points = 0

        self.split_index = 0       # 'j' in the DLGP article
        self.split_direction = None  # the direction of a split that is not along an input, as weights of the inputs
        self.split_position = 0.0  # 's' in the DLGP article
        self.overlap = DEFAULT_OVERLAP  # 'o' in the DLGP article
        self.name = name

        self.n_points_pred_perf = DEFAULT_N_POINTS_PRED_PERF
        # The residuals and sigmas of the last registered predictions, and the sigma scaler, per output
        self.residuals_list = [np.array([]) for _ in range(n_outputs)]
        self.sigma_preds_list = [np.array([]) for _ in range(n_outputs)]
        self.sigma_scalers = [DEFAULT_SIGMA_SCALER for _ in range(n_outputs)]

        logger.debug("Created node %s", self.name)


    def __getstate__(self):
        """A node is saved without its links to other nodes: the tree saves
        them, so that saving does not recurse through its levels. Its scores
        are saved as one array."""
        state = self.__dict__.copy()
        for link in NODE_LINKS:
            state.pop(link, None)
        state['_scores'] = np.array(state['_scores'])
        return state


    def __setstate__(self, state):
        self.__dict__.update(state)
        self._scores = deque(state['_scores'], maxlen=N_SCORES)
        # links that the tree has already restored are kept
        for link in NODE_LINKS:
            self.__dict__.setdefault(link, None)


    # Override the "value" attribute of Node parent class
    @property
    def n_points(self):
        """int: The number of training points currently held by this node."""
        return self.value

    
    # such that the value of a node is the number of training points
    @n_points.setter
    def n_points(self, value):
        """Sets the number of training points for this node.

        This also updates the `value` attribute inherited from the
        `binarytree.Node` parent class, which is used for display
        purposes in the tree structure.

        Args:
            value (int): The new number of training points.
        """
        self.value = value


    def init_data_set(self, n_features: int):
        """ Initialize the training set of the node. """
        self.n_features = n_features

        self.my_X_data = np.array([]).reshape((0, n_features))
        self.my_y_data = np.array([]).reshape((0, self.n_outputs))  # Multi-output support
        self.my_sigma_data = np.array([]).reshape((0, self.n_outputs))  # Per-point uncertainties per output

        self.shared_X_data = np.array([]).reshape((0, n_features))
        self.shared_y_data = np.array([]).reshape((0, self.n_outputs))  # Multi-output support
        self.shared_sigma_data = np.array([]).reshape((0, self.n_outputs))  # Shared uncertainties per output

        self.n_points_since_retrain = 0
        self.n_points = 0
        self.n_shared_points = 0

        # Initialize merge tracking - each point tracks how many merges it represents
        self.merge_counts = np.array([]).reshape((0, 1))


    def generate_children(self, n_features: int):
        """ Grow the GPtree by adding two GPNodes as children of the current GPNode. """

        # Settings that will be passed on to the child nodes
        node_config_kwargs = {
            'Nbar': self.Nbar,
            'split_position_method': self.split_position_method,
            'retrain_every_n_points': self.retrain_every_n_points,
            'split_dimension_criteria': self.split_dimension_criteria,
            'splitting_strategy': self.splitting_strategy,
            'use_standard_scaling': self.use_standard_scaling,
            'use_hyperparameter_inheritance': self.use_hyperparameter_inheritance,
            'enable_point_merging': self.enable_point_merging,
            'merge_distance_threshold': self.merge_distance_threshold,
            'min_points_before_merging': self.min_points_before_merging,
            'n_outputs': self.n_outputs,  # Pass n_outputs to children
            'output_model': self.output_model,
            'output_shape': self.output_shape,
            'output_basis': self.output_basis,
            'global_mean': self.global_mean,
            'neighbours': self.neighbours,
        }

        # Each child gets its own clone of every parent GP (one per output / score)
        # and predicts with them until its first retrain.
        left_gprs = [gpr.clone() for gpr in self.my_GPRs]
        right_gprs = [gpr.clone() for gpr in self.my_GPRs]
        node_class = type(self)
        self.left = node_class(0, my_GPR=left_gprs[0], my_GPRs=left_gprs, name=self.name + "0", **node_config_kwargs)
        self.right = node_class(0, my_GPR=right_gprs[0], my_GPRs=right_gprs, name=self.name + "1", **node_config_kwargs)
        
        self.left.is_left = True
        self.right.is_left = False

        self.children = [self.left, self.right]
        self.is_leaf = False

        for child in self.children:
            child.parent = self
            child.init_data_set(n_features)

        # Copy the registered prediction performance over to the child nodes
        self.left.residuals_list = [arr.copy() for arr in self.residuals_list]
        self.right.residuals_list = [arr.copy() for arr in self.residuals_list]
        self.left.sigma_preds_list = [arr.copy() for arr in self.sigma_preds_list]
        self.right.sigma_preds_list = [arr.copy() for arr in self.sigma_preds_list]
        self.left.sigma_scalers = self.sigma_scalers.copy()
        self.right.sigma_scalers = self.sigma_scalers.copy()

        # Inherit parent's optimized kernel hyperparameters if enabled
        # This gives children a warm-start for their GP optimization
        if self.use_hyperparameter_inheritance:
            # For each GP (output or score), copy the trained kernel to the children
            for i, gpr in enumerate(self.my_GPRs):
                if gpr.is_trained():
                    # Copy the parent's optimized kernel to the children
                    # This preserves learned length scales, amplitudes, etc.
                    trained_kernel = gpr.get_kernel()
                    self.left.my_GPRs[i].set_kernel(deepcopy(trained_kernel))
                    self.right.my_GPRs[i].set_kernel(deepcopy(trained_kernel))
            logger.debug("Inherited hyperparameters for %d output(s) from node %s to children %s and %s",
                         self.n_outputs, self.name, self.left.name, self.right.name)

        # Copy scalers so children can use parent's GP correctly
        # The children get a copy of the parent's trained GP, which was trained on
        # scaled data using the parent's scalers. So children need the same scalers
        # to properly scale inputs before prediction.
        if self.use_standard_scaling and self.X_scaler is not None:
            self.left.X_scaler = deepcopy(self.X_scaler)
            self.right.X_scaler = deepcopy(self.X_scaler)
            # Copy y_scalers (list for multi-output)
            self.left.y_scalers = [deepcopy(scaler) if scaler is not None else None for scaler in self.y_scalers]
            self.right.y_scalers = [deepcopy(scaler) if scaler is not None else None for scaler in self.y_scalers]
            self.left.y_common_scaler = deepcopy(self.y_common_scaler)
            self.right.y_common_scaler = deepcopy(self.y_common_scaler)

        # Until their first retrain the children predict with the parent's GPs, so
        # they also need the basis, per-score standardisation and global-model
        # snapshot those GPs were fitted with.
        self.left._fitted_basis = self._fitted_basis
        self.right._fitted_basis = self._fitted_basis
        self.left.z_scaler = deepcopy(self.z_scaler)
        self.right.z_scaler = deepcopy(self.z_scaler)
        self.left._fitted_global = self._fitted_global
        self.right._fitted_global = self._fitted_global
        self.left.alignment = self.right.alignment = self.alignment
        self.left._alignment_rejections = self.right._alignment_rejections = self._alignment_rejections
        self.left._alignment_wait = self.right._alignment_wait = self._alignment_wait
        self.left.sigma_profile = self.right.sigma_profile = self.sigma_profile
        self.left.fit_mean = self.right.fit_mean = self.fit_mean
        self.left.fit_std = self.right.fit_std = self.fit_std
        self.left.fit_task_id = self.right.fit_task_id = self.fit_task_id
        if self.neighbours is not None:
            for child in self.children:
                child._scores = deque(self._scores, maxlen=N_SCORES)
                child.prefers_own = self.prefers_own
                child._used()       # its GPs are copies of the parent's, fit and all


    def store_point(self, x: np.ndarray, y: Union[float, np.ndarray], sigma: Union[float, np.ndarray],
                    increment_buffer=True):
        """ Add a single data point to the node. A node with shared points from
        gradual splitting drops the last of them for it.

        Parameters
        ----------
        x : np.ndarray
            Input features (shape: 1 x n_features)
        y : float or np.ndarray
            Target value(s). For single output: float or shape (1, 1) or (1,)
            For multi-output: shape (1, n_outputs) or (n_outputs,)
        sigma : float or np.ndarray
            Uncertainty (standard deviation) for this point.
            Can be a scalar (same for all outputs) or array (per-output)
        increment_buffer : bool
            Whether to increment the retrain buffer
        """
        # Ensure y has the right shape (1, n_outputs)
        if isinstance(y, (int, float, np.floating)):
            y = np.array([[y]])  # Single output case
        else:
            y = np.atleast_2d(y)
            if y.shape[0] != 1:
                y = y.reshape(1, -1)

        # Ensure sigma has the right shape (1, n_outputs)
        if isinstance(sigma, (int, float, np.floating)):
            # Scalar sigma - broadcast to all outputs
            sigma = np.full((1, self.n_outputs), sigma)
        else:
            sigma = np.atleast_2d(sigma)
            if sigma.shape[0] != 1:
                sigma = sigma.reshape(1, -1)
            # If sigma has only one value but we have multiple outputs, broadcast
            if sigma.shape[1] == 1 and self.n_outputs > 1:
                sigma = np.tile(sigma, (1, self.n_outputs))

        # Note: Points are added to the beginning of the arrays (using np.vstack)
        self.my_X_data = np.vstack((x, self.my_X_data))
        self.my_y_data = np.vstack((y, self.my_y_data))
        self.my_sigma_data = np.vstack((sigma, self.my_sigma_data))
        # Track merge count for this new point (starts at 0)
        if self.merge_counts is not None:
            self.merge_counts = np.vstack((np.array([[0]]), self.merge_counts))
        self.n_points += 1
        if increment_buffer:
            self.n_points_since_retrain += 1

        if self.n_shared_points > 0:
            self.shared_X_data = np.delete(self.shared_X_data, -1, axis=0)
            self.shared_y_data = np.delete(self.shared_y_data, -1, axis=0)
            self.shared_sigma_data = np.delete(self.shared_sigma_data, -1, axis=0)
            self.n_shared_points -= 1


    def split_training_data(self):
        """ Assign the training samples of a node to its child nodes. """
        for x, y, sigma in zip(self.my_X_data, self.my_y_data, self.my_sigma_data):
            x = x.reshape((1, x.shape[0]))
            y = y.reshape((1, self.n_outputs))  # Multi-output support
            sigma = sigma.reshape((1, self.n_outputs))  # Multi-output support
            child = self.children[int(np.random.binomial(1, self.prob_func(x)[0][0]))]
            child.store_point(x, y, sigma, increment_buffer=False)

        
    def delete_data(self):
        """Deletes the node's own and shared points.

        This is typically called on a parent node after its data has been
        successfully split and passed down to its children. This helps to
        reduce memory consumption as the tree grows, as only leaf nodes
        or nodes about to be split actively need to store their full datasets.
        """
        del self.my_X_data
        del self.my_y_data
        del self.my_sigma_data

        del self.shared_X_data
        del self.shared_y_data
        del self.shared_sigma_data


    def delete_my_GPR(self):
        """Deletes the Gaussian Process Regressor instances (my_GPRs) from the node.

        This method is typically called on a node after it has been split and
        is no longer a leaf node. Non-leaf nodes usually do not need to
        maintain their GPR model once their responsibilities have been passed
        to their children, thus saving memory.
        """
        del self.my_GPRs
        self.alignment = None


    def find_nearest_neighbor(self, x: np.ndarray):
        """Finds the nearest neighbor to x in the node's training data.

        Args:
            x (np.ndarray): Input point to find nearest neighbor for (shape: 1 x n_features)

        Returns:
            tuple: (index, distance) of nearest neighbor, or (None, None) if no data
        """
        if self.n_points == 0:
            return None, None

        # Compute Euclidean distances to all points
        distances = np.linalg.norm(self.my_X_data - x, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_dist = distances[nearest_idx]

        return nearest_idx, nearest_dist


    def merge_with_point(self, x: np.ndarray, y: Union[float, np.ndarray],
                         sigma: Union[float, np.ndarray], nearest_idx: int):
        """Merges new point (x, y, sigma) with existing point at nearest_idx via inverse-variance weighted averaging.

        Weights are based on per-point uncertainties (standard deviations):
        - Lower uncertainty = higher weight (more confident measurement)
        - Weight = 1 / σ²

        For multi-output: merging is done independently for each output dimension.

        Args:
            x (np.ndarray): Input features of new point
            y (float or np.ndarray): Target value(s) of new point
            sigma (float or np.ndarray): Uncertainty (standard deviation) of new point
            nearest_idx (int): Index of existing point to merge with

        Returns:
            bool: True if merge was successful
        """
        # Ensure y and sigma are arrays of shape (n_outputs,)
        if isinstance(y, (int, float, np.floating)):
            y = np.array([y])
        else:
            y = np.atleast_1d(y).flatten()

        if isinstance(sigma, (int, float, np.floating)):
            sigma = np.full(self.n_outputs, sigma)
        else:
            sigma = np.atleast_1d(sigma).flatten()
            if sigma.shape[0] == 1 and self.n_outputs > 1:
                sigma = np.tile(sigma, self.n_outputs)

        # Get existing point
        x_old = self.my_X_data[nearest_idx:nearest_idx+1]
        y_old = self.my_y_data[nearest_idx, :]  # Shape: (n_outputs,)
        sigma_old = self.my_sigma_data[nearest_idx, :]  # Shape: (n_outputs,)

        # Convert standard deviations to variances
        var_old = sigma_old ** 2
        var_new = sigma ** 2

        # Add epsilon for numerical stability
        epsilon = 1e-10
        w_old = 1.0 / (var_old + epsilon)
        w_new = 1.0 / (var_new + epsilon)
        w_total = w_old + w_new

        # Weighted average for x (same for all outputs)
        # Use average weight across outputs for x merging
        w_old_x = np.mean(w_old)
        w_new_x = np.mean(w_new)
        w_total_x = w_old_x + w_new_x
        x_merged = (w_old_x * x_old + w_new_x * x) / w_total_x

        # Weighted average for y (per output)
        y_merged = (w_old * y_old + w_new * y) / w_total

        # Merge variance and convert back to standard deviation (per output)
        var_merged = 1.0 / w_total
        sigma_merged = np.sqrt(var_merged)

        # Update the existing point with merged values
        self.my_X_data[nearest_idx] = x_merged[0]
        self.my_y_data[nearest_idx, :] = y_merged
        self.my_sigma_data[nearest_idx, :] = sigma_merged

        # Update merge count for this point
        self.merge_counts[nearest_idx, 0] += 1

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Node %s: Merged points (dist=%.2e, sigma_old_mean=%.2e, sigma_new_mean=%.2e, "
                         "sigma_merged_mean=%.2e, total_merges=%d)", self.name, np.linalg.norm(x - x_old),
                         np.mean(sigma_old), np.mean(sigma), np.mean(sigma_merged), int(self.merge_counts[nearest_idx, 0]))

        return True


    def merge_candidate(self, x: np.ndarray):
        """Index of the stored point that a new point at ``x`` is merged with, or
        None: merging is enabled, the node has ``min_points_before_merging``
        points and a fitted GP, and its nearest stored point lies within
        ``merge_distance_threshold``."""
        if not self.enable_point_merging or self.n_points < self.min_points_before_merging or not self._is_fitted():
            return None
        nearest_idx, nearest_dist = self.find_nearest_neighbor(x)
        if nearest_idx is None or not nearest_dist < self.merge_distance_threshold:
            return None
        return nearest_idx


    def _is_fitted(self) -> bool:
        """True once this node can predict from data (its GPs have been fitted, or it
        inherited fitted GPs from its parent)."""
        if self.output_model == 'pca':
            return self._fitted_basis is not None
        return self.my_GPRs[0].is_trained()

    @staticmethod
    def _set_noise(gpr, alpha):
        """Hand per-point observation-noise *variances* ``alpha`` (shape (N,)) to a GP."""
        gpr.set_observation_noise(np.asarray(alpha, dtype=float).ravel())

    def _fastest_direction(self):
        """For GPs whose kernels are not along the inputs: the direction of their
        shortest length scale, as weights of the inputs, or None. The metrics of
        several GPs are added, and inputs are compared in units of their spread."""
        metrics = [gpr.get_metric(self.n_features) for gpr in self.my_GPRs if gpr.is_trained()]
        metrics = [m for m in metrics if m is not None]
        if not metrics or self.my_X_data.shape[0] < 2:
            return None
        stds = np.std(self.my_X_data, axis=0)
        stds = np.where(stds > 0, stds, 1.0)
        metric = sum(metrics)
        if not self.use_standard_scaling:
            metric = metric * np.outer(stds, stds)
        direction = np.linalg.eigh(metric)[1][:, -1] / stds
        if np.ptp(self.my_X_data @ direction) == 0:
            return None
        return direction / np.linalg.norm(direction)

    def _pooled_length_scales(self):
        """Per-dimension length scales pooled over all of this node's fitted GPs
        (elementwise minimum: the shortest scale in any output/score governs the
        resolution needed along that dimension), or None if none are available."""
        collected = []
        for gpr in self.my_GPRs:
            if not gpr.is_trained():
                continue
            ls = gpr.get_length_scales(self.n_features)
            if ls is not None:
                collected.append(np.asarray(ls, dtype=float))
        if not collected:
            return None
        return np.min(np.vstack(collected), axis=0)

    def _fit_common_y_scaler(self, y: np.ndarray):
        """Centre each output and divide all of them by one common scale.

        Used by output_model='shared': with one common scale the per-point
        observation noise stays identical across outputs in the scaled space,
        as the backend's single per-point noise level requires.
        """
        mu = y.mean(axis=0)
        scale = float((y - mu).std())
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        self.y_common_scaler = (mu, scale)
        return mu, scale

    def _independent_fit_inputs(self, X: np.ndarray, y: np.ndarray, sigma: np.ndarray, scalers=None):
        """What the GPs of output_model='independent' are fitted to, for the points
        ``(X, y, sigma)``: the inputs and, per output, the targets and the noise
        variances. With ``use_standard_scaling`` they are standardised, with
        ``scalers`` or with scalers fitted to the points, and the scalers
        ``(X_scaler, y_scalers)`` are returned last, else None. The node is not
        changed."""
        outputs = range(self.n_outputs)
        if not self.use_standard_scaling:
            return X, [y[:, i:i+1] for i in outputs], [sigma[:, i] ** 2 for i in outputs], None
        if scalers is None:
            scalers = StandardScaler().fit(X), [StandardScaler().fit(y[:, i:i+1]) for i in outputs]
        X_scaler, y_scalers = scalers
        targets = [y_scalers[i].transform(y[:, i:i+1]) for i in outputs]
        # Noise std in scaled units is sigma / y_scale; the GP takes variances
        noise = [(sigma[:, i] / y_scalers[i].scale_[0]) ** 2 for i in outputs]
        return X_scaler.transform(X), targets, noise, (X_scaler, y_scalers)

    def _set_scalers(self, scalers):
        """Take the scalers that ``_independent_fit_inputs`` returned."""
        if scalers is not None:
            self.X_scaler, self.y_scalers = scalers[0], list(scalers[1])

    def fit_due(self) -> bool:
        """True if the retrain buffer or the node is full."""
        return self.n_points_since_retrain >= self.retrain_every_n_points or self.n_points >= self.Nbar


    def fit_my_GPR(self, force_training=False):
        """Fits the node's Gaussian Process Regressor(s) to its local data.

        Training is triggered if the number of new points in the buffer
        (`n_points_since_retrain`) reaches `retrain_every_n_points`, if the node
        is full (`n_points >= Nbar`), or if `force_training` is True.

        For multi-output: trains one independent GP per output dimension.

        If `use_standard_scaling` is enabled, this method first fits StandardScalers
        on the combined training data, then transforms the data before fitting the GP.
        The raw (unscaled) data remains stored in the node.

        Args:
            force_training (bool): If True, the GPR is retrained even if the
                usual buffer or fullness conditions are not met. Defaults to False.

        Returns:
            bool: True if the GPR was trained in this call, False otherwise.
        """
        # Only train the GP if the buffer is full, the node is full, or if force_training=True
        if not (self.fit_due() or force_training):
            return False

        # Combine own points and shared points
        X_train = np.vstack((self.my_X_data, self.shared_X_data))
        y_train = np.vstack((self.my_y_data, self.shared_y_data))  # Shape: (N, n_outputs)
        sigma_train = np.vstack((self.my_sigma_data, self.shared_sigma_data))  # Shape: (N, n_outputs)
        if X_train.shape[0] == 0:
            return False

        basis = None
        if self.output_model == 'pca':
            basis = self.output_basis.current
            if basis is None:
                # No output basis yet (too few points observed). The retrain buffer
                # is left as is, so the next call trains once the basis exists.
                return False

        # With a global model the GPs are fitted on the residual of its current
        # snapshot. The stored targets stay raw.
        snapshot = None
        curves = y_train
        if self.global_mean is not None:
            snapshot = self.global_mean.current
            if snapshot is not None:
                y_train = y_train - snapshot.predict(X_train)

        self.n_points_since_retrain = 0
        previous_fit_std = self.fit_std
        self._set_fit_statistics(X_train)

        if self.output_model == 'independent':
            # One GP per output, per-output standardisation of y
            X_fit, targets, noise, scalers = self._independent_fit_inputs(X_train, y_train, sigma_train)
            self._set_scalers(scalers)
            for i, gpr in enumerate(self.my_GPRs):
                self._set_noise(gpr, noise[i])
                gpr.fit(X_fit, targets[i])

        elif self.output_model in ('shared', 'aligned'):
            previous = self.alignment
            align = self._alignment_due(X_train.shape[0])
            if align and snapshot is not None:
                # An alignment is of the curves themselves, and a leaf that holds
                # one does without the global model
                self._fit_shared(X_train, curves, sigma_train, previous, previous_fit_std, align=True)
                if self.alignment is not None:
                    snapshot = None
                else:
                    self._fit_shared(X_train, y_train, sigma_train, None, previous_fit_std, align=False)
            else:
                self._fit_shared(X_train, y_train, sigma_train, previous, previous_fit_std, align)

        else:  # 'pca'
            if self.use_standard_scaling:
                self.X_scaler = StandardScaler().fit(X_train)
                X_fit = self.X_scaler.transform(X_train)
            else:
                X_fit = X_train
            Z = basis.project(y_train)                  # (N, k) basis scores
            noise_z = basis.project_noise(sigma_train)  # (N, k) per-score noise variances
            k = basis.n_components
            # Standardise every score within the leaf (as the per-output scalers do
            # for 'independent'): the scores carry the raw output variance and a
            # leaf-local offset, while the kernel's initial amplitude and zero prior
            # mean assume centred, unit-variance targets.
            z_shift = Z.mean(axis=0)
            z_scale = Z.std(axis=0)
            z_scale = np.where(np.isfinite(z_scale) & (z_scale > 0.0), z_scale, 1.0)
            self.z_scaler = (z_shift, z_scale)
            # One GP per score: (re)size the list if the basis rank changed
            template = self.my_GPRs[0]
            while len(self.my_GPRs) < k:
                self.my_GPRs.append(template.clone())
            del self.my_GPRs[k:]
            for j in range(k):
                self._set_noise(self.my_GPRs[j], noise_z[:, j] / z_scale[j] ** 2)
                self.my_GPRs[j].fit(X_fit, (Z[:, j:j+1] - z_shift[j]) / z_scale[j])
            self._fitted_basis = basis

        self._fitted_global = snapshot
        self.asleep = False
        self._used()
        return True


    def _alignment_due(self, n_points: int) -> bool:
        """Whether this fit tries an alignment (output_model='aligned'): after a
        rejection only once the wait is over, of which this call counts one fit."""
        if self.output_model != 'aligned' or n_points < self.n_features + 3:
            return False
        if self._alignment_wait > 0:
            self._alignment_wait -= 1
            return False
        return True


    def _fit_shared(self, X_train, y_train, sigma_train, previous, previous_fit_std, align):
        """Fit one GP with one kernel for all outputs to the curves ``y_train``
        (a 2-D target). The backend takes one noise variance per point, so the
        noise variances of the outputs are averaged (exact when sigma is the
        same for all outputs, given the common output scale). With ``align`` an
        alignment of the curves is tried, starting from the ``previous`` one."""
        if self.use_standard_scaling:
            self.X_scaler = StandardScaler().fit(X_train)
            X_fit = self.X_scaler.transform(X_train)
            mu, scale = self._fit_common_y_scaler(y_train)
            y_fit = (y_train - mu) / scale
            alpha = np.mean((sigma_train / scale) ** 2, axis=1)
        else:
            X_fit, y_fit = X_train, y_train
            alpha = np.mean(sigma_train ** 2, axis=1)
        self._set_noise(self.my_GPRs[0], alpha)
        self.my_GPRs[0].fit(X_fit, y_fit)

        # What the GP takes from its leave-one-out predictions of the curves: a
        # sigma per output and, with ``align``, whether an alignment replaces it
        self.alignment, self.sigma_profile = None, None
        loo = self.my_GPRs[0].leave_one_out()
        if loo is None:
            return
        weights, variance = loo
        mu, scale = self.y_common_scaler if self.use_standard_scaling else (0.0, 1.0)
        errors = weights @ (y_train - mu) - (y_train - mu)
        variance = variance * scale ** 2
        if align:
            aligned = self._aligned_fit(X_train, X_fit, y_train, sigma_train, previous, previous_fit_std)
            if aligned is not None and np.sqrt(np.mean(aligned[2] ** 2)) * ALIGNMENT_MARGIN < np.sqrt(np.mean(errors ** 2)):
                self.my_GPRs[0], self.alignment, errors, variance, scaler = aligned
                if self.use_standard_scaling:
                    self.y_common_scaler = scaler
                self._alignment_rejections = 0
            else:
                self._alignment_rejections += 1
                self._alignment_wait = min(2 ** self._alignment_rejections, ALIGNMENT_MAX_WAIT) - 1
        # The GP has one variance for all outputs. Each output gets the factor with
        # which its sigma covers TARGET_COVERAGE of its leave-one-out errors.
        profile = np.quantile(np.abs(errors) / np.sqrt(variance)[:, None], TARGET_COVERAGE, axis=0)
        self.sigma_profile = np.maximum(profile, SIGMA_PROFILE_FLOOR * np.sqrt(np.mean(profile ** 2)))


    def _aligned_fit(self, X_train, X_fit, y_train, sigma_train, previous, previous_fit_std):
        """A warp of the grid of the curves ``y_train`` and a GP fitted to the
        aligned curves: ``(GP, AlignedCurves, leave-one-out errors and variances
        of the curves, (mu, scale) of the aligned curves)``, or None if the GP
        backend cannot."""
        gpr = self.my_GPRs[0].unfitted_copy()
        if gpr is None:
            return None
        delta = (X_train - self.fit_mean) / self.fit_std
        warp = output_warp(self.output_shape, self.n_features)
        if previous is not None:
            # the previous warp, in the units of the new offsets
            warp.start_from(previous.warp, self.fit_std / previous_fit_std)
        warp.fit(y_train, delta)
        aligned = warp.register(y_train, delta)
        curves = warp.curves(y_train, aligned, delta)
        mu, scale = 0.0, 1.0
        if self.use_standard_scaling:
            mu = aligned.mean(axis=0)
            scale = float((aligned - mu).std())
            if not np.isfinite(scale) or scale <= 0.0:
                scale = 1.0
        else:
            curves.centre_weights = np.zeros(X_train.shape[0])
        self._set_noise(gpr, np.mean((sigma_train / scale) ** 2, axis=1))
        gpr.fit(X_fit, (aligned - mu) / scale)
        weights, variance = gpr.leave_one_out()
        return gpr, curves, curves.combine(delta, weights) - y_train, variance * scale ** 2, (mu, scale)


    def _points(self, snapshot=None):
        """The node's own and shared points: inputs, targets (their residual of
        ``snapshot``, if given) and sigmas."""
        X = np.vstack((self.my_X_data, self.shared_X_data))
        y = np.vstack((self.my_y_data, self.shared_y_data))
        if snapshot is not None and X.shape[0] > 0:
            y = y - snapshot.predict(X)
        return X, y, np.vstack((self.my_sigma_data, self.shared_sigma_data))


    def _refit_on_snapshot(self, snapshot, hyperparameters=None, scalers=None):
        """Refit the GPs on the residual of ``snapshot`` (the targets, if None)
        without optimising: with ``hyperparameters``, one set per GP, in the
        scaling of ``scalers``, or with the hyperparameters the GPs have, in
        the scaling of the node's points.

        Used by global models with ``cheap_leaf_refresh`` and by
        ``apply_fit``. The retrain buffer is left as is, so the
        hyperparameters are optimised on the usual schedule. Returns False if
        this node or its GP backend cannot do it.
        """
        if self.output_model != 'independent':
            return False
        if hyperparameters is None and not all(gpr.is_trained() for gpr in self.my_GPRs):
            return False
        X_train, y_train, sigma_train = self._points(snapshot)
        if X_train.shape[0] == 0:
            return False
        X_fit, targets, noise, scalers = self._independent_fit_inputs(X_train, y_train, sigma_train, scalers)
        self._set_scalers(scalers)
        for i, gpr in enumerate(self.my_GPRs):
            self._set_noise(gpr, noise[i])
            if hyperparameters is None:
                done = gpr.refit_fixed(X_fit, targets[i])
            else:
                done = gpr.refit_fixed(X_fit, targets[i], hyperparameters[i])
            if not done:
                return False
        self._fitted_global = snapshot
        self.asleep = False
        self._used()
        self._set_fit_statistics(X_train)
        return True


    def fit_task(self):
        """The fit of the node as a task for another process: unfitted copies of
        its GPs with the inputs and targets they are to be fitted to.

        Returns the task and its frame: the scalers and the global-model
        snapshot that the inputs and targets were made with. ``solve_fit_task``
        carries the task out, and ``apply_fit`` takes what it finds together
        with the frame. The node's GPs stay as they are, and the retrain buffer
        is emptied. Returns None if the node has no points, or if it or its GP
        backend cannot hand a fit out (it is then fitted in place): this needs
        output_model='independent' and a backend with ``unfitted_copy`` (the
        scikit-learn adapter).
        """
        if self.output_model != 'independent':
            return None
        gprs = [gpr.unfitted_copy() for gpr in self.my_GPRs]
        if any(gpr is None for gpr in gprs):
            return None
        snapshot = self.global_mean.current if self.global_mean is not None else None
        X_train, y_train, sigma_train = self._points(snapshot)
        if X_train.shape[0] == 0:
            return None
        X_fit, targets, noise, scalers = self._independent_fit_inputs(X_train, y_train, sigma_train)
        for gpr, alpha in zip(gprs, noise):
            self._set_noise(gpr, alpha)
        self.n_points_since_retrain = 0
        return {'X': X_fit, 'y': targets, 'gprs': gprs}, (scalers, snapshot)


    def apply_fit(self, hyperparameters, frame):
        """Refit the GPs to the points the node holds now, with the hyperparameters
        that ``solve_fit_task`` found, one set per GP, in the ``frame`` of the
        task: the node is then as fitted in place when the task was made, with
        the points since then appended."""
        scalers, snapshot = frame
        if not self._refit_on_snapshot(snapshot, hyperparameters, scalers):
            self.fit_my_GPR(force_training=True)


    def _set_fit_statistics(self, X_train: np.ndarray):
        self.fit_mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        self.fit_std = np.where(std > 0.0, std, 1.0)


    def distance_from_fit(self, x: np.ndarray):
        """How far the rows of ``x`` lie from the points of the node's last fit:
        the distance from their mean in units of their standard deviation, the
        largest over the inputs. Zero for a node without a fit."""
        if self.fit_mean is None:
            return np.zeros(x.shape[0])
        return np.abs((x - self.fit_mean) / self.fit_std).max(axis=1)


    def append_point(self, x: np.ndarray, y: Union[float, np.ndarray], sigma: Union[float, np.ndarray]):
        """Condition the fitted GPs on one more point, at a cost quadratic in their
        number of points.

        The hyperparameters, the scaling and the global-model snapshot of the
        last fit are kept. The next fit starts from the node's points as usual.
        Returns False if the GPs are not fitted or their backend cannot do it.
        """
        if self.asleep or not self._is_fitted():
            return False
        x = np.asarray(x, dtype=float).reshape(1, -1)
        y = np.asarray(y, dtype=float).reshape(1, self.n_outputs)
        sigma = np.broadcast_to(np.asarray(sigma, dtype=float).reshape(1, -1), (1, self.n_outputs))
        if self._fitted_global is not None:
            y = y - self._fitted_global.predict(x)
        scaled = self.use_standard_scaling and self.X_scaler is not None
        xs = self._scaled_inputs(x)

        if self.output_model == 'independent':
            done = []
            for i, gpr in enumerate(self.my_GPRs):
                shift, scale = (self.y_scalers[i].mean_[0], self.y_scalers[i].scale_[0]) if scaled else (0.0, 1.0)
                done.append(gpr.append(xs, (y[0, i] - shift) / scale, (sigma[0, i] / scale) ** 2))
            return all(done)

        if self.output_model in ('shared', 'aligned'):
            mu, scale = self.y_common_scaler if scaled and self.y_common_scaler is not None else (0.0, 1.0)
            noise = float(np.mean((sigma / scale) ** 2))
            if self.alignment is None:
                return self.my_GPRs[0].append(xs, (y - mu) / scale, noise)
            delta = (x - self.fit_mean) / self.fit_std
            done = self.my_GPRs[0].append(xs, (self.alignment.warp.register(y, delta) - mu) / scale, noise)
            if done:
                self.alignment = self.alignment.with_curve(y, delta)
            return done

        z = self._fitted_basis.project(y)[0]
        noise_z = self._fitted_basis.project_noise(sigma)[0]
        z_shift, z_scale = self.z_scaler
        return all([gpr.append(xs, (z[j] - z_shift[j]) / z_scale[j], noise_z[j] / z_scale[j] ** 2)
                    for j, gpr in enumerate(self.my_GPRs)])


    def sleep(self):
        """Have the fitted GPs release what they can compute again, to save memory.

        The next prediction restores them, and they then predict as before (to
        numerical precision if points were appended). Returns False if there is
        nothing to release or the GP backend cannot.
        """
        if self.asleep or not any([gpr.release() for gpr in self.my_GPRs]):
            return False
        if self.alignment is not None:
            self.alignment.release()
        self.asleep = True
        return True


    def wake(self):
        """Restore the GPs of a sleeping node, and count them as used."""
        self._used()
        if not self.asleep:
            return
        if all([gpr.restore() for gpr in self.my_GPRs]):
            self.asleep = False
        else:
            self.fit_my_GPR(force_training=True)


    def _used(self):
        """Tell a tree that predicts from neighbours that the node's GPs are awake."""
        if self.neighbours is not None:
            self.neighbours.used(self)


    def nbytes_points(self):
        """Size in bytes of the node's own and shared points."""
        return sum(a.nbytes for a in (self.my_X_data, self.my_y_data, self.my_sigma_data,
                                      self.shared_X_data, self.shared_y_data, self.shared_sigma_data))


    def nbytes_fitted(self):
        """Size in bytes of what the node's GPs hold from their fits, as they report
        it, and of the curves of an alignment."""
        return sum(gpr.nbytes() for gpr in self.my_GPRs) + (self.alignment.nbytes() if self.alignment is not None else 0)


    def _compute_dimensional_uncertainty(self):
        """Computes uncertainty scores for each dimension to guide splitting.

        This method assesses which dimension the current GP is most uncertain about
        by analyzing how uncertainty varies along each dimension. The dimension with
        the highest uncertainty score is where splitting would be most beneficial.

        Strategy:
        1. For each dimension, create a grid of test points that vary only in that dimension
        2. Compute GP predictions and uncertainties at these test points
        3. Aggregate uncertainty (e.g., mean or max) to get a score per dimension

        Returns:
            np.ndarray: Uncertainty scores for each dimension (shape: n_features)
        """
        uncertainty_scores = np.zeros(self.n_features)

        # Use a sample of training points as reference points
        # For computational efficiency, sample at most 20 points
        n_sample_points = min(20, self.my_X_data.shape[0])
        if n_sample_points < self.my_X_data.shape[0]:
            # Randomly sample points
            sample_indices = np.random.choice(self.my_X_data.shape[0],
                                             size=n_sample_points,
                                             replace=False)
            sample_X = self.my_X_data[sample_indices, :]
        else:
            sample_X = self.my_X_data

        # For each dimension, assess uncertainty
        for dim in range(self.n_features):
            # Create test points that vary along this dimension
            # Use the range of data in this dimension
            dim_min = np.min(self.my_X_data[:, dim])
            dim_max = np.max(self.my_X_data[:, dim])

            if dim_max - dim_min < 1e-10:
                # No variation in this dimension
                uncertainty_scores[dim] = 0.0
                continue

            # Create a grid of values for this dimension (use 5 points for efficiency)
            dim_values = np.linspace(dim_min, dim_max, 5)

            # For each sampled point, vary only this dimension and measure uncertainty
            uncertainties = []
            for base_point in sample_X:
                for dim_val in dim_values:
                    test_point = base_point.copy()
                    test_point[dim] = dim_val
                    test_point = test_point.reshape(1, -1)

                    # Get prediction uncertainty
                    _, sigma = self.predict(test_point, use_calibrated_sigma=False)
                    uncertainties.append(sigma[0])

            # Aggregate uncertainties for this dimension
            # Use mean uncertainty as the score
            uncertainty_scores[dim] = np.mean(uncertainties)

        return uncertainty_scores


    def compute_split_position_and_overlap(self, theta: float):
        """Computes the split dimension, position, and overlap for node splitting.

        This method determines which feature dimension (`self.split_index`) to
        split on, typically by selecting the dimension with the maximum spread
        (range of values) in the node's current training data (`self.my_X_data`).

        The actual split position (`self.split_position`) along this dimension is
        then calculated based on the `self.split_position_method` (e.g., median,
        mean of the data in `self.my_X_data[:, self.split_index]`).

        The `overlap` region's size (`self.overlap`) is calculated as a product
        of the `theta` parameter (passed from the `GPTree` during the split
        operation, representing a fraction of the spread) and the spread (`w`)
        of the chosen `split_index`.

        Args:
            theta (float): A parameter provided by the `GPTree` that determines
                the extent of the overlap. It's a fraction of the data range
                in the chosen split dimension.
        """

        # Determine the split index based on the chosen criteria
        self.split_direction = None
        if self.split_dimension_criteria == 'max_spread':
            if self.my_X_data.shape[0] > 0:
                w = np.empty(self.n_features)
                for i in range(self.n_features):
                    w[i] = np.max(self.my_X_data[:, i]) - np.min(self.my_X_data[:, i])
                self.split_index = np.argmax(w)
            else:
                self.split_index = 0 # Default to 0 if no data
        elif self.split_dimension_criteria == 'max_variance':
            if self.my_X_data.shape[0] > 1:
                variances = np.var(self.my_X_data, axis=0)
                self.split_index = np.argmax(variances)
            else: # Fallback for single data point or no variance
                if self.my_X_data.shape[0] > 0:
                    w = np.empty(self.n_features)
                    for i in range(self.n_features):
                        w[i] = np.max(self.my_X_data[:, i]) - np.min(self.my_X_data[:, i])
                    self.split_index = np.argmax(w)
                else:
                    self.split_index = 0 # Default to 0 if no data
        elif self.split_dimension_criteria == 'max_uncertainty':
            # Split on the dimension where the GP is most uncertain
            # Strategy: compute marginal predictive uncertainty for each dimension
            if self.my_X_data.shape[0] > 1 and self._is_fitted():
                # Compute per-dimension uncertainty scores
                # We'll use the GP's predictions on the training data to assess uncertainty
                uncertainty_scores = self._compute_dimensional_uncertainty()
                self.split_index = np.argmax(uncertainty_scores)
            else:
                # Fallback to max_variance if GP not trained yet
                if self.my_X_data.shape[0] > 1:
                    variances = np.var(self.my_X_data, axis=0)
                    self.split_index = np.argmax(variances)
                elif self.my_X_data.shape[0] > 0:
                    w = np.empty(self.n_features)
                    for i in range(self.n_features):
                        w[i] = np.max(self.my_X_data[:, i]) - np.min(self.my_X_data[:, i])
                    self.split_index = np.argmax(w)
                else:
                    self.split_index = 0 # Default to 0 if no data
        elif self.split_dimension_criteria == 'min_lengthscale':
            # Split where the GP's smallest fitted ARD length scale is, i.e. where
            # the function varies fastest relative to its spread. For several GPs
            # (multi-output / basis scores) the per-dimension minimum over all of
            # them is used. Falls back to max_spread if no per-dimension length
            # scales are available (GP untrained or kernel isotropic).
            length_scales = None
            if self.my_X_data.shape[0] > 1:
                length_scales = self._pooled_length_scales()
                self.split_direction = self._fastest_direction()
            if length_scales is not None:
                # Compare length scales in units of each dimension's std so the
                # choice is scale-invariant. With standard scaling the GP frame
                # already has std ~ 1 (no-op); without it, normalise by the raw
                # per-dimension std so dimensions in different units stay comparable.
                if not self.use_standard_scaling and self.my_X_data.shape[0] > 1:
                    stds = np.std(self.my_X_data, axis=0)
                    length_scales = length_scales / np.where(stds > 0, stds, 1.0)
                # The length scale of a dimension in which the points coincide is arbitrary
                length_scales = np.where(np.ptp(self.my_X_data, axis=0) > 0, length_scales, np.inf)
                self.split_index = int(np.argmin(length_scales))
            elif self.my_X_data.shape[0] > 0:
                spreads = np.max(self.my_X_data, axis=0) - np.min(self.my_X_data, axis=0)
                self.split_index = int(np.argmax(spreads))
            else:
                self.split_index = 0
        elif self.split_dimension_criteria == 'random':
            if self.n_features > 0:
                self.split_index = np.random.randint(0, self.n_features)
            else:
                self.split_index = 0 # Default to 0 if no features
        else:
            raise ValueError(f"Unknown split_dimension_criteria: {self.split_dimension_criteria}")

        # A dimension in which all of the node's points coincide cannot separate them
        if self.my_X_data.shape[0] > 0:
            spreads = np.ptp(self.my_X_data, axis=0)
            if spreads[self.split_index] == 0:
                self.split_index = int(np.argmax(spreads))

        # Calculate spread along the split, used for overlap calculation
        # and potentially for split_position if data is scarce for other methods.
        along_split = self.split_coordinate(self.my_X_data)
        if self.my_X_data.shape[0] > 0:
            current_dim_spread = np.max(along_split) - np.min(along_split)
        else:
            current_dim_spread = 0.0 # Default if no data

        self.split_position = None
        if self.my_X_data.shape[0] == 0: # No data, place split in the middle (0 if not scaled) or handle as error?
            self.split_position = 0.0
        elif self.split_position_method == 'median':
            if self.my_X_data.shape[0] > 0:
                self.split_position = np.median(along_split)
            else: # Should not happen if split_index logic is robust
                self.split_position = 0.0
        elif self.split_position_method == 'mean':
            if self.my_X_data.shape[0] > 0:
                self.split_position = np.mean(along_split)
            else: # Should not happen
                self.split_position = 0.0
        elif self.split_position_method == 'random':
            if self.my_X_data.shape[0] > 0:
                min_val = np.min(along_split)
                max_val = np.max(along_split)
                if min_val == max_val: # All points are the same
                    self.split_position = min_val
                else:
                    self.split_position = np.random.uniform(min_val, max_val, 1)[0]
            else: # Should not happen
                self.split_position = 0.0
        elif self.split_position_method == 'randomchoice':
            if self.my_X_data.shape[0] > 0:
                self.split_position = np.random.choice(along_split)
            else: # Should not happen
                self.split_position = 0.0
        else:
            raise ValueError(f"Unknown split_position_method argument: '{self.split_position_method}'. The valid options are 'median', 'mean', 'random' and 'randomchoice'")

        self.overlap = theta * current_dim_spread


    def split_coordinate(self, x: np.ndarray):
        """The coordinate of the rows of x along the normal of this node's cut."""
        return x[:, self.split_index] if self.split_direction is None else x @ self.split_direction


    def cut_axis(self):
        """The input this node's cut is along, or None if it is along none."""
        return self.split_index if self.split_direction is None else None


    def prob_func(self, x: np.array):
        """The probability that x belongs to the right child: linear across the
        overlap region around the split position, as suggested in the DLGP article.
        Without an overlap the split is hard, and a point on it gets 0.5."""
        dist = self.split_coordinate(x) - self.split_position
        if self.overlap > 0:
            prob = dist/self.overlap + 0.5
            prob[prob < 0] = 0
            prob[prob > 1] = 1
        else:
            prob = 0.5 * (np.sign(dist) + 1.0)

        return prob.reshape(x.shape[0], 1)

    
    def marg_prob(self, x: np.ndarray):
        """ Compute the marginal probability that a test point x belongs to this node. """
        ptilde = np.ones(shape=(x.shape[0], 1))
        node = self
        while node.parent:
            is_left = node.is_left
            node = node.parent
            if is_left:
                ptilde *= (1 - node.prob_func(x))
            else:
                ptilde *= node.prob_func(x)
            
        return ptilde


    def _refresh_global(self):
        """Refit a leaf whose snapshot of the global model is older than the
        current one: a leaf that stops receiving points would otherwise keep
        an outdated snapshot whose error its residual GP cannot correct."""
        if self.global_mean is not None and self.is_leaf and self.n_points > 0:
            current = self.global_mean.current
            # a leaf with an alignment models the curves themselves, without the global model
            if current is not None and self.alignment is None and (self._fitted_global is None
                                                                    or self._fitted_global.version < current.version):
                if not (self.global_mean.cheap_leaf_refresh and self._refit_on_snapshot(current)):
                    self.fit_my_GPR(force_training=True)


    def predict(self, x: np.ndarray, use_calibrated_sigma=False):
        """Evaluates the prediction from this node's GPR(s) at input point(s) x.

        The input x is expected to be in the original (unscaled) space. If
        `use_standard_scaling` is enabled, this method transforms x to the
        standardized space before calling the GP, then inverse transforms
        the predictions back to the original space.

        For multi-output: returns predictions for all outputs as (n_samples, n_outputs) arrays.

        A leaf of a tree with ``predict_from='neighbours'`` conditions its
        kernels on the stored points nearest to each row of ``x`` in place of
        the points of its last fit (``_predict_from_neighbours``), unless its
        own GPs predicted the last points it received clearly better
        (``prefers_own``).

        Args:
            x (np.ndarray): The input point(s) at which to make predictions,
                in the original (unscaled) space.
                Shape should be (n_samples, n_features).
            use_calibrated_sigma (bool): If True, the returned `sigma_pred`
                (standard deviation) is scaled by the node's `self.sigma_scalers`.
                These scalers are intended to calibrate the uncertainty
                estimates. Defaults to False.

        Returns:
            tuple:
                - mu_pred (np.ndarray): The mean prediction(s) in original space.
                  Shape: (n_samples, n_outputs)
                - sigma_pred (np.ndarray): The standard deviation of the
                  prediction(s) in original space. Shape: (n_samples, n_outputs)
                  This is the uncertainty of the estimate of the underlying
                  function (the GP's latent posterior standard deviation), not
                  of a noisy observation. With
                  a global model it also includes the global model's own
                  estimated (epistemic) error, see ``global_mean``.
        """
        self._refresh_global()
        if self._can_condition() and not self.prefers_own:
            mu_pred, sigma_pred = self._predict_from_neighbours(x)
        else:
            mu_pred, sigma_pred = self._predict_own(x)

        # Add back the snapshot the GPs were fitted against, and its estimated error:
        # the residual GP cannot know how wrong the global model is at x.
        if self._fitted_global is not None:
            mu_pred = mu_pred + self._fitted_global.predict(x)
            err_scale = self.global_mean.error_scale
            if err_scale is not None:
                sigma_pred = np.sqrt(sigma_pred ** 2 + np.asarray(err_scale, dtype=float).reshape(1, -1) ** 2)

        # Apply calibration if requested: each output has its own factor
        if use_calibrated_sigma:
            sigma_pred = sigma_pred * np.asarray(self.sigma_scalers)

        return mu_pred, sigma_pred


    def _can_condition(self) -> bool:
        """Whether this node can predict from the stored points nearest to a
        query. A leaf with an alignment predicts from its own aligned curves."""
        return (self.neighbours is not None and self.is_leaf and self.fit_std is not None and self._is_fitted()
                and self.alignment is None)


    def _score_predictors(self, x: np.ndarray, y: np.ndarray, mu_pred: np.ndarray):
        """Record how the neighbours and the node's own GPs each predicted the
        point ``(x, y)``, of which ``mu_pred`` is the prediction of the one in
        use, and let the own GPs predict from now on if they did clearly better
        over the last points."""
        if self.prefers_own:
            mu_other, _ = self._predict_from_neighbours(x)
        else:
            mu_other, _ = self._predict_own(x)
        if self._fitted_global is not None:
            mu_other = mu_other + self._fitted_global.predict(x)
        errors = (y - mu_pred[0]) ** 2, (y - mu_other[0]) ** 2
        self._scores.append(errors[::-1] if self.prefers_own else errors)       # (neighbours, own)
        if len(self._scores) >= MIN_SCORES:
            scores = np.array(self._scores)                                    # (points, 2, n_outputs)
            from_neighbours, own = scores[:, 0].mean(axis=0) + 1e-300, scores[:, 1].mean(axis=0) + 1e-300
            self.prefers_own = bool(np.mean(np.log(own / from_neighbours)) < 2.0 * np.log(OWN_MARGIN))


    def _prior(self, Z: np.ndarray):
        """The prior of each output at the points ``Z`` in the units of the
        targets, before the global model: ``(covariances, means, scales)``
        with a matrix and a mean per output, and the factor by which the GP
        of an output scales the noise variance it is given. None unless each
        output has its own GP and the backend gives its covariance."""
        if self.output_model != 'independent':
            return None
        Zs = self._scaled_inputs(Z)
        scaled = self.use_standard_scaling and self.X_scaler is not None
        covariances, means, scales = [], np.empty(self.n_outputs), np.empty(self.n_outputs)
        for i, gpr in enumerate(self.my_GPRs):
            prior = gpr.covariance(Zs)
            if prior is None:
                return None
            shift, scale = (self.y_scalers[i].mean_[0], self.y_scalers[i].scale_[0]) if scaled else (0.0, 1.0)
            covariances.append((scale ** 2 * prior[1][0]) * prior[0])
            means[i] = shift + scale * prior[2][0]
            scales[i] = prior[1][0]
        return covariances, means, scales


    def _predicted_variance(self):
        """Per output, the median of the variances that the node's GPs gave its
        last registered predictions, before calibration, or None while there
        are fewer than MIN_SCORES of them."""
        if self._variance_unit is None and len(self.sigma_preds_list[0]) >= MIN_SCORES:
            self._variance_unit = np.array([np.median(sigma ** 2) for sigma in self.sigma_preds_list])
        return self._variance_unit


    def _predict_own(self, x: np.ndarray):
        """The prediction of the node's GPs from the points of their fit, before
        the global model is added back."""
        self.wake()
        if self.output_model == 'independent':
            return self._predict_independent(x)
        if self.output_model in ('shared', 'aligned'):
            return self._predict_shared(x)
        return self._predict_pca(x)


    def _predict_from_neighbours(self, x: np.ndarray):
        """The prediction of the node's kernels given, for each row of ``x``, the
        stored points nearest to it, before the global model is added back.
        Distances are in units of the spread of the points of the node's last
        fit. Where the neighbours fit a kernel badly the prediction is mixed
        with that of the node's own GP (``NeighbourSearch.share``)."""
        n = x.shape[0]
        mu, sigma, share = (np.zeros((n, self.n_outputs)) for _ in range(3))
        for r in range(n):
            found = self.neighbours.find(x[r], self.fit_std, leaf=self)
            share[r] = self._condition_on(found, x[r:r + 1], mu[r], sigma[r])
        rows = (share < 1.0).any(axis=1)
        if rows.any():
            mu_own, sigma_own = self._predict_own(x[rows])
            mu[rows], sigma[rows] = self.neighbours.mix(share[rows], mu[rows], sigma[rows], mu_own, sigma_own)
        return mu, sigma


    def _condition_on(self, points, xq: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
        """Fill ``mu`` and ``sigma`` (one entry per output) with the posterior at
        the single query ``xq`` of the node's kernels given ``points``, and
        return the weight of that prediction per output against the node's
        own (``NeighbourSearch.share``). An output with the weight 0 is left
        as it is."""
        X, y, noise_std, growth = points
        share = np.zeros(self.n_outputs)

        def weight(consistency):
            return self.neighbours.weight(consistency, X, xq[0], self.fit_std, growth)

        if self._fitted_global is not None:
            y = y - self._fitted_global.predict(X)
        if growth is not None:
            # neighbours that fade out: their noise grows in units of itself plus the variance of a prediction
            unit = self._predicted_variance()
            if unit is None:
                unit = 0.01 * y.var(axis=0)
            noise_std = np.sqrt(noise_std ** 2 + growth[:, None] * (noise_std ** 2 + unit))
        Xs, qs = self._scaled_inputs(X), self._scaled_inputs(xq)
        scaled = self.use_standard_scaling and self.X_scaler is not None

        if self.output_model == 'independent':
            for i, gpr in enumerate(self.my_GPRs):
                shift, scale = (self.y_scalers[i].mean_[0], self.y_scalers[i].scale_[0]) if scaled else (0.0, 1.0)
                result = gpr.condition(Xs, (y[:, i] - shift) / scale, (noise_std[:, i] / scale) ** 2, qs)
                if result is not None:
                    share[i] = weight(result[2][0])
                    if share[i] > 0.0:
                        mu[i], sigma[i] = result[0][0, 0] * scale + shift, result[1][0, 0] * scale

        elif self.output_model in ('shared', 'aligned'):
            shift, scale = self.y_common_scaler if scaled and self.y_common_scaler is not None else (0.0, 1.0)
            result = self.my_GPRs[0].condition(Xs, (y - shift) / scale, np.mean((noise_std / scale) ** 2, axis=1), qs)
            if result is not None:
                share[:] = weight(np.max(result[2]))
                if share[0] > 0.0:
                    profile = 1.0 if self.sigma_profile is None else self.sigma_profile
                    mu[:], sigma[:] = result[0][0] * scale + shift, result[1][0] * scale * profile

        else:  # 'pca'
            basis = self._fitted_basis
            Z, noise_z = basis.project(y), basis.project_noise(noise_std)
            z_shift, z_scale = self.z_scaler
            z_mean, z_var = np.empty((1, basis.n_components)), np.empty((1, basis.n_components))
            lowest = 1.0
            for j, gpr in enumerate(self.my_GPRs):
                result = gpr.condition(Xs, (Z[:, j] - z_shift[j]) / z_scale[j], noise_z[:, j] / z_scale[j] ** 2, qs)
                lowest = 0.0 if result is None else min(lowest, weight(result[2][0]))
                if lowest <= 0.0:
                    return share
                z_mean[0, j] = result[0][0, 0] * z_scale[j] + z_shift[j]
                z_var[0, j] = (result[1][0, 0] * z_scale[j]) ** 2
            mu_pred, var_pred = basis.reconstruct(z_mean, z_var)
            mu[:], sigma[:], share[:] = mu_pred[0], np.sqrt(var_pred[0]), lowest
        return share


    def _predict_independent(self, x: np.ndarray):
        """Per-output GPs (output_model='independent'); see ``predict``."""
        n_samples = x.shape[0]

        # Initialize output arrays
        mu_pred = np.zeros((n_samples, self.n_outputs))
        sigma_pred = np.zeros((n_samples, self.n_outputs))

        if self.use_standard_scaling and self.X_scaler is not None:
            # Transform input to standardized space (same for all outputs)
            x_scaled = self._scaled_inputs(x)

            # Get predictions for each output independently
            for i in range(self.n_outputs):
                # Get prediction in scaled space
                mu_scaled_i, sigma_scaled_i = self.my_GPRs[i].predict(x_scaled, return_std=True)

                # Inverse transform mean: mu_original = mu_scaled * scale_y + mean_y
                y_scaler = self.y_scalers[i]
                mu_pred[:, i] = np.asarray(mu_scaled_i, dtype=float).reshape(-1) * y_scaler.scale_[0] + y_scaler.mean_[0]

                # Transform std: sigma_original = sigma_scaled * scale_y
                # When y is scaled by scale_y, variance scales by scale_y^2, so std scales by scale_y
                sigma_pred[:, i] = sigma_scaled_i * y_scaler.scale_[0]

        else:
            # No scaling - predict on raw data for each output
            for i in range(self.n_outputs):
                mu_i, sigma_i = self.my_GPRs[i].predict(x, return_std=True)
                mu_pred[:, i] = mu_i.flatten()
                sigma_pred[:, i] = sigma_i

        return mu_pred, sigma_pred

    def _scaled_inputs(self, x: np.ndarray) -> np.ndarray:
        """``X_scaler.transform(x)`` without scikit-learn's input validation, which
        costs more than the arithmetic for a few points."""
        if self.use_standard_scaling and self.X_scaler is not None:
            return (np.asarray(x, dtype=float) - self.X_scaler.mean_) / self.X_scaler.scale_
        return x

    def _prior_prediction(self, x: np.ndarray):
        """Prediction of a node that has not been fitted yet: the prior of its first
        GP, broadcast to all outputs (what the per-output GPs would each return)."""
        n = x.shape[0]
        mu0, sd0 = self.my_GPRs[0].predict(x, return_std=True)
        mu0 = np.asarray(mu0, dtype=float).reshape(n, -1)[:, :1]
        sd0 = np.asarray(sd0, dtype=float).reshape(n, -1)[:, :1]
        return np.tile(mu0, (1, self.n_outputs)), np.tile(sd0, (1, self.n_outputs))

    def _predict_shared(self, x: np.ndarray):
        """One shared-kernel GP over all outputs (output_model='shared'), fitted
        to aligned curves if the node has an alignment (output_model='aligned')."""
        gpr = self.my_GPRs[0]
        if not gpr.is_trained():
            return self._prior_prediction(x)
        n = x.shape[0]
        xs = self._scaled_inputs(x)
        m, sd = gpr.predict(xs, return_std=True)
        m = np.asarray(m, dtype=float).reshape(n, -1)
        sd = np.asarray(sd, dtype=float).reshape(n, -1)
        if sd.shape[1] != self.n_outputs:
            # one sigma for all targets of the GP, of which an alignment can have more than there are outputs
            sd = np.tile(sd[:, :1], (1, self.n_outputs))
        if self.use_standard_scaling and self.y_common_scaler is not None:
            mu, scale = self.y_common_scaler
            m = m * scale + mu
            sd = sd * scale
        if self.sigma_profile is not None:
            sd = sd * self.sigma_profile
        if self.alignment is not None:
            m = self.alignment.combine((x - self.fit_mean) / self.fit_std, gpr.mean_weights(xs))
        return m, sd

    def _predict_pca(self, x: np.ndarray):
        """Score GPs + basis reconstruction (output_model='pca')."""
        basis = self._fitted_basis
        if basis is None:
            return self._prior_prediction(x)
        n, k = x.shape[0], basis.n_components
        xs = self._scaled_inputs(x)
        z_shift, z_scale = self.z_scaler
        z_mean = np.zeros((n, k))
        z_var = np.zeros((n, k))
        for j in range(k):
            m, sd = self.my_GPRs[j].predict(xs, return_std=True)
            z_mean[:, j] = np.asarray(m, dtype=float).ravel() * z_scale[j] + z_shift[j]
            z_var[:, j] = (np.asarray(sd, dtype=float).ravel() * z_scale[j]) ** 2
        mu_pred, var_pred = basis.reconstruct(z_mean, z_var)
        return mu_pred, np.sqrt(var_pred)


    def register_pred_perf(self, x: np.ndarray, y: Union[float, np.ndarray], only_if_fitted: bool = False,
                           prediction=None):
        """ Register the residual between prediction and true value for this data point.

        For multi-output: tracks performance per output dimension. With
        ``only_if_fitted`` the prior prediction of a node without fitted GPs is
        not registered. ``prediction`` is the mean and the uncalibrated sigma
        that the tree predicted at ``x``, each of shape (1, n_outputs), where
        that is not the node's own prediction. Returns the errors in units of
        the calibrated sigma, per output, or None if nothing was registered.
        """
        mu_pred, sigma_pred = self.predict(x, use_calibrated_sigma=False)
        if only_if_fitted and not self._is_fitted():
            return None

        # Ensure y is array of shape (n_outputs,)
        if isinstance(y, (int, float, np.floating)):
            y = np.array([y])
        else:
            y = np.atleast_1d(y).flatten()

        if self._can_condition():
            self._score_predictors(x, y[:self.n_outputs], mu_pred)
        if prediction is not None:
            mu_pred, sigma_pred = prediction

        keep_n_points = self.n_points_pred_perf - 1

        # Update for each output
        for i in range(self.n_outputs):
            self.residuals_list[i] = self.residuals_list[i][:keep_n_points]
            self.residuals_list[i] = np.insert(self.residuals_list[i], 0, y[i] - mu_pred[0, i])

            self.sigma_preds_list[i] = self.sigma_preds_list[i][:keep_n_points]
            self.sigma_preds_list[i] = np.insert(self.sigma_preds_list[i], 0, sigma_pred[0, i])
        self._variance_unit = None

        return np.abs(y[:self.n_outputs] - mu_pred[0]) / (sigma_pred[0] * np.asarray(self.sigma_scalers) + 1e-300)


    def clear_pred_perf(self):
        """Forget the registered prediction performance; sigma is then the GP's own
        until new performance is registered."""
        self.residuals_list = [np.array([]) for _ in range(self.n_outputs)]
        self.sigma_preds_list = [np.array([]) for _ in range(self.n_outputs)]
        self._variance_unit = None
        self.sigma_scalers = [1.0 for _ in range(self.n_outputs)]


    def update_sigma_scaler(self):
        """ Update the scaling factor for the prediction uncertainty.

        The scaler is the ``TARGET_COVERAGE`` quantile of the normalized
        residuals ``|residual| / sigma_pred``, i.e. the value for which a
        fraction ``TARGET_COVERAGE`` of points satisfy
        ``|residual| < scaler * sigma_pred``. For multi-output, each output is
        calibrated independently.
        """
        target_coverage = TARGET_COVERAGE

        for i in range(self.n_outputs):
            residuals_i = self.residuals_list[i]
            sigma_preds_i = self.sigma_preds_list[i]

            # Until the window is full, cover all residuals seen so far.
            if residuals_i.shape[0] < self.n_points_pred_perf:
                if len(sigma_preds_i) > 0 and np.max(sigma_preds_i) > 0:
                    self.sigma_scalers[i] = np.max(np.abs(residuals_i) / (sigma_preds_i + 1e-10))
                else:
                    self.sigma_scalers[i] = DEFAULT_SIGMA_SCALER
                continue

            ratios = np.abs(residuals_i) / (sigma_preds_i + 1e-10)
            self.sigma_scalers[i] = max(float(np.quantile(ratios, target_coverage)), 1e-9)
