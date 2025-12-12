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
import warnings
from copy import deepcopy
from sys import float_info
from typing import Callable, Optional, Type, Union

# Third-party imports
import numpy as np
from binarytree import Node
from scipy.optimize import root_scalar
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, ExpSineSquared, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

# Local imports
from pygptreeo.default_gpr import Default_GPR

# Module-level constants
DEFAULT_OVERLAP = 0.001  # Default initial overlap for node boundaries
DEFAULT_N_POINTS_PRED_PERF = 25  # Number of recent predictions tracked for calibration
DEFAULT_SIGMA_SCALER = 10.0  # Initial sigma scaling factor for uncertainty calibration
TARGET_COVERAGE = 0.68  # Target coverage for calibrated uncertainty (1 sigma)

np.set_printoptions(suppress=True)

class GPNode(Node):
    """Represents a node within the GPTree structure.

    Each GPNode is responsible for a specific region of the input space.
    It holds the training data relevant to this region, manages its own
    Gaussian Process Regressor (my_GPR) to model the data, handles the
    splitting process into child nodes when it becomes too full or complex,
    and makes local predictions within its domain.

    Attributes:
        my_GPR (sklearn.gaussian_process.GaussianProcessRegressor): The GPR
            instance associated with this node.
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
    """
    def __init__(self, *args,
                 my_GPR: GaussianProcessRegressor,
                 Nbar: Optional[int] = 100,
                 split_position_method='median',
                 retrain_every_n_points=1,
                 name="0",
                 split_dimension_criteria='max_spread',
                 splitting_strategy: Optional[str] = 'standard',
                 use_standard_scaling: Optional[bool] = True,
                 enable_point_rejection: Optional[bool] = False,
                 rejection_threshold: Optional[float] = 1e-3,
                 min_points_before_rejection: Optional[int] = 50,
                 enable_point_merging: Optional[bool] = False,
                 merge_distance_threshold: Optional[float] = 0.01,
                 min_points_before_merging: Optional[int] = 10,
                 enable_split_evaluation: Optional[bool] = False,
                 n_split_candidates: Optional[int] = 3,
                 split_eval_train_fraction: Optional[float] = 0.6,
                 split_eval_min_points: Optional[int] = 20):
        """Initializes a GPNode.

        Args:
            *args: Arguments passed to the `binarytree.Node` parent class constructor.
                Typically, the first argument is the initial `value` for the node,
                which corresponds to `n_points`.
            my_GPR (sklearn.gaussian_process.GaussianProcessRegressor): The
                Gaussian Process Regressor instance to be used by this node.
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
                'random' (random dimension). Defaults to 'max_spread'.
            use_standard_scaling (Optional[bool]): If True, standardizes both X and y
                data before fitting the GP and inverse transforms predictions.
                Defaults to True.
            enable_point_rejection (Optional[bool]): If True, rejects new points that
                are well-predicted by the current GP. Defaults to False.
            rejection_threshold (Optional[float]): Relative error threshold below which
                points are rejected. E.g., 1e-3 means reject if |y - y_pred| / |y| < 0.001.
                Defaults to 1e-3.
            min_points_before_rejection (Optional[int]): Minimum number of points required
                before rejection starts. Ensures we have enough data before being selective.
                Defaults to 50.
            enable_point_merging (Optional[bool]): If True, merges new points with nearby
                existing points via weighted averaging. Defaults to False.
            merge_distance_threshold (Optional[float]): Distance threshold below which points
                are merged. Measured as Euclidean distance in input space. Defaults to 0.01.
            min_points_before_merging (Optional[int]): Minimum number of points required
                before merging starts. Defaults to 10.
            enable_split_evaluation (Optional[bool]): If True, evaluates multiple candidate
                split dimensions before choosing the best one. Defaults to False.
            n_split_candidates (Optional[int]): Number of candidate split dimensions to
                evaluate. Defaults to 3.
            split_eval_train_fraction (Optional[float]): Fraction of points in each region
                to use for training during split evaluation. Defaults to 0.6.
            split_eval_min_points (Optional[int]): Minimum points required in a region
                to evaluate that split. Defaults to 20.
        """
        
        super().__init__(*args)

        self.Nbar = Nbar

        self.my_GPR = my_GPR
        self.parent = None
        self.children = None

        self.split_position_method = split_position_method
        self.retrain_every_n_points = retrain_every_n_points
        self.split_dimension_criteria = split_dimension_criteria
        self.splitting_strategy = splitting_strategy
        self.use_standard_scaling = use_standard_scaling

        # Point rejection parameters
        self.enable_point_rejection = enable_point_rejection
        self.rejection_threshold = rejection_threshold
        self.min_points_before_rejection = min_points_before_rejection

        # Point merging parameters
        self.enable_point_merging = enable_point_merging
        self.merge_distance_threshold = merge_distance_threshold
        self.min_points_before_merging = min_points_before_merging

        # Statistics for monitoring merging behavior
        self.n_merges = 0
        self.merge_counts = None  # Will be initialized with data

        # Split evaluation parameters
        self.enable_split_evaluation = enable_split_evaluation
        self.n_split_candidates = n_split_candidates
        self.split_eval_train_fraction = split_eval_train_fraction
        self.split_eval_min_points = split_eval_min_points

        self.parent_split_index = None
        self.parent_split_position = None

        # Standard scalers for X and y (fitted during GP training)
        self.X_scaler = None
        self.y_scaler = None

        self.is_left = None
        self.is_leaf = True
        
        self.n_points = 0
        self.n_points_since_retrain = 0

        self.my_X_data = None
        self.my_y_data = None
        self.n_features = None

        self.shared_X_data = None
        self.shared_y_data = None
        self.n_shared_points = 0

        self.split_index = 0       # 'j' in the DLGP article
        self.split_position = 0.0  # 's' in the DLGP article
        self.overlap = DEFAULT_OVERLAP  # 'o' in the DLGP article
        self.name = name

        self.n_points_pred_perf = DEFAULT_N_POINTS_PRED_PERF
        self.residuals = np.array([])
        self.mu_preds = np.array([])
        self.sigma_preds = np.array([])

        self.sigma_scaler = DEFAULT_SIGMA_SCALER
        self.sigma_scaler_init = DEFAULT_SIGMA_SCALER

        print(f"Created node {self.name}")


    def _print_debug_status(self):
        print(f"DEBUG: node {self.name}:  retrain_every_n_points: {self.retrain_every_n_points}  n_points_since_retrain: {self.n_points_since_retrain}  n_points: {self.n_points}  my_X_data.shape: {self.my_X_data.shape}  n_points: {self.n_points}  shared_X_data.shape: {self.shared_X_data.shape}  n_shared_points: {self.n_shared_points}", flush=True)


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
        self.my_y_data = np.array([]).reshape((0, 1))
        self.my_sigma_data = np.array([]).reshape((0, 1))  # Per-point uncertainties

        self.shared_X_data = np.array([]).reshape((0, n_features))
        self.shared_y_data = np.array([]).reshape((0, 1))
        self.shared_sigma_data = np.array([]).reshape((0, 1))  # Shared uncertainties

        self.n_points_since_retrain = 0
        self.n_points = 0
        self.n_shared_points = 0

        # Initialize merge tracking - each point tracks how many merges it represents
        self.merge_counts = np.array([]).reshape((0, 1))


    def generate_children(self, GPR: Type[GaussianProcessRegressor], n_features: int):
        """ Grow the GPtree by adding two GPNodes as children of the current GPNode. """

        # Settings that will be passed on to the child nodes
        node_config_kwargs = {
            'Nbar': self.Nbar,
            'split_position_method': self.split_position_method,
            'retrain_every_n_points': self.retrain_every_n_points,
            'split_dimension_criteria': self.split_dimension_criteria,
            'splitting_strategy': self.splitting_strategy,
            'use_standard_scaling': self.use_standard_scaling,
            'enable_point_rejection': self.enable_point_rejection,
            'rejection_threshold': self.rejection_threshold,
            'min_points_before_rejection': self.min_points_before_rejection,
            'enable_point_merging': self.enable_point_merging,
            'merge_distance_threshold': self.merge_distance_threshold,
            'min_points_before_merging': self.min_points_before_merging,
            'enable_split_evaluation': self.enable_split_evaluation,
            'n_split_candidates': self.n_split_candidates,
            'split_eval_train_fraction': self.split_eval_train_fraction,
            'split_eval_min_points': self.split_eval_min_points,
        }

        # Create child nodes with a copy of the parent GP
        self.left = GPNode(0, my_GPR=deepcopy(self.my_GPR), name=self.name + "0", **node_config_kwargs)
        self.right = GPNode(0, my_GPR=deepcopy(self.my_GPR), name=self.name + "1", **node_config_kwargs)
        
        self.left.is_left = True
        self.right.is_left = False

        self.children = [self.left, self.right]
        self.is_leaf = False

        for child in self.children:
            child.parent = self
            child.init_data_set(n_features)

        # Copy important numbers over to the child nodes
        self.left.residuals = self.residuals.copy()
        self.right.residuals = self.residuals.copy()

        self.left.mu_preds = self.mu_preds.copy()
        self.right.mu_preds = self.mu_preds.copy()

        self.left.sigma_preds = self.sigma_preds.copy()
        self.right.sigma_preds = self.sigma_preds.copy()

        self.left.sigma_scaler = self.sigma_scaler
        self.right.sigma_scaler = self.sigma_scaler

        self.left.sigma_scaler_init = self.sigma_scaler_init
        self.right.sigma_scaler_init = self.sigma_scaler_init

        # Copy scalers so children can use parent's GP correctly
        # The children get a copy of the parent's trained GP, which was trained on
        # scaled data using the parent's scalers. So children need the same scalers
        # to properly scale inputs before prediction.
        if self.use_standard_scaling and self.X_scaler is not None:
            self.left.X_scaler = deepcopy(self.X_scaler)
            self.left.y_scaler = deepcopy(self.y_scaler)
            self.right.X_scaler = deepcopy(self.X_scaler)
            self.right.y_scaler = deepcopy(self.y_scaler)


    def delete_point(self, index=-1, shared_point=True):
        """ Remove a single data point from the node. """
        if shared_point and self.n_shared_points > 0:
            self.shared_X_data = np.delete(self.shared_X_data, index, axis=0)
            self.shared_y_data = np.delete(self.shared_y_data, index, axis=0)
            self.shared_sigma_data = np.delete(self.shared_sigma_data, index, axis=0)
            self.n_shared_points -= 1
        elif (not shared_point) and self.n_points > 0:
            self.my_X_data = np.delete(self.my_X_data, index, axis=0)
            self.my_y_data = np.delete(self.my_y_data, index, axis=0)
            self.my_sigma_data = np.delete(self.my_sigma_data, index, axis=0)
            if self.merge_counts is not None:
                self.merge_counts = np.delete(self.merge_counts, index, axis=0)
            self.n_points -= 1
        else:
            warnings.warn(f"Node {self.name}: No point left to delete.", RuntimeWarning)

        # if remove_shared and self.n_shared_points > 0:
        #     distances = np.abs(self.shared_X_data[:, self.parent_split_index] - self.parent_split_position)
        #     index_to_discard = np.argmax(distances)
        #     self.shared_X_data = np.delete(self.shared_X_data, index_to_discard, axis=0)
        #     self.shared_y_data = np.delete(self.shared_y_data, index_to_discard, axis=0)
        #     self.n_shared_points -= 1



    def store_point(self, x: np.ndarray, y: float, sigma: float, increment_buffer=True, shared_point=False, remove_shared=True):
        """ Add a single data point to the node.

        Parameters
        ----------
        x : np.ndarray
            Input features
        y : float
            Target value
        sigma : float
            Uncertainty (standard deviation) for this point
        increment_buffer : bool
            Whether to increment the retrain buffer
        shared_point : bool
            Whether this is a shared point from gradual splitting
        remove_shared : bool
            Whether to remove one shared point when adding
        """
        # Note: Points are added to the beginning of the arrays (using np.vstack)
        if shared_point:
            self.shared_X_data = np.vstack((x, self.shared_X_data))
            self.shared_y_data = np.vstack((y, self.shared_y_data))
            self.shared_sigma_data = np.vstack((sigma, self.shared_sigma_data))
            # self.shared_X_data = np.append(self.shared_X_data, x, axis=0)
            # self.shared_y_data = np.append(self.shared_y_data, y, axis=0)
            self.n_shared_points += 1
        else:
            self.my_X_data = np.vstack((x, self.my_X_data))
            self.my_y_data = np.vstack((y, self.my_y_data))
            self.my_sigma_data = np.vstack((sigma, self.my_sigma_data))
            # self.my_X_data = np.append(self.my_X_data, x, axis=0)
            # self.my_y_data = np.append(self.my_y_data, y, axis=0)
            # Track merge count for this new point (starts at 0)
            if self.merge_counts is not None:
                self.merge_counts = np.vstack((np.array([[0]]), self.merge_counts))
            self.n_points += 1
            if increment_buffer:
                self.n_points_since_retrain += 1

        if remove_shared and self.n_shared_points > 0:
            self.delete_point(shared_point=True)

    
    def split_training_data(self):
        """ Assign the training samples of a node to its child nodes. """
        for x, y, sigma in zip(self.my_X_data, self.my_y_data, self.my_sigma_data):
            x = x.reshape((1, x.shape[0]))
            y = y.reshape((1, 1))
            sigma = sigma.reshape((1, 1))
            child = self.children[int(np.random.binomial(1, self.prob_func(x)[0][0]))]
            child.store_point(x, y, sigma, increment_buffer=False)

        
    def delete_data(self, delete_own_data=True, delete_shared_data=True):
        """Deletes the data (my_X_data, my_y_data, my_sigma_data, and possibly shared_X_data, shared_y_data, shared_sigma_data) from the node.

        This is typically called on a parent node after its data has been
        successfully split and passed down to its children. This helps to
        reduce memory consumption as the tree grows, as only leaf nodes
        or nodes about to be split actively need to store their full datasets.
        """
        if delete_own_data:
            del self.my_X_data
            del self.my_y_data
            del self.my_sigma_data

        if delete_shared_data:
            del self.shared_X_data
            del self.shared_y_data
            del self.shared_sigma_data


    def delete_my_GPR(self):
        """Deletes the Gaussian Process Regressor (my_GPR) instance from the node.

        This method is typically called on a node after it has been split and
        is no longer a leaf node. Non-leaf nodes usually do not need to
        maintain their GPR model once their responsibilities have been passed
        to their children, thus saving memory.
        """
        del self.my_GPR


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


    def merge_with_point(self, x: np.ndarray, y: float, sigma: float, nearest_idx: int):
        """Merges new point (x, y, sigma) with existing point at nearest_idx via inverse-variance weighted averaging.

        Weights are based on per-point uncertainties (standard deviations):
        - Lower uncertainty = higher weight (more confident measurement)
        - Weight = 1 / σ²

        Args:
            x (np.ndarray): Input features of new point
            y (float): Target value of new point
            sigma (float): Uncertainty (standard deviation) of new point
            nearest_idx (int): Index of existing point to merge with

        Returns:
            bool: True if merge was successful
        """
        # Get existing point
        x_old = self.my_X_data[nearest_idx:nearest_idx+1]
        y_old = self.my_y_data[nearest_idx, 0]
        sigma_old = self.my_sigma_data[nearest_idx, 0]

        # Convert standard deviations to variances
        var_old = sigma_old ** 2
        var_new = sigma ** 2

        # Add epsilon for numerical stability
        epsilon = 1e-10
        w_old = 1.0 / (var_old + epsilon)
        w_new = 1.0 / (var_new + epsilon)
        w_total = w_old + w_new

        # Weighted average for both x and y (using inverse-variance weighting)
        x_merged = (w_old * x_old + w_new * x) / w_total
        y_merged = (w_old * y_old + w_new * y) / w_total

        # Merge variance and convert back to standard deviation
        var_merged = 1.0 / w_total
        sigma_merged = np.sqrt(var_merged)

        # Update the existing point with merged values
        self.my_X_data[nearest_idx] = x_merged[0]
        self.my_y_data[nearest_idx, 0] = y_merged
        self.my_sigma_data[nearest_idx, 0] = sigma_merged

        # Update merge count for this point
        self.merge_counts[nearest_idx, 0] += 1
        self.n_merges += 1

        print(f"Node {self.name}: Merged points (dist={np.linalg.norm(x - x_old):.2e}, "
              f"sigma_old={sigma_old:.2e}, sigma_new={sigma:.2e}, sigma_merged={sigma_merged:.2e}, "
              f"total_merges={int(self.merge_counts[nearest_idx, 0])})")

        return True


    def should_merge_point(self, x: np.ndarray, y: float, sigma: float):
        """Determines if a new point should be merged with an existing nearby point.

        A point is merged if:
        1. Point merging is enabled
        2. Node has enough points (>= min_points_before_merging)
        3. GP has been trained
        4. Nearest neighbor is within merge_distance_threshold

        If merging occurs, the existing point is updated in-place.

        Args:
            x (np.ndarray): Input features of new point
            y (float): Target value of new point
            sigma (float): Uncertainty (standard deviation) of new point

        Returns:
            bool: True if point was merged (don't add it), False otherwise (add as new point)
        """
        # Point merging disabled or not enough points yet
        if not self.enable_point_merging:
            return False

        if self.n_points < self.min_points_before_merging:
            return False

        # GP not trained yet
        if not hasattr(self.my_GPR, 'kernel_'):
            return False

        # Find nearest neighbor
        nearest_idx, nearest_dist = self.find_nearest_neighbor(x)
        if nearest_idx is None:
            return False

        # Check if distance is below threshold
        if nearest_dist < self.merge_distance_threshold:
            # Merge the points
            return self.merge_with_point(x, y, sigma, nearest_idx)

        return False


    def should_reject_point(self, x: np.ndarray, y: float):
        """Determines if a new training point should be rejected.

        A point is rejected if:
        1. Point rejection is enabled
        2. Node has enough points (>= min_points_before_rejection)
        3. GP has been trained (has kernel_)
        4. The point is well-predicted (relative error < rejection_threshold)

        Args:
            x (np.ndarray): Input features of the new point
            y (float): Target value of the new point

        Returns:
            bool: True if point should be rejected, False if it should be stored
        """
        # Point rejection disabled or not enough points yet
        if not self.enable_point_rejection:
            return False

        if self.n_points < self.min_points_before_rejection:
            return False

        # GP not trained yet
        if not hasattr(self.my_GPR, 'kernel_'):
            return False

        # Get prediction for this point
        try:
            y_pred, _ = self.predict(x, return_std=True, use_calibrated_sigma=False)
        except Exception:
            # If prediction fails, don't reject (be conservative)
            return False

        # Compute relative error
        abs_error = np.abs(y - y_pred[0])
        relative_error = abs_error / np.maximum(np.abs(y), 1e-10)

        # Reject if error is below threshold
        is_rejected = bool(relative_error < self.rejection_threshold)

        if is_rejected:
            print(f"Node {self.name}: Rejected point (rel_err={float(relative_error):.2e} < {self.rejection_threshold:.2e})")

        return is_rejected

    def _fit_scalers(self, X: np.ndarray, y: np.ndarray):
        """Fits StandardScalers on the combined training data.

        This method creates and fits separate StandardScaler instances for
        the feature data (X) and target data (y). These scalers are stored
        in the node and used to transform data before GP training and to
        inverse transform predictions.

        Args:
            X (np.ndarray): Feature data to fit the X scaler on.
                Shape: (n_samples, n_features)
            y (np.ndarray): Target data to fit the y scaler on.
                Shape: (n_samples, 1)

        Note:
            StandardScaler handles edge cases automatically:
            - Single point: sets scale=1, mean=point_value
            - Zero variance in a dimension: sets scale=1 for that dimension
        """
        # Fit X scaler (per-feature standardization)
        self.X_scaler = StandardScaler()
        self.X_scaler.fit(X)

        # Fit y scaler (single output standardization)
        self.y_scaler = StandardScaler()
        self.y_scaler.fit(y)


    def fit_my_GPR(self, force_training=False):
        """Fits the node's Gaussian Process Regressor (GPR) to its local data.

        Training is triggered if the number of new points in the buffer
        (`n_points_since_retrain`) reaches `retrain_every_n_points`, if the node
        is full (`n_points >= Nbar`), or if `force_training` is True.

        If `use_standard_scaling` is enabled, this method first fits StandardScalers
        on the combined training data, then transforms the data before fitting the GP.
        The raw (unscaled) data remains stored in the node.

        Args:
            force_training (bool): If True, the GPR is retrained even if the
                usual buffer or fullness conditions are not met. Defaults to False.

        Returns:
            bool: True if the GPR was trained in this call, False otherwise.
        """
        did_train = False
        # Only train the GP if the buffer is full, the node is full, or if force_training=True
        if (self.n_points_since_retrain >= self.retrain_every_n_points) or (self.n_points >= self.Nbar) or force_training:
            self.n_points_since_retrain = 0

            # Combine own points and shared points
            X_train = np.vstack((self.my_X_data, self.shared_X_data))
            y_train = np.vstack((self.my_y_data, self.shared_y_data))
            sigma_train = np.vstack((self.my_sigma_data, self.shared_sigma_data))

            if self.use_standard_scaling and X_train.shape[0] > 0:
                # Fit scalers on the combined training data
                self._fit_scalers(X_train, y_train)

                # Transform data to standardized space
                X_train_scaled = self.X_scaler.transform(X_train)
                y_train_scaled = self.y_scaler.transform(y_train)

                # Transform uncertainties (std dev → variance → scaled variance)
                # σ_scaled = σ_original / y_scale
                # α_scaled = σ_scaled² = σ_original² / y_scale²
                y_scale = self.y_scaler.scale_[0]
                sigma_train_scaled = sigma_train / y_scale
                alpha_train_scaled = sigma_train_scaled ** 2  # Convert to variance

                # Set GP alpha and train
                self.my_GPR.alpha = alpha_train_scaled.flatten()
                self.my_GPR.fit(X_train_scaled, y_train_scaled)
            else:
                # No scaling - convert std dev to variance
                alpha_train = sigma_train ** 2  # Convert to variance
                self.my_GPR.alpha = alpha_train.flatten()
                self.my_GPR.fit(X_train, y_train)

            did_train = True
        return did_train


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
                    _, sigma = self.predict(test_point, return_std=True, use_calibrated_sigma=False)
                    uncertainties.append(sigma[0])

            # Aggregate uncertainties for this dimension
            # Use mean uncertainty as the score
            uncertainty_scores[dim] = np.mean(uncertainties)

        return uncertainty_scores


    def evaluate_candidate_split(self, split_index: int, split_position: float, overlap: float):
        """Evaluates prediction performance of a candidate split.

        For a given split (dimension and position), this method:
        1. Partitions data into left/right regions using prob_func
        2. For each region:
           - Splits into train/test subsets
           - Trains a small GP on train subset
           - Evaluates RMSE on test subset
        3. Returns combined error score (lower is better)

        Args:
            split_index (int): Dimension to split on
            split_position (float): Position along dimension to split at
            overlap (float): Overlap parameter for prob_func

        Returns:
            float: Combined RMSE score (weighted average of left and right errors)
                   Returns np.inf if evaluation fails or insufficient data
        """
        if self.my_X_data.shape[0] < 2 * self.split_eval_min_points:
            # Not enough data to evaluate split
            return np.inf

        # Temporarily set split parameters for prob_func
        old_split_index = self.split_index
        old_split_position = self.split_position
        old_overlap = self.overlap

        self.split_index = split_index
        self.split_position = split_position
        self.overlap = overlap

        try:
            # Partition data into left/right using prob_func
            left_indices = []
            right_indices = []

            for i in range(self.my_X_data.shape[0]):
                x = self.my_X_data[i:i+1]
                prob_right = self.prob_func(x)[0][0]
                # Deterministically assign based on probability
                if prob_right < 0.5:
                    left_indices.append(i)
                else:
                    right_indices.append(i)

            left_indices = np.array(left_indices)
            right_indices = np.array(right_indices)

            # Check if we have enough points in each region
            if len(left_indices) < self.split_eval_min_points or len(right_indices) < self.split_eval_min_points:
                return np.inf

            # Evaluate left region
            X_left = self.my_X_data[left_indices]
            y_left = self.my_y_data[left_indices]

            n_train_left = max(int(len(left_indices) * self.split_eval_train_fraction), 10)
            n_train_left = min(n_train_left, len(left_indices) - 5)  # Leave at least 5 for testing

            # Randomly split into train/test
            indices_left = np.random.permutation(len(left_indices))
            train_idx_left = indices_left[:n_train_left]
            test_idx_left = indices_left[n_train_left:]

            if len(test_idx_left) == 0:
                return np.inf

            # Train small GP on left train subset
            gp_left = deepcopy(self.my_GPR)
            X_train_left = X_left[train_idx_left]
            y_train_left = y_left[train_idx_left]

            # Apply scaling if enabled
            if self.use_standard_scaling:
                from sklearn.preprocessing import StandardScaler
                scaler_X_left = StandardScaler()
                scaler_y_left = StandardScaler()
                X_train_left_scaled = scaler_X_left.fit_transform(X_train_left)
                y_train_left_scaled = scaler_y_left.fit_transform(y_train_left)
                gp_left.fit(X_train_left_scaled, y_train_left_scaled)

                # Predict on test set
                X_test_left_scaled = scaler_X_left.transform(X_left[test_idx_left])
                y_pred_left_scaled = gp_left.predict(X_test_left_scaled, return_std=False)
                y_pred_left = scaler_y_left.inverse_transform(y_pred_left_scaled.reshape(-1, 1)).flatten()
            else:
                gp_left.fit(X_train_left, y_train_left)
                y_pred_left = gp_left.predict(X_left[test_idx_left], return_std=False)

            y_true_left = y_left[test_idx_left].flatten()
            rmse_left = np.sqrt(np.mean((y_true_left - y_pred_left)**2))

            # Evaluate right region
            X_right = self.my_X_data[right_indices]
            y_right = self.my_y_data[right_indices]

            n_train_right = max(int(len(right_indices) * self.split_eval_train_fraction), 10)
            n_train_right = min(n_train_right, len(right_indices) - 5)

            indices_right = np.random.permutation(len(right_indices))
            train_idx_right = indices_right[:n_train_right]
            test_idx_right = indices_right[n_train_right:]

            if len(test_idx_right) == 0:
                return np.inf

            gp_right = deepcopy(self.my_GPR)
            X_train_right = X_right[train_idx_right]
            y_train_right = y_right[train_idx_right]

            if self.use_standard_scaling:
                from sklearn.preprocessing import StandardScaler
                scaler_X_right = StandardScaler()
                scaler_y_right = StandardScaler()
                X_train_right_scaled = scaler_X_right.fit_transform(X_train_right)
                y_train_right_scaled = scaler_y_right.fit_transform(y_train_right)
                gp_right.fit(X_train_right_scaled, y_train_right_scaled)

                X_test_right_scaled = scaler_X_right.transform(X_right[test_idx_right])
                y_pred_right_scaled = gp_right.predict(X_test_right_scaled, return_std=False)
                y_pred_right = scaler_y_right.inverse_transform(y_pred_right_scaled.reshape(-1, 1)).flatten()
            else:
                gp_right.fit(X_train_right, y_train_right)
                y_pred_right = gp_right.predict(X_right[test_idx_right], return_std=False)

            y_true_right = y_right[test_idx_right].flatten()
            rmse_right = np.sqrt(np.mean((y_true_right - y_pred_right)**2))

            # Combined score: weighted average by number of test points
            n_test_left = len(test_idx_left)
            n_test_right = len(test_idx_right)
            combined_rmse = (n_test_left * rmse_left + n_test_right * rmse_right) / (n_test_left + n_test_right)

            return combined_rmse

        except Exception as e:
            # If evaluation fails, return infinity (worst score)
            warnings.warn(f"Node {self.name}: Split evaluation failed: {e}", RuntimeWarning)
            return np.inf

        finally:
            # Restore original split parameters
            self.split_index = old_split_index
            self.split_position = old_split_position
            self.overlap = old_overlap


    def find_best_split_dimension(self, theta: float):
        """Finds the best split dimension by evaluating multiple candidates.

        Tests several candidate dimensions and selects the one with lowest
        prediction error. Candidates are chosen using different criteria
        (max_variance, max_spread, max_uncertainty).

        Args:
            theta (float): Theta parameter for computing overlap

        Returns:
            tuple: (best_split_index, best_split_position) or (None, None) if evaluation fails
        """
        if self.my_X_data.shape[0] < 2 * self.split_eval_min_points:
            # Not enough data for evaluation
            return None, None

        # Generate candidate dimensions to test
        candidates = []

        # Candidate 1: Max variance
        if self.my_X_data.shape[0] > 1:
            variances = np.var(self.my_X_data, axis=0)
            dim_max_var = np.argmax(variances)
            candidates.append(('max_variance', dim_max_var))

        # Candidate 2: Max spread
        if self.my_X_data.shape[0] > 0:
            spreads = np.max(self.my_X_data, axis=0) - np.min(self.my_X_data, axis=0)
            dim_max_spread = np.argmax(spreads)
            candidates.append(('max_spread', dim_max_spread))

        # Candidate 3: Max uncertainty (if GP is trained)
        if hasattr(self.my_GPR, 'kernel_') and self.my_X_data.shape[0] > 1:
            uncertainty_scores = self._compute_dimensional_uncertainty()
            dim_max_unc = np.argmax(uncertainty_scores)
            candidates.append(('max_uncertainty', dim_max_unc))

        # Remove duplicates (keep unique dimensions)
        unique_candidates = []
        seen_dims = set()
        for name, dim in candidates:
            if dim not in seen_dims:
                unique_candidates.append((name, dim))
                seen_dims.add(dim)

        # Limit to n_split_candidates
        unique_candidates = unique_candidates[:self.n_split_candidates]

        if len(unique_candidates) == 0:
            return None, None

        print(f"Node {self.name}: Evaluating {len(unique_candidates)} candidate splits...")

        # Evaluate each candidate
        best_score = np.inf
        best_index = None
        best_position = None

        for name, split_index in unique_candidates:
            # Compute split position (use median)
            split_position = np.median(self.my_X_data[:, split_index])

            # Compute overlap
            spread = np.max(self.my_X_data[:, split_index]) - np.min(self.my_X_data[:, split_index])
            overlap = theta * spread

            # Evaluate this candidate
            score = self.evaluate_candidate_split(split_index, split_position, overlap)

            print(f"  Candidate: {name} (dim {split_index}), score: {score:.4e}")

            if score < best_score:
                best_score = score
                best_index = split_index
                best_position = split_position

        if best_index is not None:
            print(f"  Best: dim {best_index} with score {best_score:.4e}")

        return best_index, best_position


    def compute_split_position_and_overlap(self, theta: float):
        """Computes the split dimension, position, and overlap for node splitting.

        This method determines which feature dimension (`self.split_index`) to
        split on, typically by selecting the dimension with the maximum spread
        (range of values) in the node's current training data (`self.my_X_data`).

        If `enable_split_evaluation` is True, it evaluates multiple candidate
        splits and selects the best one based on prediction performance.

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

        # If split evaluation is enabled, find best split by testing candidates
        if self.enable_split_evaluation:
            best_index, best_position = self.find_best_split_dimension(theta)
            if best_index is not None:
                self.split_index = best_index
                self.split_position = best_position
                # Compute overlap for the chosen dimension
                current_dim_spread = np.max(self.my_X_data[:, self.split_index]) - np.min(self.my_X_data[:, self.split_index])
                self.overlap = theta * current_dim_spread
                return  # Done - we have split_index, split_position, and overlap
            # If evaluation failed, fall through to use default criteria

        # Determine the split index based on the chosen criteria
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
            if self.my_X_data.shape[0] > 1 and hasattr(self.my_GPR, 'kernel_'):
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
        elif self.split_dimension_criteria == 'random':
            if self.n_features > 0:
                self.split_index = np.random.randint(0, self.n_features)
            else:
                self.split_index = 0 # Default to 0 if no features
        else:
            raise ValueError(f"Unknown split_dimension_criteria: {self.split_dimension_criteria}")

        # Calculate spread for the chosen split_index, used for overlap calculation
        # and potentially for split_position if data is scarce for other methods.
        if self.my_X_data.shape[0] > 0:
            current_dim_spread = np.max(self.my_X_data[:, self.split_index]) - np.min(self.my_X_data[:, self.split_index])
        else:
            current_dim_spread = 0.0 # Default if no data

        self.split_position = None
        if self.my_X_data.shape[0] == 0: # No data, place split in the middle (0 if not scaled) or handle as error?
            self.split_position = 0.0 
        elif self.split_position_method == 'median':
            if self.my_X_data.shape[0] > 0:
                self.split_position = np.median(self.my_X_data[:, self.split_index])
            else: # Should not happen if split_index logic is robust
                self.split_position = 0.0
        elif self.split_position_method == 'mean':
            if self.my_X_data.shape[0] > 0:
                self.split_position = np.mean(self.my_X_data[:, self.split_index])
            else: # Should not happen
                self.split_position = 0.0
        elif self.split_position_method == 'random':
            if self.my_X_data.shape[0] > 0:
                min_val = np.min(self.my_X_data[:, self.split_index])
                max_val = np.max(self.my_X_data[:, self.split_index])
                if min_val == max_val: # All points are the same
                    self.split_position = min_val
                else:
                    self.split_position = np.random.uniform(min_val, max_val, 1)[0]
            else: # Should not happen
                self.split_position = 0.0
        elif self.split_position_method == 'randomchoice':
            if self.my_X_data.shape[0] > 0:
                self.split_position = np.random.choice(self.my_X_data[:, self.split_index])
            else: # Should not happen
                self.split_position = 0.0
        else:
            raise ValueError(f"Unknown split_position_method argument: '{self.split_position_method}'. The valid options are 'median', 'mean', 'random' and 'randomchoice'")

        self.overlap = theta * current_dim_spread


    def prob_func(self, x: np.array):
        """ The default probability function as suggested in the DLGP article. """
        prob = (x[:, self.split_index] - self.split_position)/self.overlap + 0.5
        prob[prob < 0] = 0
        prob[prob > 1] = 1

        prob.shape = (x.shape[0], 1)

        return prob

    
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


    def predict(self, x: np.ndarray, return_std=True, use_calibrated_sigma=False):
        """Evaluates the prediction from this node's GPR at input point(s) x.

        The input x is expected to be in the original (unscaled) space. If
        `use_standard_scaling` is enabled, this method transforms x to the
        standardized space before calling the GP, then inverse transforms
        the predictions back to the original space.

        Args:
            x (np.ndarray): The input point(s) at which to make predictions,
                in the original (unscaled) space.
                Shape should be (n_samples, n_features).
            return_std (bool): Whether to return the standard deviation of the
                prediction. Defaults to True.
            use_calibrated_sigma (bool): If True, the returned `sigma_pred`
                (standard deviation) is scaled by the node's `self.sigma_scaler`
                attribute. This scaler is intended to calibrate the uncertainty
                estimates. Defaults to False.

        Returns:
            tuple:
                - mu_pred (np.ndarray): The mean prediction(s) in original space.
                - sigma_pred (np.ndarray): The standard deviation of the
                  prediction(s) in original space. Only returned if `return_std` is True.
        """
        if self.use_standard_scaling and self.X_scaler is not None:
            # Transform input to standardized space
            x_scaled = self.X_scaler.transform(x)

            # Get prediction in scaled space
            mu_scaled, sigma_scaled = self.my_GPR.predict(x_scaled, return_std=return_std)

            # Inverse transform mean: mu_original = mu_scaled * scale_y + mean_y
            mu_pred = self.y_scaler.inverse_transform(mu_scaled.reshape(-1, 1)).flatten()

            # Transform std: sigma_original = sigma_scaled * scale_y
            # When y is scaled by scale_y, variance scales by scale_y^2, so std scales by scale_y
            sigma_pred = sigma_scaled * self.y_scaler.scale_[0]
        else:
            # No scaling - predict on raw data
            mu_pred, sigma_pred = self.my_GPR.predict(x, return_std=return_std)

        if use_calibrated_sigma:
            sigma_pred = sigma_pred * self.sigma_scaler

        return mu_pred, sigma_pred


    def register_pred_perf(self, x: np.ndarray, y: float):
        """ Register the residual between prediction and true value at for this data point. """
        mu_pred, sigma_pred = self.predict(x, return_std=True, use_calibrated_sigma=False)

        keep_n_points = self.n_points_pred_perf - 1

        self.residuals = self.residuals[:keep_n_points]
        self.residuals = np.insert(self.residuals, 0, y - mu_pred)

        self.mu_preds = self.mu_preds[:keep_n_points]
        self.mu_preds = np.insert(self.mu_preds, 0, mu_pred)

        self.sigma_preds = self.sigma_preds[:keep_n_points]
        self.sigma_preds = np.insert(self.sigma_preds, 0, sigma_pred)


    def update_sigma_scaler(self):
        """ Update the scaling factor for the prediction uncertainty. """
        target_coverage = TARGET_COVERAGE

        # Before we have collected self.n_points_pred_perf points, just set the 
        # self.sigma_scaler such that all residuals are covered 
        if self.residuals.shape[0] < self.n_points_pred_perf:
            self.sigma_scaler = np.max(np.abs(self.residuals) / self.sigma_preds)
            self.sigma_scaler_init = self.sigma_scaler
            return 

        def coverage_deviation(x):
            deviation = np.sum(np.abs(self.residuals) < x * self.sigma_preds) / self.n_points_pred_perf - target_coverage
            return deviation

        # Make sure we start from a range that brackets the root of coverage_deviation
        x_bracket = [0.0, 2 * self.sigma_scaler_init]
        while coverage_deviation(x_bracket[0]) * coverage_deviation(x_bracket[1]) > 0:
            self.sigma_scaler_init *= 2
            x_bracket[1] = 2 * self.sigma_scaler_init

        sol = root_scalar(coverage_deviation, x0=self.sigma_scaler_init, bracket=x_bracket, maxiter=50)
        if sol.converged:
            self.sigma_scaler = np.max([sol.root, 1e-9])
