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
                 use_standard_scaling: Optional[bool] = True):
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
                split dimension. Defaults to 'max_spread'.
            use_standard_scaling (Optional[bool]): If True, standardizes both X and y
                data before fitting the GP and inverse transforms predictions.
                Defaults to True.
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

        self.shared_X_data = np.array([]).reshape((0, n_features))
        self.shared_y_data = np.array([]).reshape((0, 1))

        self.n_points_since_retrain = 0
        self.n_points = 0
        self.n_shared_points = 0


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
            self.n_shared_points -= 1
        elif (not shared_point) and self.n_points > 0:
            self.my_X_data = np.delete(self.my_X_data, index, axis=0)
            self.my_y_data = np.delete(self.my_y_data, index, axis=0)
            self.n_points -= 1
        else:
            warnings.warn(f"Node {self.name}: No point left to delete.", RuntimeWarning)

        # if remove_shared and self.n_shared_points > 0:
        #     distances = np.abs(self.shared_X_data[:, self.parent_split_index] - self.parent_split_position)
        #     index_to_discard = np.argmax(distances)
        #     self.shared_X_data = np.delete(self.shared_X_data, index_to_discard, axis=0)
        #     self.shared_y_data = np.delete(self.shared_y_data, index_to_discard, axis=0)
        #     self.n_shared_points -= 1



    def store_point(self, x: np.ndarray, y: float, increment_buffer=True, shared_point=False, remove_shared=True):
        """ Add a single data point to the node. """
        # Note: Points are added to the beginning of the arrays (using np.vstack)
        if shared_point:
            self.shared_X_data = np.vstack((x, self.shared_X_data))
            self.shared_y_data = np.vstack((y, self.shared_y_data))
            # self.shared_X_data = np.append(self.shared_X_data, x, axis=0)
            # self.shared_y_data = np.append(self.shared_y_data, y, axis=0)
            self.n_shared_points += 1
        else:
            self.my_X_data = np.vstack((x, self.my_X_data))
            self.my_y_data = np.vstack((y, self.my_y_data))
            # self.my_X_data = np.append(self.my_X_data, x, axis=0)
            # self.my_y_data = np.append(self.my_y_data, y, axis=0)
            self.n_points += 1
            if increment_buffer:
                self.n_points_since_retrain += 1

        if remove_shared and self.n_shared_points > 0:
            self.delete_point(shared_point=True)

    
    def split_training_data(self):
        """ Assign the training samples of a node to its child nodes. """
        for x, y in zip(self.my_X_data, self.my_y_data):
            x = x.reshape((1, x.shape[0]))
            y = y.reshape((1, 1))
            child = self.children[int(np.random.binomial(1, self.prob_func(x)[0][0]))]
            child.store_point(x, y, increment_buffer=False)

        
    def delete_data(self, delete_own_data=True, delete_shared_data=True):
        """Deletes the data (my_X_data, my_y_data, and possibly shared_X_data, shared_y_data) from the node.

        This is typically called on a parent node after its data has been
        successfully split and passed down to its children. This helps to
        reduce memory consumption as the tree grows, as only leaf nodes
        or nodes about to be split actively need to store their full datasets.
        """
        if delete_own_data:
            del self.my_X_data
            del self.my_y_data

        if delete_shared_data:
            del self.shared_X_data
            del self.shared_y_data


    def delete_my_GPR(self):
        """Deletes the Gaussian Process Regressor (my_GPR) instance from the node.

        This method is typically called on a node after it has been split and
        is no longer a leaf node. Non-leaf nodes usually do not need to
        maintain their GPR model once their responsibilities have been passed
        to their children, thus saving memory.
        """
        del self.my_GPR


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

            if self.use_standard_scaling and X_train.shape[0] > 0:
                # Fit scalers on the combined training data
                self._fit_scalers(X_train, y_train)

                # Transform data to standardized space
                X_train_scaled = self.X_scaler.transform(X_train)
                y_train_scaled = self.y_scaler.transform(y_train)

                # Train GP on scaled data
                self.my_GPR.fit(X_train_scaled, y_train_scaled)
            else:
                # No scaling - train on raw data
                self.my_GPR.fit(X_train, y_train)

            did_train = True
        return did_train


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
