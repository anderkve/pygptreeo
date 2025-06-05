import numpy as np
from binarytree import Node
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ExpSineSquared, ConstantKernel, WhiteKernel
from typing import Callable, Optional, Type, Union
from scipy.optimize import root_scalar
from copy import deepcopy
from sys import float_info
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

from pygptreeo.default_gpr import Default_GPR

np.set_printoptions(suppress=True)

class GPNode(Node):
    """Represents a node within the GPTree structure.

    Each GPNode is responsible for a specific region of the input space.
    It holds the training data relevant to this region, manages its list of
    Gaussian Process Regressors (my_GPRs) to model the data, handles the
    splitting process into child nodes when it becomes too full or complex,
    and makes local predictions within its domain. If `n_GPs_per_node > 1`,
    predictions are combined using Product of Experts, and fitting may involve
    data partitioning.

    Attributes:
        my_GPRs (list[sklearn.gaussian_process.GaussianProcessRegressor]): List of GPR
            instances associated with this node.
        n_GPs_per_node (int): The number of Gaussian Process Regressors managed by this node.
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
                 my_GPRs: Union[GaussianProcessRegressor, list[GaussianProcessRegressor]],
                 Nbar: Optional[int] = 100,
                 split_position_method='median',
                 retrain_every_n_points=1,
                 name="0",
                 split_dimension_criteria='max_spread',
                 splitting_strategy: Optional[str] = 'standard',
                 n_GPs_per_node: int = 1,
                 n_train: Optional[int] = None):
        """Initializes a GPNode.

        Args:
            *args: Arguments passed to the `binarytree.Node` parent class constructor.
                Typically, the first argument is the initial `value` for the node,
                which corresponds to `n_points`.
            my_GPRs (Union[sklearn.gaussian_process.GaussianProcessRegressor, list[sklearn.gaussian_process.GaussianProcessRegressor]]): The
                Gaussian Process Regressor instance or a list of GPR instances to be used by this node.
                If a single GPR is provided, it's wrapped in a list. The number of GPRs in this list
                should ideally match `n_GPs_per_node` if `n_GPs_per_node > 1`, or `GPNode` will operate
                with the provided GPRs (e.g., using the first one if `len(my_GPRs) < n_GPs_per_node` in some operations,
                or all of them if `len(my_GPRs) == n_GPs_per_node`).
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
            n_GPs_per_node (int): The number of Gaussian Process Regressors this node
                should manage. Defaults to 1. If `len(my_GPRs)` does not match
                `n_GPs_per_node`, behavior might vary based on method (e.g. fitting might
                train all GPRs in `my_GPRs`, prediction might use all in `my_GPRs` for PoE).
            n_train (Optional[int]): The number of training points. Stored as an attribute.
        """
        
        super().__init__(*args)
        self.n_train = n_train

        self.Nbar = Nbar
        self.n_GPs_per_node = n_GPs_per_node

        if not isinstance(my_GPRs, list):
            my_GPRs = [my_GPRs]
        self.my_GPRs = my_GPRs
        self.parent = None
        self.children = None

        self.split_position_method = split_position_method
        self.retrain_every_n_points = retrain_every_n_points
        self.split_dimension_criteria = split_dimension_criteria
        self.splitting_strategy = splitting_strategy
        
        self.parent_split_index = None
        self.parent_split_position = None

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
        self.overlap = 0.001       # 'o' in the DLGP article
        self.name = name

        self.n_points_pred_perf = 25
        self.residuals = np.array([])
        self.mu_preds = np.array([])
        self.sigma_preds = np.array([])

        self.sigma_scaler = 10
        self.sigma_scaler_init = 10
        self.gp_rmse_scores = None

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
            'n_train': self.n_train,
        }

        # Create child nodes with a copy of the parent GP
        self.left = GPNode(0, my_GPRs=deepcopy(self.my_GPRs), name=self.name + "0", n_GPs_per_node=self.n_GPs_per_node, **node_config_kwargs)
        self.right = GPNode(0, my_GPRs=deepcopy(self.my_GPRs), name=self.name + "1", n_GPs_per_node=self.n_GPs_per_node, **node_config_kwargs)
        
        self.left.is_left = True
        self.right.is_left = False

        self.children = [self.left, self.right]
        self.is_leaf = False

        for child in self.children:
            child.parent = self
            child.init_data_set(n_features)
            child.gp_rmse_scores = None # Initialize for new children

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


    def delete_my_GPRs(self):
        """Deletes the list of Gaussian Process Regressors (my_GPRs) from the node.

        This method is typically called on a node after it has been split and
        is no longer a leaf node. Non-leaf nodes usually do not need to
        maintain their GPR model once their responsibilities have been passed
        to their children, thus saving memory.
        """
        del self.my_GPRs


    def fit_my_GPR(self, force_training=False):
        """Fits the node's Gaussian Process Regressor (GPR) to its local data.

        Training is triggered if the number of new points in the buffer
        (`n_points_since_retrain`) reaches `retrain_every_n_points`, if the node
        is full (`n_points >= Nbar`), or if `force_training` is True.

        The method iterates through `kernel_alternatives` for each GPR.
        If `self.n_GPs_per_node == 1`, `self.my_GPRs[0]` is trained on all data in the node.
        If `self.n_GPs_per_node > 1`, the data in the node (`X_train`, `y_train`) is
        partitioned among the GPRs in `self.my_GPRs`. Each GPR is then trained
        on its assigned subset of data. If the total data in the node is insufficient
        for robust partitioning (e.g., fewer than `n_GPs_per_node * 2` points), all GPRs
        in the node may be trained on all available data points in the node to ensure stability.
        The kernel that yields the best (lowest) log-marginal-likelihood (LML) is
        selected for each respective GPR.

        Args:
            force_training (bool): If True, the GPR is retrained even if the
                usual buffer or fullness conditions are not met. Defaults to False.

        Returns:
            bool: True if the GPR was trained in this call, False otherwise.
        """
        did_train = False
        MIN_SAMPLES_FOR_VALIDATION = 5 # Or another suitable number like 10

        if (self.n_points_since_retrain >= self.retrain_every_n_points) or \
           (self.n_points >= self.Nbar) or force_training:

            self.n_points_since_retrain = 0
            self.gp_rmse_scores = [] # Reset scores

            X_data_full = np.vstack((self.my_X_data, self.shared_X_data))
            y_data_full = np.vstack((self.my_y_data, self.shared_y_data))

            if X_data_full.shape[0] == 0:
                warnings.warn(f"Node {self.name}: No data points available for training.", RuntimeWarning)
                return False # Cannot train with no data

            X_train_gpr, X_val, y_train_gpr, y_val = (None, None, None, None)

            if X_data_full.shape[0] < MIN_SAMPLES_FOR_VALIDATION:
                warnings.warn(f"Node {self.name}: Insufficient data ({X_data_full.shape[0]} points) for train-validation split. Training on all data. RMSE will be NaN.", RuntimeWarning)
                X_train_gpr = X_data_full
                y_train_gpr = y_data_full
                self.gp_rmse_scores = [np.nan] * self.n_GPs_per_node # Mark RMSE as not applicable for all GPs
            else:
                X_train_gpr, X_val, y_train_gpr, y_val = train_test_split(
                    X_data_full, y_data_full, test_size=0.2, random_state=42 # Fixed random_state for reproducibility
                )
                if X_val.shape[0] == 0: # If test_size results in empty validation set
                    warnings.warn(f"Node {self.name}: Validation set is empty after split. Training on all data. RMSE will be NaN.", RuntimeWarning)
                    X_train_gpr = X_data_full # Fallback to using all data for training
                    y_train_gpr = y_data_full
                    self.gp_rmse_scores = [np.nan] * self.n_GPs_per_node


            # Calculate ranges, bounds, init_points based on the actual training data portion (X_train_gpr)
            # This needs to be done after X_train_gpr is defined.
            if X_train_gpr.shape[0] == 0: # Should not happen if checks above are correct
                warnings.warn(f"Node {self.name}: X_train_gpr is empty before GPR parameter calculation. Skipping training.", RuntimeWarning)
                return False

            x_max_vals = [np.max(X_train_gpr[:,i]) for i in range(self.n_features)]
            x_min_vals = [np.min(X_train_gpr[:,i]) for i in range(self.n_features)]
            x_ranges = [x_max_vals[i] - x_min_vals[i] if x_max_vals[i] > x_min_vals[i] else 1e-6 for i in range(self.n_features)]


            min_ls_for_bounds = self.my_GPRs[0].min_length_scale
            if hasattr(self.my_GPRs[0], 'min_length_scale'):
                 min_ls_for_bounds = self.my_GPRs[0].min_length_scale
            else:
                 warnings.warn("min_length_scale not defined on GPR, using default for bounds.", RuntimeWarning)
                 min_ls_for_bounds = 0.001

            use_bounds = [(np.max([min_ls_for_bounds, 0.01*x_ranges[i]]), np.max([10*min_ls_for_bounds, 10*x_ranges[i]])) for i in range(self.n_features)]
            use_init_points = [0.1*x_ranges[i] if x_ranges[i] > 0 else min_ls_for_bounds for i in range(self.n_features)]

            # Initialize self.gp_rmse_scores as a list to store RMSEs of initial GPRs
            self.gp_rmse_scores = []

            if self.n_GPs_per_node == 1:
                if X_train_gpr.shape[0] == 0:
                    warnings.warn(f"Node {self.name}, GP 0: No training data. This GP will not be trained.", RuntimeWarning)
                    self.gp_rmse_scores.append(np.nan)
                    # No need to return False yet, final GP might still be creatable if logic allows, though unlikely here
                else:
                    X_subset_for_this_gp, y_subset_for_this_gp = X_train_gpr, y_train_gpr
                    if self.n_train is not None and 0 < self.n_train < X_train_gpr.shape[0]:
                        X_subset_for_this_gp, y_subset_for_this_gp = resample(
                            X_train_gpr, y_train_gpr, replace=False, n_samples=self.n_train, random_state=42
                        )

                    best_lml = float_info.max
                    successfully_fitted_this_gp = False
                    for kernel_idx, kernel in enumerate(self.my_GPRs[0].kernel_alternatives):
                        params = kernel.get_params(deep=True)
                        new_params = {}
                        for k,v in params.items():
                            if k.endswith("__length_scale_bounds"): new_params[k] = use_bounds
                            elif k.endswith("__length_scale"): new_params[k] = use_init_points
                        current_kernel_alternative = deepcopy(kernel)
                        current_kernel_alternative.set_params(**new_params)

                        temp_GPR_instance = deepcopy(self.my_GPRs[0])
                        temp_GPR_instance.kernel = current_kernel_alternative

                        try:
                            temp_GPR_instance.fit(X_subset_for_this_gp, y_subset_for_this_gp)
                            lml = temp_GPR_instance.log_marginal_likelihood_value_
                            if lml < best_lml:
                                self.my_GPRs[0] = deepcopy(temp_GPR_instance)
                                best_lml = lml
                                successfully_fitted_this_gp = True
                        except Exception as e:
                            print(f"Node {self.name}, GP 0, Kernel {kernel_idx}: Error during GPR fit with kernel {current_kernel_alternative}. Error: {e}")

                    if X_val is not None and X_val.shape[0] > 0 and successfully_fitted_this_gp:
                        try:
                            y_pred_val, _ = self.my_GPRs[0].predict(X_val, return_std=True)
                            y_pred_val = np.asarray(y_pred_val).reshape(-1,1)
                            rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
                            self.gp_rmse_scores.append(rmse)
                        except Exception as e:
                            print(f"Node {self.name}, GP 0: Error during RMSE calculation. Error: {e}")
                            self.gp_rmse_scores.append(np.nan)
                    else:
                        self.gp_rmse_scores.append(np.nan)

            else: # n_GPs_per_node > 1
                n_total_points_in_train_gpr = X_train_gpr.shape[0]
                if n_total_points_in_train_gpr == 0:
                    warnings.warn(f"Node {self.name}: No training data for multi-GP. No GPs will be trained.", RuntimeWarning)
                    self.gp_rmse_scores = [np.nan] * self.n_GPs_per_node
                else:
                    MIN_POINTS_PER_GP_FOR_PARTITIONING = 2
                    partition_data_among_gps = (n_total_points_in_train_gpr >= self.n_GPs_per_node * MIN_POINTS_PER_GP_FOR_PARTITIONING)

                    gp_data_indices_list = []
                    if partition_data_among_gps:
                        indices = np.arange(n_total_points_in_train_gpr)
                        np.random.shuffle(indices) # Shuffle before splitting for more random partitions
                        gp_data_indices_list = np.array_split(indices, self.n_GPs_per_node)

                    for i in range(self.n_GPs_per_node):
                        X_pool_for_gp_i, y_pool_for_gp_i = X_train_gpr, y_train_gpr
                        if partition_data_among_gps:
                            gp_indices = gp_data_indices_list[i]
                            if len(gp_indices) == 0:
                                warnings.warn(f"Node {self.name}, GP {i}: No data points assigned after partitioning. This GP will not be trained. RMSE will be NaN.", RuntimeWarning)
                                self.gp_rmse_scores.append(np.nan)
                                continue
                            X_pool_for_gp_i = X_train_gpr[gp_indices]
                            y_pool_for_gp_i = y_train_gpr[gp_indices]

                        X_subset_for_this_gp, y_subset_for_this_gp = X_pool_for_gp_i, y_pool_for_gp_i
                        if self.n_train is not None and 0 < self.n_train < X_pool_for_gp_i.shape[0]:
                            X_subset_for_this_gp, y_subset_for_this_gp = resample(
                                X_pool_for_gp_i, y_pool_for_gp_i, replace=False, n_samples=self.n_train, random_state=42+i # Vary random state per GP
                            )

                        if X_subset_for_this_gp.shape[0] == 0:
                            warnings.warn(f"Node {self.name}, GP {i}: Final X_subset_for_this_gp is empty. This GP will not be trained. RMSE will be NaN.", RuntimeWarning)
                            self.gp_rmse_scores.append(np.nan)
                            continue

                        current_gp_to_train = self.my_GPRs[i]
                        best_lml_for_this_gp = float_info.max
                        successfully_fitted_this_gp = False

                        if not hasattr(current_gp_to_train, 'kernel_alternatives') or not current_gp_to_train.kernel_alternatives:
                            warnings.warn(f"Node {self.name}, GP {i}: No kernel alternatives found. Fitting with current kernel.", RuntimeWarning)
                            try:
                                if current_gp_to_train.kernel is None:
                                    warnings.warn(f"Node {self.name}, GP {i}: Kernel is None. Assigning a default Matern kernel.", RuntimeWarning)
                                    current_gp_to_train.kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1.0)

                                gpr_copy_for_fit = deepcopy(current_gp_to_train)
                                gpr_copy_for_fit.fit(X_subset_for_this_gp, y_subset_for_this_gp)
                                self.my_GPRs[i] = gpr_copy_for_fit
                                successfully_fitted_this_gp = True
                            except Exception as e:
                                print(f"Node {self.name}, GP {i}: Error during GPR fit with current kernel: {e}")
                        else:
                            temp_gpr_for_kernel_search = deepcopy(current_gp_to_train)
                            for kernel_idx, kernel_alt in enumerate(current_gp_to_train.kernel_alternatives):
                                params = kernel_alt.get_params(deep=True)
                                new_kernel_params = {}
                                for k, v_param in params.items():
                                    if k.endswith("__length_scale_bounds"): new_kernel_params[k] = use_bounds
                                    elif k.endswith("__length_scale"): new_kernel_params[k] = use_init_points

                                current_kernel_to_try = deepcopy(kernel_alt)
                                current_kernel_to_try.set_params(**new_kernel_params)

                                gpr_instance_for_this_kernel = deepcopy(temp_gpr_for_kernel_search)
                                gpr_instance_for_this_kernel.kernel = current_kernel_to_try

                                try:
                                    gpr_instance_for_this_kernel.fit(X_subset_for_this_gp, y_subset_for_this_gp)
                                    lml = gpr_instance_for_this_kernel.log_marginal_likelihood_value_
                                    if lml < best_lml_for_this_gp:
                                        self.my_GPRs[i] = deepcopy(gpr_instance_for_this_kernel)
                                        best_lml_for_this_gp = lml
                                        successfully_fitted_this_gp = True
                                except Exception as e:
                                    # print(f"Node {self.name}, GP {i}, Kernel {kernel_idx}: Error during GPR fit with kernel {current_kernel_to_try}. Error: {e}")
                                    pass

                        if X_val is not None and X_val.shape[0] > 0 and successfully_fitted_this_gp:
                            try:
                                y_pred_val_i, _ = self.my_GPRs[i].predict(X_val, return_std=True)
                                y_pred_val_i = np.asarray(y_pred_val_i).reshape(-1,1)
                                rmse_i = np.sqrt(mean_squared_error(y_val, y_pred_val_i))
                                self.gp_rmse_scores.append(rmse_i)
                            except Exception as e:
                                print(f"Node {self.name}, GP {i}: Error during RMSE calculation. Error: {e}")
                                self.gp_rmse_scores.append(np.nan)
                        else:
                            self.gp_rmse_scores.append(np.nan)

            # Select best initial GP
            best_gp_index = 0
            if not self.gp_rmse_scores: # Empty list
                warnings.warn(f"Node {self.name}: No RMSE scores available for initial GPs. Defaulting to GP at index 0.", RuntimeWarning)
            else:
                # Replace NaNs with a large number for argmin, but only if there are non-NaNs
                # If all are NaN, argmin([np.nan, np.nan]) raises error.
                # np.nanargmin behaves correctly if all are NaN, returning 0.
                # However, let's be explicit.
                if all(np.isnan(score) for score in self.gp_rmse_scores):
                    warnings.warn(f"Node {self.name}: All initial GP RMSEs are NaN. Defaulting to GP at index 0.", RuntimeWarning)
                    best_gp_index = 0
                else:
                    try:
                        best_gp_index = np.nanargmin(self.gp_rmse_scores)
                    except ValueError: # Should be caught by all(np.isnan(...))
                         warnings.warn(f"Node {self.name}: Error finding best GP index from RMSEs {self.gp_rmse_scores}. Defaulting to 0.", RuntimeWarning)
                         best_gp_index = 0


            best_initial_gp = self.my_GPRs[best_gp_index]

            # Create and train final GP
            if X_train_gpr.shape[0] > 0:
                if hasattr(best_initial_gp, 'kernel_') and best_initial_gp.kernel_ is not None:
                    final_gp = deepcopy(self.my_GPRs[0]) # Use first GP as template for type and non-kernel params
                    final_gp.kernel = deepcopy(best_initial_gp.kernel_)
                    final_gp.optimizer = None # Disable hyperparameter optimization

                    try:
                        final_gp.fit(X_train_gpr, y_train_gpr)

                        final_gp_rmse = np.nan
                        if X_val is not None and X_val.shape[0] > 0:
                            try:
                                y_pred_val_final, _ = final_gp.predict(X_val, return_std=True)
                                y_pred_val_final = np.asarray(y_pred_val_final).reshape(-1,1)
                                final_gp_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val_final))
                            except Exception as e:
                                print(f"Node {self.name}, Final GP: Error during RMSE calculation. Error: {e}")

                        self.gp_rmse_scores.append(final_gp_rmse) # Add RMSE of the final GP
                        self.my_GPRs.append(final_gp) # Add the final GP to the list

                    except Exception as e:
                        print(f"Node {self.name}: Error during final GP training. Error: {e}")
                        # self.gp_rmse_scores.append(np.nan) # If final GP fails, its RMSE is NaN

                else:
                    warnings.warn(f"Node {self.name}: Best initial GP (index {best_gp_index}) has no fitted kernel. Skipping final GP creation.", RuntimeWarning)
                    # self.gp_rmse_scores.append(np.nan) # Add a NaN for the final GP's slot if it's not created
            else:
                warnings.warn(f"Node {self.name}: X_train_gpr is empty. Skipping final GP creation.", RuntimeWarning)
                # self.gp_rmse_scores.append(np.nan) # Add a NaN for the final GP's slot


            # Print statements
            x_range_strs = [f"({x_min_vals[i_f]:.2e},{x_max_vals[i_f]:.2e})" for i_f in range(self.n_features)]
            x_range_str = "[" + ", ".join(x_range_strs) + "]"

            # RMSE print string needs to handle initial GPs and the final one
            rmse_print_parts = []
            num_initial_gps = self.n_GPs_per_node
            for i in range(num_initial_gps):
                if i < len(self.gp_rmse_scores) and not np.isnan(self.gp_rmse_scores[i]):
                    rmse_print_parts.append(f"GP{i}_RMSE: {self.gp_rmse_scores[i]:.3f}")
                else:
                    rmse_print_parts.append(f"GP{i}_RMSE: NaN")

            if len(self.my_GPRs) > num_initial_gps: # Final GP was added
                 final_gp_rmse_idx = num_initial_gps # RMSE for final GP is at this index in gp_rmse_scores
                 if final_gp_rmse_idx < len(self.gp_rmse_scores) and not np.isnan(self.gp_rmse_scores[final_gp_rmse_idx]):
                     rmse_print_parts.append(f"FinalGP_RMSE: {self.gp_rmse_scores[final_gp_rmse_idx]:.3f}")
                 else:
                     rmse_print_parts.append("FinalGP_RMSE: NaN")

            rmse_print_str = ", ".join(rmse_print_parts)

            print(f"Trained node {self.name} ({len(self.my_GPRs)} GPs total): x_range: {x_range_str} [{rmse_print_str}]")
            for i, gp in enumerate(self.my_GPRs):
                gp_label = f"Initial GP {i}" if i < num_initial_gps else f"Final GP (from GP{best_gp_index})"
                kernel_str = str(gp.kernel_) if hasattr(gp, 'kernel_') and gp.kernel_ is not None else \
                             (str(gp.kernel) + " (default, not optimized)" if hasattr(gp, 'kernel') and gp.kernel is not None else "Not trained")
                print(f"  {gp_label} kernel: {kernel_str}")


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

        # TODO: Introduce alternative ways to compute the split position, e.g. median
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
        """Evaluates the prediction from this node's final GPR at input point(s) x.

        The method attempts to use the last GPR in `self.my_GPRs`, which is
        expected to be the "final GP" trained on all node data with a selected kernel.
        Fallbacks are in place if this GP is not available or trained.

        Args:
            x (np.ndarray): The input point(s) at which to make predictions.
                Shape should be (n_samples, n_features).
            return_std (bool): Whether to return the standard deviation of the
                prediction. Defaults to True.
            use_calibrated_sigma (bool): If True, the returned `sigma_pred`
                (standard deviation) is scaled by the node's `self.sigma_scaler`
                attribute.

        Returns:
            tuple:
                - mu_pred (np.ndarray): The mean prediction(s).
                - sigma_pred (np.ndarray): The standard deviation of the
                  prediction(s). Only returned if `return_std` is True.
        """
        predicting_gp = None

        if not self.my_GPRs:
            warnings.warn(f"Node {self.name}: No GPRs available for prediction. Returning default predictions.", RuntimeWarning)
            # Default predictions handled below if predicting_gp remains None
        # Check if final GP exists (list length is n_GPs_per_node + 1)
        elif len(self.my_GPRs) == self.n_GPs_per_node + 1:
            predicting_gp = self.my_GPRs[-1]
        elif self.my_GPRs: # Final GP does not exist, fallback to first initial GP if available
            warnings.warn(f"Node {self.name}: Final GP not found (expected {self.n_GPs_per_node + 1} GPRs, found {len(self.my_GPRs)}). Using first GPR from initial set for prediction.", RuntimeWarning)
            predicting_gp = self.my_GPRs[0]
        else: # Should be caught by 'if not self.my_GPRs:'
            warnings.warn(f"Node {self.name}: Unexpected state of my_GPRs. Returning default predictions.", RuntimeWarning)
            # Default predictions handled below

        if predicting_gp is None or not hasattr(predicting_gp, 'kernel_') or predicting_gp.kernel_ is None:
            if predicting_gp is None and self.my_GPRs : #This means it fell through the logic above without assigning (e.g. my_GPRs not empty but failed other conditions)
                 warnings.warn(f"Node {self.name}: No valid GPR selected for prediction (my_GPRs has {len(self.my_GPRs)} elements). Returning default predictions.", RuntimeWarning)
            elif predicting_gp: # GP was selected but not trained
                 warnings.warn(f"Node {self.name}: Selected GPR for prediction (type: {type(predicting_gp)}) is not trained. Returning default predictions.", RuntimeWarning)
            # If predicting_gp is None and self.my_GPRs is empty, the first warning already covered it.

            mu_pred = np.zeros((x.shape[0], 1))
            sigma_pred = np.ones((x.shape[0], 1)) if return_std else None

            if return_std:
                return mu_pred, sigma_pred
            else:
                return mu_pred

        # Proceed with prediction using the chosen predicting_gp
        try:
            mu_pred_temp, sigma_pred_temp = predicting_gp.predict(x, return_std=return_std)
        except Exception as e:
            warnings.warn(f"Node {self.name}: Error during GPR predict call: {e}. Returning default predictions.", RuntimeWarning)
            mu_pred = np.zeros((x.shape[0], 1))
            sigma_pred = np.ones((x.shape[0], 1)) if return_std else None
            if return_std:
                return mu_pred, sigma_pred
            else:
                return mu_pred

        mu_pred = np.asarray(mu_pred_temp).reshape(-1, 1)

        if return_std:
            if sigma_pred_temp is not None:
                sigma_pred = np.asarray(sigma_pred_temp).reshape(-1, 1)
            else:
                # This case should ideally be handled by the GPR itself if return_std=True
                sigma_pred = np.ones_like(mu_pred)
                warnings.warn(f"Node {self.name}: GPR predict() returned None for std even when return_std=True. Using np.ones_like(mu_pred).", RuntimeWarning)

            if use_calibrated_sigma and sigma_pred is not None: # Ensure sigma_pred is not None before scaling
                sigma_pred = sigma_pred * self.sigma_scaler
            return mu_pred, sigma_pred
        else:
            return mu_pred

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
        target_coverage = 0.68

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
