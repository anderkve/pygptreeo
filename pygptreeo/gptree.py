import numpy as np
from binarytree import Node
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.utils import resample
from typing import Callable, Optional, Type, Union
from copy import deepcopy
from tqdm import tqdm
import joblib # Ensure joblib is imported
from copy import deepcopy # Add this import
from sklearn.gaussian_process import GaussianProcessRegressor # For type checking

from pygptreeo.default_gpr import Default_GPR
from pygptreeo.gpnode import GPNode


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
        GPR (sklearn.gaussian_process.GaussianProcessRegressor): The base GPR
            configuration used as a template for the GPR in each new GPNode.
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

    Methods:
        fit(X_train, y_train, show_progress=False, shuffle=True, forward_GPR_to_next_leaf=False):
            Constructs the tree from a batch of training data.
        update_tree(x, y, allow_training=True):
            Adds a single data point to the tree, potentially causing node
            splits and GPR retraining.
        predict(X_test, mode='recursive', show_progress=False):
            Predicts target values for X_test using the ensemble of GPRs in
            the leaf nodes.
        save(path):
            Saves the trained GPTree object to a file.
    """
    def __init__(self,
                 GPR: Optional[GaussianProcessRegressor] = Default_GPR(),
                 Nbar: Optional[int] = 100,
                 theta: Optional[float] = 0.0001,
                 use_calibrated_sigma: Optional[bool] = True,
                 split_dimension_criteria: Optional[str] = 'max_spread',
                 splitting_strategy: Optional[str] = 'standard',
                 n_GPs_per_node: int = 1,
                 **kwargs):
        """Initializes the GPTree.

        Args:
            GPR (Optional[GaussianProcessRegressor]): A Gaussian Process Regressor instance or class
                to be used as a template for GPRs in the tree nodes. Defaults to `Default_GPR()`.
                This template is deepcopied for each GP in each node.
            Nbar (Optional[int]): Maximum number of training points a node can hold before it
                considers splitting. Defaults to 100.
            theta (Optional[float]): Parameter influencing the overlap region between sibling nodes
                after a split. The overlap size is `theta * range_of_data_in_split_dimension`.
                Defaults to 0.0001.
            use_calibrated_sigma (Optional[bool]): If True, enables scaling of predictive standard
                deviations in GPNodes using their `sigma_scaler` attribute. Defaults to True.
            split_dimension_criteria (Optional[str]): Method used by GPNodes to determine the
                feature dimension for splitting. Valid options include 'max_spread', 'max_variance',
                'random'. Defaults to 'max_spread'.
            splitting_strategy (Optional[str]): Strategy used by GPNodes for data handling
                post-split (e.g., 'standard', 'gradual'). Defaults to 'standard'.
            n_GPs_per_node (Optional[int]): The number of Gaussian Process Regressors to be
                instantiated and managed within each GPNode of the tree. Defaults to 1.
                This parameter is passed to each GPNode.
            **kwargs: Additional keyword arguments that are filtered. Arguments relevant to
                `sklearn.gaussian_process.GaussianProcessRegressor` are used if `GPR` is a class type
                to instantiate the `GPR_template`. Remaining arguments are passed to the constructor
                of the root `GPNode` (e.g., `split_position_method`, `retrain_every_n_points`).
        """
        
        self.GPR_template = GPR
        self.splitting_strategy = splitting_strategy
        self.n_GPs_per_node = n_GPs_per_node

        # Instantiate the GPR template if it's a class type
        if isinstance(self.GPR_template, type) and issubclass(self.GPR_template, GaussianProcessRegressor):
            # Pass only GPR-specific kwargs if possible, or all if they are distinct enough
            # For Default_GPR, it doesn't take many args in its __init__ other than sklearn's
            gpr_init_kwargs = {k: v for k, v in kwargs.items() if k in GaussianProcessRegressor().get_params().keys()}
            base_gpr_instance = self.GPR_template(**gpr_init_kwargs)
        elif isinstance(self.GPR_template, GaussianProcessRegressor):
            base_gpr_instance = self.GPR_template
        else:
            # Fallback or error if GPR is not a class or instance
            base_gpr_instance = Default_GPR() # Default fallback
            warnings.warn("GPR argument was not a class or instance of GaussianProcessRegressor. Using Default_GPR().", RuntimeWarning)

        # Ensure base_gpr_instance has kernel_alternatives and min_length_scale
        # This is particularly for plain sklearn GPRs; Default_GPR handles this itself.
        if not hasattr(base_gpr_instance, 'kernel_alternatives'):
            if hasattr(base_gpr_instance, 'kernel') and base_gpr_instance.kernel is not None:
                base_gpr_instance.kernel_alternatives = [base_gpr_instance.kernel]
            else: # Default kernel if none present
                default_kernel = ConstantKernel(1.0) * Matern(nu=1.5)
                base_gpr_instance.kernel = default_kernel
                base_gpr_instance.kernel_alternatives = [default_kernel]
        if not hasattr(base_gpr_instance, 'min_length_scale'):
            base_gpr_instance.min_length_scale = 0.001 # A sensible default

        gpr_list_for_root = [deepcopy(base_gpr_instance) for _ in range(self.n_GPs_per_node)]

        # Filter kwargs for GPNode constructor to avoid passing GPR instantiation args
        gpnode_kwargs = {k: v for k, v in kwargs.items() if k not in GaussianProcessRegressor().get_params().keys()}

        # GPNode expects value as the first positional argument if provided through *args
        self.root = GPNode(0, # value for n_points, passed as first positional arg
                           my_GPRs=gpr_list_for_root,
                           Nbar=Nbar,
                           split_dimension_criteria=split_dimension_criteria,
                           splitting_strategy=self.splitting_strategy,
                           n_GPs_per_node=self.n_GPs_per_node,
                           **gpnode_kwargs)

        self.theta = theta

        self.n_features = 0

        self.use_calibrated_sigma = use_calibrated_sigma

        self.first_point = True


    def update_tree(self, x: np.ndarray, y: float, allow_training=True):
        """Updates the tree structure and node GPRs with a new data point (x, y).

        This method implements a process similar to Algorithm 1 in the DLGP
        (Deep Locally-Weighted Gaussian Processes) article, with some modifications.
        It navigates from the root to find an appropriate leaf node for the new
        data point `x` based on the probabilistic splitting functions of intermediate
        nodes. The point is added to the chosen leaf's training data.

        If `allow_training` is True, the leaf node's GPR may be retrained based
        on its buffer conditions (e.g., `retrain_every_n_points`) or if it becomes
        full. If the node's capacity `Nbar` is reached, it splits into two
        child nodes, its data is partitioned, and its own GPR and data are
        typically deleted to save memory.

        Args:
            x (np.ndarray): The new input data point (features), expected as a
                1D array or a 2D array with one row.
            y (float): The corresponding target value for the data point.
            allow_training (bool): If True (default), the GPR model in the
                selected leaf node can be retrained if its conditions
                (e.g., buffer full, node full) are met. If False, GPR retraining
                is suppressed during this update. This is useful, for example,
                during the initial `fit` method where tree construction is the
                priority before a final training pass on all leaves.
        """

        # The first input point is used to determine self.n_features
        if self.first_point:
            self.n_features = x.size
            self.root.init_data_set(self.n_features)
            self.first_point = False

        # Find a leaf node for the new (x,y) point
        # - Start from the root node
        # - For each level, pick a branch according node.prob_func(x), until a leaf node is reached
        current_node = self.root # Renamed 'node' to 'current_node' for clarity
        # while current_node.children: # This was the old check
        while not current_node.is_leaf: # Correct check for leaf
            current_node = current_node.children[int(np.random.binomial(1, current_node.prob_func(x)[0][0]))]

        # DEBUG
        # current_node._print_debug_status()

        # Add new point and register prediction performance
        current_node.store_point(x, y, remove_shared=True)
        current_node.register_pred_perf(x, y)

        # Update the uncertainty scaler for this node?
        if self.use_calibrated_sigma:
            current_node.update_sigma_scaler()

        # Retrain GP? The current_node will decide based Nbar and/or its buffer of training points
        if allow_training:
            did_retrain = current_node.fit_my_GPR()

        # If the current_node is full, generate child nodes
        if current_node.n_points >= current_node.Nbar:
            # Create child nodes. Each child node gets a copy of the current parent GP list.
            # The GPR_template (self.GPR_template) is passed but GPNode.generate_children
            # currently uses deepcopy(self.my_GPRs).
            current_node.generate_children(GPR=self.GPR_template, n_features=self.n_features)
            
            # Compute parameters for the probability function 
            current_node.compute_split_position_and_overlap(self.theta)

            # Now pass parent's split info to children
            current_node.children[0].parent_split_index = current_node.split_index
            current_node.children[0].parent_split_position = current_node.split_position
            current_node.children[1].parent_split_index = current_node.split_index
            current_node.children[1].parent_split_position = current_node.split_position

            if current_node.splitting_strategy == 'gradual':

                # First distribute data as usual between the two child nodes
                current_node.split_training_data()

                # Since we are doing gradual splitting, give 
                # each child a copy of the other child's data
                order = current_node.children[1].my_X_data[:,current_node.split_index].argsort()
                current_node.children[0].shared_X_data = current_node.children[1].my_X_data[order]
                current_node.children[0].shared_y_data = current_node.children[1].my_y_data[order]
                current_node.children[0].n_shared_points = current_node.children[0].shared_X_data.shape[0]

                order = current_node.children[0].my_X_data[:,current_node.split_index].argsort()[::-1]
                current_node.children[1].shared_X_data = current_node.children[0].my_X_data[order]
                current_node.children[1].shared_y_data = current_node.children[0].my_y_data[order]
                current_node.children[1].n_shared_points = current_node.children[1].shared_X_data.shape[0]

            else:
                # Standard splitting
                current_node.split_training_data()

            # Retrain the child-node GPs?
            # for child in current_node.children:
            #     child.fit_my_GPR()

            # GP and training data of non-leaf nodes is not needed
            current_node.delete_data(delete_own_data=True, delete_shared_data=True)
            current_node.delete_my_GPRs() # Changed from delete_my_GPR


    def fit(self, X_train: np.ndarray, y_train: np.ndarray, show_progress: Optional[bool]=False, shuffle: Optional[bool]=True, 
            forward_GPR_to_next_leaf: Optional[bool]=False):
        """
        Construct the binary tree by assigning a set of training samples to nodes and train the leaf-node GPs.

        Arguments
        ----------
        X_train: np.ndarray
            The training data in feature space. Has shape=(N_train, n_features).

        y_train: np.ndarray
            The training data in target space. Has shape=(N_train, 1) (only scalar targets implemented).

        show_progress: Optional[bool]=False
            Display a progress bar in the terminal using tqdm.

        shuffle: Optional[bool]=True
            Shuffle the training set to avoid an unbalanced tree.

        forward_GPR_to_next_leaf: Optional[bool]=False
            When training the leaf-node GPs, let the next leaf start from a copy of the trained GP 
            from the previous leaf.
        """
        self.n_features = X_train.shape[1]
        N = X_train.shape[0]
        self.root.init_data_set(self.n_features)

        if shuffle:
            X_train, y_train = resample(X_train, y_train, replace=False)

        # Construct the tree
        for x, y in tqdm(zip(X_train, y_train), total=N, disable=not show_progress, desc="Building binary tree"):
            x = x.reshape((1, x.shape[0]))
            y = y.reshape((1, 1))

            self.update_tree(x, y, allow_training=False)
        
        # Train all the leaves
        for i, leaf in tqdm(enumerate(self.root.leaves), total=len(self.root.leaves), disable=not show_progress, desc="Training"):
            # leaf.is_leaf = True # This should already be true for leaves
            leaf.fit_my_GPR() # This will fit all GPs in leaf.my_GPRs
            if forward_GPR_to_next_leaf and i != len(self.root.leaves) - 1:
                # This logic needs care with multiple GPRs.
                # Do we copy all GPRs, or just the first?
                # For now, let's assume we copy the whole list of GPRs.
                self.root.leaves[i+1].my_GPRs = deepcopy(leaf.my_GPRs)

            
            """ kernel = leaf.my_GPR.kernel_
            with open("hyperparameters.txt", 'a') as infile:
                infile.write(f"Leaf node {i}")
                infile.write("##############")
                for hyperparameter, hyperparameter_value in zip(kernel.hyperparameters, kernel.theta):
                    infile.write(f"{hyperparameter} {np.exp(hyperparameter_value)} \n") """
                
                
    def predict_recursive(self, X_test: np.ndarray, show_progress: Optional[bool]=False):
        """ 
        A predict function that uses a recursive function to collect contributing leaves.
        Should be the fastest alternative when N_train >> Nbar. 
        
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
        

        mean_DLGP = np.zeros((X_test.shape[0], 1))
        var_DLGP = np.zeros((X_test.shape[0], 1))

        for i, x in tqdm(enumerate(X_test), total=X_test.shape[0], disable=not show_progress, desc="Predicting"):
            x = x.reshape((1, x.shape[0]))

            sum_probs = 0
            collection_done = False

            leaves = []
            pred_leaf_probs = []

            collect_leaves(x, self.root, 1.0)
        
            for leaf, ptilde in zip(leaves, pred_leaf_probs):

                mu_leaf, sigma_leaf = leaf.predict(x, return_std=True, use_calibrated_sigma=self.use_calibrated_sigma)

                # mean_DLGP[i] += ptilde*mu_leaf[0] # Old comment
                # mu_leaf and sigma_leaf are (1,1) from GPNode.predict for a single x
                mean_DLGP[i, 0] += ptilde * mu_leaf[0, 0]
                var_DLGP[i, 0] += ptilde * (sigma_leaf[0, 0]**2 + mu_leaf[0, 0]**2)
            
            var_DLGP[i, 0] += -mean_DLGP[i, 0]**2 # Corrected calculation for variance
        
        return mean_DLGP, np.sqrt(np.maximum(0, var_DLGP)) # Ensure var_DLGP is not negative due to float errors


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
        mean_DLGP = np.zeros((X_test.shape[0], 1))
        var_DLGP = np.zeros((X_test.shape[0], 1))

        for leaf in tqdm(self.root.leaves, disable=not show_progress, desc="Predicting"):
            
            ptilde = leaf.marg_prob(X_test)
            ptilde = ptilde.reshape(mean_DLGP.shape)

            # We can skip this leaf if its prediction contribute zero for all points in X_test
            if np.all(ptilde == 0.0):
                continue

            mu_leaf, sigma_leaf = leaf.predict(X_test, return_std=True, use_calibrated_sigma=self.use_calibrated_sigma)
            mu_leaf = mu_leaf.reshape(mean_DLGP.shape)
            sigma_leaf = sigma_leaf.reshape(mean_DLGP.shape)

            mean_DLGP += ptilde*mu_leaf
            var_DLGP += ptilde*(sigma_leaf*sigma_leaf + mu_leaf*mu_leaf)
        
        var_DLGP += -mean_DLGP*mean_DLGP

        return mean_DLGP, np.sqrt(var_DLGP)


    def predict_each(self, X_test):
        """ Get the prediction of each leaf node individually.  """
        res = []
        for leaf in self.root.leaves:
            ptilde = leaf.marg_prob(X_test)
            ptilde = ptilde.reshape(mean_DLGP.shape)

            mu_leaf, sigma_leaf = leaf.predict(X_test, return_std=True, use_calibrated_sigma=self.use_calibrated_sigma)
            res.append((mu_leaf, sigma_leaf, ptilde))
        return res
    

    def predict(self, X_test: np.ndarray, mode: Optional[str]='recursive', show_progress: Optional[bool]=False):
        """ Main predict function that calls a specific predict function according to the 'mode' argument.  """
        if mode == 'recursive':
            return self.predict_recursive(X_test, show_progress=show_progress)
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
