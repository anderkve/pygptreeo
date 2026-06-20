"""GPTree: Main tree structure for online regression.

This module implements the GPTree class, which provides the core logic for
dynamically growing a binary tree of Gaussian Process regressors. The tree
adapts to incoming data by splitting nodes and training local GP models.

The GPTree manages:
    - Dynamic tree growth based on data accumulation
    - Routing of data points to appropriate leaf nodes
    - Prediction aggregation across multiple leaf nodes
    - Training coordination for all GP models

This implementation is suitable for online learning scenarios where data
arrives sequentially and the model needs to adapt continuously.
"""

# Standard library imports
from copy import deepcopy
from typing import Callable, Optional, Type, Union

# Third-party imports
import joblib
import numpy as np
from sklearn.utils import resample
from tqdm import tqdm

# Local imports
from pygptreeo.default_gpr import Default_GPR
from pygptreeo.gpnode import GPNode
from pygptreeo.gp_interface import GPRegressorInterface


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
                 GPR: Optional[GPRegressorInterface] = None,
                 Nbar: Optional[int] = 100,
                 theta: Optional[float] = 0.0001,
                 use_calibrated_sigma: Optional[bool] = True,
                 split_dimension_criteria: Optional[str] = 'max_spread',
                 splitting_strategy: Optional[str] = 'standard',
                 max_n_pred_leaves: Optional[int] = None,
                 aggregation: Optional[str] = "default",
                 n_outputs: Optional[int] = 1,
                 incremental_updates: Optional[bool] = True,
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
                dimension. Options: 'max_spread', 'max_variance', 'max_uncertainty',
                'random'. Defaults to 'max_spread'.
            splitting_strategy (Optional[str]): Strategy for splitting nodes.
                'standard' or 'gradual'. Defaults to 'standard'.
            max_n_pred_leaves (Optional[int]): Maximum number of leaves to use
                for prediction. Defaults to None (use all).
            aggregation (Optional[str]): Method for aggregating predictions.
                'default'/'moe' or 'poe'. Defaults to 'default'.
            n_outputs (Optional[int]): Number of output dimensions. Defaults to 1 (single output).
                For multi-output GPs, independent GPs are trained for each output.
            incremental_updates (Optional[bool]): If True, leaf GPs incorporate
                each new point via an exact rank-1 Cholesky update between full
                re-fits. This only has an effect for GP backends that support it
                (e.g. IncrementalGP); for other backends it is a harmless no-op.
                Defaults to True.
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
        self.incremental_updates = incremental_updates

        self.root = GPNode(0, my_GPR=GPR, Nbar=Nbar, split_dimension_criteria=split_dimension_criteria,
                          splitting_strategy=self.splitting_strategy, n_outputs=n_outputs,
                          incremental_updates=incremental_updates, **kwargs)  # Initialize root node of the GPTree

        self.theta = theta

        self.n_features = 0

        self.use_calibrated_sigma = use_calibrated_sigma

        self.max_n_pred_leaves = max_n_pred_leaves

        self.aggregation = aggregation

        self.first_point = True


    def update_tree(self, x: np.ndarray, y: Union[float, np.ndarray], sigma: Union[float, np.ndarray], allow_training=True):
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
        typically deleted to save memory.

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
        """

        # The first input point is used to determine self.n_features
        if self.first_point:
            self.n_features = x.size
            self.root.init_data_set(self.n_features)
            self.first_point = False

        # Find a leaf node for the new (x,y,sigma) point
        # - Start from the root node
        # - For each level, pick a branch according node.prob_func(x), until a leaf node is reached
        node = self.root
        while not node.is_leaf:
            node = node.children[int(np.random.binomial(1, node.prob_func(x)[0][0]))]

        # Check if this point should be merged with a nearby point (updates existing point in-place)
        if node.should_merge_point(x, y, sigma):
            # Point was merged with existing point, don't add as new point
            # Still register prediction performance and update sigma scaler
            node.register_pred_perf(x, y)
            if self.use_calibrated_sigma:
                node.update_sigma_scaler()
            return

        # Check if this point should be rejected (if well-predicted by current GP)
        if node.should_reject_point(x, y):
            # Point is well-predicted, don't store it
            return

        # Add new point and register prediction performance
        node.store_point(x, y, sigma, remove_shared=True)
        node.register_pred_perf(x, y)

        # Update the uncertainty scaler for this node?
        if self.use_calibrated_sigma:
            node.update_sigma_scaler()

        # Retrain GP? The node will decide based Nbar and/or its buffer of training points
        if allow_training:
            did_retrain = node.fit_my_GPR()
            # If no full (re-)fit happened this step, incorporate the new point
            # cheaply via an exact rank-1 update (no-op unless the GP backend
            # supports incremental updates and has been fitted at least once).
            if not did_retrain:
                node.incremental_update_gp(x, y, sigma)

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
                order = node.children[1].my_X_data[:,node.split_index].argsort()
                node.children[0].shared_X_data = node.children[1].my_X_data[order]
                node.children[0].shared_y_data = node.children[1].my_y_data[order]
                node.children[0].shared_sigma_data = node.children[1].my_sigma_data[order]
                node.children[0].n_shared_points = node.children[0].shared_X_data.shape[0]

                order = node.children[0].my_X_data[:,node.split_index].argsort()[::-1]
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


    def fit(self, X_train: np.ndarray, y_train: np.ndarray, sigma_train: np.ndarray,
            show_progress: Optional[bool]=False, shuffle: Optional[bool]=True,
            forward_GPR_to_next_leaf: Optional[bool]=False):
        """
        Construct the binary tree by assigning a set of training samples to nodes and train the leaf-node GPs.

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
        """
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
            if forward_GPR_to_next_leaf and i != len(self.root.leaves) - 1:
                # For multi-output, forward only the first GP
                self.root.leaves[i+1].my_GPRs[0] = deepcopy(leaf.my_GPRs[0])
                if self.n_outputs == 1:
                    self.root.leaves[i+1].my_GPR = self.root.leaves[i+1].my_GPRs[0]
                
                
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

                for leaf, ptilde in zip(leaves, pred_leaf_probs):

                    mu_leaf, sigma_leaf = leaf.predict(x, return_std=True, use_calibrated_sigma=self.use_calibrated_sigma)
                    # mu_leaf and sigma_leaf have shape (1, n_outputs)

                    mean_DLGP[i, :] += ptilde * mu_leaf[0, :]
                    var_DLGP[i, :] += ptilde * (sigma_leaf[0, :]**2 + mu_leaf[0, :]**2)

                var_DLGP[i, :] += -mean_DLGP[i, :]**2

            # Generalized product of experts
            elif self.aggregation == "poe":

                # Collect individual predictions
                mus = []
                vars_ = []
                betas = pred_leaf_probs
                for leaf, ptilde in zip(leaves, pred_leaf_probs):
                    mu_leaf, sigma_leaf = leaf.predict(x, return_std=True, use_calibrated_sigma=self.use_calibrated_sigma)
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

        for leaf in tqdm(self.root.leaves, disable=not show_progress, desc="Predicting"):

            ptilde = leaf.marg_prob(X_test)  # Shape: (n_test, 1)

            # We can skip this leaf if its prediction contribute zero for all points in X_test
            if np.all(ptilde == 0.0):
                continue

            mu_leaf, sigma_leaf = leaf.predict(X_test, return_std=True, use_calibrated_sigma=self.use_calibrated_sigma)
            # mu_leaf and sigma_leaf have shape (n_test, n_outputs)

            # Broadcast ptilde to match shape (n_test, n_outputs)
            mean_DLGP += ptilde * mu_leaf
            var_DLGP += ptilde * (sigma_leaf**2 + mu_leaf**2)

        var_DLGP += -mean_DLGP**2

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
