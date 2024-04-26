import numpy as np
from binarytree import Node
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.utils import resample
from typing import Callable, Optional, Type, Union
from copy import deepcopy
from tqdm import tqdm

from pygptreeo.default_gpr import Default_GPR
from pygptreeo.gpnode import GPNode


class GPTree:
    """ Class for GPTree regression (only scalar target functions implemented).
    
        Attributes
        ----------
        Nbar: Optional[int] = 100
            Maximum number of training points that each node can have.

        theta: Optional[float] = 0.0001
            Parameter in probability function when assigning training samples to nodes.

        Methods
        -------
    """
    def __init__(self,
                 GPR: Optional[GaussianProcessRegressor] = Default_GPR(),
                 Nbar: Optional[int] = 100,
                 theta: Optional[float] = 0.0001,
                 use_calibrated_sigma: Optional[bool] = True,
                 **kwargs):
        
        self.GPR = GPR
        self.root = GPNode(0, my_GPR=GPR, Nbar=Nbar, **kwargs)  # Initialize root node of the GPTree

        self.theta = theta

        self.n_features = 0

        self.use_calibrated_sigma = use_calibrated_sigma

        self.first_point = True


    def update_tree(self, x: np.ndarray, y: float, allow_training=True):
        """ Algorithm 1 in DLGP article, with some tweaks of our own """

        # The first input point is used to determine self.n_features
        if self.first_point:
            self.n_features = x.size
            self.root.init_training_set(self.n_features)
            self.first_point = False

        # Find a leaf node for the new (x,y) point
        # - Start from the root node
        # - For each level, pick a branch according node.prob_func(x), until a leaf node is reached
        node = self.root
        # while node.children:
        while not node.is_leaf:
            node = node.children[int(np.random.binomial(1, node.prob_func(x)))]

        # Add new point and register prediction performance
        node.add_training_data(x, y)
        node.register_pred_perf(x, y)

        # Update the uncertainty scaler for this node?
        if self.use_calibrated_sigma:
            node.update_sigma_scaler()

        # Retrain GP? The node will decide based Nbar and/or its buffer of training points
        if allow_training:
            did_retrain = node.fit_my_GPR()

        # If the node is full, generate child nodes
        if node.num_training_points == node.Nbar:

            # Create child nodes. Each child node gets a copy of the current parent GP.
            node.generate_children(self.GPR, self.n_features)
            
            # Compute parameters for the probability function 
            node.compute_split_position_and_overlap(self.theta)

            # Divide the training data between the two child nodes
            node.split_training_data()

            # Retrain the child-node GPs?
            """ for child in node.children:
                child.fit_my_GPR()
                pass """

            # GP and training data of non-leaf nodes is not needed
            node.delete_training_data()
            node.delete_my_GPR()


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
        self.root.init_training_set(self.n_features)

        if shuffle:
            X_train, y_train = resample(X_train, y_train, replace=False)

        # Construct the tree
        for x, y in tqdm(zip(X_train, y_train), total=N, disable=not show_progress, desc="Building binary tree"):
            x = x.reshape((1, x.shape[0]))
            y = y.reshape((1, 1))

            self.update_tree(x, y, allow_training=False)
        
        # Train all the leaves
        for i, leaf in tqdm(enumerate(self.root.leaves), total=len(self.root.leaves), disable=not show_progress, desc="Training"):
            # leaf.is_leaf = True
            leaf.fit_my_GPR()
            if forward_GPR_to_next_leaf and i != len(self.root.leaves) - 1:
                self.root.leaves[i+1].my_GPR = deepcopy(leaf.my_GPR)

            
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
            """ Recursive function to collect contributing leaves for prediction at single test point x.  """

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

                # mean_DLGP[i] += ptilde*mu_leaf[0]
                mean_DLGP[i] += ptilde*mu_leaf
                var_DLGP[i] += ptilde*(sigma_leaf*sigma_leaf + mu_leaf*mu_leaf)
            
            var_DLGP[i] += -mean_DLGP[i]*mean_DLGP[i]
        
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
        joblib.dump(self, path)
