import numpy as np
from binarytree import Node
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ExpSineSquared, ConstantKernel, WhiteKernel
from typing import Callable, Optional, Type, Union
from copy import deepcopy
from sys import float_info

from pygptreeo.default_gpr import Default_GPR

np.set_printoptions(suppress=True)

class GPNode(Node):
    def __init__(self, *args, 
                 my_GPR: GaussianProcessRegressor, 
                 Nbar: Optional[int] = 100,
                 split_position_method='median', 
                 retrain_every_n_points=1,
                 name="0"):
        
        super().__init__(*args)

        self.Nbar = Nbar

        self.my_GPR = my_GPR
        self.parent = None
        self.children = None

        self.split_position_method = split_position_method
        self.retrain_every_n_points = retrain_every_n_points
        
        self.is_left = None
        self.is_leaf = True
        
        self.num_training_points = 0
        self.num_buffer_points = 0

        self.my_X_data = None
        self.my_y_data = None
        self.n_features = None

        self.split_index = 0       # 'j' in the DLGP article
        self.split_position = 0.0  # 's' in the DLGP article
        self.overlap = 0.001       # 'o' in the DLGP article
        self.name = name

        self.n_points_pred_perf = 25
        self.residuals = np.zeros(self.n_points_pred_perf)
        self.mu_preds = np.zeros(self.n_points_pred_perf)
        self.sigma_preds = np.zeros(self.n_points_pred_perf)

        print(f"Created node {self.name}")


    # Override the "value" attribute of Node parent class 
    @property
    def num_training_points(self):
        return self.value

    
    # such that the value of a node is the number of training points
    @num_training_points.setter
    def num_training_points(self, value):
        self.value = value


    def init_training_set(self, n_features: int):
        """ Initialize the training set of the node. """
        self.my_X_data = np.array([]).reshape((0, n_features))
        self.my_y_data = np.array([]).reshape((0, 1))
        self.n_features = n_features
        self.num_buffer_points = 0
        self.num_training_points = 0


    def generate_children(self, GPR: Type[GaussianProcessRegressor], n_features: int):
        """ Grow the GPtree by adding two GPNodes as children of the current GPNode. """

        # Settings that will be passed on to the child nodes
        node_config_kwargs = {
            'Nbar': self.Nbar,
            'split_position_method': self.split_position_method,
            'retrain_every_n_points': self.retrain_every_n_points,
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
            child.init_training_set(n_features)

        # Copy the registered residuals and predictions
        self.left.residuals = self.residuals.copy()
        self.right.residuals = self.residuals.copy()

        self.left.mu_preds = self.mu_preds.copy()
        self.right.mu_preds = self.mu_preds.copy()

        self.left.sigma_preds = self.sigma_preds.copy()
        self.right.sigma_preds = self.sigma_preds.copy()


    def add_training_data(self, x: np.ndarray, y: float, increment_buffer=True):
        """ Add a single training sample to the training set of the node. """
        self.my_X_data = np.append(self.my_X_data, x, axis=0)
        self.my_y_data = np.append(self.my_y_data, y, axis=0)
        self.num_training_points += 1
        if increment_buffer:
            self.num_buffer_points += 1

        
    def split_training_data(self):
        """ Assign the training samples of a node to its child nodes. """
        for x, y in zip(self.my_X_data, self.my_y_data):
            x = x.reshape((1, x.shape[0]))
            y = y.reshape((1, 1))
            child = self.children[int(np.random.binomial(1, self.prob_func(x)))]
            child.add_training_data(x, y, increment_buffer=False)

        
    def delete_training_data(self):
        del self.my_X_data, self.my_y_data


    def delete_my_GPR(self):
        del self.my_GPR


    def fit_my_GPR(self, force_training=False):
        """ Fit the GP of the node with sklearn. """
        did_train = False
        # Only train the GP if the buffer is full, the node is full, or if force_training=True
        if (self.num_buffer_points == self.retrain_every_n_points) or (self.num_training_points == self.Nbar) or force_training:
            self.num_buffer_points = 0

            x_max_vals = [np.max(self.my_X_data[:,i]) for i in range(self.n_features)]
            x_min_vals = [np.min(self.my_X_data[:,i]) for i in range(self.n_features)]
            x_ranges = [x_max_vals[i] - x_min_vals[i] for i in range(self.n_features)]
            # x_ranges = [np.max(self.my_X_data[:,i])-np.min(self.my_X_data[:,i]) for i in range(self.n_features)]

            use_bounds = [(np.max([self.my_GPR.min_length_scale, 0.01*x_ranges[i]]), np.max([10*self.my_GPR.min_length_scale, 10*x_ranges[i]])) for i in range(self.n_features)]
            use_init_points = [0.1*x_ranges[i] for i in range(self.n_features)]


            # Loop over kernel alternatives
            # TODO: Only try the alternative kernels with a certain probability?
            # TODO: Make this code more efficient. It should be unnecessary to copy the
            #       entire my_GPR object like we do below...
            best_lml = float_info.max
            temp_GPR = self.my_GPR
            for kernel in self.my_GPR.kernel_alternatives:

                params = kernel.get_params(deep=True)

                # Example params dict: 
                # params: {
                #     'k1': 1**2, 
                #     'k2': Matern(length_scale=[1, 1], nu=1.5), 
                #     'k1__constant_value': 1.0, 
                #     'k1__constant_value_bounds': (0.001, 100000000.0), 
                #     'k2__length_scale': [1.0, 1.0], 
                #     'k2__length_scale_bounds': [(0.001, 1000.0), (0.001, 1000.0)], 
                #     'k2__nu': 1.5
                # }

                new_params = {}
                for k,v in params.items():
                    if k[-21:] == "__length_scale_bounds":
                        new_params[k] = use_bounds
                    elif k[-14:] == "__length_scale":
                        new_params[k] = use_init_points
                kernel.set_params(**new_params)

                temp_GPR.kernel = kernel
                temp_GPR.fit(self.my_X_data, self.my_y_data)

                lml = temp_GPR.log_marginal_likelihood_value_
                theta = temp_GPR.kernel_.theta
                # print(f"kernel: {kernel}   lml: {lml}   theta: {theta}   exp(theta): {np.exp(theta)}")

                # Keep this kernel?
                if lml < best_lml:
                    self.my_GPR = deepcopy(temp_GPR)
                    best_lml = lml

            x_range_strs = []
            for i in range(self.n_features):
                x_range_strs.append( "({:.2e},{:.2e})".format(x_min_vals[i],x_max_vals[i]))
            x_range_str = "[" + ", ".join(x_range_strs) + "]"
            print(f"Trained node {self.name}:  "
                  # f"x_data_range: {[(x_max_vals[i],x_min_vals[i]) for i in range(self.n_features)]}  "
                  f"x_range: {x_range_str}  "
                  f"kernel: {self.my_GPR.kernel_}"
            )

            did_train = True
        return did_train


    def compute_split_position_and_overlap(self, theta: float):
        """ Find the position of the dividing hyperplane and the size of the overlapping region. """

        # TODO: Introduce alternative ways to choose the split index, 
        #       e.g. max spread per lengthscale, principal component, random, ...
        #       Currently this is using the "max spread" approach.
        w = np.empty(self.n_features)
        for i in range(self.n_features):
            w[i] = np.max(self.my_X_data[:, i]) - np.min(self.my_X_data[:, i])
        self.split_index = np.argmax(w)

        # TODO: Introduce alternative ways to compute the split position, e.g. median

        self.split_position = None
        if self.split_position_method == 'median':
            self.split_position = np.median(self.my_X_data[:, self.split_index])
        elif self.split_position_method == 'mean':
            self.split_position = np.mean(self.my_X_data[:, self.split_index])
        elif self.split_position_method == 'random':
            self.split_position = np.random.uniform(np.min(self.my_X_data[:, self.split_index]), np.max(self.my_X_data[:, self.split_index]), 1) 
        elif self.split_position_method == 'randomchoice':
            self.split_position = np.random.choice(self.my_X_data[:, self.split_index])
        else:
            raise ValueError(f"Unknown split_position_method argument: '{self.split_position_method}'. The valid options are 'median', 'mean', 'random' and 'randomchoice'")

        self.overlap = theta*w[self.split_index]


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


    def predict(self, x: np.ndarray, return_std=True):
        """ Evaluate the prediction from this node's GP at input point x. """
        mu_pred, sigma_pred = self.my_GPR.predict(x, return_std=return_std)
        return mu_pred, sigma_pred


    def register_pred_perf(self, x: np.ndarray, y: float):
        """ Register the residual between prediction and true value at for this data point. """
        mu_pred, sigma_pred = self.predict(x, return_std=True)

        self.residuals = self.residuals[:-1]
        self.residuals = np.insert(self.residuals, 0, y - mu_pred)

        self.mu_preds = self.mu_preds[:-1]
        self.mu_preds = np.insert(self.mu_preds, 0, mu_pred)

        self.sigma_preds = self.sigma_preds[:-1]
        self.sigma_preds = np.insert(self.sigma_preds, 0, sigma_pred)

