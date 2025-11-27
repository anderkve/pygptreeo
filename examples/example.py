"""Basic example script demonstrating the usage of the GPTree class.

The example also defines a custom GPR class `my_GPR_class` to illustrate
how specific kernel configurations can be passed to the `GPTree`.
"""
import numpy as np
from pygptreeo import GPTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared, ConstantKernel, WhiteKernel
import sys

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

from target_functions import Eggholder, Himmelblau, Rosenbrock, Rastrigin, Levy, Custom

target_dict = {
    'eggholder': Eggholder,
    'himmelblau': Himmelblau,
    'rosenbrock': Rosenbrock,
    'rastrigin': Rastrigin,
    'levy': Levy,
    'custom': Custom,
}

np.random.seed(512312)
# np.random.seed(49235)
# np.random.seed(int(sys.argv[-1]))


#
# Test settings
#

target_name = "eggholder"
target = target_dict[target_name]

n_dims = 2
n_pts = 1000

Nbar = 100
theta = 1e-4
retrain_step = 20

x_min = 0.0
x_max = 1.0

X_input = np.random.uniform(x_min, x_max, n_dims * n_pts).reshape(n_pts, n_dims)
# X_input = np.random.normal(0.4, 0.1, n_dims * n_pts).reshape(n_pts, n_dims)
y_input = target(X_input.T)


class my_GPR_class(GaussianProcessRegressor):
    """Custom Gaussian Process Regressor for use with GPTree.

    This class inherits from `sklearn.gaussian_process.GaussianProcessRegressor`
    and is tailored for this example to define specific kernel configurations
    and default parameters for the GPR instances used within the `GPTree`.

    Attributes:
        kernel: The kernel to be used.
        min_length_scale (float): A minimum bound for the kernel's length
            scale hyperparameters. This is used by `GPNode` to constrain
            the kernel optimization.
        alpha (float or ndarray): Value added to the diagonal of the kernel
            matrix during fitting. Passed to `GaussianProcessRegressor`.
        optimizer (str or callable): The optimizer used for fitting the
            kernel's hyperparameters. Passed to `GaussianProcessRegressor`.
        n_restarts_optimizer (int): The number of times the optimizer is
            restarted. Passed to `GaussianProcessRegressor`.
        normalize_y (bool): Whether the target values y are normalized before
            fitting. Passed to `GaussianProcessRegressor`.
        copy_X_train (bool): If True, a persistent copy of the training data
            is stored. Passed to `GaussianProcessRegressor`.
        n_targets (int): The number of dimensions of the target values.
            Passed to `GaussianProcessRegressor`.
        random_state (int, RandomState instance or None): Controls the
            randomness of the initialization. Passed to `GaussianProcessRegressor`.
    """
    def __init__(self, kernel=None, *, alpha=1e-6, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, n_targets=None, random_state=None):
        super().__init__()
        self.kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3,1e8)) * Matern(nu=1.5, length_scale=[1.0]*n_dims, length_scale_bounds=[(1e-3, 1e3)]*n_dims)
        self.min_length_scale = 0.001

        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.n_targets = n_targets
        self.random_state = random_state


mygpr = my_GPR_class()

# Construct GPTree
gpt = GPTree(
    GPR=my_GPR_class(), 
    Nbar=Nbar,
    theta=theta, 
    split_position_method='median',
    split_dimension_criteria='max_variance',
    retrain_every_n_points=retrain_step,
    use_calibrated_sigma=True,
    splitting_strategy='gradual',
    # splitting_strategy='standard',
)


# Run through points one point at a time
point_i = 0
for x,y in zip(X_input, y_input):

    point_i += 1

    x = x.reshape((1, x.shape[0]))
    y = y.reshape((1,1))

    # Compute prediction
    y_pred, y_pred_std = gpt.predict(x, show_progress=False)

    # Update gpt with training point
    gpt.update_tree(x, y)

    # Print point summary comparing predicted y to true y
    print(f"point {point_i}:  x: {x[0]}  y: {y[0][0]}  y_pred: {y_pred[0][0]}  y_pred_std: {y_pred_std[0][0]}")

print()
print(gpt.root)
print()
print("Done.")
print()



