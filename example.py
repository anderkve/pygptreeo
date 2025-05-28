import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import seaborn as sns
from pygptreeo import GPTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared, ConstantKernel, WhiteKernel
import sys

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

from example_target_functions import Eggholder, Himmelblau, Rosenbrock, Rastrigin, Levy, Custom

target_dict = {
    'eggholder': Eggholder,
    'himmelblau': Himmelblau,
    'rosenbrock': Rosenbrock,
    'rastrigin': Rastrigin,
    'levy': Levy,
    'custom': Custom,
}

# plt.rcParams['text.usetex'] = True

np.random.seed(512312)
# np.random.seed(49235)
# np.random.seed(int(sys.argv[-1]))


#
# Test settings
#

target_name = "eggholder"
target = target_dict[target_name]

n_dims = 2
n_pts = 10000

Nbar = 200
theta = 1e-4
retrain_step = 200

x_min = 0.0
x_max = 1.0

X_input = np.random.uniform(x_min, x_max, n_dims * n_pts).reshape(n_pts, n_dims)
# X_input = np.random.normal(0.4, 0.1, n_dims * n_pts).reshape(n_pts, n_dims)
y_input = target(X_input.T)


class my_GPR_class(GaussianProcessRegressor):
    def __init__(self, kernel=None, *, alpha=1e-6, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=True, copy_X_train=True, n_targets=None, random_state=None):
        super().__init__()
        self.kernel_alternatives = [
            ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3,1e8)) * Matern(nu=1.5, length_scale=[1.0]*n_dims, length_scale_bounds=[(1e-3, 1e3)]*n_dims),
            # ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3,1e8)) * RBF(length_scale=[1.0]*n_dims, length_scale_bounds=[(1e-3, 1e3)]*n_dims),
        ]

        self.kernel = self.kernel_alternatives[0]
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
    retrain_every_n_points=retrain_step,
    use_calibrated_sigma=True,
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
print("Done.")
print()

