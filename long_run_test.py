"""Performance test script for GPTree focusing on long runs and CSV output.

The script uses a custom GPR class `my_GPR_class` for specific kernel
configurations, similar to `example.py`.
"""
import numpy as np
from pygptreeo import GPTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared, ConstantKernel, WhiteKernel
import sys
import csv
import time

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
n_pts = 2500

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
        kernel_alternatives (list): A list of scikit-learn kernel objects.
            The `GPNode` within `GPTree` will iterate through these kernels
            during its fitting process and select the one that maximizes
            the log-marginal-likelihood.
        min_length_scale (float): A minimum bound for the kernel's length
            scale hyperparameters. This is used by `GPNode` to constrain
            the kernel optimization.
        kernel: The default kernel to be used. Initially set to the first
            kernel in `kernel_alternatives`.
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
    split_dimension_criteria='max_variance',
    retrain_every_n_points=retrain_step,
    use_calibrated_sigma=True,
    splitting_strategy='gradual',
    # splitting_strategy='standard',
)

results_buffer = []
csv_file_name = "run_output.csv"
csv_header = ["x_coordinates", "true_y", "predicted_y", "prediction_uncertainty", "predict_time_s", "update_tree_time_s"]

with open(csv_file_name, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)

# Run through points one point at a time
point_i = 0
for x,y in zip(X_input, y_input):

    point_i += 1

    x = x.reshape((1, x.shape[0]))
    y = y.reshape((1,1))

    # Compute prediction
    start_predict_time = time.perf_counter()
    y_pred, y_pred_std = gpt.predict(x, show_progress=False)
    end_predict_time = time.perf_counter()
    predict_time = end_predict_time - start_predict_time

    # Update gpt with training point
    start_update_time = time.perf_counter()
    gpt.update_tree(x, y)
    end_update_time = time.perf_counter()
    update_tree_time = end_update_time - start_update_time

    # Convert x to a string representation, handling multi-dimensional case
    x_coords_list = [f"{xi:.4e}" for xi in x[0]]
    x_coords_str = ";".join(x_coords_list)
    # x_coords_str = ";".join(map(str, x[0]))

    current_result = [
        x_coords_str,
        f"{y[0][0]:.6e}",
        f"{y_pred[0][0]:.6e}",
        f"{y_pred_std[0][0]:.6e}",
        f"{predict_time:.3e}",
        f"{update_tree_time:.3e}"
    ]
    results_buffer.append(current_result)

    if len(results_buffer) >= 2000:
        with open(csv_file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(results_buffer)
        results_buffer.clear()
        print(f"Processed and wrote {point_i} points to {csv_file_name}") # Optional: for progress indication

# Write any remaining data in the buffer
if results_buffer:
    with open(csv_file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results_buffer)
    print(f"Wrote remaining {len(results_buffer)} points to {csv_file_name}") # Optional
    results_buffer.clear()

print("\nDone.\n") # Keep a final "Done" message or make it more specific.
