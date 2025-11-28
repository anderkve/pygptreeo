"""Basic example script demonstrating the usage of the GPTree class.

This script showcases how to set up and use the `GPTree` from the
`pygptreeo` library for a regression task. It performs the following steps:
1.  Defines or imports a target function (e.g., Eggholder, Himmelblau).
2.  Sets up parameters for the `GPTree` and the custom GPR (`my_GPR_class`).
3.  Initializes a `GPTree` instance with the custom GPR.
4.  Generates random input data (`X_input`) and corresponding target
    values (`y_input`).
5.  Iterates through the data points:
    a.  Makes a prediction using the current state of the `GPTree`.
    b.  Updates the `GPTree` with the new data point.
    c.  Prints a comparison of the true and predicted values.

The example also defines a custom GPR class `my_GPR_class` to illustrate
how specific kernel configurations can be passed to the `GPTree`.
"""
import numpy as np
import matplotlib.pyplot as plt
from pygptreeo import GPTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared, ConstantKernel, WhiteKernel
import sys
import time

# from warnings import simplefilter
# from sklearn.exceptions import ConvergenceWarning
# simplefilter("ignore", category=ConvergenceWarning)

from target_functions import Eggholder, Himmelblau, Rosenbrock, Rastrigin, Levy, Custom

target_dict = {
    'eggholder': Eggholder,
    'himmelblau': Himmelblau,
    'rosenbrock': Rosenbrock,
    'rastrigin': Rastrigin,
    'levy': Levy,
    'custom': Custom,
}


def is_within_percentage(value, target, percentage):
    if target == 0: # Avoid division by zero if target is 0
        # If target is 0, value must also be 0 to be 'within percentage'
        return value == 0
    return (np.abs(value - target) / np.abs(target)) <= (0.01 * percentage)

np.random.seed(512312)
# np.random.seed(49235)
# np.random.seed(int(sys.argv[-1]))


#
# Test settings
#

make_plot = True

target_name = "eggholder"
target = target_dict[target_name]

n_dims = 3
n_pts = 100000

Nbar = 100
theta = 1e-4
retrain_step = 100

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
    def __init__(self, kernel=None, *, alpha=1e-6, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=1, normalize_y=False, copy_X_train=True, n_targets=None, random_state=None):
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

# Initialize for plotting
fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
fig.suptitle('Performance metrics', fontsize=16)

# Data storage for plots
points_processed_history = []
avg_predict_times_history = []
avg_update_times_history = []
nrmse_history = []
within_1_percent_history = []
within_2_percent_history = []
within_4_percent_history = []
within_8_percent_history = []
within_16_percent_history = []
coverage_history = []

# Data storage for current batch
current_batch_predict_times = []
current_batch_update_times = []
current_batch_actual_values = []
current_batch_predicted_values = []
current_batch_predicted_std_devs = []

# Batch size and plot update frequency
batch_size = 2000
plot_update_frequency = batch_size


# Run through points one point at a time
point_i = 0
for x,y in zip(X_input, y_input):

    point_i += 1

    x = x.reshape((1, x.shape[0]))
    y = y.reshape((1,1))

    if not make_plot:

        # Compute prediction
        y_pred, y_pred_std = gpt.predict(x, show_progress=False)

        # Update tree
        gpt.update_tree(x, y)

        # Print point summary
        abs_err = np.abs(y_pred[0][0] - y[0][0])
        rel_err = abs_err / np.max([np.abs(y[0][0]), 1e-10])
        print(f"point {point_i}:  x: {x[0]}  y: {y[0][0]:.4e}  y_pred: {y_pred[0][0]:.4e}  y_pred_std: {y_pred_std[0][0]:.3e}  abs_err: {abs_err:.3e}  rel_err: {rel_err:.3e}")

        continue

    # OK, we have make_plot = True

    # Compute prediction
    start_time = time.time()
    y_pred, y_pred_std = gpt.predict(x, show_progress=False)
    predict_time = time.time() - start_time
    current_batch_predict_times.append(predict_time)

    # Update tree
    start_time = time.time()
    gpt.update_tree(x, y)
    update_time = time.time() - start_time
    current_batch_update_times.append(update_time)
    current_batch_actual_values.append(y[0][0])
    current_batch_predicted_values.append(y_pred[0][0])
    current_batch_predicted_std_devs.append(y_pred_std[0][0])

    if point_i % plot_update_frequency == 0 and point_i > 0:
        # Calculate metrics for the last 'plot_update_frequency' points
        avg_predict_time_batch = np.sum(current_batch_predict_times) / float(batch_size)
        avg_update_time_batch = np.sum(current_batch_update_times) / float(batch_size)

        actual_vals = np.array(current_batch_actual_values)
        predicted_vals = np.array(current_batch_predicted_values)
        std_devs = np.array(current_batch_predicted_std_devs)

        # NRMSE
        nrmse_batch = np.sqrt(np.mean((actual_vals - predicted_vals)**2)) / (np.max(actual_vals) - np.min(actual_vals))

        # Accuracy (within 5% of true value)
        within_1_percent_flags = [is_within_percentage(p, a, 1) for p, a in zip(predicted_vals, actual_vals)]
        within_2_percent_flags = [is_within_percentage(p, a, 2) for p, a in zip(predicted_vals, actual_vals)]
        within_4_percent_flags = [is_within_percentage(p, a, 4) for p, a in zip(predicted_vals, actual_vals)]
        within_8_percent_flags = [is_within_percentage(p, a, 8) for p, a in zip(predicted_vals, actual_vals)]
        within_16_percent_flags = [is_within_percentage(p, a, 16) for p, a in zip(predicted_vals, actual_vals)]
        within_1_percent_batch = np.mean(within_1_percent_flags)
        within_2_percent_batch = np.mean(within_2_percent_flags)
        within_4_percent_batch = np.mean(within_4_percent_flags)
        within_8_percent_batch = np.mean(within_8_percent_flags)
        within_16_percent_batch = np.mean(within_16_percent_flags)

        # Empirical Coverage
        coverage_batch = np.mean(np.abs(actual_vals - predicted_vals) <= std_devs)

        # Append to history for plotting
        points_processed_history.append(point_i)
        avg_predict_times_history.append(avg_predict_time_batch)
        avg_update_times_history.append(avg_update_time_batch)
        nrmse_history.append(nrmse_batch)
        within_1_percent_history.append(within_1_percent_batch)
        within_2_percent_history.append(within_2_percent_batch)
        within_4_percent_history.append(within_4_percent_batch)
        within_8_percent_history.append(within_8_percent_batch)
        within_16_percent_history.append(within_16_percent_batch)
        coverage_history.append(coverage_batch)

        # Update plots
        axs[0].clear()
        axs[0].plot(points_processed_history, avg_predict_times_history, marker='.')
        axs[0].set_ylabel('Time (s)')
        axs[0].set_title('Avg. predict time per point')

        axs[1].clear()
        axs[1].plot(points_processed_history, avg_update_times_history, marker='.')
        axs[1].set_ylabel('Time (s)')
        axs[1].set_title('Avg. update time per point')

        axs[2].clear()
        axs[2].plot(points_processed_history, nrmse_history, marker='.')
        axs[2].set_ylabel('Batch NRMSE')
        axs[2].set_title('Batch NRMSE')
        axs[2].set_ylim([0.001, 1.0])
        # axs[2].set_ylim([0.0, np.max([0.1, np.max(nrmse_history)])])
        axs[2].set_yscale('log')

        axs[3].clear()
        axs[3].plot(points_processed_history, within_1_percent_history, marker='.', label="within 1%")
        axs[3].plot(points_processed_history, within_2_percent_history, marker='.', label="within 2%")
        axs[3].plot(points_processed_history, within_4_percent_history, marker='.', label="within 4%")
        axs[3].plot(points_processed_history, within_8_percent_history, marker='.', label="within 8%")
        axs[3].plot(points_processed_history, within_16_percent_history, marker='.', label="within 16%")
        axs[3].legend()
        axs[3].set_ylabel('Fraction within error threshold')
        axs[3].set_title('Fraction within error threshold')
        axs[3].set_ylim([0, 1])

        axs[4].clear()
        axs[4].plot(points_processed_history, coverage_history, marker='.')
        axs[4].plot([points_processed_history[0], points_processed_history[-1]], [0.68, 0.68], '--')
        axs[4].set_ylabel('Coverage')
        axs[4].set_title('Empirical coverage of prediction uncertainty')
        axs[4].set_xlabel('Number of points processed') # X-label only on the last plot
        axs[4].set_ylim([0, 1])

        # Remove x-axis labels from other subplots if they were set
        for i in range(4):
            axs[i].set_xlabel('')

        for ax_idx, ax in enumerate(axs):
            ax.grid(True)
            # Re-apply y-labels as clear() might remove them
            if ax_idx == 0: ax.set_ylabel('Time (s)')
            elif ax_idx == 1: ax.set_ylabel('Time (s)')
            elif ax_idx == 2: ax.set_ylabel('Batch NRMSE')
            elif ax_idx == 3: ax.set_ylabel('Fraction within error threshold')
            elif ax_idx == 4: ax.set_ylabel('Coverage')


        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("plot.png")
        # plt.pause(0.01)

        # Reset batch lists
        current_batch_predict_times.clear()
        current_batch_update_times.clear()
        current_batch_actual_values.clear()
        current_batch_predicted_values.clear()
        current_batch_predicted_std_devs.clear()

    # Print point summary
    abs_err = np.abs(y_pred[0][0] - y[0][0])
    rel_err = abs_err / np.max([np.abs(y[0][0]), 1e-10])
    print(f"point {point_i}:  x: {x[0]}  y: {y[0][0]:.4e}  y_pred: {y_pred[0][0]:.4e}  y_pred_std: {y_pred_std[0][0]:.3e}  abs_err: {abs_err:.3e}  rel_err: {rel_err:.3e}")



print()
print("Done.")
print()

