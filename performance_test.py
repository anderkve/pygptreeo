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
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import seaborn as sns
from pygptreeo import GPTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared, ConstantKernel, WhiteKernel
import sys
import time

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

def is_within_percentage(value, target, percentage):
    if target == 0: # Avoid division by zero if target is 0
        # If target is 0, value must also be 0 to be 'within percentage'
        return value == 0
    return np.abs(value - target) <= (percentage / 100.0) * np.abs(target)

np.random.seed(512312)
# np.random.seed(49235)
# np.random.seed(int(sys.argv[-1]))


#
# Test settings
#

target_name = "eggholder"
target = target_dict[target_name]

n_dims = 2
n_pts = 100000

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


# Make sure to use the global n_dims from example.py for data generation

def test_split_criteria():
    print("\nTesting Split Dimension Criteria...\n")

    n_pts_test = 50
    Nbar_test = 10

    # Use a fixed seed for reproducibility of test data
    np.random.seed(42)
    # Ensure X_test_data uses the global n_dims from example.py
    # (n_dims is available from the outer scope of example.py)
    X_test_data = np.random.rand(n_pts_test, n_dims)
    y_test_data = np.sum(X_test_data, axis=1).reshape(-1, 1)

    criteria_to_test = ['max_spread', 'max_variance', 'random']

    for criterion in criteria_to_test:
        print(f"--- Testing criterion: {criterion} ---")
        try:
            # my_GPR_class is defined in the global scope of example.py
            gpr_instance_for_test = my_GPR_class()

            gpt_test = GPTree(
                GPR=gpr_instance_for_test,
                Nbar=Nbar_test,
                split_dimension_criteria=criterion,
                retrain_every_n_points=Nbar_test
            )

            gpt_test.fit(X_test_data, y_test_data, show_progress=False)

            print(f"Successfully trained GPTree with criterion: {criterion}")
            if gpt_test.root and not gpt_test.root.is_leaf:
                print(f"Root node split on dimension: {gpt_test.root.split_index} using {criterion}")
            elif gpt_test.root and gpt_test.root.is_leaf:
                print(f"Root node did not split with Nbar={Nbar_test} (capacity) and {n_pts_test} points using {criterion}. It remained a leaf.")
            else:
                print(f"No root node OR root node is None after fit for {criterion}.")

        except Exception as e:
            print(f"Error during testing criterion {criterion}: {e}")
            import traceback
            traceback.print_exc()
        print(f"--- End Test for {criterion} ---\n")


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
    # splitting_strategy='gradual',
    splitting_strategy='standard',
)

# Initialize for plotting
fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
fig.suptitle('Performance Metrics Over Time', fontsize=16)

axs[0].set_ylabel('Predict Time (s)')
axs[1].set_ylabel('Update Time (s)')
axs[2].set_ylabel('RMSE')
axs[3].set_ylabel('Accuracy (<5% error)')
axs[4].set_ylabel('Coverage (within uncertainty)')
axs[4].set_xlabel('Number of Points Processed')

# Data storage for plots
points_processed_history = []
predict_times_history = []
update_times_history = []
rmse_history = []
accuracy_5_percent_history = []
coverage_history = []

# Data storage for current batch
current_batch_predict_times = []
current_batch_update_times = []
current_batch_actual_values = []
current_batch_predicted_values = []
current_batch_predicted_std_devs = []

# Plotting update frequency
plot_update_frequency = 2000


# Run through points one point at a time
point_i = 0
for x,y in zip(X_input, y_input):

    point_i += 1

    x = x.reshape((1, x.shape[0]))
    y = y.reshape((1,1))

    # Compute prediction
    start_time = time.time()
    y_pred, y_pred_std = gpt.predict(x, show_progress=False)
    predict_time = time.time() - start_time
    current_batch_predict_times.append(predict_time)

    # Update gpt with training point
    start_time = time.time()
    gpt.update_tree(x, y)
    update_time = time.time() - start_time
    current_batch_update_times.append(update_time)
    current_batch_actual_values.append(y[0][0])
    current_batch_predicted_values.append(y_pred[0][0])
    current_batch_predicted_std_devs.append(y_pred_std[0][0])

    if point_i % plot_update_frequency == 0 and point_i > 0:
        # Calculate metrics for the last 'plot_update_frequency' points
        total_predict_time_batch = np.sum(current_batch_predict_times)
        total_update_time_batch = np.sum(current_batch_update_times)

        actual_vals = np.array(current_batch_actual_values)
        predicted_vals = np.array(current_batch_predicted_values)
        std_devs = np.array(current_batch_predicted_std_devs)

        # RMSE
        rmse_batch = np.sqrt(np.mean((actual_vals - predicted_vals)**2))

        # Accuracy (within 5% of true value)
        within_5_percent_flags = [is_within_percentage(p, a, 5) for p, a in zip(predicted_vals, actual_vals)]
        accuracy_5_percent_batch = np.mean(within_5_percent_flags)

        # Empirical Coverage
        coverage_batch = np.mean(np.abs(actual_vals - predicted_vals) <= std_devs)

        # Append to history for plotting
        points_processed_history.append(point_i)
        predict_times_history.append(total_predict_time_batch)
        update_times_history.append(total_update_time_batch)
        rmse_history.append(rmse_batch)
        accuracy_5_percent_history.append(accuracy_5_percent_batch)
        coverage_history.append(coverage_batch)

        # Update plots
        axs[0].clear()
        axs[0].plot(points_processed_history, predict_times_history, marker='.')
        axs[0].set_ylabel('Predict Time (s)')
        axs[0].set_title('Predict Time per Batch')

        axs[1].clear()
        axs[1].plot(points_processed_history, update_times_history, marker='.')
        axs[1].set_ylabel('Update Time (s)')
        axs[1].set_title('Update Time per Batch')

        axs[2].clear()
        axs[2].plot(points_processed_history, rmse_history, marker='.')
        axs[2].set_ylabel('RMSE')
        axs[2].set_title('RMSE per Batch')

        axs[3].clear()
        axs[3].plot(points_processed_history, accuracy_5_percent_history, marker='.')
        axs[3].set_ylabel('Accuracy (<5% error)')
        axs[3].set_title('Fraction of Predictions within 5% Error')

        axs[4].clear()
        axs[4].plot(points_processed_history, coverage_history, marker='.')
        axs[4].set_ylabel('Coverage')
        axs[4].set_title('Empirical Coverage of Prediction Uncertainty')
        axs[4].set_xlabel('Number of Points Processed') # X-label only on the last plot

        # Remove x-axis labels from other subplots if they were set
        for i in range(4):
            axs[i].set_xlabel('')

        for ax_idx, ax in enumerate(axs):
            ax.grid(True)
            # Re-apply y-labels as clear() might remove them
            if ax_idx == 0: ax.set_ylabel('Predict Time (s)')
            elif ax_idx == 1: ax.set_ylabel('Update Time (s)')
            elif ax_idx == 2: ax.set_ylabel('RMSE')
            elif ax_idx == 3: ax.set_ylabel('Accuracy (<5% error)')
            elif ax_idx == 4: ax.set_ylabel('Coverage')


        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.pause(0.01)

        # Reset batch lists
        current_batch_predict_times.clear()
        current_batch_update_times.clear()
        current_batch_actual_values.clear()
        current_batch_predicted_values.clear()
        current_batch_predicted_std_devs.clear()

    # Print point summary comparing predicted y to true y
    # print(f"point {point_i}:  x: {x[0]}  y: {y[0][0]}  y_pred: {y_pred[0][0]}  y_pred_std: {y_pred_std[0][0]}")

print()
print(gpt.root)
print()
print("Done.")
print()




# test_split_criteria()
plt.show()
