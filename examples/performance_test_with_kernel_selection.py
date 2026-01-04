"""Performance test demonstrating automatic kernel selection.

This script showcases the automatic kernel selection feature of the `GPTree`
from the `pygptreeo` library for a regression task. It performs the following steps:
1.  Defines or imports a target function (e.g., Eggholder, Himmelblau).
2.  Sets up parameters for the `GPTree` with automatic kernel selection enabled.
3.  Initializes a `GPTree` instance with `enable_kernel_selection=True`.
4.  Generates random input data (`X_input`) and corresponding target
    values (`y_input`).
5.  Iterates through the data points:
    a.  Makes a prediction using the current state of the `GPTree`.
    b.  Updates the `GPTree` with the new data point.
    c.  Prints a comparison of the true and predicted values.
6.  At the end, reports which kernel types were selected across the tree.

With automatic kernel selection, each node tests its current kernel against a
randomly selected alternative at split time, and selects the better performing
one based on log marginal likelihood. This allows the tree to adapt kernel
complexity to different regions: complex kernels for large regions, simpler
kernels for small regions.
"""
import numpy as np
import matplotlib.pyplot as plt
from pygptreeo import GPTree
from pygptreeo.default_gpr import Default_GPR
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

# target_name = "rosenbrock"
target_name = "eggholder"
# target_name = "himmelblau"
target = target_dict[target_name]

n_dims = 3
n_pts = 100000

Nbar = 200
theta = 1e-4  #0.10 # 1e-4
retrain_step = 200

x_min = 0.0
x_max = 1.0

X_input = np.random.uniform(x_min, x_max, n_dims * n_pts).reshape(n_pts, n_dims)
# X_input = np.random.normal(0.4, 0.1, n_dims * n_pts).reshape(n_pts, n_dims)
y_input = target(X_input.T)


# Construct GPTree with automatic kernel selection
# Instead of defining a custom GPR class, we enable automatic kernel selection
# The tree will start with kernel type 0: Const*(RBF + Matern(nu=1.5))
# At each split, nodes will test their kernel against a random alternative
# and select the better performing one based on log marginal likelihood

# Create a GPR with custom settings to use as template
# The kernel will be automatically set based on kernel selection
my_gpr = Default_GPR(
    n_restarts_optimizer=3,  # Match the original performance_test.py settings
    alpha=1e-6,
)

gpt = GPTree(
    GPR=my_gpr,  # Pass custom GPR as template for settings
    enable_kernel_selection=True,  # Enable automatic kernel selection
    Nbar=Nbar,
    theta=theta,
    split_position_method='median',
    # split_dimension_criteria='max_variance',
    split_dimension_criteria='max_uncertainty',
    retrain_every_n_points=retrain_step,
    use_calibrated_sigma=True,
    splitting_strategy='gradual',
    # splitting_strategy='standard',
    max_n_pred_leaves=3,
    aggregation='moe',
    # aggregation='poe',
    #
    use_hyperparameter_inheritance=False,
    use_standard_scaling=True,
    #
    enable_point_rejection=False,
    rejection_threshold=1e-2,
    min_points_before_rejection=25,
    #
    enable_point_merging=False,
    merge_distance_threshold=0.01,
    min_points_before_merging=10,
    #
    enable_split_evaluation=True,
    n_split_candidates=4,
    split_eval_train_fraction=0.4,
    split_eval_min_points=20,
)

# Initialize for plotting
fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)
fig.suptitle('PyGPTreeo performance metrics (with automatic kernel selection)', fontsize=16)

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
        y_pred, y_pred_std, leaf_names = gpt.predict(x, show_progress=False, return_leaf_names=True)

        # Update tree
        gpt.update_tree(x, y, 0.001 * np.abs(y))

        # Print point summary
        abs_err = np.abs(y_pred[0][0] - y[0][0])
        rel_err = abs_err / np.max([np.abs(y[0][0]), 1e-10])
        print(f"point {point_i}:  x: {x[0]}  y: {y[0][0]:.4e}  y_pred: {y_pred[0][0]:.4e}  y_pred_std: {y_pred_std[0][0]:.3e}  abs_err: {abs_err:.3e}  rel_err: {rel_err:.3e}  n_leaves: {len(leaf_names)}")

        continue

    # OK, we have make_plot = True

    # Compute prediction
    start_time = time.time()
    y_pred, y_pred_std, leaf_names = gpt.predict(x, show_progress=False, return_leaf_names=True)
    predict_time = time.time() - start_time
    current_batch_predict_times.append(predict_time)

    # Update tree
    start_time = time.time()
    gpt.update_tree(x, y, 0.001 * np.abs(y))
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

        # Accuracy
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
        axs[0].plot(points_processed_history, avg_predict_times_history, label="Avg. predict time (last 2000 pts)", linewidth=2.0)
        axs[0].set_xlim([0, n_pts])
        axs[0].set_ylabel('Time (s)')
        axs[0].set_title('Average prediction time per point')
        axs[0].legend()

        axs[1].clear()
        axs[1].plot(points_processed_history, avg_update_times_history, label="Avg. update time (last 2000 pts)", color='orange', linewidth=2.0)
        axs[1].set_xlim([0, n_pts])
        axs[1].set_ylabel('Time (s)')
        axs[1].set_title('Average tree update time per point')
        axs[1].legend()

        axs[2].clear()
        axs[2].plot(points_processed_history, nrmse_history, label="NRMSE (last 2000 pts)", color='green', linewidth=2.0)
        axs[2].set_ylabel('NRMSE')
        axs[2].set_title('NRMSE for predictions')
        axs[2].set_xlim([0, n_pts])
        axs[2].set_ylim([0.001, 1.0])
        # axs[2].set_ylim([0.0, np.max([0.1, np.max(nrmse_history)])])
        axs[2].set_yscale('log')
        axs[2].legend()

        axs[3].clear()
        axs[3].plot(points_processed_history, within_16_percent_history, label="Fraction < 16% Error (last 2000 pts)", linewidth=2.0)
        axs[3].plot(points_processed_history, within_8_percent_history, label="Fraction < 8% Error (last 2000 pts)", linewidth=2.0)
        axs[3].plot(points_processed_history, within_4_percent_history, label="Fraction < 4% Error (last 2000 pts)", linewidth=2.0)
        axs[3].plot(points_processed_history, within_2_percent_history, label="Fraction < 2% Error (last 2000 pts)", linewidth=2.0)
        axs[3].plot(points_processed_history, within_1_percent_history, label="Fraction < 1% Error (last 2000 pts)", linewidth=2.0)
        axs[3].set_ylabel('Fraction')
        axs[3].set_title('Fraction of predictions within x% of true value')
        axs[3].set_xlim([0, n_pts])
        axs[3].set_ylim([0, 1])
        axs[3].legend()

        axs[4].clear()
        axs[4].plot(points_processed_history, coverage_history, label="Empirical coverage (last 2000 pts)", color='purple', linewidth=2.0)
        axs[4].plot([points_processed_history[0], points_processed_history[-1]], [0.68, 0.68], "--", color='black', linewidth=2.0)
        axs[4].set_ylabel('Fraction')
        axs[4].set_title('Empirical coverage of prediction uncertainty')
        axs[4].set_xlabel('Total points processed') # X-label only on the last plot
        axs[4].set_xlim([0, n_pts])
        axs[4].set_ylim([0, 1])
        axs[4].legend()

        # Remove x-axis labels from other subplots if they were set
        for i in range(4):
            axs[i].set_xlabel('')

        for ax in axs:
            ax.grid(True)


        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("plot_2.png")
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
    print(f"point {point_i}:  x: {x[0]}  y: {y[0][0]:.4e}  y_pred: {y_pred[0][0]:.4e}  y_pred_std: {y_pred_std[0][0]:.3e}  abs_err: {abs_err:.3e}  rel_err: {rel_err:.3e}  n_leaves: {len(leaf_names)}")


print()
print("Done.")
print()

# Report kernel type distribution across the tree
print("=" * 80)
print("Kernel Type Distribution Across Tree")
print("=" * 80)

kernel_names = {
    0: "Const*(RBF + Matern(nu=1.5))",
    1: "Const*(RQ + Matern(nu=1.5))",
    2: "Const*(RQ + RBF)",
    3: "Const*RQ",
    4: "Const*Matern(nu=1.5)",
    5: "Const*RBF"
}

# Collect kernel type statistics from all leaves
kernel_types = {}
for leaf in gpt.root.leaves:
    kernel_idx = leaf.kernel_type_idx
    if kernel_idx is not None:
        kernel_types[kernel_idx] = kernel_types.get(kernel_idx, 0) + 1

print(f"\nTotal number of leaf nodes: {len(gpt.root.leaves)}")
print("\nKernel types selected:")
for idx in sorted(kernel_types.keys()):
    count = kernel_types[idx]
    name = kernel_names.get(idx, f"Unknown ({idx})")
    percentage = 100.0 * count / len(gpt.root.leaves)
    print(f"  Kernel {idx}: {count:3d} leaves ({percentage:5.1f}%) - {name}")

print()
print("This demonstrates how automatic kernel selection allows different regions")
print("of the tree to use different kernel types based on their local data.")
print("=" * 80)
