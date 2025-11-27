"""Animated visualization of the GPTree learning process.

This script demonstrates the capabilities of the GPTree by providing an
animated visualization as it learns a target function incrementally.
It performs the following main actions:

1.  **Setup**:
    *   Takes command-line arguments to configure the test, including:
        *   `target_name`: The name of the target function to learn (e.g., 'eggholder', 'himmelblau').
        *   `n_pts`: Total number of training points to generate and process.
        *   `Nbar`: GPTree parameter for max points per leaf node.
        *   `retrain_step`: GPTree parameter for retraining frequency in GPNode.
        *   `update_step`: Frequency (in terms of Nbar multiples) for updating plots.
        *   `live_update`: Boolean (0 or 1) to enable/disable live plot updates.
    *   Initializes the selected target function and a `GPTree` instance,
        optionally with a custom GPR configuration (`my_GPR_class`).
    *   Generates a set of 2D random training data points (`X_input`, `y_input`).

2.  **Plotting Setup**:
    *   Creates a Matplotlib figure with multiple subplots to display:
        *   Four 1D slices of the true target function versus the GPTree's
          current predictions and uncertainty bounds.
        *   A 2D scatter plot showing the locations of training data points used so far.
        *   A 2D contour plot of the true target function for reference.

3.  **Incremental Learning and Animation**:
    *   Feeds the training data points to the `GPTree` one by one using
      `gpt.update_tree()`.
    *   Periodically (controlled by `update_step`), the `update_plots`
      function is called to refresh the visualizations with the GPTree's
      latest predictions and the newly added training data.
    *   If `live_update` is enabled, plots are updated interactively.
      Otherwise, a final plot is shown after all points are processed.

The script is primarily intended for visualizing how the GPTree adapts its
predictions and structure as it encounters more data.
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

# plt.rcParams['text.usetex'] = True

# np.random.seed(512312)
# np.random.seed(49235)
np.random.seed(int(sys.argv[-1]))


#
# Test settings
#

target_name = sys.argv[1].lower()
target = target_dict[target_name]

n_dims = 2
n_pts = int(sys.argv[2])

Nbar = int(sys.argv[3])
theta = 1e-4
retrain_step = int(sys.argv[4])

update_step = int(sys.argv[5])
live_update = bool(int(sys.argv[6]))

x_min = 0.0
x_max = 1.0

X_input = np.random.uniform(x_min, x_max, n_dims * n_pts).reshape(n_pts, n_dims)
# X_input = np.random.normal(0.4, 0.1, n_dims * n_pts).reshape(n_pts, n_dims)
y_input = target(X_input.T)




class my_GPR_class(GaussianProcessRegressor):
    """Custom Gaussian Process Regressor for use with GPTree in this script.

    This class inherits from `sklearn.gaussian_process.GaussianProcessRegressor`
    and is tailored for this visualization script to define specific kernel
    configurations and default parameters for the GPR instances used within
    the `GPTree`.

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
    def __init__(self, kernel=None, *, alpha=1e-6, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=True, copy_X_train=True, n_targets=None, random_state=None):
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
    retrain_every_n_points=retrain_step,
    use_calibrated_sigma=True,
)



#
# Evaluate true function
#

n_plot_pts = 400
n_predict_pts = 200

# Plot 1
plot_1_x2_val = 0.75
X_true_plot_1 = np.column_stack((
    np.linspace(x_min, x_max, n_plot_pts),
    np.ones(n_plot_pts) * plot_1_x2_val
))
x_true_plot_1 = X_true_plot_1[:,0]
y_true_plot_1 = target(X_true_plot_1.T)

X_predict_plot_1 = np.column_stack((
    np.linspace(x_min, x_max, n_predict_pts),
    np.ones(n_predict_pts) * plot_1_x2_val
))
x_predict_plot_1 = X_predict_plot_1[:,0]
y_predict_plot_1, y_predict_std_plot_1 = gpt.predict(X_predict_plot_1)
y_predict_plot_1 = y_predict_plot_1.reshape(n_predict_pts,)
y_predict_std_plot_1 = y_predict_std_plot_1.reshape(n_predict_pts,)


# Plot 2
plot_2_x2_val = 0.25
X_true_plot_2 = np.column_stack((
    np.linspace(x_min, x_max, n_plot_pts),
    np.ones(n_plot_pts) * plot_2_x2_val
))
x_true_plot_2 = X_true_plot_2[:,0]
y_true_plot_2 = target(X_true_plot_2.T)

X_predict_plot_2 = np.column_stack((
    np.linspace(x_min, x_max, n_predict_pts),
    np.ones(n_predict_pts) * plot_2_x2_val
))
x_predict_plot_2 = X_predict_plot_2[:,0]
y_predict_plot_2, y_predict_std_plot_2 = gpt.predict(X_predict_plot_2)
y_predict_plot_2 = y_predict_plot_2.reshape(n_predict_pts,)
y_predict_std_plot_2 = y_predict_std_plot_2.reshape(n_predict_pts,)


# Plot 3
plot_3_x1_val = 0.4
X_true_plot_3 = np.column_stack((
    np.ones(n_plot_pts) * plot_3_x1_val,
    np.linspace(x_min, x_max, n_plot_pts),
))
x_true_plot_3 = X_true_plot_3[:,1]
y_true_plot_3 = target(X_true_plot_3.T)

X_predict_plot_3 = np.column_stack((
    np.ones(n_predict_pts) * plot_3_x1_val,
    np.linspace(x_min, x_max, n_predict_pts),
))
x_predict_plot_3 = X_predict_plot_3[:,1]
y_predict_plot_3, y_predict_std_plot_3 = gpt.predict(X_predict_plot_3)
y_predict_plot_3 = y_predict_plot_3.reshape(n_predict_pts,)
y_predict_std_plot_3 = y_predict_std_plot_3.reshape(n_predict_pts,)


# Plot 4
X_true_plot_4 = np.column_stack((
    np.linspace(x_min, x_max, n_plot_pts),
    np.linspace(x_min, x_max, n_plot_pts),
    # np.ones(n_plot_pts) * 0.2
))
x_true_plot_4 = X_true_plot_4[:,0]
y_true_plot_4 = target(X_true_plot_4.T)

X_predict_plot_4 = np.column_stack((
    np.linspace(x_min, x_max, n_predict_pts),
    np.linspace(x_min, x_max, n_predict_pts),
    # np.ones(n_predict_pts) * 0.2
))
x_predict_plot_4 = X_predict_plot_4[:,0]
y_predict_plot_4, y_predict_std_plot_4 = gpt.predict(X_predict_plot_4)
y_predict_plot_4 = y_predict_plot_4.reshape(n_predict_pts,)
y_predict_std_plot_4 = y_predict_std_plot_4.reshape(n_predict_pts,)



#
# Compute data for 2D color plot of the target function
#

n_pr_dim = 401
x_2D = np.linspace(x_min, x_max, n_pr_dim)
y_2D = np.linspace(x_min, x_max, n_pr_dim)
z_2D = np.array([target(np.array([x,y])) for y in y_2D for x in x_2D])

Z_2D = z_2D.reshape(n_pr_dim, n_pr_dim)
X_2D, Y_2D = np.meshgrid(x_2D, y_2D)


#
# Prepare figure with multiple panels 
#

# sns.set()

fig = plt.figure(figsize=(20.0, 12.0)) 
gs = gridspec.GridSpec(4, 2, height_ratios=[1,1,1,1], width_ratios=[3,2], figure=fig, hspace=0.35)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[2,0])
ax4 = fig.add_subplot(gs[3,0])
ax5 = fig.add_subplot(gs[0:2,1])
ax6 = fig.add_subplot(gs[2:4,1])

# gs.tight_layout(fig)

cmap = plt.get_cmap("tab10")

# Plot 1
ax1.plot(x_true_plot_1, y_true_plot_1, '-', linewidth=3.0, c=cmap(0), label='True')
# plot_1, = ax1.plot(x_predict_plot_1, y_predict_plot_1, '.', c='black', linewidth=10, label='GPTreeO')
plot_1, = ax1.plot(x_predict_plot_1, y_predict_plot_1, '-', c='black', linewidth=2.0, label='GPTreeO')
plot_1_std_up, = ax1.plot(x_predict_plot_1, y_predict_plot_1 + y_predict_std_plot_1, '-', c='0.5', linewidth=1.0)
plot_1_std_dn, = ax1.plot(x_predict_plot_1, y_predict_plot_1 - y_predict_std_plot_1, '-', c='0.5', linewidth=1.0)
ax1.set_xlabel("x_1")
ax1.set_ylabel("y")
ax1.legend()
ax1.text(0.98, 0.05, f"x_2 = {plot_1_x2_val}", fontsize=10, horizontalalignment='right', transform = ax1.transAxes)


# Plot 2
ax2.plot(x_true_plot_2, y_true_plot_2, '-', linewidth=3.0, c=cmap(1), label='True')
# plot_2, = ax2.plot(x_predict_plot_2, y_predict_plot_2, '.', c='black', linewidth=10, label='GPTreeO')
plot_2, = ax2.plot(x_predict_plot_2, y_predict_plot_2, '-', c='black', linewidth=2.0, label='GPTreeO')
plot_2_std_up, = ax2.plot(x_predict_plot_2, y_predict_plot_2 + y_predict_std_plot_2, '-', c='0.5', linewidth=1.0)
plot_2_std_dn, = ax2.plot(x_predict_plot_2, y_predict_plot_2 - y_predict_std_plot_2, '-', c='0.5', linewidth=1.0)
ax2.set_xlabel("x_1")
ax2.set_ylabel("y")
ax2.legend()
ax2.text(0.98, 0.05, f"x_2 = {plot_2_x2_val}", fontsize=10, horizontalalignment='right', transform = ax2.transAxes)



# Plot 3
ax3.plot(x_true_plot_3, y_true_plot_3, '-', linewidth=3.0, c=cmap(2), label='True')
# plot_3, = ax3.plot(x_predict_plot_3, y_predict_plot_3, '.', c='black', linewidth=10, label='GPTreeO')
plot_3, = ax3.plot(x_predict_plot_3, y_predict_plot_3, '-', c='black', linewidth=2.0, label='GPTreeO')
plot_3_std_up, = ax3.plot(x_predict_plot_3, y_predict_plot_3 + y_predict_std_plot_3, '-', c='0.5', linewidth=1.0)
plot_3_std_dn, = ax3.plot(x_predict_plot_3, y_predict_plot_3 - y_predict_std_plot_3, '-', c='0.5', linewidth=1.0)
ax3.set_xlabel("x_2")
ax3.set_ylabel("y")
ax3.legend()
ax3.text(0.98, 0.05, f"x_1 = {plot_3_x1_val}", fontsize=10, horizontalalignment='right', transform = ax3.transAxes)


# Plot 4
ax4.plot(x_true_plot_4, y_true_plot_4, '-', linewidth=3.0, c=cmap(3), label='True')
# plot_4, = ax4.plot(x_predict_plot_4, y_predict_plot_4, '.', c='black', linewidth=10, label='GPTreeO')
plot_4, = ax4.plot(x_predict_plot_4, y_predict_plot_4, '-', c='black', linewidth=2.0, label='GPTreeO')
plot_4_std_up, = ax4.plot(x_predict_plot_4, y_predict_plot_4 + y_predict_std_plot_4, '-', c='0.5', linewidth=1.0)
plot_4_std_dn, = ax4.plot(x_predict_plot_4, y_predict_plot_4 - y_predict_std_plot_4, '-', c='0.5', linewidth=1.0)
ax4.set_xlabel("x_1 = x_2")
ax4.set_ylabel("y")
ax4.legend()
ax4.text(0.98, 0.05, f"x_1 = x_2", fontsize=10, horizontalalignment='right', transform = ax4.transAxes)


# Plot 5
ax5.plot([X_true_plot_1[0,0], X_true_plot_1[-1,0]], [X_true_plot_1[0,1], X_true_plot_1[-1,1]], '-', c=cmap(0), linewidth=2.5)
ax5.plot([X_true_plot_2[0,0], X_true_plot_2[-1,0]], [X_true_plot_2[0,1], X_true_plot_2[-1,1]], '-', c=cmap(1), linewidth=2.5)
ax5.plot([X_true_plot_3[0,0], X_true_plot_3[-1,0]], [X_true_plot_3[0,1], X_true_plot_3[-1,1]], '-', c=cmap(2), linewidth=2.5)
ax5.plot([X_true_plot_4[0,0], X_true_plot_4[-1,0]], [X_true_plot_4[0,1], X_true_plot_4[-1,1]], '-', c=cmap(3), linewidth=2.5)
plot_5 = ax5.scatter([], [], s=0.2, c="black", zorder=5)
ax5.set_xlim([0,1])
ax5.set_ylim([0,1])
ax5.set_xlabel("x_1")
ax5.set_ylabel("x_2")
plot_5_text = ax5.text(0.0, 1.02, f"{target.__name__}, {n_dims}D, 0 training points", fontsize=12, transform = ax5.transAxes)


# Plot 6
ax6.contourf(X_2D, Y_2D, Z_2D, 10)
ax6.set_xlim([0,1])
ax6.set_ylim([0,1])
ax6.set_xlabel("x_1")
ax6.set_ylabel("x_2")
plot_6_text = ax6.text(0.0, 1.02, f"{target.__name__}, {n_dims}D", fontsize=12, transform = ax6.transAxes)

if live_update:
    plt.ion()
    plt.draw()
    plt.pause(0.01)


# Function for updating plots
def update_plots(point_i, n_last_points):
    """Updates the Matplotlib plots with new data and GPTree predictions.

    This function is called periodically during the training loop to refresh
    the visualizations. It queries the global `gpt` object for its current
    predictions on predefined slices of the input space and updates the
    corresponding line plots (mean prediction and uncertainty bounds).
    It also updates a scatter plot to show the locations of the most recent
    training points.

    Args:
        point_i (int): The total number of training points processed so far by
            the `GPTree`. This is used, for example, in plot titles.
        n_last_points (int): The number of most recent training data points
            from `X_input` to display on the scatter plot (ax5).

    Note:
        This function relies on several globally defined Matplotlib axes objects
        (ax1, ax2, etc.), plot line objects (plot_1, plot_1_std_up, etc.),
        and the global `gpt` (GPTree instance) and `X_input` (training data)
        variables.
    """

    # Plot 1
    y_predict_plot_1, y_predict_std_plot_1 = gpt.predict(X_predict_plot_1, show_progress=False)
    plot_1.set_ydata(y_predict_plot_1)
    plot_1_std_up.set_ydata(y_predict_plot_1 + y_predict_std_plot_1)
    plot_1_std_dn.set_ydata(y_predict_plot_1 - y_predict_std_plot_1)

    # Plot 2
    y_predict_plot_2, y_predict_std_plot_2 = gpt.predict(X_predict_plot_2, show_progress=False)
    plot_2.set_ydata(y_predict_plot_2)
    plot_2_std_up.set_ydata(y_predict_plot_2 + y_predict_std_plot_2)
    plot_2_std_dn.set_ydata(y_predict_plot_2 - y_predict_std_plot_2)

    # Plot 3
    y_predict_plot_3, y_predict_std_plot_3 = gpt.predict(X_predict_plot_3, show_progress=False)
    plot_3.set_ydata(y_predict_plot_3)
    plot_3_std_up.set_ydata(y_predict_plot_3 + y_predict_std_plot_3)
    plot_3_std_dn.set_ydata(y_predict_plot_3 - y_predict_std_plot_3)

    # Plot 4
    y_predict_plot_4, y_predict_std_plot_4 = gpt.predict(X_predict_plot_4, show_progress=False)
    plot_4.set_ydata(y_predict_plot_4)
    plot_4_std_up.set_ydata(y_predict_plot_4 + y_predict_std_plot_4)
    plot_4_std_dn.set_ydata(y_predict_plot_4 - y_predict_std_plot_4)

    # Plot 5
    scat = plot_5.get_offsets()
    # scat = np.append(scat, X_input[(point_i - Nbar*update_step):point_i], axis=0)
    scat = np.append(scat, X_input[(point_i - n_last_points):point_i], axis=0)
    plot_5.set_offsets(scat)
    plot_5_text.set_text(f"{target.__name__}, {n_dims}D, {point_i} training points")

    # Draw plots
    plt.draw()
    plt.pause(0.01)

    # if point_i % 1000 == 0:
    #     plt.savefig(f"plot__{point_i}_pts.png")



# Feed in training data one point at a time
point_i = 0
update_count = 0
for x,y in zip(X_input, y_input):

    point_i += 1
    # print(f"point_i: {point_i}   x: {x}   y: {y}")
    if point_i % 100 == 0:
        print(f"point_i: {point_i}")

    x = x.reshape((1, x.shape[0]))
    y = y.reshape((1,1))

    # Update gpt
    gpt.update_tree(x, y)

    # Update plot
    if point_i % Nbar == 0:
        update_count += 1

    # if (live_update) and (point_i % Nbar == 0):
    if (live_update) and (update_count == update_step):
        n_last_points = Nbar * update_step
        update_plots(point_i, n_last_points)
        update_count = 0

update_plots(point_i, n_pts-1)

print()
print("Done.")
print()
# print(gpt.root)
# print()

if live_update:
    plt.ioff()

plt.show()
# plt.savefig("plot.png")

