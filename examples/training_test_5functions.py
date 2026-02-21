"""Training test: 5 GPTree instances on 5 smooth functions in 15 dimensions.

This script trains 5 separate GPTree instances on 5 different smoothly varying
functions in 15 dimensions. Training is done in batches of several thousand
points, with a similarly sized batch of test points evaluated after each
training batch. This is repeated for 10 iterations and the results are
summarised in a plot.

Usage:
    python training_test_5functions.py
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from pygptreeo import GPTree, Default_GPR


# ============================================================
# Settings
# ============================================================

n_dims = 15
n_iterations = 40
batch_size_train = 5000
batch_size_test = 5000

x_min = 0.0
x_max = 1.0

Nbar = 1000
theta = 1e-4
retrain_step = 1000

np.random.seed(42)


# ============================================================
# Define 5 smoothly varying target functions in n_dims dimensions
# ============================================================

# Each function takes x of shape (n_dims, n_points) and returns
# an array of shape (n_points,). Values of x are in [0, 1].


def smooth_quadratic(x):
    """Weighted sum of shifted quadratics.

    A smooth bowl-shaped function with a single minimum.
    """
    dim = x.shape[0]
    centers = np.linspace(0.3, 0.7, dim)
    weights = np.linspace(1.0, 3.0, dim)
    result = 0.0
    for i in range(dim):
        result = result + weights[i] * (x[i] - centers[i])**2
    return result


def smooth_sinusoidal(x):
    """Sum of low-frequency sinusoids with varying phases.

    A smooth, gently oscillating function.
    """
    dim = x.shape[0]
    frequencies = np.linspace(0.5, 2.0, dim)
    phases = np.linspace(0, np.pi, dim)
    result = 0.0
    for i in range(dim):
        result = result + np.sin(2 * np.pi * frequencies[i] * x[i] + phases[i])
    return result


def smooth_gaussian_hills(x):
    """Sum of broad Gaussian bumps at different centers.

    A smooth landscape with three broad hills.
    """
    dim = x.shape[0]
    sigma = 1.0
    centers_list = [
        np.linspace(0.2, 0.4, dim),
        np.linspace(0.5, 0.7, dim),
        np.linspace(0.6, 0.9, dim),
    ]
    amplitudes = [2.0, 1.5, 1.0]
    result = 0.0
    for centers, amp in zip(centers_list, amplitudes):
        exponent = 0.0
        for i in range(dim):
            exponent = exponent + (x[i] - centers[i])**2
        result = result + amp * np.exp(-exponent / (2 * sigma**2))
    return result


def smooth_polynomial(x):
    """Mixed cubic polynomial with pairwise interactions.

    A smooth polynomial surface with gentle curvature.
    """
    dim = x.shape[0]
    a = np.linspace(0.5, 2.0, dim)
    b = np.linspace(-1.0, 1.0, dim)
    result = 0.0
    for i in range(dim):
        result = result + a[i] * (x[i] - 0.5)**3 + b[i] * (x[i] - 0.5)**2
    # Add pairwise interaction terms
    for i in range(0, dim - 1, 2):
        result = result + 0.5 * (x[i] - 0.5) * (x[i+1] - 0.5)
    return result


def smooth_exp_cosine(x):
    """Smooth exponential-cosine combination.

    A smooth function combining a Gaussian envelope with cosine modulation.
    """
    dim = x.shape[0]
    weights = np.linspace(0.5, 1.5, dim)
    cos_sum = 0.0
    quad_sum = 0.0
    for i in range(dim):
        cos_sum = cos_sum + weights[i] * np.cos(np.pi * (x[i] - 0.5))
        quad_sum = quad_sum + (x[i] - 0.5)**2
    result = np.exp(-0.5 * quad_sum) * (1.0 + cos_sum)
    return result


functions = [
    ("Quadratic", smooth_quadratic),
    ("Sinusoidal", smooth_sinusoidal),
    ("Gaussian hills", smooth_gaussian_hills),
    ("Polynomial", smooth_polynomial),
    ("Exp-cosine", smooth_exp_cosine),
]

n_functions = len(functions)


# ============================================================
# Create GPTree instances
# ============================================================

trees = []
for i in range(n_functions):
    gpt = GPTree(
        Nbar=Nbar,
        theta=theta,
        retrain_every_n_points=retrain_step,
        use_calibrated_sigma=True,
        split_dimension_criteria='max_spread',
        splitting_strategy='gradual',
        use_standard_scaling=True,
    )
    trees.append(gpt)


# ============================================================
# Training and testing loop
# ============================================================

# Metrics storage: shape (n_functions, n_iterations)
nrmse_results = np.zeros((n_functions, n_iterations))
mae_results = np.zeros((n_functions, n_iterations))
coverage_results = np.zeros((n_functions, n_iterations))
train_times = np.zeros((n_functions, n_iterations))
test_times = np.zeros((n_functions, n_iterations))

for iteration in range(n_iterations):
    print(f"\n{'='*70}")
    print(f"Iteration {iteration + 1}/{n_iterations}")
    print(f"{'='*70}")

    # Generate training batch
    X_train = np.random.uniform(x_min, x_max, (batch_size_train, n_dims))

    # Generate test batch
    X_test = np.random.uniform(x_min, x_max, (batch_size_test, n_dims))

    for f_idx, (func_name, func) in enumerate(functions):

        # Compute training targets
        y_train = func(X_train.T)  # shape (n_pts,)

        # Train: feed batch point by point using update_tree
        t_start = time.time()
        for j in range(batch_size_train):
            x_j = X_train[j:j+1, :]            # shape (1, n_dims)
            y_j = y_train[j].reshape(1, 1)      # shape (1, 1)
            sigma_j = np.maximum(0.001 * np.abs(y_j), 1e-6)
            trees[f_idx].update_tree(x_j, y_j, sigma_j)
        train_times[f_idx, iteration] = time.time() - t_start

        # Test: predict on test batch
        y_test_true = func(X_test.T)  # shape (n_pts,)

        t_start = time.time()
        y_pred, y_std = trees[f_idx].predict(X_test)
        test_times[f_idx, iteration] = time.time() - t_start

        y_pred_flat = y_pred[:, 0]
        y_std_flat = y_std[:, 0]

        # Compute metrics
        residuals = y_test_true - y_pred_flat
        y_range = np.max(y_test_true) - np.min(y_test_true)
        if y_range < 1e-10:
            y_range = 1.0
        nrmse = np.sqrt(np.mean(residuals**2)) / y_range
        mae = np.mean(np.abs(residuals))
        coverage = np.mean(np.abs(residuals) <= y_std_flat)

        nrmse_results[f_idx, iteration] = nrmse
        mae_results[f_idx, iteration] = mae
        coverage_results[f_idx, iteration] = coverage

        total_pts = (iteration + 1) * batch_size_train
        print(f"  {func_name:20s}  |  NRMSE: {nrmse:.4e}  |  MAE: {mae:.4e}"
              f"  |  Coverage: {coverage:.3f}  |  Train: {train_times[f_idx, iteration]:.1f}s"
              f"  |  Test: {test_times[f_idx, iteration]:.1f}s  |  Total pts: {total_pts}")


# ============================================================
# Plot results
# ============================================================

iterations = np.arange(1, n_iterations + 1)
cumulative_train_pts = iterations * batch_size_train

fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
fig.suptitle(
    f'GPTree training performance \u2014 5 smooth functions in {n_dims}D\n'
    f'(batch size: {batch_size_train} train / {batch_size_test} test, Nbar={Nbar})',
    fontsize=14,
)

colors = plt.cm.tab10(np.linspace(0, 0.5, n_functions))

# Panel 1: NRMSE
ax = axs[0]
for f_idx, (func_name, _) in enumerate(functions):
    ax.plot(cumulative_train_pts, nrmse_results[f_idx], 'o-',
            label=func_name, color=colors[f_idx], linewidth=2, markersize=5)
ax.set_ylabel('NRMSE')
ax.set_title('Normalised Root Mean Square Error')
ax.set_yscale('log')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Panel 2: MAE
ax = axs[1]
for f_idx, (func_name, _) in enumerate(functions):
    ax.plot(cumulative_train_pts, mae_results[f_idx], 's-',
            label=func_name, color=colors[f_idx], linewidth=2, markersize=5)
ax.set_ylabel('MAE')
ax.set_title('Mean Absolute Error')
ax.set_yscale('log')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Panel 3: Coverage
ax = axs[2]
for f_idx, (func_name, _) in enumerate(functions):
    ax.plot(cumulative_train_pts, coverage_results[f_idx], '^-',
            label=func_name, color=colors[f_idx], linewidth=2, markersize=5)
ax.axhline(y=0.68, color='black', linestyle='--', linewidth=1.5, label='1\u03c3 target (0.68)')
ax.set_ylabel('Coverage fraction')
ax.set_title('Empirical 1\u03c3 coverage')
ax.set_xlabel('Cumulative training points')
ax.set_ylim([0, 1.05])
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('gptree_training_test_results.png', dpi=150, bbox_inches='tight')

print(f"\nPlot saved to: gptree_training_test_results.png")
print("Done.")
