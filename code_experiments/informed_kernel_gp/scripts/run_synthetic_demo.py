#!/usr/bin/env python
"""Synthetic demonstration of informed-kernel GP regression.

Compares four GP kernel variants on a 2D synthetic function with
rich nonlinear structure:

    f(x, y) = cos(3x) · exp(-y²/2) + tanh(xy) / (1 + x²)
              + 0.5 · sin(πy) · log(1 + x²)

This function is challenging because it combines:
  - oscillatory × decaying interactions  (cos · exp)
  - saturating nonlinear coupling        (tanh(xy) / (1+x²))
  - mixed transcendental products        (sin · log)

Runs two scenarios:
    1. Well-specified basis  — four φ functions that span f exactly,
       plus a distractor feature (x² + y²) to test ARD
    2. Misspecified basis    — drops the tanh interaction term

Kernels compared:
    k_specific_ard, k_total_ard, k_product_ard, k_rbf
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent directory to path so we can import the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from informed_kernel_gp.benchmark import run_benchmark, build_kernels, run_single_trial
from informed_kernel_gp.kernels import ARDDotProductFeatureKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


# --- True function ---

def true_function(X):
    """f(x, y) = 100 + 10·[cos(3x)·exp(-y²/2) + tanh(xy)/(1+x²) + 0.5·sin(πy)·log(1+x²)]"""
    x = X[:, 0]
    y = X[:, 1]
    return 100.0 + 10.0 * (
        np.cos(3 * x) * np.exp(-y ** 2 / 2)
        + np.tanh(x * y) / (1 + x ** 2)
        + 0.5 * np.sin(np.pi * y) * np.log1p(x ** 2)
    )


# --- Basis functions ---

def phi_cos_exp(X):
    """cos(3x) · exp(-y²/2)"""
    return np.cos(3 * X[:, 0]) * np.exp(-X[:, 1] ** 2 / 2)


def phi_tanh_interaction(X):
    """tanh(xy) / (1 + x²)"""
    return np.tanh(X[:, 0] * X[:, 1]) / (1 + X[:, 0] ** 2)


def phi_sin_log(X):
    """sin(πy) · log(1 + x²)"""
    return np.sin(np.pi * X[:, 1]) * np.log1p(X[:, 0] ** 2)


def phi_distractor(X):
    """x² + y²  (irrelevant distractor to test ARD)"""
    return X[:, 0] ** 2 + X[:, 1] ** 2


# Well-specified: four features that span f exactly (with coefficients 1, 1, 0.5)
# plus a distractor that should be downweighted by ARD
BASIS_WELL_SPECIFIED = [phi_cos_exp, phi_tanh_interaction, phi_sin_log, phi_distractor]

# Misspecified: missing the tanh interaction term — basis cannot represent f exactly
BASIS_MISSPECIFIED = [phi_cos_exp, phi_sin_log, phi_distractor]

# --- Configuration ---

DOMAIN_BOUNDS = [(-2.0, 3.0), (-2.0, 2.0)]
N_TRAIN_VALUES = [5, 10, 15, 20, 30, 50, 75, 100]
N_TRIALS = 20
N_TEST_PER_DIM = 50
NOISE_STD = 0.02
RANDOM_SEED = 42
SURFACE_N_TRAIN = 15

KERNEL_NAMES = [
    "k_specific_ard", "k_total_ard",
    "k_product_ard", "k_rbf",
]
KERNEL_LABELS = {
    "k_specific_ard": r"$k_{\mathrm{specific}}^{\mathrm{ARD}}$",
    "k_total_ard": r"$k_{\mathrm{total}}^{\mathrm{ARD}}$",
    "k_product_ard": r"$k_{\mathrm{product}}^{\mathrm{ARD}}$",
    "k_rbf": r"$k_{\mathrm{RBF}}$",
}
KERNEL_COLORS = {
    "k_specific_ard": "C0",
    "k_total_ard": "C1",
    "k_product_ard": "C2",
    "k_rbf": "C3",
}


def plot_learning_curves(results, n_train_values, output_dir, subtitle=""):
    """Plot RMSE and MLPD vs N_train."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for metric_idx, (metric, ylabel) in enumerate([("rmse", "RMSE"), ("mlpd", "MLPD")]):
        ax = axes[metric_idx]
        for name in KERNEL_NAMES:
            means = [np.mean(results[n][name][metric]) for n in n_train_values]
            stds = [np.std(results[n][name][metric]) for n in n_train_values]
            means = np.array(means)
            stds = np.array(stds)
            ax.plot(n_train_values, means, "o-", label=KERNEL_LABELS[name],
                    color=KERNEL_COLORS[name])
            ax.fill_between(n_train_values, means - stds, means + stds,
                            alpha=0.2, color=KERNEL_COLORS[name])
        ax.set_xlabel(r"$N_{\mathrm{train}}$")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if metric == "rmse":
            ax.set_title("Learning Curve: RMSE")
        else:
            ax.set_title("Learning Curve: MLPD")

    if subtitle:
        fig.suptitle(subtitle, fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "learning_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_predictive_surfaces(X_test, y_test, n_test_per_dim, output_dir,
                             domain_bounds, basis_functions,
                             n_train, noise_std, random_seed, subtitle=""):
    """Plot predicted mean and std for all kernels alongside ground truth."""
    rng = np.random.RandomState(random_seed)

    # Generate training data
    X_train = np.column_stack([
        rng.uniform(lo, hi, size=n_train)
        for lo, hi in domain_bounds
    ])
    y_train = true_function(X_train)
    if noise_std > 0:
        y_train = y_train + rng.normal(0, noise_std, size=n_train)

    kernels = build_kernels(basis_functions)

    # Fit all GPs
    gps = {}
    _pure_specific = {"k_specific_ard"}
    for name, kernel in kernels.items():
        use_normalize = name not in _pure_specific
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=5,
            random_state=0, normalize_y=use_normalize,
        )
        gp.fit(X_train, y_train)
        gps[name] = gp

    # Reshape test grid for plotting
    x_grid = np.linspace(*domain_bounds[0], n_test_per_dim)
    y_grid = np.linspace(*domain_bounds[1], n_test_per_dim)
    Xg, Yg = np.meshgrid(x_grid, y_grid, indexing="ij")
    y_true_grid = y_test.reshape(n_test_per_dim, n_test_per_dim)

    # Pre-compute all predictions
    predictions = {}
    for name in KERNEL_NAMES:
        y_pred, y_std = gps[name].predict(X_test, return_std=True)
        predictions[name] = {
            "mean": y_pred.reshape(n_test_per_dim, n_test_per_dim),
            "std": y_std.reshape(n_test_per_dim, n_test_per_dim),
        }

    # Shared color scales
    vmin, vmax = y_true_grid.min(), y_true_grid.max()
    std_vmax = predictions["k_rbf"]["std"].max()
    rbf_rel_err = np.abs(predictions["k_rbf"]["mean"] - y_true_grid) / (np.abs(y_true_grid) + 1e-10)
    err_vmax = rbf_rel_err.max()

    n_kernels = len(KERNEL_NAMES)
    fig, axes = plt.subplots(3, n_kernels + 1, figsize=(5 * (n_kernels + 1), 15))

    # Ground truth
    im = axes[0, 0].pcolormesh(Xg, Yg, y_true_grid, cmap="viridis",
                                vmin=vmin, vmax=vmax, shading="auto")
    axes[0, 0].set_title("Ground Truth")
    axes[0, 0].scatter(X_train[:, 0], X_train[:, 1], c="red", s=15, zorder=5,
                        label="Training pts")
    axes[0, 0].legend(fontsize=8)
    plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
    axes[1, 0].set_visible(False)
    axes[2, 0].set_visible(False)

    for col_idx, name in enumerate(KERNEL_NAMES, start=1):
        y_pred_grid = predictions[name]["mean"]
        y_std_grid = predictions[name]["std"]

        im_mean = axes[0, col_idx].pcolormesh(Xg, Yg, y_pred_grid, cmap="viridis",
                                     vmin=vmin, vmax=vmax, shading="auto")
        axes[0, col_idx].set_title(f"{KERNEL_LABELS[name]} Mean")
        axes[0, col_idx].scatter(X_train[:, 0], X_train[:, 1], c="red", s=15, zorder=5)
        plt.colorbar(im_mean, ax=axes[0, col_idx], shrink=0.8)

        im_std = axes[1, col_idx].pcolormesh(Xg, Yg, y_std_grid, cmap="magma",
                                              vmin=0, vmax=std_vmax, shading="auto")
        axes[1, col_idx].set_title(f"{KERNEL_LABELS[name]} Std")
        axes[1, col_idx].scatter(X_train[:, 0], X_train[:, 1], c="white", s=15, zorder=5)
        plt.colorbar(im_std, ax=axes[1, col_idx], shrink=0.8)

        rel_err_grid = np.abs(y_pred_grid - y_true_grid) / (np.abs(y_true_grid) + 1e-10)
        im_err = axes[2, col_idx].pcolormesh(Xg, Yg, rel_err_grid, cmap="RdYlGn_r",
                                              vmin=0, vmax=err_vmax, shading="auto")
        axes[2, col_idx].set_title(f"{KERNEL_LABELS[name]} |Rel. Error|")
        axes[2, col_idx].scatter(X_train[:, 0], X_train[:, 1], c="black", s=15, zorder=5)
        plt.colorbar(im_err, ax=axes[2, col_idx], shrink=0.8)

    for ax in axes.flat:
        if ax.get_visible():
            ax.set_aspect("equal")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
    suptitle = f"Predictive Surfaces (N_train = {n_train})"
    if subtitle:
        suptitle += f" — {subtitle}"
    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "predictive_surfaces.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def print_table(results, n_train_values):
    """Print RMSE summary table."""
    header = f"{'N_train':>8}"
    for name in KERNEL_NAMES:
        header += f"  {name + ' RMSE':>20}"
    print(header)
    print("-" * (8 + 22 * len(KERNEL_NAMES)))
    for n in n_train_values:
        row = f"{n:>8}"
        for name in KERNEL_NAMES:
            mean = np.mean(results[n][name]["rmse"])
            std = np.std(results[n][name]["rmse"])
            row += f"  {mean:8.4f} ± {std:6.4f}"
        print(row)
    print()


def run_scenario(basis_functions, label, output_dir):
    """Run one benchmark scenario end-to-end."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== {label} ===")
    print(f"  Basis: {[f.__doc__ for f in basis_functions]}")
    print(f"  N_train values: {N_TRAIN_VALUES}")
    print(f"  N_trials: {N_TRIALS}")
    print(f"  Noise std: {NOISE_STD}")
    print()

    results, X_test, y_test = run_benchmark(
        true_function=true_function,
        basis_functions=basis_functions,
        domain_bounds=DOMAIN_BOUNDS,
        n_train_values=N_TRAIN_VALUES,
        n_trials=N_TRIALS,
        n_test_per_dim=N_TEST_PER_DIM,
        noise_std=NOISE_STD,
        random_seed=RANDOM_SEED,
    )

    print_table(results, N_TRAIN_VALUES)

    print("Generating plots...")
    plot_learning_curves(results, N_TRAIN_VALUES, output_dir, subtitle=label)
    plot_predictive_surfaces(
        X_test, y_test, N_TEST_PER_DIM, output_dir,
        DOMAIN_BOUNDS, basis_functions,
        SURFACE_N_TRAIN, NOISE_STD, RANDOM_SEED,
        subtitle=label,
    )
    print()


def main():
    base_output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")

    # Scenario 1: well-specified basis (4 features + distractor)
    run_scenario(
        BASIS_WELL_SPECIFIED,
        "Well-specified basis (4 features incl. distractor)",
        os.path.join(base_output_dir, "well_specified"),
    )

    # Scenario 2: misspecified basis (missing tanh interaction)
    run_scenario(
        BASIS_MISSPECIFIED,
        "Misspecified basis (3 features, no tanh interaction)",
        os.path.join(base_output_dir, "misspecified"),
    )

    print("Done.")


if __name__ == "__main__":
    main()
