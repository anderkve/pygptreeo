#!/usr/bin/env python
"""Demo: feature suggestion from a vague user description.

Simulates the agentic workflow where the user provides only a qualitative
description of their target function.  The system:

  1. Parses the description into structured FunctionTraits
  2. Suggests a set of basis functions with rationale
  3. Builds GP kernels using the suggested features
  4. Evaluates on a synthetic target that *matches* the description
     (but whose exact form is unknown to the suggestion engine)

The synthetic target is a smooth log-likelihood-like function on [0,1]²
with 3 well-spread local optima — exactly the kind of function the user
described.

This tests whether the *generic* suggested features (derived only from
the vague description) are useful even when the system has no access to
the true functional form.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from informed_kernel_gp.suggest_features import (
    parse_vague_description,
    suggest_features,
)
from informed_kernel_gp.benchmark import run_benchmark, build_kernels, run_single_trial
from informed_kernel_gp.kernels import ARDDotProductFeatureKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


# =====================================================================
# 1. The vague user description
# =====================================================================

USER_DESCRIPTION = (
    "The target function is a smoothly varying log-likelihood function "
    "of two input parameters, with three or four local optima. The optima "
    "are fairly well-spread out in input space. The input range is [0,1] "
    "in each coordinate."
)


# =====================================================================
# 2. A synthetic target that matches the description
#    (this is the "ground truth" the suggestion engine does NOT see)
# =====================================================================

def true_function(X):
    """Smooth log-likelihood-like function on [0,1]² with 3 local optima.

    Constructed as a mixture of three Gaussian bumps plus gentle curvature,
    passed through a concave transform to look like a log-likelihood.

    Optima located approximately at (0.2, 0.7), (0.7, 0.3), (0.5, 0.8).
    """
    x = X[:, 0]
    y = X[:, 1]

    # Three Gaussian bumps at well-spread locations
    bump1 = 2.0 * np.exp(-((x - 0.2)**2 + (y - 0.7)**2) / (2 * 0.08**2))
    bump2 = 1.5 * np.exp(-((x - 0.7)**2 + (y - 0.3)**2) / (2 * 0.10**2))
    bump3 = 1.8 * np.exp(-((x - 0.5)**2 + (y - 0.8)**2) / (2 * 0.09**2))

    # Gentle overall curvature (concave envelope)
    envelope = 0.5 * x * (1 - x) * y * (1 - y)

    # Combine: log-likelihood = log(mixture + small floor)
    # The log makes it concave and creates the right "feel"
    mixture = bump1 + bump2 + bump3 + 4.0 * envelope + 0.01
    return np.log(mixture)


# =====================================================================
# 3. Configuration
# =====================================================================

DOMAIN_BOUNDS = [(0.0, 1.0), (0.0, 1.0)]
N_TRAIN_VALUES = [5, 10, 15, 20, 30, 50, 75, 100]
N_TRIALS = 20
N_TEST_PER_DIM = 50
NOISE_STD = 0.05
RANDOM_SEED = 123
SURFACE_N_TRAIN = 20

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


# =====================================================================
# 4. Plotting helpers (similar to run_synthetic_demo.py)
# =====================================================================

def plot_learning_curves(results, n_train_values, output_dir, subtitle=""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for metric_idx, (metric, ylabel) in enumerate([("rmse", "RMSE"), ("mlpd", "MLPD")]):
        ax = axes[metric_idx]
        for name in KERNEL_NAMES:
            means = [np.mean(results[n][name][metric]) for n in n_train_values]
            stds = [np.std(results[n][name][metric]) for n in n_train_values]
            means, stds = np.array(means), np.array(stds)
            ax.plot(n_train_values, means, "o-", label=KERNEL_LABELS[name],
                    color=KERNEL_COLORS[name])
            ax.fill_between(n_train_values, means - stds, means + stds,
                            alpha=0.2, color=KERNEL_COLORS[name])
        ax.set_xlabel(r"$N_{\mathrm{train}}$")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Learning Curve: {ylabel}")
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
    rng = np.random.RandomState(random_seed)
    X_train = np.column_stack([
        rng.uniform(lo, hi, size=n_train) for lo, hi in domain_bounds
    ])
    y_train = true_function(X_train)
    if noise_std > 0:
        y_train += rng.normal(0, noise_std, size=n_train)

    kernels = build_kernels(basis_functions)
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


# =====================================================================
# 5. Main
# =====================================================================

def main():
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "outputs", "vague_description"
    )
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Parse the vague description ---
    print("=" * 70)
    print("STEP 1: Parse vague user description")
    print("=" * 70)
    print(f"\nUser says:\n  \"{USER_DESCRIPTION}\"\n")

    traits = parse_vague_description(USER_DESCRIPTION)
    print(f"Extracted traits:")
    print(f"  n_dims          = {traits.n_dims}")
    print(f"  domain          = {traits.domain}")
    print(f"  smoothness      = {traits.smoothness}")
    print(f"  n_optima        = {traits.n_optima}")
    print(f"  optima_spread   = {traits.optima_spread}")
    print(f"  is_log_likelihood = {traits.is_log_likelihood}")
    print(f"  has_symmetry    = {traits.has_symmetry}")
    print()

    # --- Step 2: Suggest features ---
    print("=" * 70)
    print("STEP 2: Suggest basis functions")
    print("=" * 70)

    suggestion = suggest_features(traits)
    print()
    print(suggestion.summary())
    print()

    basis_functions = suggestion.basis_functions()
    print(f"Total features: {len(basis_functions)}")
    print()

    # --- Step 3: Run benchmark ---
    print("=" * 70)
    print("STEP 3: Run GP benchmark with suggested features")
    print("=" * 70)
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

    # --- Step 4: Plots ---
    print("=" * 70)
    print("STEP 4: Generate plots")
    print("=" * 70)

    plot_learning_curves(
        results, N_TRAIN_VALUES, output_dir,
        subtitle="Vague description → suggested features"
    )
    plot_predictive_surfaces(
        X_test, y_test, N_TEST_PER_DIM, output_dir,
        DOMAIN_BOUNDS, basis_functions,
        SURFACE_N_TRAIN, NOISE_STD, RANDOM_SEED,
        subtitle="Vague description → suggested features",
    )

    # --- Step 5: Summary ---
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Key question: Do the *generic* features suggested from a vague")
    print("description help the GP, compared to a pure RBF baseline?")
    print()
    for n in [10, 20, 50]:
        if n in results:
            rmse_rbf = np.mean(results[n]["k_rbf"]["rmse"])
            rmse_total = np.mean(results[n]["k_total_ard"]["rmse"])
            rmse_specific = np.mean(results[n]["k_specific_ard"]["rmse"])
            improvement = (rmse_rbf - rmse_total) / rmse_rbf * 100
            print(f"  N_train={n:>3}:  RBF RMSE={rmse_rbf:.4f}  "
                  f"total_ard RMSE={rmse_total:.4f}  "
                  f"({improvement:+.1f}% vs RBF)")
    print()
    print(f"Output saved to: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
