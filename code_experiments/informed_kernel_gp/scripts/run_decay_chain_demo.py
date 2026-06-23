#!/usr/bin/env python
"""Demo: physics-informed feature suggestion for a decay-chain log-likelihood.

Simulates the workflow where the user tells the agent what *type* of
computation underlies their log-likelihood — in this case, a radioactive
decay chain — and the agent uses domain knowledge (Bateman equations)
to suggest physically motivated basis functions.

The synthetic ground-truth is a Gaussian log-likelihood for fitting a
three-stage decay chain  A → B → C  to noisy count-rate data.  The two
free parameters are the decay rates λ₁ and λ₂, mapped to [0,1]².

This tests whether *physics-informed* features (exponentials, products,
rational ridge terms) outperform generic Fourier/RBF features.
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
# 1. The user description (mentions decay chain but not the exact form)
# =====================================================================

USER_DESCRIPTION = (
    "The target function is a log-likelihood from fitting a radioactive "
    "decay chain model with two or three decay stages to experimental "
    "count-rate data. The two free parameters are the decay rates of the "
    "first and second stages. The function is smooth and has a single "
    "global optimum, but the surface has an elongated ridge structure "
    "near the line where the two rates are equal. The input range is "
    "[0,1] in each coordinate."
)


# =====================================================================
# 2. Synthetic ground truth: Bateman decay-chain log-likelihood
# =====================================================================

# Physical setup: chain A -> B -> C
# We "observe" the activity of species B at a set of measurement times.
# The log-likelihood is a function of (λ₁, λ₂) ∈ [λ_min, λ_max]²,
# rescaled to [0,1]².

LAMBDA_MIN = 0.5   # minimum decay rate
LAMBDA_MAX = 5.0   # maximum decay rate
N0 = 100.0         # initial number of A atoms
T_OBS = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
SIGMA_OBS = 2.0    # measurement noise std

# True parameter values used to generate the "observed data"
LAMBDA1_TRUE = 2.0
LAMBDA2_TRUE = 3.5

# Pre-compute observed data (fixed for all evaluations)
_rng_data = np.random.RandomState(42)


def _bateman_NB(t, lam1, lam2):
    """Activity of species B in the chain A -> B -> C (Bateman equation).

    N_B(t) = N0 * lam1 / (lam2 - lam1) * [exp(-lam1*t) - exp(-lam2*t)]

    For numerical stability near lam1 ≈ lam2, use a Taylor expansion.
    """
    dt = lam2 - lam1
    stable = np.abs(dt) > 1e-8
    result = np.zeros_like(t, dtype=float)

    if np.any(stable):
        result[stable] = (
            N0 * lam1 / dt
            * (np.exp(-lam1 * t[stable]) - np.exp(-lam2 * t[stable]))
        )
    if np.any(~stable):
        # First-order Taylor: N_B ≈ N0 * lam1 * t * exp(-lam1 * t)
        result[~stable] = N0 * lam1 * t[~stable] * np.exp(-lam1 * t[~stable])

    return result


# Generate synthetic "observed" count rates
Y_OBS = _bateman_NB(T_OBS, LAMBDA1_TRUE, LAMBDA2_TRUE)
Y_OBS += _rng_data.normal(0, SIGMA_OBS, size=len(T_OBS))


def true_function(X):
    """Gaussian log-likelihood for the decay-chain model.

    Parameters
    ----------
    X : ndarray, shape (n, 2)
        Each row is (u, v) ∈ [0,1]², mapped to (λ₁, λ₂) ∈ [λ_min, λ_max]².

    Returns
    -------
    loglik : ndarray, shape (n,)
    """
    n = X.shape[0]
    loglik = np.empty(n)

    for i in range(n):
        lam1 = LAMBDA_MIN + X[i, 0] * (LAMBDA_MAX - LAMBDA_MIN)
        lam2 = LAMBDA_MIN + X[i, 1] * (LAMBDA_MAX - LAMBDA_MIN)
        y_pred = _bateman_NB(T_OBS, lam1, lam2)
        residuals = Y_OBS - y_pred
        loglik[i] = -0.5 * np.sum(residuals**2) / SIGMA_OBS**2

    return loglik


# =====================================================================
# 3. Configuration (same structure as the other demos)
# =====================================================================

DOMAIN_BOUNDS = [(0.0, 1.0), (0.0, 1.0)]
N_TRAIN_VALUES = [5, 10, 15, 20, 30, 50, 75, 100]
N_TRIALS = 20
N_TEST_PER_DIM = 50
NOISE_STD = 0.5  # GP observation noise (on the log-likelihood scale)
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
# 4. Plotting helpers
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
            ax.set_xlabel(r"$\lambda_1$ (rescaled)")
            ax.set_ylabel(r"$\lambda_2$ (rescaled)")
    suptitle = f"Predictive Surfaces (N_train = {n_train})"
    if subtitle:
        suptitle += f"\n{subtitle}"
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
        os.path.dirname(__file__), "..", "outputs", "decay_chain"
    )
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Parse the user description ---
    print("=" * 70)
    print("STEP 1: Parse user description")
    print("=" * 70)
    print(f"\nUser says:\n  \"{USER_DESCRIPTION}\"\n")

    traits = parse_vague_description(USER_DESCRIPTION)
    print(f"Extracted traits:")
    print(f"  n_dims            = {traits.n_dims}")
    print(f"  domain            = {traits.domain}")
    print(f"  smoothness        = {traits.smoothness}")
    print(f"  n_optima          = {traits.n_optima}")
    print(f"  optima_spread     = {traits.optima_spread}")
    print(f"  is_log_likelihood = {traits.is_log_likelihood}")
    print(f"  has_symmetry      = {traits.has_symmetry}")
    print(f"  function_class    = {traits.function_class}")
    print()

    # --- Step 2: Suggest features ---
    print("=" * 70)
    print("STEP 2: Suggest basis functions (physics-informed)")
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
    print("STEP 3: Run GP benchmark with physics-informed features")
    print("=" * 70)
    print(f"\nTrue function: Bateman decay-chain log-likelihood")
    print(f"  Chain: A → B → C")
    print(f"  True rates: λ₁={LAMBDA1_TRUE}, λ₂={LAMBDA2_TRUE}")
    print(f"  Observation times: {T_OBS}")
    print(f"  Measurement noise σ = {SIGMA_OBS}")
    print(f"  Parameters (λ₁, λ₂) ∈ [{LAMBDA_MIN}, {LAMBDA_MAX}]²")
    print(f"  Rescaled to [0,1]²\n")

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
        subtitle="Physics-informed features (decay chain)"
    )
    plot_predictive_surfaces(
        X_test, y_test, N_TEST_PER_DIM, output_dir,
        DOMAIN_BOUNDS, basis_functions,
        SURFACE_N_TRAIN, NOISE_STD, RANDOM_SEED,
        subtitle="Physics-informed features (decay chain)",
    )

    # --- Step 5: Summary ---
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Target: Bateman decay-chain log-likelihood on [0,1]²")
    print("Key question: Do physics-informed features (exponentials, products,")
    print("rational ridge terms) outperform generic Fourier/RBF features?")
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
