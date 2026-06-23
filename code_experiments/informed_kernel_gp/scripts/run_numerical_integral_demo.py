#!/usr/bin/env python
"""Demo: emulating a numerical function with no closed-form expression.

This demo tests whether an agent that can *inspect the code* of a
numerical function can suggest useful basis functions for the GP kernel,
even when no closed-form expression exists.

The true function is a parametric integral computed with scipy:

    f(a, b) = ∫₀^∞  exp(-a·t) · sin(t) · log(1 + b·t) / (1 + t²)  dt

where (a, b) are mapped from (u, v) ∈ [0,1]².  This integral has no
elementary closed form for general (a, b).

An agent inspecting this code can read the integrand and reason about
the parametric dependence:

  • For large a, exp(-a·t) cuts off at t ~ 1/a, making the integral
    dominated by the small-t expansion: f ≈ b · ∫ t² exp(-at) dt ∝ b/a³.
  • For small a, the 1/(1+t²) factor and sin(t) oscillation determine
    the structure, giving Laplace-transform–like rational dependence.
  • The b parameter enters through log(1+b·t). For large b, this ≈
    log(b) + log(t), giving logarithmic scaling.
  • The interaction between a and b arises because exp(-a·t) sets the
    effective integration window [0, ~1/a], over which log(1+b·t)
    is sampled.

From this analysis, an agent would suggest basis functions like:
  1/a^n  (power-law decay),  log(1+b)  (logarithmic scaling),
  b/a^n  (interaction),  1/(1+a²)  (rational/Laplace structure).
"""

import sys
import os
import numpy as np
from scipy import integrate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from informed_kernel_gp.kernels import ARDDotProductFeatureKernel
from informed_kernel_gp.benchmark import run_benchmark, build_kernels
from sklearn.gaussian_process import GaussianProcessRegressor


# =====================================================================
# 1. True function: parametric integral (no closed form)
# =====================================================================

A_MIN, A_MAX = 0.3, 5.0    # damping rate
B_MIN, B_MAX = 0.1, 5.0    # log-growth rate
F_OFFSET = 1.0             # shift so all values ≥ 1 (avoids relative-error artifacts)


def _integrand(t, a, b):
    """Integrand: exp(-a·t) · sin(t) · log(1 + b·t) / (1 + t²)."""
    return np.exp(-a * t) * np.sin(t) * np.log(1.0 + b * t) / (1.0 + t ** 2)


def _f_scalar(a, b):
    """Evaluate the parametric integral for a single (a, b) pair."""
    result, _ = integrate.quad(_integrand, 0, np.inf, args=(a, b),
                               limit=100, epsabs=1e-10, epsrel=1e-10)
    return result


def true_function(X):
    """Parametric integral on [0,1]² → (a, b) parameter space.

    Parameters
    ----------
    X : ndarray, shape (n, 2)
        Each row is (u, v) ∈ [0,1]², mapped to (a, b).

    Returns
    -------
    y : ndarray, shape (n,)
    """
    n = X.shape[0]
    y = np.empty(n)
    for i in range(n):
        a = A_MIN + X[i, 0] * (A_MAX - A_MIN)
        b = B_MIN + X[i, 1] * (B_MAX - B_MIN)
        y[i] = _f_scalar(a, b) + F_OFFSET
    return y


# =====================================================================
# 2. Agent-suggested basis functions
# =====================================================================
# These features are what an agent would suggest after reading the
# integrand code and reasoning about the parametric dependence.

def _get_agent_suggested_features():
    """Construct basis functions as an agent would from code inspection.

    The agent reads the integrand:
        exp(-a·t) · sin(t) · log(1 + b·t) / (1 + t²)
    and reasons about how f(a, b) depends on the parameters.
    """
    features = []
    labels = []

    # --- Power-law features in a ---
    # Reasoning: exp(-a·t) concentrates the integral near t ~ 1/a.
    # For large a, f ~ ∫₀^{1/a} t · bt dt ~ b/a³.
    # Including several power laws to span the transition from
    # small-a (rational) to large-a (power-law) regimes.
    features.append(lambda X: (A_MIN + X[:, 0] * (A_MAX - A_MIN) + 0.1) ** (-1))
    labels.append("a^{-1}")
    features.append(lambda X: (A_MIN + X[:, 0] * (A_MAX - A_MIN) + 0.1) ** (-2))
    labels.append("a^{-2}")
    features.append(lambda X: (A_MIN + X[:, 0] * (A_MAX - A_MIN) + 0.1) ** (-3))
    labels.append("a^{-3}")

    # --- Rational / Laplace-transform features in a ---
    # Reasoning: ∫₀^∞ exp(-at) sin(t) dt = 1/(1+a²) exactly (when other
    # factors are absent).  This rational structure persists approximately.
    features.append(lambda X: 1.0 / (1.0 + (A_MIN + X[:, 0] * (A_MAX - A_MIN)) ** 2))
    labels.append("1/(1+a²)")

    # --- Logarithmic features in b ---
    # Reasoning: log(1+b·t) ≈ log(b) + log(t) for large b, giving
    # f ~ log(b) × ∫ exp(-at) sin(t) log(t) / (1+t²) dt.
    features.append(lambda X: np.log(1.0 + B_MIN + X[:, 1] * (B_MAX - B_MIN)))
    labels.append("log(1+b)")

    # --- Linear feature in b ---
    # Reasoning: for small b, log(1+bt) ≈ bt, so f ≈ b × const(a).
    features.append(lambda X: B_MIN + X[:, 1] * (B_MAX - B_MIN))
    labels.append("b (linear)")

    # --- Interaction features ---
    # Reasoning: the effective integral window is [0, ~1/a], over which
    # log(1+bt) is sampled.  The dominant interaction is b/a^n.
    features.append(lambda X: (
        (B_MIN + X[:, 1] * (B_MAX - B_MIN))
        / (A_MIN + X[:, 0] * (A_MAX - A_MIN) + 0.1) ** 2
    ))
    labels.append("b/a²")
    features.append(lambda X: (
        (B_MIN + X[:, 1] * (B_MAX - B_MIN))
        / (A_MIN + X[:, 0] * (A_MAX - A_MIN) + 0.1) ** 3
    ))
    labels.append("b/a³")

    # --- Compound: log(1+b) / (1+a²) ---
    # Reasoning: combines the logarithmic b-dependence with the
    # rational a-dependence from the Laplace transform structure.
    features.append(lambda X: (
        np.log(1.0 + B_MIN + X[:, 1] * (B_MAX - B_MIN))
        / (1.0 + (A_MIN + X[:, 0] * (A_MAX - A_MIN)) ** 2)
    ))
    labels.append("log(1+b)/(1+a²)")

    # --- Exponential feature in a ---
    # Reasoning: the integrand's dominant factor is exp(-at); this
    # structure can propagate into the integral's a-dependence.
    features.append(lambda X: np.exp(-(A_MIN + X[:, 0] * (A_MAX - A_MIN))))
    labels.append("exp(-a)")

    return features, labels


# =====================================================================
# 3. Configuration
# =====================================================================

DOMAIN_BOUNDS = [(0.0, 1.0), (0.0, 1.0)]
N_TRAIN_VALUES = [10, 20, 50]
N_TRIALS = 5
N_TEST_PER_DIM = 30
NOISE_STD = 0.01    # low noise: emulating a deterministic function
RANDOM_SEED = 321
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
            ax.set_xlabel(r"$a$ (damping rate, rescaled)")
            ax.set_ylabel(r"$b$ (log-growth rate, rescaled)")
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
            row += f"  {mean:8.6f} ± {std:8.6f}"
        print(row)
    print()


# =====================================================================
# 5. Main
# =====================================================================

def main():
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "outputs", "numerical_integral"
    )
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Display the true function ---
    print("=" * 70)
    print("STEP 1: True function (numerical integral, no closed form)")
    print("=" * 70)
    print()
    print("  f(a, b) = ∫₀^∞ exp(-a·t) · sin(t) · log(1+b·t) / (1+t²) dt")
    print()
    print(f"  a ∈ [{A_MIN}, {A_MAX}]  (damping rate)")
    print(f"  b ∈ [{B_MIN}, {B_MAX}]  (log-growth rate)")
    print(f"  Mapped from (u, v) ∈ [0,1]²")
    print()

    # Spot-check a few values
    print("  Spot-check values:")
    for u, v in [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)]:
        val = true_function(np.array([[u, v]]))[0]
        a = A_MIN + u * (A_MAX - A_MIN)
        b = B_MIN + v * (B_MAX - B_MIN)
        print(f"    f(a={a:.1f}, b={b:.1f}) = {val:.6f}")
    print()

    # --- Step 2: Agent-suggested features ---
    print("=" * 70)
    print("STEP 2: Agent-suggested basis functions (from code inspection)")
    print("=" * 70)
    print()
    print("  An agent inspecting the integrand code would reason:")
    print("    • exp(-a·t) concentrates mass near t~1/a → power-law features in a")
    print("    • ∫ exp(-at) sin(t) dt = 1/(1+a²) → rational features in a")
    print("    • log(1+b·t) ≈ b·t for small b → linear feature in b")
    print("    • log(1+b·t) ≈ log(b) for large b → logarithmic feature in b")
    print("    • exp(-at) sets window over which log(1+bt) is sampled → b/a^n interaction")
    print()

    basis_functions, feature_labels = _get_agent_suggested_features()
    print(f"  Suggested {len(basis_functions)} features:")
    for i, label in enumerate(feature_labels, 1):
        print(f"    {i:>2}. {label}")
    print()

    # --- Step 3: Run benchmark ---
    print("=" * 70)
    print("STEP 3: Benchmark — emulating the numerical integral")
    print("=" * 70)
    print(f"\n  GP training noise σ = {NOISE_STD}")
    print(f"  (Low noise: emulating a deterministic numerical function)\n")

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
        subtitle="Numerical integral — agent-suggested features from code inspection",
    )
    plot_predictive_surfaces(
        X_test, y_test, N_TEST_PER_DIM, output_dir,
        DOMAIN_BOUNDS, basis_functions,
        SURFACE_N_TRAIN, NOISE_STD, RANDOM_SEED,
        subtitle="Numerical integral — agent-suggested features from code inspection",
    )

    # --- Step 5: Summary ---
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Target: f(a,b) = ∫₀^∞ exp(-at)·sin(t)·log(1+bt)/(1+t²) dt")
    print("(Numerical integral with no closed-form solution)")
    print()
    for n in N_TRAIN_VALUES:
        if n in results:
            rmse_rbf = np.mean(results[n]["k_rbf"]["rmse"])
            rmse_total = np.mean(results[n]["k_total_ard"]["rmse"])
            rmse_product = np.mean(results[n]["k_product_ard"]["rmse"])
            impr_total = (rmse_rbf - rmse_total) / rmse_rbf * 100
            impr_product = (rmse_rbf - rmse_product) / rmse_rbf * 100
            print(f"  N_train={n:>3}:  RBF RMSE={rmse_rbf:.6f}  "
                  f"total_ard={rmse_total:.6f} ({impr_total:+.1f}%)  "
                  f"product_ard={rmse_product:.6f} ({impr_product:+.1f}%)")
    print()
    print(f"Output saved to: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
