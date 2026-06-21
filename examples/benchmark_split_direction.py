"""Benchmark: comparing split-dimension criteria for GPTree.

This script compares every available `split_dimension_criteria` on the same
streaming data set, following the pygptreeo "house style" of tracking batch
metrics as a function of the number of processed points:

    max_spread       : split the dimension with the largest data range (default)
    max_variance     : split the dimension with the largest data variance
    max_uncertainty  : split where the GP is most uncertain (grid-based, costly)
    min_lengthscale  : split the dimension with the smallest fitted ARD length
                       scale, i.e. where the GP says the function varies fastest
    oblique          : split perpendicular to the estimated dominant direction of
                       variation (a non-axis-aligned cut)
    random           : split a random dimension

`min_lengthscale` (idea #1) reuses the GP's already-optimized ARD length scales,
so it is much cheaper than `max_uncertainty` while targeting the same goal:
split where the local function has the most structure.

For each criterion and each batch of `BATCH_SIZE` points we record:
    - batch NRMSE
    - fraction of predictions within 4% of the true value
    - empirical coverage of the 1-sigma uncertainty band
    - number of leaves in the tree
    - average prediction time per point
    - average tree-update time per point

The script writes a comparison figure (one panel per metric, one line per
criterion) and a CSV of the raw batch metrics.

Usage
-----
    OMP_NUM_THREADS=1 python examples/benchmark_split_direction.py [target] [n_points]

    target   : 'aniso_chirp' (default), 'diagonal', or 'eggholder'
    n_points : total number of streamed points (default 20000)

The 'diagonal' target is a plane wave along the (1,1,...) diagonal, flat in the
perpendicular directions -- the case the oblique criterion is designed for, and
which every axis-aligned criterion must resolve by staircasing many cuts.

The 'aniso_chirp' target is anisotropic and heterogeneous: rough and chirped
along x0 (frequency grows with x0), but smooth/linear along x1 -- and x1 has the
LARGEST input spread, which deliberately misleads the spread-based criteria into
splitting the wrong (smooth) dimension. This is where the choice of split
dimension matters most. The 'eggholder' target is roughly isotropic and
uniformly rough, so all criteria pick comparable dimensions (a null reference).
"""

import os
import sys
import time
import contextlib
import warnings

import numpy as np
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

# Allow running from the repo root or from examples/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from pygptreeo import GPTree
from pygptreeo.adapters import SklearnGPAdapter
from target_functions import Eggholder


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
TARGET = sys.argv[1] if len(sys.argv) > 1 else "aniso_chirp"
N_POINTS = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
BATCH_SIZE = 2000

N_DIMS = 3
NBAR = 200
THETA = 1e-4
RETRAIN_STEP = 50          # < Nbar so the GP (and its length scales) is fresh at a split

WITHIN_FRACTION = 0.04     # "within 4%" accuracy threshold
SEED = 512312

CRITERIA = ["max_spread", "max_variance", "max_uncertainty", "min_lengthscale", "oblique", "random"]


# ----------------------------------------------------------------------------
# Target functions
# ----------------------------------------------------------------------------
def make_data(target, n_points, n_dims, seed):
    """Return (X, y) for the requested target on a fixed random stream."""
    rng = np.random.RandomState(seed)

    if target == "eggholder":
        X = rng.uniform(0.0, 1.0, (n_points, n_dims))
        y = Eggholder(X.T)
        return X, y

    if target == "aniso_chirp":
        # Anisotropic + heterogeneous target.
        #   x0 in [0, 1] : rough, chirped (instantaneous frequency grows with x0)
        #   x1 in [0, 3] : smooth/linear, but the LARGEST input spread
        #   x2 in [0, 1] : mild low-frequency variation
        lo = np.array([0.0, 0.0, 0.0])
        hi = np.array([1.0, 3.0, 1.0])
        X = rng.uniform(lo[:n_dims], hi[:n_dims], (n_points, n_dims))
        x0 = X[:, 0]
        y = np.sin(2 * np.pi * (1.0 + 8.0 * x0) * x0)
        if n_dims > 1:
            y = y + 0.3 * X[:, 1]
        if n_dims > 2:
            y = y + 0.2 * np.sin(2 * np.pi * 0.3 * X[:, 2])
        return X, y

    if target == "diagonal":
        # Diagonal plane wave: the function varies only along the (1,1,...)
        # diagonal and is flat in the perpendicular directions. Axis-aligned
        # splits must "staircase" many cuts to resolve the diagonal wavefronts;
        # an oblique split aligned with the diagonal resolves it directly.
        X = rng.uniform(0.0, 1.0, (n_points, n_dims))
        s = X.sum(axis=1) / np.sqrt(n_dims)
        y = np.sin(2 * np.pi * 2.5 * s)
        return X, y

    raise ValueError(f"Unknown target '{target}'")


def make_gpr(n_dims):
    """Anisotropic Matern GPR (ARD), required for length-scale-based splitting."""
    kernel = ConstantKernel(1.0, (1e-3, 1e8)) * Matern(
        nu=1.5,
        length_scale=[1.0] * n_dims,
        length_scale_bounds=[(1e-5, 1e5)] * n_dims,
    )
    return SklearnGPAdapter(
        GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=1)
    )


# ----------------------------------------------------------------------------
# Single-criterion streaming run
# ----------------------------------------------------------------------------
def run_criterion(criterion, X, y, n_dims):
    """Stream all points through one GPTree config, returning batch-metric history."""
    y_range = float(np.max(y) - np.min(y))

    gpt = GPTree(
        GPR=make_gpr(n_dims),
        Nbar=NBAR,
        theta=THETA,
        retrain_every_n_points=RETRAIN_STEP,
        use_standard_scaling=True,
        use_calibrated_sigma=True,
        splitting_strategy="gradual",
        max_n_pred_leaves=3,
        aggregation="moe",
        split_dimension_criteria=criterion,
    )

    hist = {k: [] for k in [
        "points", "predict_time", "update_time", "nrmse",
        "within", "coverage", "n_leaves",
    ]}

    bt_pred, bt_upd, bt_true, bt_pred_val, bt_std = [], [], [], [], []

    # Suppress the library's per-node prints during the long run.
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        for i, (xi, yi) in enumerate(zip(X, y), start=1):
            xr = xi.reshape(1, -1)
            yr = np.array([[yi]])

            t0 = time.time()
            y_pred, y_std = gpt.predict(xr)
            bt_pred.append(time.time() - t0)

            t0 = time.time()
            gpt.update_tree(xr, yr, 0.001 * np.abs(yr) + 1e-9)
            bt_upd.append(time.time() - t0)

            bt_true.append(yi)
            bt_pred_val.append(y_pred[0, 0])
            bt_std.append(y_std[0, 0])

            if i % BATCH_SIZE == 0:
                actual = np.array(bt_true)
                pred = np.array(bt_pred_val)
                std = np.array(bt_std)
                abs_err = np.abs(actual - pred)

                hist["points"].append(i)
                hist["predict_time"].append(np.mean(bt_pred))
                hist["update_time"].append(np.mean(bt_upd))
                hist["nrmse"].append(np.sqrt(np.mean(abs_err ** 2)) / y_range)
                rel = abs_err / np.maximum(np.abs(actual), 1e-10)
                hist["within"].append(np.mean(rel <= WITHIN_FRACTION))
                hist["coverage"].append(np.mean(abs_err <= std))
                hist["n_leaves"].append(len(gpt.root.leaves))

                bt_pred, bt_upd, bt_true, bt_pred_val, bt_std = [], [], [], [], []

    for k in hist:
        hist[k] = np.array(hist[k])

    total = (np.sum(hist["update_time"]) + np.sum(hist["predict_time"])) * BATCH_SIZE
    sys.stderr.write(
        f"  {criterion:16s} done: leaves={hist['n_leaves'][-1]:5d}  "
        f"final NRMSE={hist['nrmse'][-1]:.4f}  total time={total:7.1f}s\n"
    )
    return hist


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_comparison(results, target, outfile):
    panels = [
        ("nrmse", "NRMSE", True),
        ("within", f"Fraction within {int(WITHIN_FRACTION*100)}%", False),
        ("coverage", "Empirical 1-sigma coverage", False),
        ("n_leaves", "Number of leaves", False),
        ("predict_time", "Avg predict time / point (s)", True),
        ("update_time", "Avg update time / point (s)", True),
    ]
    colors = {
        "max_spread": "tab:gray",
        "max_variance": "tab:brown",
        "max_uncertainty": "tab:orange",
        "min_lengthscale": "tab:green",
        "oblique": "tab:blue",
        "random": "tab:red",
    }

    fig, axs = plt.subplots(3, 2, figsize=(15, 13), sharex=True)
    fig.suptitle(
        f"pyGPTreeo: split-dimension criteria — target='{target}'",
        fontsize=15,
    )
    axs = axs.ravel()

    for ax, (key, label, logy) in zip(axs, panels):
        for name, hist in results.items():
            lw = 2.6 if name in ("min_lengthscale", "oblique") else 1.8
            ax.plot(hist["points"], hist[key], marker="o", markersize=3,
                    linewidth=lw, label=name, color=colors.get(name))
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.4)
        if logy:
            ax.set_yscale("log")
        if key == "coverage":
            ax.axhline(0.68, ls="--", color="black", lw=1.5)
            ax.set_ylim(0, 1)
        if key == "within":
            ax.set_ylim(0, 1)

    for ax in axs[-2:]:
        ax.set_xlabel("Total points processed")
    axs[0].legend(loc="best", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(outfile, dpi=110)
    sys.stderr.write(f"\nSaved figure: {outfile}\n")


def save_csv(results, outfile):
    keys = ["predict_time", "update_time", "nrmse", "within", "coverage", "n_leaves"]
    with open(outfile, "w") as f:
        f.write("criterion,points," + ",".join(keys) + "\n")
        for name, hist in results.items():
            for j, p in enumerate(hist["points"]):
                row = [f"{hist[k][j]:.6g}" for k in keys]
                f.write(f"{name},{int(p)}," + ",".join(row) + "\n")
    sys.stderr.write(f"Saved CSV:    {outfile}\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    X, y = make_data(TARGET, N_POINTS, N_DIMS, SEED)

    sys.stderr.write(
        f"Benchmark: target='{TARGET}'  N={N_POINTS}  n_dims={N_DIMS}  "
        f"Nbar={NBAR}  retrain={RETRAIN_STEP}\n"
    )

    results = {}
    for criterion in CRITERIA:
        sys.stderr.write(f"Running criterion: {criterion} ...\n")
        # Same data stream and seed for every criterion (fair comparison).
        np.random.seed(SEED)
        results[criterion] = run_criterion(criterion, X, y, N_DIMS)

    suffix = f"{TARGET}_{N_POINTS}"
    plot_comparison(results, TARGET, f"benchmark_split_direction_{suffix}.png")
    save_csv(results, f"benchmark_split_direction_{suffix}.csv")

    # Console summary (final batch).
    sys.stderr.write("\n=== Final-batch summary ===\n")
    sys.stderr.write(f"{'criterion':16s} {'leaves':>7s} {'NRMSE':>9s} "
                     f"{'within':>8s} {'coverage':>9s}\n")
    for name, h in results.items():
        sys.stderr.write(
            f"{name:16s} {int(h['n_leaves'][-1]):7d} {h['nrmse'][-1]:9.4f} "
            f"{h['within'][-1]:8.3f} {h['coverage'][-1]:9.3f}\n"
        )


if __name__ == "__main__":
    main()
