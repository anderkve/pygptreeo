"""Compare split-dimension criteria for GPTree on a streaming benchmark.

Runs each `split_dimension_criteria` ('max_spread', 'max_variance',
'max_uncertainty', 'min_lengthscale', 'random') on the same data stream and
plots batch NRMSE vs the number of processed points.

Usage:
    OMP_NUM_THREADS=1 python examples/benchmark_split_direction.py [target] [n_points]

`target` is a standard function ('eggholder', 'himmelblau', 'rosenbrock',
'rastrigin', 'levy', 'custom') or the synthetic 'aniso_chirp' (default), which is
anisotropic and heterogeneous (rough along x0, smooth along a wider x1 that
misleads the spread-based criteria) so the choice of split axis matters.
"""

import os
import sys
import contextlib
import warnings

import numpy as np
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from pygptreeo import GPTree
from pygptreeo.adapters import SklearnGPAdapter
import target_functions as tf

TARGET = sys.argv[1] if len(sys.argv) > 1 else "aniso_chirp"
N_POINTS = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
BATCH = 2000
N_DIMS, NBAR, THETA, RETRAIN, SEED = 3, 200, 1e-4, 50, 512312
CRITERIA = ["max_spread", "max_variance", "max_uncertainty", "min_lengthscale", "random"]
STANDARD = {n: getattr(tf, n.capitalize()) for n in
            ["eggholder", "himmelblau", "rosenbrock", "rastrigin", "levy", "custom"]}


def make_data(target, n, nd, seed):
    rng = np.random.RandomState(seed)
    if target in STANDARD:
        X = rng.uniform(0.0, 1.0, (n, nd))
        return X, STANDARD[target](X.T)
    if target == "aniso_chirp":
        # x0: rough/chirped; x1: smooth but widest spread; x2: mild.
        X = rng.uniform([0, 0, 0][:nd], [1, 3, 1][:nd], (n, nd))
        y = np.sin(2 * np.pi * (1.0 + 8.0 * X[:, 0]) * X[:, 0])
        if nd > 1:
            y = y + 0.3 * X[:, 1]
        if nd > 2:
            y = y + 0.2 * np.sin(2 * np.pi * 0.3 * X[:, 2])
        return X, y
    raise ValueError(f"Unknown target '{target}'")


def make_gpr(nd):
    kernel = ConstantKernel(1.0, (1e-3, 1e8)) * Matern(
        nu=1.5, length_scale=[1.0] * nd, length_scale_bounds=[(1e-5, 1e5)] * nd)
    return SklearnGPAdapter(
        GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=1))


def run(criterion, X, y, nd):
    """Stream all points through one config; return (points, nrmse, leaves, coverage)."""
    y_range = float(np.max(y) - np.min(y))
    gpt = GPTree(GPR=make_gpr(nd), Nbar=NBAR, theta=THETA, retrain_every_n_points=RETRAIN,
                 use_standard_scaling=True, use_calibrated_sigma=True,
                 splitting_strategy="gradual", max_n_pred_leaves=3, aggregation="moe",
                 split_dimension_criteria=criterion)
    pts, nrmse, leaves, cover = [], [], [], []
    err, std = [], []
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        for i, (xi, yi) in enumerate(zip(X, y), 1):
            xr, yr = xi.reshape(1, -1), np.array([[yi]])
            yp, ys = gpt.predict(xr)
            gpt.update_tree(xr, yr, 0.001 * np.abs(yr) + 1e-9)
            err.append(abs(yp[0, 0] - yi)); std.append(ys[0, 0])
            if i % BATCH == 0:
                e = np.array(err)
                pts.append(i)
                nrmse.append(np.sqrt(np.mean(e ** 2)) / y_range)
                cover.append(np.mean(e <= np.array(std)))
                leaves.append(len(gpt.root.leaves))
                err, std = [], []
    return np.array(pts), np.array(nrmse), np.array(leaves), np.array(cover)


def main():
    X, y = make_data(TARGET, N_POINTS, N_DIMS, SEED)
    sys.stderr.write(f"target={TARGET} N={N_POINTS} dims={N_DIMS}\n")
    sys.stderr.write(f"{'criterion':16s}{'NRMSE':>9s}{'leaves':>8s}{'coverage':>10s}\n")

    plt.figure(figsize=(8, 5))
    for crit in CRITERIA:
        np.random.seed(SEED)  # same stream for every criterion
        pts, nrmse, leaves, cover = run(crit, X, y, N_DIMS)
        plt.plot(pts, nrmse, marker="o", label=crit)
        sys.stderr.write(f"{crit:16s}{nrmse[-1]:9.4f}{leaves[-1]:8d}{cover[-1]:10.3f}\n")

    plt.yscale("log"); plt.grid(True, alpha=0.4); plt.legend()
    plt.xlabel("points processed"); plt.ylabel("batch NRMSE")
    plt.title(f"split-dimension criteria — {TARGET}")
    plt.tight_layout()
    out = f"benchmark_split_direction_{TARGET}_{N_POINTS}.png"
    plt.savefig(out, dpi=110)
    sys.stderr.write(f"saved {out}\n")


if __name__ == "__main__":
    main()
