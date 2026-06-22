"""Streaming sample-efficiency curves: additive leaf kernel vs default Matern.

For each standard target this streams points one at a time and, every 2000
points, evaluates held-out NRMSE on a fixed independent test set. It then plots
held-out NRMSE versus the number of processed points for the default full-D
Matern kernel and for the additive kernel (`make_additive_kernel`, depth 2 +
rescue), one subplot per target.

Usage:
    OMP_NUM_THREADS=1 python examples/benchmark_additive_curve.py \
        [targets] [dims] [n_points]

    targets : comma list from eggholder,himmelblau,rosenbrock,rastrigin,levy,custom
              (default: all six)
    dims    : single input dimension (default: 4)
    n_points: training stream length (default: 20000)
"""

import os
import sys
import contextlib
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from pygptreeo import GPTree
from pygptreeo.adapters import SklearnGPAdapter
from pygptreeo.kernels import make_additive_kernel
import target_functions as tf

STANDARD = {n: getattr(tf, n.capitalize()) for n in
            ["eggholder", "himmelblau", "rosenbrock", "rastrigin", "levy", "custom"]}

BATCH = 2000
NTEST = 2000
NBAR, THETA, RETRAIN, SEED = 200, 1e-4, 50, 20240617


def kernel_baseline(nd):
    return ConstantKernel(1.0, (1e-3, 1e8)) * Matern(
        nu=1.5, length_scale=[1.0] * nd, length_scale_bounds=[(1e-5, 1e5)] * nd)


CONFIGS = {
    "baseline (full-D Matern)": kernel_baseline,
    "additive (depth-2 + rescue)": lambda nd: make_additive_kernel(nd, interaction_depth=2, rescue=True),
}


def make_data(target, n, nd, seed):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0.0, 1.0, (n, nd))
    return X, STANDARD[target](X.T)


def run(kernel_fn, X, y, Xte, yte, nd):
    """Stream X,y; every BATCH points evaluate held-out NRMSE on a fixed test set.

    Returns per-checkpoint (points, held-out-NRMSE, leaves, coverage).
    """
    y_range = float(np.max(y) - np.min(y))
    gpr = SklearnGPAdapter(GaussianProcessRegressor(
        kernel=kernel_fn(nd), alpha=1e-6, n_restarts_optimizer=1))
    gpt = GPTree(GPR=gpr, Nbar=NBAR, theta=THETA, retrain_every_n_points=RETRAIN,
                 use_standard_scaling=True, use_calibrated_sigma=True,
                 splitting_strategy="gradual", max_n_pred_leaves=3, aggregation="moe",
                 split_dimension_criteria="min_lengthscale")
    pts, nrmse, leaves, cover = [], [], [], []
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        for i, (xi, yi) in enumerate(zip(X, y), 1):
            gpt.update_tree(xi.reshape(1, -1), np.array([[yi]]), 0.001 * abs(yi) + 1e-9)
            if i % BATCH == 0:
                yp, ys = gpt.predict(Xte)
                e = np.abs(yp[:, 0] - yte)
                pts.append(i)
                nrmse.append(np.sqrt(np.mean(e ** 2)) / y_range)
                cover.append(float(np.mean(e <= ys[:, 0])))
                leaves.append(len(gpt.root.leaves))
    return np.array(pts), np.array(nrmse), np.array(leaves), np.array(cover)


def main():
    targets = (sys.argv[1].split(",") if len(sys.argv) > 1 else list(STANDARD.keys()))
    nd = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    npoints = int(sys.argv[3]) if len(sys.argv) > 3 else 20000

    ncol = 3 if len(targets) > 1 else 1
    nrow = int(np.ceil(len(targets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.8 * nrow), squeeze=False)
    colors = {"baseline (full-D Matern)": "tab:gray",
              "additive (depth-2 + rescue)": "tab:red"}

    for k, target in enumerate(targets):
        ax = axes[k // ncol][k % ncol]
        X, y = make_data(target, npoints, nd, SEED)
        Xte, yte = make_data(target, NTEST, nd, SEED + 777)
        finals = {}
        for name, kfn in CONFIGS.items():
            np.random.seed(SEED)  # identical stream for both kernels
            pts, nrmse, leaves, cover = run(kfn, X, y, Xte, yte, nd)
            ax.plot(pts, nrmse, marker="o", ms=4, color=colors[name],
                    label=f"{name} (leaves={leaves[-1]}, cov={cover[-1]:.2f})")
            finals[name] = nrmse[-1]
            sys.stderr.write(f"{target:11s} {name:30s} final NRMSE={nrmse[-1]:.4f} "
                             f"leaves={leaves[-1]} cov={cover[-1]:.2f}\n")
        b = finals["baseline (full-D Matern)"]
        a = finals["additive (depth-2 + rescue)"]
        delta = 100.0 * (a - b) / b
        ax.set_title(f"{target}  (additive {delta:+.1f}% vs baseline)", fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("points processed")
        ax.set_ylabel("held-out NRMSE")
        ax.legend(fontsize=7, loc="upper right")
        sys.stderr.write(f"  -> {target}: additive {delta:+.1f}% vs baseline\n")

    for j in range(len(targets), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle(f"Additive leaf kernel vs default Matern — {nd}D, N={npoints}, "
                 f"batches of {BATCH}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(os.path.dirname(__file__),
                       f"additive_curve_{nd}D_N{npoints}.png")
    fig.savefig(out, dpi=120)
    sys.stderr.write(f"saved {out}\n")
    print(out)


if __name__ == "__main__":
    main()
