"""Benchmark the additive leaf kernel (`make_additive_kernel`) vs the default.

Each leaf GP normally uses a full-dimensional Matern kernel, whose sample
complexity grows steeply with input dimension. `make_additive_kernel` builds a
low-order additive kernel (sum of 1-D and pairwise terms) plus a full-D Matern
"rescue" term whose learnable amplitude lets the model fall back to the default
on non-additive targets. This script measures sample efficiency: held-out NRMSE
as a function of the number of streamed training points, on the standard
benchmark functions.

Usage:
    OMP_NUM_THREADS=1 python examples/benchmark_additive_kernel.py \
        [targets] [dims] [n_points]

    targets : comma list from eggholder,himmelblau,rosenbrock,rastrigin,levy,custom
              (default: all six)
    dims    : comma list of input dimensions (default: 4)
    n_points: training stream length (default: 3000)
"""

import os
import sys
import contextlib
import warnings

import numpy as np

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

NBAR = int(os.environ.get("GPTREE_NBAR", "80"))
THETA, RETRAIN, SEED = 1e-4, 50, 20240617
NTEST = 1500
CHECKPOINTS_FRAC = (0.5, 1.0)

# Kernel builders compared against the full-D Matern baseline.
def kernel_baseline(nd):
    return ConstantKernel(1.0, (1e-3, 1e8)) * Matern(
        nu=1.5, length_scale=[1.0] * nd, length_scale_bounds=[(1e-5, 1e5)] * nd)

KERNELS = {
    "baseline":       lambda nd: kernel_baseline(nd),
    "add_d2_rescue":  lambda nd: make_additive_kernel(nd, interaction_depth=2, rescue=True),
    "add_d2_norescue": lambda nd: make_additive_kernel(nd, interaction_depth=2, rescue=False),
}


def make_data(target, n, nd, seed):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0.0, 1.0, (n, nd))
    return X, STANDARD[target](X.T)


def run(kernel_fn, X, y, Xte, yte, nd, checkpoints):
    """Stream X,y; at each checkpoint record held-out NRMSE and coverage."""
    y_range = float(np.max(y) - np.min(y))
    gpr = SklearnGPAdapter(GaussianProcessRegressor(
        kernel=kernel_fn(nd), alpha=1e-6, n_restarts_optimizer=1))
    gpt = GPTree(GPR=gpr, Nbar=NBAR, theta=THETA, retrain_every_n_points=RETRAIN,
                 use_standard_scaling=True, use_calibrated_sigma=True,
                 splitting_strategy="gradual", max_n_pred_leaves=3, aggregation="moe",
                 split_dimension_criteria="min_lengthscale")
    nrmse, cover = {}, {}
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        for i, (xi, yi) in enumerate(zip(X, y), 1):
            gpt.update_tree(xi.reshape(1, -1), np.array([[yi]]), 0.001 * abs(yi) + 1e-9)
            if i in checkpoints:
                yp, ys = gpt.predict(Xte)
                e = np.abs(yp[:, 0] - yte)
                nrmse[i] = np.sqrt(np.mean(e ** 2)) / y_range
                cover[i] = float(np.mean(e <= ys[:, 0]))
    return nrmse, cover


def main():
    targets = (sys.argv[1].split(",") if len(sys.argv) > 1
               else list(STANDARD.keys()))
    dims = ([int(d) for d in sys.argv[2].split(",")] if len(sys.argv) > 2 else [4])
    npoints = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    checkpoints = sorted(set(int(npoints * f) for f in CHECKPOINTS_FRAC))
    cfinal = checkpoints[-1]

    for nd in dims:
        sys.stderr.write(f"\n##### dims={nd}  N={npoints}  Nbar={NBAR} "
                         f"(held-out NRMSE at N={cfinal}; lower is better) #####\n")
        sys.stderr.write(f"{'target':12s}{'baseline':>11s}{'add_d2_resc':>13s}"
                         f"{'add_d2_nores':>14s}{'  best vs base':>16s}\n")
        for target in targets:
            X, y = make_data(target, npoints, nd, SEED)
            Xte, yte = make_data(target, NTEST, nd, SEED + 777)
            vals = {}
            for name, kfn in KERNELS.items():
                np.random.seed(SEED)
                nrmse, cover = run(kfn, X, y, Xte, yte, nd, checkpoints)
                vals[name] = nrmse[cfinal]
            base = vals["baseline"]
            best_add = min(vals["add_d2_rescue"], vals["add_d2_norescue"])
            delta = 100.0 * (best_add - base) / base
            sys.stderr.write(
                f"{target:12s}{vals['baseline']:11.4f}{vals['add_d2_rescue']:13.4f}"
                f"{vals['add_d2_norescue']:14.4f}{delta:+14.1f}%\n")


if __name__ == "__main__":
    main()
