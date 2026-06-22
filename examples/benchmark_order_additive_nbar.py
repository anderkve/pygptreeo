"""Benchmark: per-order additive kernel vs baseline (and explicit additive) in a GPTree.

Streams each benchmark function through a GPTree and, at the end, reports held-out
NRMSE, 1-sigma coverage, and total streaming (build + retrain) time for three leaf
kernels:

    baseline : plain full-D Matern (the library default)
    additive : make_additive_kernel  (explicit O(d^D) term enumeration)
    order    : make_order_additive_kernel  (Newton-Girard, O(d*D))

Run it at a small and a large leaf size to see how the kernel choice trades off
against Nbar (small leaves: additive structure helps accuracy most; large leaves:
kernel-assembly cost matters more, where the per-order kernel is cheapest).

Usage:
    OMP_NUM_THREADS=1 python examples/benchmark_order_additive_nbar.py NBAR [N] [dims] [targets]
"""
import os, sys, time, warnings, contextlib
import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from pygptreeo import GPTree
from pygptreeo.adapters import SklearnGPAdapter
from pygptreeo.kernels import make_additive_kernel, make_order_additive_kernel
import target_functions as tf

FUNCS = {n: getattr(tf, n.capitalize()) for n in
         ["eggholder", "himmelblau", "rosenbrock", "rastrigin", "levy", "custom"]}
THETA, SEED, NTEST = 1e-4, 20240617, 1500


def baseline_kernel(d):
    return ConstantKernel(1.0, (1e-3, 1e8)) * Matern(
        nu=1.5, length_scale=[1.0] * d, length_scale_bounds=[(1e-5, 1e5)] * d)


def build_kernel(kind, d):
    # RESCUE=0 drops the full-D Matern rescue term from the additive kernels
    # (leaving just the additive block), e.g. to isolate the additive structure
    # or speed up large-Nbar runs. Defaults to on.
    rescue = os.environ.get("RESCUE", "1") != "0"
    if kind == "baseline":
        return baseline_kernel(d)
    if kind == "additive":
        return make_additive_kernel(d, interaction_depth=2, rescue=rescue)
    if kind == "order":
        return make_order_additive_kernel(d, max_order=2, rescue=rescue)
    raise ValueError(kind)


def run(kind, X, y, Xte, yte, d, Nbar, retrain):
    yr = float(np.max(y) - np.min(y))
    gpr = SklearnGPAdapter(GaussianProcessRegressor(
        kernel=build_kernel(kind, d), alpha=1e-6, n_restarts_optimizer=0))
    gpt = GPTree(GPR=gpr, Nbar=Nbar, theta=THETA, retrain_every_n_points=retrain,
                 use_standard_scaling=True, use_calibrated_sigma=True,
                 splitting_strategy="gradual", max_n_pred_leaves=3, aggregation="moe",
                 split_dimension_criteria="min_lengthscale")
    t0 = time.perf_counter()
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        for xi, yi in zip(X, y):
            gpt.update_tree(xi.reshape(1, -1), np.array([[yi]]), 0.001 * abs(yi) + 1e-9)
    tstream = time.perf_counter() - t0
    yp, ys = gpt.predict(Xte)
    e = np.abs(yp[:, 0] - yte)
    nrmse = np.sqrt(np.mean(e ** 2)) / yr
    cover = float(np.mean(e <= ys[:, 0]))
    n_leaves = len(gpt.root.leaves)
    return nrmse, cover, tstream, n_leaves


def main():
    Nbar = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    dims = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [6]
    targets = sys.argv[4].split(",") if len(sys.argv) > 4 else list(FUNCS)
    # retrain_every_n_points. Default heuristic, overridable via RETRAIN env var;
    # RETRAIN=lazy means retrain only when a leaf is full (== Nbar), i.e. each
    # node is trained once, just before it splits (cheapest, fewest fits).
    env_retrain = os.environ.get("RETRAIN")
    if env_retrain == "lazy":
        retrain = Nbar
    elif env_retrain is not None:
        retrain = int(env_retrain)
    else:
        retrain = max(50, Nbar // 5)

    for d in dims:
        print(f"\n##### Nbar={Nbar}  d={d}  N={N}  retrain_every={retrain} "
              f"(held-out NRMSE / coverage / stream time) #####", flush=True)
        print(f"{'target':11s}| {'NRMSE base':>10s}{'add':>8s}{'order':>8s} | "
              f"{'cov base':>8s}{'add':>6s}{'order':>6s} | "
              f"{'t_base':>7s}{'t_add':>7s}{'t_ord':>7s} | {'ord/base':>8s}{'ord/add':>8s}",
              flush=True)
        for name in targets:
            f = FUNCS[name]
            rng = np.random.RandomState(SEED)
            X = rng.uniform(0, 1, (N, d))
            try:
                y = f(X.T)
            except Exception:
                continue  # himmelblau only defined for d <= 6
            Xte = np.random.RandomState(SEED + 777).uniform(0, 1, (NTEST, d))
            yte = f(Xte.T)
            res = {}
            # KINDS env var selects conditions (e.g. KINDS=baseline,order to skip
            # the expensive explicit additive kernel at large Nbar).
            kinds = os.environ.get("KINDS", "baseline,additive,order").split(",")
            for kind in ("baseline", "additive", "order"):
                if kind not in kinds:
                    res[kind] = (float("nan"),) * 4
                    continue
                np.random.seed(SEED)
                res[kind] = run(kind, X, y, Xte, yte, d, Nbar, retrain)
            b, a, o = res["baseline"], res["additive"], res["order"]
            print(f"{name:11s}| {b[0]:10.4f}{a[0]:8.4f}{o[0]:8.4f} | "
                  f"{b[1]:8.2f}{a[1]:6.2f}{o[1]:6.2f} | "
                  f"{b[2]:7.1f}{a[2]:7.1f}{o[2]:7.1f} | "
                  f"{b[2]/o[2]:7.2f}x{a[2]/o[2]:7.2f}x", flush=True)


if __name__ == "__main__":
    main()
