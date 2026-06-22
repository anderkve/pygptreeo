"""Benchmark automatic per-region kernel selection (AutoSelectGPR) in a GPTree.

Compares three leaf-kernel strategies on the benchmark functions:

    baseline : plain full-D Matern
    additive : per-order additive kernel, no rescue (make_order_additive_kernel)
    auto     : AutoSelectGPR -- per region, fit both and keep the higher-evidence
               one (margined log marginal likelihood), with the additive verdict
               inherited by descendants

The point: `auto` should track whichever fixed kernel is better on each target --
the big additive wins on additive targets (e.g. levy, rastrigin) and the baseline
on genuinely non-additive ones (bump, coswave) -- without the user knowing in
advance, and without paying for the rescue term's expensive blend.

Usage:
    OMP_NUM_THREADS=1 python examples/benchmark_auto_select.py [N] [Nbar] [targets]
"""
import os, sys, time, warnings, contextlib
import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from pygptreeo import GPTree, make_auto_gpr
from pygptreeo.adapters import SklearnGPAdapter
from pygptreeo.kernels import make_order_additive_kernel
import target_functions as tf

FUNCS = {n: getattr(tf, n.capitalize()) for n in
         ["levy", "rastrigin", "rosenbrock", "eggholder", "custom", "bump", "coswave"]}
SEED, NTEST, THETA, NR = 20240617, 1500, 1e-4, 2


def baseline_gpr(d):
    return SklearnGPAdapter(GaussianProcessRegressor(
        kernel=ConstantKernel(1.0, (1e-3, 1e8)) * Matern(
            nu=1.5, length_scale=[1.0] * d, length_scale_bounds=[(1e-5, 1e5)] * d),
        alpha=1e-6, n_restarts_optimizer=NR))


def additive_gpr(d):
    return SklearnGPAdapter(GaussianProcessRegressor(
        kernel=make_order_additive_kernel(d, 2, rescue=False), alpha=1e-6,
        n_restarts_optimizer=NR))


def run(make_gpr, X, y, Xte, yte, d, Nbar):
    yr = float(y.max() - y.min())
    gpt = GPTree(GPR=make_gpr(d), Nbar=Nbar, theta=THETA, retrain_every_n_points=Nbar,
                 use_standard_scaling=True, use_calibrated_sigma=True,
                 splitting_strategy="gradual", max_n_pred_leaves=3, aggregation="moe",
                 split_dimension_criteria="min_lengthscale")
    t0 = time.perf_counter()
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        for xi, yi in zip(X, y):
            gpt.update_tree(xi.reshape(1, -1), np.array([[yi]]), 0.001 * abs(yi) + 1e-9)
    t = time.perf_counter() - t0
    yp, _ = gpt.predict(Xte)
    nrmse = np.sqrt(np.mean((yp[:, 0] - yte) ** 2)) / yr
    verds = [getattr(l.my_GPRs[0], "verdict", None) for l in gpt.root.leaves]
    n_add = sum(1 for v in verds if v == "additive")
    return nrmse, t, n_add, len(verds)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    Nbar = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    targets = sys.argv[3].split(",") if len(sys.argv) > 3 else list(FUNCS)
    d = 6
    print(f"\n##### auto kernel selection: d={d} N={N} Nbar={Nbar} (lazy retrain) #####")
    print(f"{'target':12s}| {'NRMSE base':>10s}{'add':>8s}{'auto':>8s} | "
          f"{'t_base':>7s}{'t_add':>7s}{'t_auto':>7s} | {'auto picks':>16s}")
    for name in targets:
        f = FUNCS[name]
        rng = np.random.RandomState(SEED)
        X = rng.uniform(0, 1, (N, d))
        try:
            y = f(X.T)
        except Exception:
            continue
        Xte = np.random.RandomState(SEED + 777).uniform(0, 1, (NTEST, d))
        yte = f(Xte.T)
        np.random.seed(SEED); nb, tb, _, _ = run(baseline_gpr, X, y, Xte, yte, d, Nbar)
        np.random.seed(SEED); na, ta, _, _ = run(additive_gpr, X, y, Xte, yte, d, Nbar)
        np.random.seed(SEED); nu, tu, nadd, nl = run(make_auto_gpr, X, y, Xte, yte, d, Nbar)
        print(f"{name:12s}| {nb:10.4f}{na:8.4f}{nu:8.4f} | "
              f"{tb:7.1f}{ta:7.1f}{tu:7.1f} | {f'{nadd}/{nl} additive':>16s}", flush=True)


if __name__ == "__main__":
    main()
