"""Speed experiment: per-order additive kernel (Newton-Girard) vs the explicit one.

Three parts:
  (1) Correctness  -- analytic gradient of OrderAdditiveKernel vs finite differences.
  (2) Assembly cost -- wall time of one K + dK evaluation, explicit O(d^D) term
      enumeration vs Newton-Girard O(d*D), as d and depth grow.
  (3) Leaf-sized fits -- GP fit time and held-out NRMSE on the benchmark functions,
      baseline Matern vs make_additive_kernel vs make_order_additive_kernel.

Usage:  OMP_NUM_THREADS=1 python examples/benchmark_oak_kernel.py
"""
import os, sys, time, warnings, contextlib
import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from pygptreeo.kernels import AdditiveKernel, make_additive_kernel
from pygptreeo.kernels_oak import OrderAdditiveKernel, make_order_additive_kernel
import target_functions as tf

FUNCS = {n: getattr(tf, n.capitalize()) for n in
         ["eggholder", "himmelblau", "rosenbrock", "rastrigin", "levy", "custom"]}


# --------------------------------------------------------------------------- #
# (1) gradient correctness
# --------------------------------------------------------------------------- #
def check_gradient(d=4, D=2, n=6, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, (n, d))
    k = OrderAdditiveKernel(d, max_order=D, base_kernel="matern",
                            length_scale=rng.uniform(0.5, 1.5, d),
                            order_variance=rng.uniform(0.5, 1.5, D))
    _, g_analytic = k(X, eval_gradient=True)

    theta = k.theta.copy()
    eps = 1e-6
    g_fd = np.zeros_like(g_analytic)
    for i in range(len(theta)):
        tp = theta.copy(); tp[i] += eps
        tm = theta.copy(); tm[i] -= eps
        k.theta = tp; Kp = k(X)
        k.theta = tm; Km = k(X)
        g_fd[:, :, i] = (Kp - Km) / (2 * eps)
    k.theta = theta
    err = np.max(np.abs(g_analytic - g_fd))
    print(f"(1) gradient check d={d} D={D}: max|analytic - finite_diff| = {err:.2e}  "
          f"{'OK' if err < 1e-5 else 'FAIL'}")


# --------------------------------------------------------------------------- #
# (2) assembly microbenchmark
# --------------------------------------------------------------------------- #
def assembly(n=120, reps=40):
    print("\n(2) assembly cost: one K + gradient eval (ms), explicit vs Newton-Girard")
    print(f"{'d':>4}{'depth':>7}{'#terms':>8}{'explicit':>11}{'order(NG)':>11}{'speedup':>9}")
    rng = np.random.RandomState(0)
    for D in (2, 3):
        for d in (4, 6, 8, 10, 12):
            if d < D:
                continue
            X = rng.uniform(0, 1, (n, d))
            ke = AdditiveKernel(d, interaction_depth=D, base_kernel="matern")
            ko = OrderAdditiveKernel(d, max_order=D, base_kernel="matern")
            n_terms = ke.n_terms

            def timeit(k):
                t0 = time.perf_counter()
                for _ in range(reps):
                    k(X, eval_gradient=True)
                return 1e3 * (time.perf_counter() - t0) / reps

            te, to = timeit(ke), timeit(ko)
            print(f"{d:>4}{D:>7}{n_terms:>8}{te:>10.2f}m{to:>10.2f}m{te/to:>8.2f}x")


# --------------------------------------------------------------------------- #
# (3) leaf-sized fit comparison
# --------------------------------------------------------------------------- #
def baseline_kernel(d):
    return ConstantKernel(1.0, (1e-3, 1e8)) * Matern(
        nu=1.5, length_scale=[1.0] * d, length_scale_bounds=[(1e-5, 1e5)] * d)


def fit_compare(d=6, n_leaf=80, n_test=2000, seed=1):
    print(f"\n(3) leaf-sized GP fit: d={d}, N={n_leaf}  (held-out NRMSE | fit time s)")
    print(f"{'target':11s}| {'base':>7s}{'explicit':>9s}{'order':>7s} | "
          f"{'t_base':>7s}{'t_expl':>7s}{'t_ord':>7s} | {'ord/expl':>9s}")
    rng = np.random.RandomState(seed)
    Xtr = rng.uniform(0, 1, (n_leaf, d))
    Xte = rng.uniform(0, 1, (n_test, d))
    builders = {
        "base": lambda: baseline_kernel(d),
        "explicit": lambda: make_additive_kernel(d, interaction_depth=2, rescue=True),
        "order": lambda: make_order_additive_kernel(d, max_order=2, rescue=True),
    }
    for name, f in FUNCS.items():
        try:
            ytr, yte = f(Xtr.T), f(Xte.T)
        except Exception:
            continue  # some targets (e.g. himmelblau) are only defined for d <= 6
        yr = float(np.max(yte) - np.min(yte))
        nrmse, tfit = {}, {}
        for tag, build in builders.items():
            g = GaussianProcessRegressor(kernel=build(), alpha=1e-6, n_restarts_optimizer=1)
            ys = (ytr - ytr.mean()) / (ytr.std() + 1e-12)
            t0 = time.perf_counter()
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                g.fit(Xtr, ys)
            tfit[tag] = time.perf_counter() - t0
            yp = g.predict(Xte) * (ytr.std() + 1e-12) + ytr.mean()
            nrmse[tag] = np.sqrt(np.mean((yp - yte) ** 2)) / yr
        print(f"{name:11s}| {nrmse['base']:7.4f}{nrmse['explicit']:9.4f}{nrmse['order']:7.4f} | "
              f"{tfit['base']:7.2f}{tfit['explicit']:7.2f}{tfit['order']:7.2f} | "
              f"{tfit['explicit']/tfit['order']:8.2f}x")


if __name__ == "__main__":
    check_gradient(4, 2)
    check_gradient(6, 3)
    assembly()
    fit_compare(6)
