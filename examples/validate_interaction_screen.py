"""Stage-1 validation for interaction pruning (no tree plumbing yet).

Checks two things the whole feature rests on:
  (1) Discovery: streaming a benchmark function through one PairInteractionScreen
      recovers the right active pairs (adjacent (i,i+1) for coupled targets;
      NONE for the separable rastrigin/levy).
  (2) Mechanics + payoff: a kernel pruned to the discovered pairs (a) matches the
      full depth-2 additive kernel's held-out accuracy on a leaf-sized sample and
      (b) is meaningfully cheaper to evaluate.
"""
import os, sys, time, warnings, contextlib
import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.gaussian_process import GaussianProcessRegressor
from pygptreeo.interaction_screen import PairInteractionScreen
from pygptreeo.kernels import make_additive_kernel, prune_additive_kernel
from pygptreeo.adapters import SklearnGPAdapter
import target_functions as tf

FUNCS = {n: getattr(tf, n.capitalize()) for n in
         ["eggholder", "himmelblau", "rosenbrock", "rastrigin", "levy", "custom"]}
SEPARABLE = {"rastrigin", "levy"}  # expect NO active pairs


def adjacent(d):
    return [(i, i + 1) for i in range(d - 1)]


def discovery(d=4, N=6000, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, (N, d))
    print(f"\n===== discovery: d={d}, N={N} =====")
    print(f"{'target':11s}{'active pairs (score)':50s}  expected")
    for name, f in FUNCS.items():
        y = f(X.T)
        scr = PairInteractionScreen(d, warmup_points=0)
        for xi, yi in zip(X, y):
            scr.update(xi, yi)
        sc = scr.scores()
        active = scr.active_pairs() or []
        exp = [] if name in SEPARABLE else adjacent(d)
        astr = ", ".join(f"{p}:{sc[p]:.3f}" for p in sorted(active)) or "(none)"
        ok = set(active) == set(exp)
        print(f"{name:11s}{astr:50.50s}  {exp if exp else 'separable'}  "
              f"{'OK' if ok else 'MISMATCH'}")


def payoff(d=6, n_leaf=80, n_test=2000, seed=1):
    """On a leaf-sized sample, compare full vs adjacent-pruned additive kernel."""
    print(f"\n===== payoff: d={d}, leaf N={n_leaf} =====")
    print(f"{'target':11s}{'full NRMSE':>12s}{'pruned NRMSE':>13s}"
          f"{'full fit s':>12s}{'pruned fit s':>14s}{'kern speedup':>14s}")
    rng = np.random.RandomState(seed)
    Xtr = rng.uniform(0, 1, (n_leaf, d))
    Xte = rng.uniform(0, 1, (n_test, d))
    for name, f in FUNCS.items():
        ytr, yte = f(Xtr.T), f(Xte.T)
        yr = float(np.max(yte) - np.min(yte))
        exp_pairs = [] if name in SEPARABLE else adjacent(d)

        res = {}
        for tag, keep in (("full", None), ("pruned", exp_pairs)):
            k = make_additive_kernel(d, interaction_depth=2, rescue=True)
            if keep is not None:
                k = prune_additive_kernel(k, keep)
            g = GaussianProcessRegressor(kernel=k, alpha=1e-6, n_restarts_optimizer=1)
            ys = (ytr - ytr.mean()) / (ytr.std() + 1e-12)
            t0 = time.perf_counter()
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                g.fit(Xtr, ys)
            tfit = time.perf_counter() - t0
            yp = g.predict(Xte) * (ytr.std() + 1e-12) + ytr.mean()
            nrmse = np.sqrt(np.mean((yp - yte) ** 2)) / yr
            # Kernel-eval cost on the fitted kernel.
            kf = g.kernel_
            t0 = time.perf_counter()
            for _ in range(50):
                kf(Xtr, eval_gradient=True)
            res[tag] = (nrmse, tfit, time.perf_counter() - t0)
        speed = res["full"][2] / res["pruned"][2]
        print(f"{name:11s}{res['full'][0]:12.4f}{res['pruned'][0]:13.4f}"
              f"{res['full'][1]:12.2f}{res['pruned'][1]:14.2f}{speed:13.2f}x")


if __name__ == "__main__":
    discovery(4)
    discovery(6)
    payoff(6)
