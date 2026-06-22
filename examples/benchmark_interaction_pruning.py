"""Stage-2 end-to-end validation: region-local interaction pruning in a GPTree.

Streams each benchmark function through two otherwise-identical GPTrees built on
the depth-2 additive leaf kernel -- one with prune_interactions=False, one with
True -- and reports held-out NRMSE, total streaming (fit) time, and the resulting
leaf-kernel term counts. The hypothesis: on separable targets (rastrigin, levy)
the pruner collapses leaf kernels toward main-effects-only -> faster, no accuracy
loss; on coupled targets it keeps the live pairs -> accuracy preserved.
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
from pygptreeo.kernels import make_additive_kernel, AdditiveKernel
import target_functions as tf

FUNCS = {n: getattr(tf, n.capitalize()) for n in
         ["eggholder", "himmelblau", "rosenbrock", "rastrigin", "levy", "custom"]}

NBAR, THETA, RETRAIN, SEED, NTEST = 80, 1e-4, 50, 20240617, 1500


def make_data(f, n, d, seed):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, (n, d))
    return X, f(X.T)


def leaf_term_counts(gpt):
    """Number of additive terms in each leaf's kernel (None if not additive)."""
    counts = []
    for leaf in gpt.root.leaves:
        k = leaf.my_GPRs[0].get_kernel()
        def find(kk):
            if isinstance(kk, AdditiveKernel): return kk
            p = kk.get_params(deep=False)
            if "k1" in p:
                return find(kk.k1) or find(kk.k2)
            return None
        ak = find(k)
        counts.append(ak.n_terms if ak is not None else None)
    return [c for c in counts if c is not None]


def baseline_kernel(d):
    """Plain full-D Matern: the library default, no additive structure."""
    return ConstantKernel(1.0, (1e-3, 1e8)) * Matern(
        nu=1.5, length_scale=[1.0] * d, length_scale_bounds=[(1e-5, 1e5)] * d)


def run(kind, X, y, Xte, yte, d):
    """kind in {'baseline', 'add', 'add_prune'}."""
    yr = float(np.max(y) - np.min(y))
    if kind == "baseline":
        kernel = baseline_kernel(d)
    else:
        kernel = make_additive_kernel(d, interaction_depth=2, rescue=True)
    gpr = SklearnGPAdapter(GaussianProcessRegressor(
        kernel=kernel, alpha=1e-6, n_restarts_optimizer=1))
    gpt = GPTree(GPR=gpr, Nbar=NBAR, theta=THETA, retrain_every_n_points=RETRAIN,
                 use_standard_scaling=True, use_calibrated_sigma=True,
                 splitting_strategy="gradual", max_n_pred_leaves=3, aggregation="moe",
                 split_dimension_criteria="min_lengthscale",
                 prune_interactions=(kind == "add_prune"))
    t0 = time.perf_counter()
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        for xi, yi in zip(X, y):
            gpt.update_tree(xi.reshape(1, -1), np.array([[yi]]), 0.001 * abs(yi) + 1e-9)
    tstream = time.perf_counter() - t0
    yp, _ = gpt.predict(Xte)
    nrmse = np.sqrt(np.mean((yp[:, 0] - yte) ** 2)) / yr
    terms = leaf_term_counts(gpt) if kind != "baseline" else []
    return nrmse, tstream, terms


def main():
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    targets = sys.argv[3].split(",") if len(sys.argv) > 3 else list(FUNCS)
    full_terms = d + d * (d - 1) // 2
    print(f"\n##### d={d} N={N} Nbar={NBAR}  (full depth-2 additive = {full_terms} terms) #####")
    print("NRMSE (held-out) and total streaming time for: plain Matern baseline | "
          "additive (no prune) | additive (region-local prune)")
    print(f"{'target':11s}| {'base':>7s}{'add':>8s}{'add+pr':>8s} | "
          f"{'t_base':>7s}{'t_add':>7s}{'t_pr':>7s} | "
          f"{'pr/add':>7s}{'pr/base':>8s} | {'leaf terms':>11s}")
    for name in targets:
        f = FUNCS[name]
        X, y = make_data(f, N, d, SEED)
        Xte, yte = make_data(f, NTEST, d, SEED + 777)
        np.random.seed(SEED); n_b, t_b, _ = run("baseline", X, y, Xte, yte, d)
        np.random.seed(SEED); n_a, t_a, _ = run("add", X, y, Xte, yte, d)
        np.random.seed(SEED); n_p, t_p, terms_p = run("add_prune", X, y, Xte, yte, d)
        mt = np.mean(terms_p) if terms_p else float("nan")
        print(f"{name:11s}| {n_b:7.4f}{n_a:8.4f}{n_p:8.4f} | "
              f"{t_b:7.1f}{t_a:7.1f}{t_p:7.1f} | "
              f"{t_a / t_p:6.2f}x{t_b / t_p:7.2f}x | {mt:5.1f}/{full_terms}", flush=True)


if __name__ == "__main__":
    main()
