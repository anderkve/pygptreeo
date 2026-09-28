"""
Multi-output GPTree for learning functions f(t; x): B-splines vs. a PCA output basis

This example revisits the problem of ``multioutput_function_learning_bsplines.py``:
learn a function f(t; x) of a variable t (e.g. time) that depends on a 2-D
parameter vector x = (x1, x2), from training points where the whole curve
f(t; x) is observed on a t-grid.

The B-spline approach represents each curve by its B-spline coefficients and
learns one GP per coefficient. Here that is compared with GPTree's
``output_model='pca'``, which learns a global linear basis of the output space
from the data and models only the leading basis scores, and with dropping the
spline step altogether: the raw curve values on the t-grid are the outputs,
and the PCA basis *is* the (data-adapted) function basis.

Configurations compared, all on the same GPTree settings:

    (A) B-spline coefficients as outputs, output_model='independent'  (original approach)
    (B) B-spline coefficients as outputs, output_model='pca'
    (C) f(t) values on the t-grid as outputs, output_model='pca'      (no splines)
    (D) f(t) values on the t-grid as outputs, output_model='independent'
        (one GP per grid point: the slow brute-force reference)

Usage:
    python examples/multioutput_function_learning_pca.py [n_train] [easy|hard] [noise_std]

    n_train    number of training parameter points (default 60)
    easy|hard  target family: 'easy' is the target of the B-spline example, 'hard'
               varies faster in x (default 'easy')
    noise_std  Gaussian noise added to the observed curves (default 0.0)
"""

import sys
import time
import numpy as np
from scipy.interpolate import BSpline, splrep

from pygptreeo import GPTree, Default_GPR

np.random.seed(42)


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def target_easy(t, x1, x2):
    """The target of examples/multioutput_function_learning_bsplines.py."""
    return (np.sin(2 * np.pi * t) * x1
            + np.cos(4 * np.pi * t) * x2
            + 0.5 * np.sin(6 * np.pi * t) * x1 * x2
            + 0.3 * np.exp(-((t - 0.5) ** 2) / (0.1 + 0.1 * x1 ** 2)))


def target_hard(t, x1, x2):
    """Same ingredients, but the coefficient functions vary faster in x."""
    return (np.sin(2 * np.pi * t) * np.sin(3 * np.pi * x1)
            + np.cos(4 * np.pi * t) * np.cos(3 * np.pi * x2)
            + 0.5 * np.sin(6 * np.pi * t) * np.sin(2 * np.pi * (x1 + x2))
            + 0.3 * np.exp(-((t - 0.5) ** 2) / (0.05 + 0.1 * x1 ** 2))
            + 0.2 * np.sin(10 * np.pi * t) * np.cos(4 * np.pi * x1 * x2))


# --------------------------------------------------------------------------- #
# B-spline helpers (as in the B-spline example)
# --------------------------------------------------------------------------- #
def bspline_coefficients(t, f, interior_knots):
    knots, coeffs, _ = splrep(t, f, s=0, k=3, t=interior_knots)
    return coeffs, knots


def bspline_reconstruct(t, coeffs, knots):
    f = BSpline(knots, coeffs, 3, extrapolate=False)(t)
    return np.nan_to_num(f, nan=0.0)


# --------------------------------------------------------------------------- #
def make_tree(n_outputs, output_model):
    return GPTree(
        GPR=Default_GPR(nu=2.5, n_restarts_optimizer=1),
        Nbar=30,
        theta=0.001,
        n_outputs=n_outputs,
        output_model=output_model,
        output_basis_components='noise',  # keep the components above the observation-noise level
        output_basis_min_points=20,
        use_calibrated_sigma=False,
        splitting_strategy='standard',
        use_standard_scaling=True,
    )


def run(name, X_train, Y_train, sigma, X_test, output_model, to_curves):
    """Fit a tree on outputs Y_train, predict at X_test, map predictions to curves."""
    t0 = time.time()
    gpt = make_tree(Y_train.shape[1], output_model)
    gpt.fit(X_train, Y_train, sigma, show_progress=False, shuffle=True)
    t_fit = time.time() - t0
    t0 = time.time()
    Y_pred, Y_std = gpt.predict(X_test, mode='recursive')
    t_pred = time.time() - t0
    F_pred = to_curves(Y_pred)
    n_gps = int(np.mean([len(leaf.my_GPRs) for leaf in gpt.root.leaves]))
    basis = gpt.output_basis.current if gpt.output_basis is not None else None
    return dict(name=name, F=F_pred, t_fit=t_fit, t_pred=t_pred, n_gps=n_gps,
                n_leaves=len(gpt.root.leaves), basis=basis, Y_std=Y_std)


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    family = sys.argv[2] if len(sys.argv) > 2 else 'easy'
    noise = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    target = {'easy': target_easy, 'hard': target_hard}[family]
    n_test = 200
    t = np.linspace(0, 1, 100)
    n_interior_knots = 12

    print("=" * 78)
    print(f"f(t; x) learning: B-splines vs PCA output basis   "
          f"(n_train={n_train}, target={family}, noise={noise})")
    print("=" * 78)

    rng = np.random.RandomState(0)
    X_train = rng.rand(n_train, 2)
    X_test = rng.rand(n_test, 2)
    F_train = np.array([target(t, *x) for x in X_train])
    F_train_obs = F_train + noise * rng.randn(*F_train.shape)
    F_test = np.array([target(t, *x) for x in X_test])

    # B-spline coefficient targets (fitted to the observed, possibly noisy, curves)
    interior = np.linspace(0, 1, n_interior_knots + 2)[1:-1]
    C_train, knots = [], None
    for f in F_train_obs:
        c, knots = bspline_coefficients(t, f, interior)
        C_train.append(c)
    C_train = np.array(C_train)
    n_coeffs = C_train.shape[1]
    print(f"t-grid points: {len(t)}   B-spline coefficients: {n_coeffs}   test points: {n_test}\n")

    # Observation noise handed to the tree: the curve noise (or a small floor). For
    # the spline coefficients the same value is used, as in the B-spline example.
    sig = max(noise, 1e-3)
    sigma_C = np.full(C_train.shape, sig)
    sigma_F = np.full(F_train_obs.shape, sig)

    results = []
    results.append(run("(A) B-spline coeffs, independent GPs", X_train, C_train, sigma_C, X_test,
                       'independent', lambda C: np.array([bspline_reconstruct(t, c, knots) for c in C])))
    results.append(run("(B) B-spline coeffs, PCA basis", X_train, C_train, sigma_C, X_test,
                       'pca', lambda C: np.array([bspline_reconstruct(t, c, knots) for c in C])))
    results.append(run("(C) f(t) on grid, PCA basis (no splines)", X_train, F_train_obs, sigma_F, X_test,
                       'pca', lambda F: F))
    results.append(run("(D) f(t) on grid, independent GPs", X_train, F_train_obs, sigma_F, X_test,
                       'independent', lambda F: F))

    print(f"{'configuration':<44s} {'GPs/leaf':>8s} {'leaves':>6s} {'fit [s]':>8s} {'pred [s]':>8s} "
          f"{'RMSE f(t)':>10s} {'max |err|':>10s}")
    for r in results:
        err = r['F'] - F_test
        rmse = np.sqrt(np.mean(err ** 2)); mx = np.max(np.abs(err))
        extra = ""
        if r['basis'] is not None:
            b = r['basis']
            extra = f"   basis: {b.n_components} comps, {b.explained_variance_ratio.sum():.5f} of variance"
        print(f"{r['name']:<44s} {r['n_gps']:>8d} {r['n_leaves']:>6d} {r['t_fit']:>8.1f} {r['t_pred']:>8.2f} "
              f"{rmse:>10.4f} {mx:>10.4f}{extra}")

    # Spline truncation floor: how well the B-spline basis itself can represent the test curves
    C_test = np.array([bspline_coefficients(t, f, interior)[0] for f in F_test])
    F_spline_best = np.array([bspline_reconstruct(t, c, knots) for c in C_test])
    print(f"\nB-spline representation floor (exact coefficients at the test x): "
          f"RMSE {np.sqrt(np.mean((F_spline_best - F_test) ** 2)):.4f}")

    # Optional plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    idx = [0, n_test // 3, 2 * n_test // 3, n_test - 1]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, i in zip(axes.ravel(), idx):
        ax.plot(t, F_test[i], 'k-', lw=2.5, alpha=0.6, label='true f(t)')
        for r, style in zip(results, ['C0--', 'C1-.', 'C2-', 'C3:']):
            ax.plot(t, r['F'][i], style, lw=1.5, label=r['name'])
        ax.set_title(f"x = ({X_test[i, 0]:.2f}, {X_test[i, 1]:.2f})")
        ax.set_xlabel('t'); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"f(t; x) reconstruction, n_train={n_train}, target={family}, noise={noise}")
    fig.tight_layout()
    out = f"multioutput_pca_comparison_{family}_n{n_train}.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
