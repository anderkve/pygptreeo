"""Benchmark: global model + residual GPTree under different input-stream types.

Compares a plain GPTree with a GPTree that models the *residual* of a global
model (a GP with a low-order additive + Matern kernel, fitted on a coverage
reservoir of the stream), across input streams of different character:

    uniform    independent uniform samples over the unit cube
    focusing   differential-evolution-like: a broad uniform phase, then a
               Gaussian cluster around the target's minimum whose width shrinks
               geometrically (0.3 -> 0.02)
    sweeping   a Gaussian cluster (width 0.12) whose centre moves along a smooth
               path through the cube
    walker     MCMC-like: every proposal of a Metropolis random walk on
               exp(-f / T), accepted or not

and reports, per (target, stream, configuration, seed), the prequential NRMSE
on the stream after warm-up, the NRMSE on a uniform test set (global accuracy),
and the NRMSE on a "focus" test set drawn where the stream ended up.

Configurations:
    tree          plain GPTree
    global_pkg    the package implementation,
                  ``GPTree(global_mean=AdditiveGPGlobalMean(...))``
    global        prototype of the same design (see below); ``global_fresh``
                  adds the package's refresh rule and should match ``global_pkg``
    frozen        prototype with the global model frozen after the warm-up
    global_damped, global_damped_tight, global_damped_wide
                  prototype whose global contribution is damped by a coverage
                  confidence (1 - relative predictive variance of a reference GP
                  on the reservoir, length scale 1x / 0.5x / 2x the reservoir
                  spacing)
    *_lin         leaf kernel with an added linear-trend term

Global-model design shared by the prototype and the package:
    * a coverage reservoir (maximin design, 500 points) rather than a
      uniform-in-time sample, so the model represents the explored region
      whatever the stream does;
    * refit only when >= 25 % of the reservoir has turned over since the last
      fit (and at most once per reservoir-size points);
    * every refit runs the hyperparameter optimiser with random restarts on
      top of the warm start;
    * versioned snapshots: a leaf subtracts the current snapshot at fit time
      and adds back the same snapshot at predict time.

The prototype configurations inject the residual mechanism into GPNode by
wrapping ``fit_my_GPR``, ``predict`` and ``generate_children`` at import time;
do not import this module from library code.

Usage:
    python examples/benchmark_global_mean_streams.py --target rotated_rosenbrock \\
        --streams uniform,focusing,sweeping,walker --configs tree,global --seeds 1,2,3
    python examples/benchmark_global_mean_streams.py --summarize results/*.jsonl

Each run prints one JSON line (prefixed with ``RESULT ``); ``--summarize``
aggregates such lines (from files or stdin) into a markdown table.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, Matern

from pygptreeo import GPTree, Default_GPR, AdditiveMaternKernel, AdditiveGPGlobalMean
from pygptreeo.gpnode import GPNode
import target_functions as tf

TARGETS = {
    'rotated_rosenbrock': tf.RotatedRosenbrock,
    'gaussian_peaks': tf.GaussianPeaks,
    'rosenbrock': tf.Rosenbrock,
    'rastrigin': tf.Rastrigin,
    'levy': tf.Levy,
}


# --------------------------------------------------------------------------- #
# Streams
# --------------------------------------------------------------------------- #
def _find_minimum(target, d, rng):
    X0 = rng.rand(400, d)
    f0 = target(X0.T)
    best = None
    for i in np.argsort(f0)[:5]:
        res = minimize(lambda x: float(target(x)), X0[i], method='L-BFGS-B', bounds=[(0, 1)] * d)
        if best is None or res.fun < best.fun:
            best = res
    return np.clip(best.x, 0, 1)


def make_stream(kind, target, d, N, rng):
    """Return (X, X_focus): the stream inputs and a 3000-point test set drawn
    where the stream ended up (for 'uniform' this is another uniform sample)."""
    if kind == 'uniform':
        return rng.rand(N, d), rng.rand(3000, d)

    if kind == 'focusing':
        n_broad = max(500, N // 6)
        xmin = _find_minimum(target, d, rng)
        width = 0.3 * (0.02 / 0.3) ** (np.clip(np.arange(N) - n_broad, 0, None) / (N - n_broad))
        broad = rng.rand(N, d)
        cluster = np.clip(xmin + width[:, None] * rng.randn(N, d), 0, 1)
        X = np.where((np.arange(N) < n_broad)[:, None], broad, cluster)
        return X, np.clip(xmin + 0.03 * rng.randn(3000, d), 0, 1)

    if kind == 'sweeping':
        s = np.linspace(0, 1, N)
        centre = 0.5 + 0.35 * np.column_stack([np.sin(2 * np.pi * (s * 0.75 + 0.2 * j)) for j in range(d)])
        X = np.clip(centre + 0.12 * rng.randn(N, d), 0, 1)
        s_end = rng.uniform(0.9, 1.0, 3000)
        c_end = 0.5 + 0.35 * np.column_stack([np.sin(2 * np.pi * (s_end * 0.75 + 0.2 * j)) for j in range(d)])
        return X, np.clip(c_end + 0.12 * rng.randn(3000, d), 0, 1)

    if kind == 'walker':
        # Metropolis random walk on exp(-f/T); the stream is every proposal.
        X0 = rng.rand(400, d)
        f0 = target(X0.T)
        T = 0.05 * (np.percentile(f0, 90) - f0.min())
        x = X0[np.argmin(f0)].copy(); fx = float(target(x))
        X = np.empty((N, d)); visited = []
        for i in range(N):
            prop = np.clip(x + 0.05 * rng.randn(d), 0, 1)
            fp = float(target(prop))
            X[i] = prop
            if np.log(rng.rand()) < -(fp - fx) / T:
                x, fx = prop, fp
            visited.append(x.copy())
        visited = np.array(visited[N // 2:])
        idx = rng.randint(0, visited.shape[0], 3000)
        return X, np.clip(visited[idx] + 0.02 * rng.randn(3000, d), 0, 1)

    raise ValueError(f"unknown stream kind '{kind}'")


# --------------------------------------------------------------------------- #
# Global model prototype: coverage reservoir + turnover-triggered refits + snapshots
# --------------------------------------------------------------------------- #
class Snapshot:
    """An immutable fitted global model: version + scalers + fitted GP (+ damping).

    Damping (``damping_scale > 0``): the model's contribution is multiplied by a
    confidence ``w(x) in [0, 1]`` so that, in standardised units, the prediction is
    ``w(x) * m(x)``. Where the model has data ``w ~ 1``; where it extrapolates
    ``w -> 0`` and the prediction reverts to the (constant) prior mean, so the
    residual a leaf models reverts to the raw target exactly where the global
    model is guessing.

    ``w`` is one minus the *relative posterior variance of a reference GP* on the
    same reservoir points: an RBF kernel with length scale
    ``damping_scale * (median nearest-neighbour spacing of the reservoir)`` and
    unit amplitude, so ``var_ref(x) / 1`` is 0 on the data and 1 far from all of
    it. The fitted global kernel's own predictive variance cannot be used for
    this: marginal likelihood picks catch-all length scales 100x the domain
    width, and its posterior variance is then < 1e-3 of the prior everywhere,
    even far outside the data (see the benchmark results document).
    ``damping_power`` sharpens the transition (w -> w ** power).
    """

    def __init__(self, version, xs, ys, gp, ref_X=None, ref_L=None, ref_ell=None, damping_power=1.0):
        self.version, self.xs, self.ys, self.gp = version, xs, ys, gp
        self.ref_X, self.ref_L, self.ref_ell, self.damping_power = ref_X, ref_L, ref_ell, damping_power

    @property
    def damped(self):
        return self.ref_L is not None

    def confidence(self, X):
        """w(x) in [0, 1]: 1 - relative posterior variance of the reference GP (1 = fully trusted)."""
        Xs = self.xs.transform(X)
        d2 = ((Xs[:, None, :] - self.ref_X[None, :, :]) ** 2).sum(-1)          # (m, n)
        k_star = np.exp(-0.5 * d2 / self.ref_ell ** 2)                          # unit-amplitude RBF
        v = solve_triangular(self.ref_L, k_star.T, lower=True)                  # (n, m)
        explained = np.clip((v ** 2).sum(axis=0), 0.0, 1.0)                     # k*^T K^-1 k*  in [0, 1]
        return explained ** self.damping_power

    def predict(self, X):
        m = self.gp.predict(self.xs.transform(X))
        if self.damped:
            m = self.confidence(X) * m
        return self.ys.inverse_transform(m[:, None]).ravel()


class CoverageReservoir:
    """Fixed-size maximin design of the stream: a newcomer replaces one point of the
    closest pair if that increases the minimum pairwise separation."""

    def __init__(self, size, d):
        self.size = size
        self.X = np.empty((0, d)); self.y = np.empty(0)
        self.D = None; self.turnover = 0

    def add(self, x, yv):
        if self.X.shape[0] < self.size:
            self.X = np.vstack((self.X, x)); self.y = np.append(self.y, yv); self.turnover += 1
            if self.X.shape[0] == self.size:
                diff = self.X[:, None, :] - self.X[None, :, :]
                self.D = np.sqrt((diff ** 2).sum(-1)); np.fill_diagonal(self.D, np.inf)
            return
        dnew = np.sqrt(((self.X - x) ** 2).sum(1))
        i, j = np.unravel_index(np.argmin(self.D), self.D.shape)
        if dnew.min() > self.D[i, j]:
            victim = i if self.D[i].min() <= self.D[j].min() else j
            self.X[victim] = x; self.y[victim] = yv
            self.D[victim, :] = dnew; self.D[:, victim] = dnew; self.D[victim, victim] = np.inf
            self.turnover += 1


class GlobalMeanModel:
    def __init__(self, d, seed, reservoir_size=500, min_points=200, min_turnover=0.25,
                 restarts=2, freeze_after=None, damping_scale=0.0, damping_power=1.0):
        self.d, self.seed = d, seed
        self.damping_scale, self.damping_power = damping_scale, damping_power   # damping_scale 0 = off
        self.res = CoverageReservoir(reservoir_size, d)
        self.min_points, self.min_turnover, self.restarts = min_points, min_turnover, restarts
        self.freeze_after = freeze_after
        self.current = None
        self.n_seen = 0; self.n_at_fit = 0; self.turnover_at_fit = 0
        self.n_refits = 0; self.n_skipped = 0; self.t_fit = 0.0

    def observe(self, x, yv):
        self.n_seen += 1
        self.res.add(x, yv)
        if self.n_seen < self.min_points:
            return
        if self.current is None:
            self.fit(); return
        if self.freeze_after is not None and self.n_seen > self.freeze_after:
            return
        due = self.n_seen >= 2 * self.n_at_fit or self.n_seen - self.n_at_fit >= self.res.size
        if not due:
            return
        if self.res.turnover - self.turnover_at_fit < self.min_turnover * self.res.size:
            self.n_skipped += 1; self.n_at_fit = self.n_seen   # nothing new at the model's scale
            return
        self.fit()

    def fit(self):
        t0 = time.time()
        Xr, yr = self.res.X, self.res.y
        xs = StandardScaler().fit(Xr); ys = StandardScaler().fit(yr[:, None])
        kernel = self.current.gp.kernel_ if self.current is not None else AdditiveMaternKernel(d=self.d, order=2)
        gp = GaussianProcessRegressor(kernel, alpha=1e-6, n_restarts_optimizer=self.restarts, random_state=self.seed)
        gp.fit(xs.transform(Xr), ys.transform(yr[:, None]).ravel())
        version = 1 if self.current is None else self.current.version + 1
        ref = {}
        if self.damping_scale > 0:
            Xs = xs.transform(Xr)
            D = np.sqrt(((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
            ell = self.damping_scale * float(np.median(D.min(axis=1)))       # x reservoir spacing
            K = np.exp(-0.5 * (np.where(np.isinf(D), 0.0, D) ** 2) / ell ** 2) + 1e-8 * np.eye(Xs.shape[0])
            ref = dict(ref_X=Xs, ref_L=cholesky(K, lower=True), ref_ell=ell, damping_power=self.damping_power)
        self.current = Snapshot(version, xs, ys, gp, **ref)
        self.n_at_fit = self.n_seen; self.turnover_at_fit = self.res.turnover
        self.n_refits += 1; self.t_fit += time.time() - t0


# Inject the residual mechanism into GPNode (prototype; see module docstring).
_orig_fit, _orig_predict, _orig_children = GPNode.fit_my_GPR, GPNode.predict, GPNode.generate_children
_ACTIVE = {'model': None, 'refresh_stale': False, 'n_stale_refits': 0}


def _fit_with_residual(self, force_training=False):
    model = _ACTIVE['model']
    snap = model.current if model is not None else None
    if snap is None:
        return _orig_fit(self, force_training)
    raw_y, raw_sh = self.my_y_data, self.shared_y_data
    try:
        if raw_y.shape[0]:
            self.my_y_data = raw_y - snap.predict(self.my_X_data)[:, None]
        if raw_sh.shape[0]:
            self.shared_y_data = raw_sh - snap.predict(self.shared_X_data)[:, None]
        did = _orig_fit(self, force_training)
    finally:
        self.my_y_data, self.shared_y_data = raw_y, raw_sh
    if did:
        self._fitted_mean = snap
    return did


def _predict_with_residual(self, x, return_std=True, use_calibrated_sigma=False):
    model = _ACTIVE['model']
    snap = getattr(self, '_fitted_mean', None)
    if (_ACTIVE['refresh_stale'] and model is not None and model.current is not None and self.is_leaf
            and self.n_points > 0 and (snap is None or snap.version < model.current.version)):
        # A newer global version exists: refit this leaf against it before predicting, so no
        # prediction ever combines a stale global snapshot with a residual GP that cannot
        # correct that snapshot's extrapolation error.
        _fit_with_residual(self, force_training=True)
        _ACTIVE['n_stale_refits'] += 1
        snap = getattr(self, '_fitted_mean', None)
    mu, sd = _orig_predict(self, x, return_std, use_calibrated_sigma)
    if snap is not None:
        mu = mu + snap.predict(x)[:, None]
    return mu, sd


def _children_with_residual(self, GPR, n_features):
    _orig_children(self, GPR, n_features)
    for child in self.children:
        child._fitted_mean = getattr(self, '_fitted_mean', None)


GPNode.fit_my_GPR = _fit_with_residual
GPNode.predict = _predict_with_residual
GPNode.generate_children = _children_with_residual


# --------------------------------------------------------------------------- #
def run_one(target_name, stream, config, seed, d, N, nbar, calibrate=False):
    target = TARGETS[target_name]
    rng = np.random.RandomState(seed)
    X, X_focus = make_stream(stream, target, d, N, rng)
    y = target(X.T); sig = np.maximum(1e-3 * np.abs(y), 1e-6)
    X_uni = rng.rand(3000, d); y_uni = target(X_uni.T); y_focus = target(X_focus.T)
    yrange = y_uni.max() - y_uni.min()
    warmup = max(500, N // 6)

    model = None
    linear = config.endswith('_lin')          # leaf kernel with an explicit linear-trend term
    config = config[:-4] if linear else config
    refresh = config.endswith('_fresh')       # refit a leaf on first use after a newer global version
    config = config[:-6] if refresh else config
    _ACTIVE['refresh_stale'] = refresh; _ACTIVE['n_stale_refits'] = 0
    package_learner = None                    # 'global_pkg': the package implementation (pygptreeo.global_mean)
    if config == 'global_pkg':
        package_learner = AdditiveGPGlobalMean(reservoir_size=500, min_points=200, min_turnover=0.25,
                                               n_restarts_optimizer=2, random_state=seed)
    elif config == 'global':
        model = GlobalMeanModel(d, seed)
    elif config == 'frozen':
        model = GlobalMeanModel(d, seed, freeze_after=warmup)
    elif config == 'global_damped':          # reference length scale = 1 x reservoir spacing
        model = GlobalMeanModel(d, seed, damping_scale=1.0)
    elif config == 'global_damped_tight':    # 0.5 x spacing: trusts the model only right on its data
        model = GlobalMeanModel(d, seed, damping_scale=0.5)
    elif config == 'global_damped_wide':     # 2 x spacing: mild damping
        model = GlobalMeanModel(d, seed, damping_scale=2.0)
    elif config not in ('tree', 'global_pkg', 'neural'):
        raise ValueError(config)
    _ACTIVE['model'] = model

    np.random.seed(seed)
    neural_learner = None
    if config == 'neural':
        # Neural-linear leaves: one shared feature network (trained on every point seen,
        # refit at each doubling), Bayesian linear regression on its features per leaf.
        from pygptreeo import FeatureNetLearner, NeuralLinearGPR
        neural_learner = FeatureNetLearner(random_state=seed)
        gpr = NeuralLinearGPR(neural_learner)
    elif linear:
        # ARD Matern + Bayesian linear trend: the leaf GP then extrapolates with its own local slope
        leaf_kernel = (ConstantKernel() * Matern(nu=1.5, length_scale=np.ones(d))
                       + ConstantKernel(1.0, (1e-5, 1e5)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3)))
        gpr = Default_GPR(kernel=leaf_kernel, n_restarts_optimizer=1)
    else:
        gpr = Default_GPR(n_restarts_optimizer=1)
    gpt = GPTree(GPR=gpr, Nbar=nbar, theta=1e-4,
                 retrain_every_n_points=25, splitting_strategy='gradual', use_calibrated_sigma=calibrate,
                 global_mean=package_learner)
    t0 = time.time(); errs = np.empty(N); sds = np.empty(N)
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(N):
            xi = X[i:i + 1]
            if model is not None:
                model.observe(xi, y[i])
            mu, sd = gpt.predict(xi)
            errs[i] = mu[0, 0] - y[i]; sds[i] = sd[0, 0]
            gpt.update_tree(xi, np.array([[y[i]]]), np.array([[sig[i]]]))
        P_uni, S_uni = gpt.predict(X_uni, mode='loop')
        P_focus, S_focus = gpt.predict(X_focus, mode='loop')
    elapsed = time.time() - t0
    e_uni = P_uni[:, 0] - y_uni; e_focus = P_focus[:, 0] - y_focus
    out = {
        'target': target_name, 'stream': stream, 'config': config + ('_fresh' if refresh else '') + ('_lin' if linear else ''),
        'seed': seed, 'd': d, 'N': N, 'calibrated': bool(calibrate),
        'prequential_nrmse': float(np.sqrt(np.mean(errs[warmup:] ** 2)) / yrange),
        'uniform_nrmse': float(np.sqrt(np.mean(e_uni ** 2)) / yrange),
        'focus_nrmse': float(np.sqrt(np.mean(e_focus ** 2)) / yrange),
        # uncertainty: empirical 1-sigma coverage (target 0.68) and RMS predicted sigma over RMS error
        'coverage_prequential': float(np.mean(np.abs(errs[warmup:]) <= sds[warmup:])),
        'coverage_uniform': float(np.mean(np.abs(e_uni) <= S_uni[:, 0])),
        'coverage_focus': float(np.mean(np.abs(e_focus) <= S_focus[:, 0])),
        'sigma_over_rmse_prequential': float(np.sqrt(np.mean(sds[warmup:] ** 2)) / np.sqrt(np.mean(errs[warmup:] ** 2))),
        'sigma_over_rmse_uniform': float(np.sqrt(np.mean(S_uni[:, 0] ** 2)) / np.sqrt(np.mean(e_uni ** 2))),
        'sigma_over_rmse_focus': float(np.sqrt(np.mean(S_focus[:, 0] ** 2)) / np.sqrt(np.mean(e_focus ** 2))),
        'leaves': len(gpt.root.leaves), 'seconds': round(elapsed, 1),
    }
    if neural_learner is not None:
        out.update(refits=neural_learner.n_refits, fit_seconds=round(neural_learner.fit_seconds, 1))
    if package_learner is not None:
        out.update(refits=package_learner.n_refits, refits_skipped=package_learner.n_skipped,
                   snapshots_alive=len({l._fitted_global.version for l in gpt.root.leaves
                                        if getattr(l, '_fitted_global', None) is not None}))
    if model is not None:
        out.update(refits=model.n_refits, refits_skipped=model.n_skipped, fit_seconds=round(model.t_fit, 1),
                   stale_leaf_refits=_ACTIVE['n_stale_refits'],
                   snapshots_alive=len({l._fitted_mean.version for l in gpt.root.leaves
                                        if getattr(l, '_fitted_mean', None) is not None}))
        if model.current is not None and model.current.damped:
            # mean damping weight of the final snapshot on the two test sets (1 = undamped)
            out.update(confidence_uniform=round(float(model.current.confidence(X_uni).mean()), 3),
                       confidence_focus=round(float(model.current.confidence(X_focus).mean()), 3),
                       damping_ell=round(float(model.current.ref_ell), 3))
    _ACTIVE['model'] = None; _ACTIVE['refresh_stale'] = False
    return out


def summarize(lines):
    import statistics as st
    rows = {}
    for line in lines:
        line = line.strip()
        if not line.startswith('RESULT '):
            continue
        r = json.loads(line[len('RESULT '):])
        rows.setdefault((r['target'], r['stream'], r['config']), []).append(r)
    with_cov = any('coverage_prequential' in r for rs in rows.values() for r in rs)
    if with_cov:
        out = ["| target | stream | config | prequential NRMSE | uniform-test NRMSE | focus-test NRMSE | "
               "coverage prequential / uniform / focus (target 0.68) | sigma/RMSE prequential / uniform / focus | time [s] |",
               "|---|---|---|---|---|---|---|---|---|"]
    else:
        out = ["| target | stream | config | prequential NRMSE | uniform-test NRMSE | focus-test NRMSE | refits | time [s] |",
               "|---|---|---|---|---|---|---|---|"]

    def cell(vals):
        vals = list(vals)
        if len(vals) == 1:
            return f"{vals[0]:.4f}"
        return f"{st.mean(vals):.4f} ({min(vals):.4f}..{max(vals):.4f})"

    for (tgt, strm, cfg), rs in sorted(rows.items()):
        if with_cov:
            cov = " / ".join(f"{st.mean([r[k] for r in rs]):.2f}" for k in
                             ('coverage_prequential', 'coverage_uniform', 'coverage_focus'))
            rat = " / ".join(f"{st.mean([r[k] for r in rs]):.2f}" for k in
                             ('sigma_over_rmse_prequential', 'sigma_over_rmse_uniform', 'sigma_over_rmse_focus'))
            out.append(f"| {tgt} | {strm} | {cfg} | {cell(r['prequential_nrmse'] for r in rs)} | "
                       f"{cell(r['uniform_nrmse'] for r in rs)} | {cell(r['focus_nrmse'] for r in rs)} | {cov} | {rat} | "
                       f"{st.mean([r['seconds'] for r in rs]):.0f} |")
            continue
        refits = f"{st.mean([r.get('refits', 0) for r in rs]):.0f}"
        out.append(f"| {tgt} | {strm} | {cfg} | {cell(r['prequential_nrmse'] for r in rs)} | "
                   f"{cell(r['uniform_nrmse'] for r in rs)} | {cell(r['focus_nrmse'] for r in rs)} | {refits} | "
                   f"{st.mean([r['seconds'] for r in rs]):.0f} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target', default='rotated_rosenbrock', choices=sorted(TARGETS))
    ap.add_argument('--streams', default='uniform,focusing,sweeping,walker')
    ap.add_argument('--configs', default='tree,global',
                    help="comma-separated subset of tree,global,global_pkg,frozen,global_damped,global_damped_tight,"
                         "global_damped_wide; append _fresh (e.g. global_fresh) to refit a leaf on first use after a newer "
                         "global version, and/or _lin (e.g. tree_lin) for a leaf kernel with a linear-trend term")
    ap.add_argument('--seeds', default='1')
    ap.add_argument('--d', type=int, default=6)
    ap.add_argument('--N', type=int, default=4000)
    ap.add_argument('--nbar', type=int, default=100)
    ap.add_argument('--calibrate', action='store_true',
                    help='use_calibrated_sigma=True and report 1-sigma coverage and sigma/RMSE ratios')
    ap.add_argument('--summarize', nargs='*', help='aggregate RESULT lines from these files (or stdin) into a table')
    args = ap.parse_args()

    if args.summarize is not None:
        lines = []
        for f in (args.summarize or ['-']):
            lines += (sys.stdin if f == '-' else open(f)).readlines()
        print(summarize(lines))
        return

    for seed in [int(s) for s in args.seeds.split(',')]:
        for stream in args.streams.split(','):
            for config in args.configs.split(','):
                r = run_one(args.target, stream, config, seed, args.d, args.N, args.nbar, calibrate=args.calibrate)
                print("RESULT " + json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
