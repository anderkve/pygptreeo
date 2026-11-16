"""Benchmark harness for the partitioning study: data, one prequential run, records.

A model is anything with ``predict(X) -> (mean, sigma)`` (arrays of shape (n, 1))
and ``update_tree(x, y, sigma)``. Each run stores its per-point predictions, so
any metric can be computed afterwards (``report.py``).
"""
import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "examples"))
# long batches run against a frozen copy of the package, if there is one
if os.path.isdir(os.path.join(HERE, "snapshot", "pygptreeo")) and not os.environ.get("PYGPTREEO_LIVE"):
    sys.path.insert(0, os.path.join(HERE, "snapshot"))
import target_functions as tf                                   # noqa: E402
from target_functions_physics import TARGETS as PHYSICS         # noqa: E402

from sklearn.exceptions import ConvergenceWarning               # noqa: E402
warnings.simplefilter("ignore", category=ConvergenceWarning)

RESULTS = os.path.join(HERE, "results")


def AnisoChirp(x):
    """Rough along the first input, smooth along the others: 5 + sin(2 pi 6 x_0^2) + sum_i>0 x_i^2."""
    x = np.asarray(x, dtype=float)
    return 5.0 + np.sin(2 * np.pi * 6.0 * x[0] ** 2) + np.sum(x[1:] ** 2, axis=0)


def DiagonalRidge(x):
    """A narrow ridge and a kink on oblique hyperplanes: no axis-aligned cut follows either."""
    x = np.asarray(x, dtype=float)
    d = x.shape[0]
    u = (x[0] + x[1] - 1.0) / np.sqrt(2.0)
    w = np.ones(d) / np.sqrt(d)
    t = np.tensordot(w, x - 0.5, axes=1)
    return 3.0 + np.exp(-0.5 * (u / 0.04) ** 2) + 2.0 * np.abs(t) + 0.5 * np.sum((x - 0.5) ** 2, axis=0)


ANALYTIC = {
    'eggholder': (tf.Eggholder, 3), 'rotated_rosenbrock': (tf.RotatedRosenbrock, 6),
    'gaussian_peaks': (tf.GaussianPeaks, 6), 'active_peaks': (tf.ActiveSubspacePeaks, 10),
    'michalewicz': (tf.Michalewicz, 5), 'ackley': (tf.Ackley, 6), 'griewank': (tf.Griewank, 6),
    'step_ridge': (tf.StepRidge, 6), 'chirp': (tf.Chirp, 4), 'rastrigin': (tf.Rastrigin, 4),
    'aniso_chirp': (AnisoChirp, 3), 'diagonal_ridge': (DiagonalRidge, 4),
}
ALL_TARGETS = sorted(ANALYTIC) + sorted(PHYSICS)


def target_dim(name):
    return PHYSICS[name].d if name in PHYSICS else ANALYTIC[name][1]


def make_data(target, stream, N, seed, n_test=2000):
    """Stream inputs with the noise-free target, the observations and their declared
    sigma, and two test sets: uniform points, and points held out of the stream
    itself (same distribution in space and along the stream)."""
    rng = np.random.RandomState(seed)
    n_hist = min(n_test, N // 5)
    if target in PHYSICS:
        t = PHYSICS[target]
        d = t.d

        def by_column(x):
            x = np.asarray(x)
            return t(x.T) if x.ndim > 1 else float(t(x)[0])
        X_all = tf.make_stream(stream, by_column, d, N + n_hist, rng)
        y_obs_all, sigma_all, y_all = t.sample(X_all, rng)
        X_uni = rng.rand(n_test, d)
        y_uni = t(X_uni)
    else:
        f, d = ANALYTIC[target]
        X_all = tf.make_stream(stream, f, d, N + n_hist, rng)
        y_all = f(X_all.T)
        y_obs_all = y_all
        sigma_all = np.maximum(1e-3 * np.abs(y_all), 1e-6)
        X_uni = rng.rand(n_test, d)
        y_uni = f(X_uni.T)
    held = np.zeros(N + n_hist, dtype=bool)
    held[rng.choice(N + n_hist, n_hist, replace=False)] = True
    return dict(X=X_all[~held], y=y_all[~held], y_obs=y_obs_all[~held], sigma=sigma_all[~held],
                X_uni=X_uni, y_uni=y_uni, X_hist=X_all[held], y_hist=y_all[held], d=d)


def n_leaves(model):
    if hasattr(model, 'n_leaves'):
        return model.n_leaves
    return len(model.root.leaves)


def run_stream(model, data, seed, checkpoint=1000):
    """Predict each stream point, then update with it. Returns the record of the run."""
    np.random.seed(seed)
    X, y_obs, sigma = data['X'], data['y_obs'], data['sigma']
    N = len(X)
    pred, std, t_pred, t_upd = (np.empty(N) for _ in range(4))
    leaves, depth, searched = [], [], []
    clock = time.perf_counter
    for i in range(N):
        xi = X[i:i + 1]
        t0 = clock()
        mu, sd = model.predict(xi)
        t1 = clock()
        model.update_tree(xi, y_obs[i:i + 1, None], sigma[i:i + 1, None])
        t2 = clock()
        pred[i], std[i], t_pred[i], t_upd[i] = mu[0, 0], sd[0, 0], t1 - t0, t2 - t1
        if (i + 1) % checkpoint == 0:
            leaves.append(n_leaves(model))
            if hasattr(model, 'root'):
                names = [len(leaf.name) - 1 for leaf in model.root.leaves]
                depth.append((max(names), float(np.mean(names))))
            search = getattr(model, 'neighbours', None)
            if search is not None:
                searched.append((search.n_searches, search.n_leaves_searched))
    rec = dict(pred=pred, std=std, y=data['y'], t_pred=t_pred, t_upd=t_upd, leaves=np.array(leaves),
               depth=np.array(depth), searched=np.array(searched))
    for name in ('uni', 'hist'):
        t0 = clock()
        mu, sd = model.predict(data['X_' + name])
        rec['t_' + name] = (clock() - t0) / len(data['X_' + name])
        rec['pred_' + name], rec['std_' + name], rec['y_' + name] = mu[:, 0], sd[:, 0], data['y_' + name]
    if hasattr(model, 'extra_record'):
        rec.update(model.extra_record())
    return rec


def record_path(variant, target, stream, N, seed):
    return os.path.join(RESULTS, variant, f"{target}__{stream}__N{N}__s{seed}.npz")


def run_one(variant, target, stream, N, seed, overwrite=False):
    """Run one (variant, target, stream, N, seed) and save the record; skip if it exists."""
    from variants import make_model
    path = record_path(variant, target, stream, N, seed)
    if os.path.exists(path) and not overwrite:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = make_data(target, stream, N, seed)
    np.random.seed(seed)
    model = make_model(variant, data['d'], seed)
    t0 = time.time()
    rec = run_stream(model, data, seed)
    rec['wall'] = time.time() - t0
    np.savez_compressed(path, **rec)
    return path
