"""Replay: a leaf chooses between its own GP and the neighbours by how each
predicted the points it received lately.

For each query of a stream (from `start` on): the leaf's name, the truth, the
leaf's own prediction and that of its kernel on the 100 nearest stored points
with their consistency. `simulate` then applies a rule leaf by leaf.

    python diag_select.py run dev 5000        # the replays
    python diag_select.py report 5000         # the rules
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # before numpy is loaded
import glob
import itertools
import os
import sys
from collections import deque
from multiprocessing import Pool

import numpy as np
from scipy.linalg.lapack import dpotrf, dtrtrs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench                                                    # noqa: E402
from run_matrix import SETS                                      # noqa: E402

FOLDER = os.path.join(bench.RESULTS, '_diags')


def replay(args):
    from pygptreeo import GPTree
    from variants import BASE
    target, stream, N, seed = args
    path = os.path.join(FOLDER, f"{target}__{stream}__N{N}__s{seed}.npz")
    if os.path.exists(path):
        return args
    k = 100
    data = bench.make_data(target, stream, N, seed)
    X, y, y_obs, sigma = data['X'], data['y'], data['y_obs'], data['sigma']
    np.random.seed(seed)
    gpt = GPTree(**BASE)
    rows, names = [], []
    for i in range(N):
        xi = X[i:i + 1]
        if i > 2 * k:
            leaf = gpt._leaf_of(xi)
            if leaf._is_fitted() and leaf.X_scaler is not None:
                mu_l, sd_l = leaf.predict(xi, return_std=True, use_calibrated_sigma=False)
                kernel = leaf.my_GPRs[0].sklearn_gpr.kernel_
                xm, xs = leaf.X_scaler.mean_, leaf.X_scaler.scale_
                ym, ysc = leaf.y_scalers[0].mean_[0], leaf.y_scalers[0].scale_[0]
                d2 = (((X[:i] - xi[0]) / xs) ** 2).sum(axis=1)
                idx = np.argpartition(d2, k - 1)[:k]
                K_all = kernel(np.vstack(((X[idx] - xm) / xs, (xi - xm) / xs)))
                K = K_all[:k, :k].copy()
                K.flat[::k + 1] += (sigma[idx] / ysc) ** 2
                L, info = dpotrf(K, lower=True, overwrite_a=True)
                if info == 0:
                    solved, _ = dtrtrs(L, np.column_stack(((y_obs[idx] - ym) / ysc, K_all[:k, k:])), lower=True)
                    v, u = solved[:, 0], solved[:, 1]
                    rows.append((i, y[i], y_obs[i], mu_l[0, 0], sd_l[0, 0], u @ v * ysc + ym,
                                 np.sqrt(max(K_all[k, k] - u @ u, 0.0)) * ysc, v @ v / k))
                    names.append(leaf.name)
        gpt.update_tree(xi, y_obs[i:i + 1, None], sigma[i:i + 1, None])
    fields = ('index', 'y', 'y_obs', 'leaf', 'leaf_sd', 'knn', 'knn_sd', 'chi2')
    os.makedirs(FOLDER, exist_ok=True)
    np.savez(path, name=np.array(names), **{f: np.array([r[j] for r in rows]) for j, f in enumerate(fields)})
    return args


def simulate(d, window=25, margin=1.0, minimum=10, limit=10.0, tail=0.5):
    """Errors of the rule over the last `tail` of the queries: a leaf predicts
    with its own GP while the mean squared error of its own GP over the last
    `window` points it received is below `margin` squared times that of the
    neighbours. Children start with their parent's record."""
    records = {}
    n = len(d['y'])
    chosen = np.empty(n)
    own_used = np.zeros(n, dtype=bool)
    for i in range(n):
        name = str(d['name'][i])
        if name not in records:
            parent = name[:-1]
            while parent and parent not in records:
                parent = parent[:-1]
            records[name] = deque(records[parent], maxlen=window) if parent in records else deque(maxlen=window)
        rec = records[name]
        nb = d['knn'][i] if d['chi2'][i] <= limit else d['leaf'][i]
        use_own = False
        if len(rec) >= minimum:
            e_nb, e_own = np.mean([r[0] for r in rec]), np.mean([r[1] for r in rec])
            use_own = e_own < margin ** 2 * e_nb
        chosen[i] = d['leaf'][i] if use_own else nb
        own_used[i] = use_own
        # the point arrives: both predictions are scored against its observed value
        rec.append(((nb - d['y_obs'][i]) ** 2, (d['leaf'][i] - d['y_obs'][i]) ** 2))
    t = int(n * (1 - tail))
    err = lambda p: np.sqrt(np.mean((p[t:] - d['y'][t:]) ** 2))
    guarded = np.where(d['chi2'] <= limit, d['knn'], d['leaf'])
    return err(chosen) / err(d['leaf']), err(guarded) / err(d['leaf']), own_used[t:].mean()


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        targets = SETS[sys.argv[2]] if sys.argv[2] in SETS else sys.argv[2].split(',')
        N = int(sys.argv[3])
        streams = sys.argv[4].split(',') if len(sys.argv) > 4 else ('uniform', 'walker', 'sweeping')
        procs = int(sys.argv[5]) if len(sys.argv) > 5 else 4
        jobs = [(t, s, N, 1) for t, s in itertools.product(targets, streams)]
        with Pool(procs) as pool:
            for n, a in enumerate(pool.imap_unordered(replay, jobs), 1):
                print(n, len(jobs), a, flush=True)
    else:
        N = int(sys.argv[2])
        runs = {}
        for path in sorted(glob.glob(os.path.join(FOLDER, f"*__N{N}__s*.npz"))):
            target, stream = os.path.basename(path)[:-4].split('__')[:2]
            runs[(target, stream)] = dict(np.load(path))
        g = lambda x: float(np.exp(np.mean(np.log(x))))
        print(f"{len(runs)} runs at N={N}; errors over the second half, relative to the leaf's own GP")
        for window, margin, minimum in [(25, 1.0, 10), (25, 0.9, 10), (25, 0.8, 10), (25, 0.7, 10), (50, 1.0, 10),
                                        (50, 0.9, 10), (50, 0.8, 10), (100, 0.9, 20), (10, 0.8, 5)]:
            out = {key: simulate(d, window, margin, minimum) for key, d in runs.items()}
            r = np.array([v[0] for v in out.values()])
            base = np.array([v[1] for v in out.values()])
            print(f"window {window:3d} margin {margin:.1f}: {g(r):.3f} (neighbours with the test alone {g(base):.3f}), worst {r.max():.2f} "
                  f"(alone {base.max():.2f}), runs more than 2% worse {int(np.sum(r > 1.02))} (alone {int(np.sum(base > 1.02))}), "
                  f"own GP for {100 * np.mean([v[2] for v in out.values()]):.0f}% of the queries")
        if len(runs) <= 12:
            out = {key: simulate(d, 25, 0.9, 10) for key, d in runs.items()}
            for key, v in out.items():
                print(f"   {key[0]:24s} {key[1]:9s} rule {v[0]:.3f}  test alone {v[1]:.3f}  own GP {100 * v[2]:.0f}%")
