"""Steps of the predicted mean along short straight paths: python jitter.py target stream N [variants separated by ;]

Builds the default tree and the tree that predicts from neighbours, then walks
from stored points to their 200th nearest stored point in 1000 steps. A step
counts as a discontinuity by how far it departs from the mean of the steps on
either side of it, in units of the predicted sigma there.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # before numpy is loaded
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ['PYGPTREEO_LIVE'] = '1'
sys.path.insert(0, HERE)
warnings.simplefilter('ignore')
import numpy as np                                              # noqa: E402
import bench                                                    # noqa: E402
from variants import make_model                                 # noqa: E402

target, stream, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
STEPS, PATHS = 1000, 60
data = bench.make_data(target, stream, N, 1)
X, y_obs, sigma = data['X'], data['y_obs'], data['sigma']
rng = np.random.RandomState(0)
scale = X.std(axis=0)
starts = rng.choice(np.arange(N - 5000, N), PATHS, replace=False)
paths = []
for a in starts:
    dist = (((X - X[a]) / scale) ** 2).sum(axis=1)
    b = np.argpartition(dist, 200)[200]
    paths.append(X[a] + np.linspace(0.0, 1.0, STEPS + 1)[:, None] * (X[b] - X[a]))
for v in (sys.argv[4].split(';') if len(sys.argv) > 4 else ('base', 'nbf')):
    m = make_model(v, data['d'], 1)
    np.random.seed(1)
    for i in range(N):
        m.update_tree(X[i:i + 1], y_obs[i:i + 1, None], sigma[i:i + 1, None])
    counts = {0.01: [], 0.1: [], 1.0: []}
    total, largest, leaves_crossed = [], [], []
    for p in paths:
        mu, sd = m.predict(p)
        mu, sd = mu[:, 0], sd[:, 0]
        step = np.diff(mu)
        # what a step adds to the mean of its two neighbours, in sigma
        jump = np.abs(step[1:-1] - 0.5 * (step[:-2] + step[2:])) / sd[1:-2]
        for level in counts:
            counts[level].append(np.sum(jump > level))
        total.append(jump[jump > 0.01].sum())
        largest.append(jump.max())
        located = [m.leaf_index.locate(q) for q in p[::10]]
        leaves_crossed.append(len({id(leaf) for leaf in located if leaf is not None}))
    print(f"{target} {stream} N={N} {v:16s} leaves on a path {np.mean(leaves_crossed):4.1f}   steps off by more than "
          + "  ".join(f"{level} sigma: {np.mean(c):6.1f}" for level, c in counts.items())
          + f"   sum of them {np.mean(total):6.2f} sigma (median {np.median(total):5.2f})   largest {np.median(largest):5.2f} sigma (worst path {np.max(largest):6.1f})",
          flush=True)
