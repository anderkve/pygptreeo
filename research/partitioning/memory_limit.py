"""A memory limit with predictions from the leaves and from neighbours:

    python memory_limit.py target stream N leaf|neighbours [limits in MB separated by ,; 'none' for no limit]

Prints the largest size the tree reported along the stream, what is left of
the points and leaves at the end, and the error of the predictions made
before each point of the last quarter.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # before numpy is loaded
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ['PYGPTREEO_LIVE'] = '1'
sys.path.insert(0, HERE)
warnings.simplefilter('ignore')
import numpy as np                                              # noqa: E402
import bench                                                    # noqa: E402
from pygptreeo import GPTree                                    # noqa: E402

BASE = dict(Nbar=100, theta=1e-4, retrain_every_n_points=25, splitting_strategy='gradual')
target, stream, N, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
limits = [None if v == 'none' else float(v) for v in (sys.argv[5].split(',') if len(sys.argv) > 5 else ['none'])]
data = bench.make_data(target, stream, N, 1)
X, y_obs, sigma, y = data['X'], data['y_obs'], data['sigma'], data['y']
for limit in limits:
    np.random.seed(1)
    m = GPTree(predict_from=mode, max_memory_mb=limit, **BASE)
    pred, sd, largest, true_largest = np.empty(N), np.empty(N), 0.0, 0.0
    t_pred = t_upd = 0.0
    for i in range(N):
        t0 = time.perf_counter()
        mu, s = m.predict(X[i:i + 1])
        t1 = time.perf_counter()
        m.update_tree(X[i:i + 1], y_obs[i:i + 1, None], sigma[i:i + 1, None])
        t2 = time.perf_counter()
        pred[i], sd[i] = mu[0, 0], s[0, 0]
        t_pred += t1 - t0
        t_upd += t2 - t1
        if i % 200 == 199:
            largest = max(largest, m.memory_mb)
            true_largest = max(true_largest, (m._fixed_size() + sum(m._leaf_size(leaf)[0] for leaf in m.root.leaves)) / 1e6)
    leaves = m.root.leaves
    k = 3 * N // 4
    err = np.abs(pred[k:] - y[k:])
    print(f"{target} {stream} N={N} {mode:10s} limit {'none' if limit is None else limit:>5}  largest size reported {largest:5.1f} MB, counted {true_largest:5.1f} MB"
          f"  leaves {len(leaves):4d} asleep {sum(leaf.asleep for leaf in leaves):4d}  points kept {sum(leaf.n_points for leaf in leaves):6d}"
          f"  rmse {np.sqrt(np.mean(err ** 2)):.5f}  cover {np.mean(err <= sd[k:]):.3f}  predict {1e3 * t_pred / N:.3f} ms  update {1e3 * t_upd / N:.2f} ms", flush=True)
