"""Cost along a steadily drifting stream: update and prediction time per window.

    python drift_bench.py live|snapshot leaf|neighbours [N] [Nbar]
"""
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if sys.argv[1] == 'snapshot':
    sys.path.insert(0, os.path.join(HERE, 'snapshot'))
else:
    sys.path.insert(0, os.path.join(HERE, '..', '..'))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np                                              # noqa: E402
from pygptreeo import GPTree                                     # noqa: E402

warnings.simplefilter('ignore')
mode = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
Nbar = int(sys.argv[4]) if len(sys.argv) > 4 else 100
d = 4
rng = np.random.RandomState(0)
X = 0.5 + 0.03 * rng.randn(N, d)
X[:, 0] = np.arange(N) * (3.0 / 12000) + 0.03 * rng.randn(N)          # the cluster moves by its width every 120 points
y = np.sin(6 * X[:, 0]) + X[:, 1] ** 2 + 0.3 * X[:, 2] * X[:, 3]
np.random.seed(0)
gpt = GPTree(Nbar=Nbar, retrain_every_n_points=25, predict_from=mode)
t_pred, t_upd, err = [], [], []
clock = time.perf_counter
window = N // 6
for i in range(N):
    t0 = clock()
    mu, sd = gpt.predict(X[i:i + 1])
    t1 = clock()
    gpt.update_tree(X[i:i + 1], y[i:i + 1, None], np.array([[1e-4]]))
    t2 = clock()
    t_pred.append(t1 - t0); t_upd.append(t2 - t1); err.append(mu[0, 0] - y[i])
    if (i + 1) % window == 0:
        depths = [len(leaf.name) - 1 for leaf in gpt.root.leaves]
        print(f"{sys.argv[1]:8s} {mode:10s} N={i + 1:6d}  leaves {len(depths):4d}  depth of the cuts: mean {np.mean(depths):6.1f} max {max(depths):4d}"
              f"   predict {1e3 * np.mean(t_pred[-window:]):.3f} ms   update {1e3 * np.mean(t_upd[-window:]):.3f} ms"
              f"   rmse {np.sqrt(np.mean(np.square(err[-window:]))):.2e}", flush=True)
