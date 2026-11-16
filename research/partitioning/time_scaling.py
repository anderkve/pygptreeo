"""Update and prediction times along one long stream, with one process:

    python time_scaling.py live|snapshot target stream N [variants separated by ;]

At 5000, 20 000, 50 000, ... points: the mean update time since the last
checkpoint and the mean time of 1000 single-point predictions at the points
that follow.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # before numpy is loaded
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if sys.argv[1] == 'live':
    os.environ['PYGPTREEO_LIVE'] = '1'
sys.path.insert(0, HERE)
VARIANTS = sys.argv[5].split(';') if len(sys.argv) > 5 else ('base', 'nbf')
import numpy as np                                              # noqa: E402
import bench                                                    # noqa: E402
from variants import make_model                                 # noqa: E402

target, stream, N = sys.argv[2], sys.argv[3], int(sys.argv[4])
data = bench.make_data(target, stream, N + 1000, 1)
X, y_obs, sigma = data['X'], data['y_obs'], data['sigma']
checkpoints = [n for n in (5000, 20000, 50000, 100000, 200000, 400000) if n <= N]
for v in VARIANTS:
    m = make_model(v, data['d'], 1)
    np.random.seed(1)
    t_upd, last = 0.0, 0
    for i in range(N):
        t0 = time.perf_counter()
        m.update_tree(X[i:i + 1], y_obs[i:i + 1, None], sigma[i:i + 1, None])
        t_upd += time.perf_counter() - t0
        if i + 1 in checkpoints:
            t0 = time.perf_counter()
            P = np.array([m.predict(X[j:j + 1])[0][0, 0] for j in range(i + 1, i + 1001)])
            t_pred = (time.perf_counter() - t0) / 1000
            print(f"{sys.argv[1]:8s} {target} {stream} {v:6s} N={i + 1:7d}  leaves {len(m.root.leaves):5d}  "
                  f"update {1e3 * t_upd / (i + 1 - last):.2f} ms  predict {1e3 * t_pred:.3f} ms  "
                  f"rmse {np.sqrt(np.mean((P - data['y'][i + 1:i + 1001]) ** 2)):.4f}", flush=True)
            t_upd, last = 0.0, i + 1
