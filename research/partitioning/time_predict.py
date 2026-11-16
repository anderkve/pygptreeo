"""Update time along N points and prediction time after them:

    python time_predict.py live|snapshot target stream N [variants separated by ;]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if sys.argv[1] == 'live':
    os.environ['PYGPTREEO_LIVE'] = '1'
sys.path.insert(0, HERE)
VARIANTS = sys.argv[5].split(';') if len(sys.argv) > 5 else ('base', 'nb')
import numpy as np                                              # noqa: E402
import bench                                                    # noqa: E402
import pygptreeo                                                # noqa: E402
from variants import make_model                                 # noqa: E402

target, stream, N = sys.argv[2], sys.argv[3], int(sys.argv[4])
data = bench.make_data(target, stream, N + 1000, 1)
X, y_obs, sigma = data['X'], data['y_obs'], data['sigma']
for v in VARIANTS:
    m = make_model(v, data['d'], 1)
    np.random.seed(1)
    t0 = time.perf_counter()
    for i in range(N):
        m.update_tree(X[i:i + 1], y_obs[i:i + 1, None], sigma[i:i + 1, None])
    t_upd = (time.perf_counter() - t0) / N
    if m.neighbours is not None:
        m.neighbours.n_searches = m.neighbours.n_leaves_searched = 0
    t0 = time.perf_counter()
    P = np.array([m.predict(X[i:i + 1])[0][0, 0] for i in range(N, N + 1000)])
    t_pred = (time.perf_counter() - t0) / 1000
    extra = f"  leaves searched {m.neighbours.n_leaves_searched / max(m.neighbours.n_searches, 1):5.1f}" if m.neighbours is not None else ''
    print(f"{sys.argv[1]:8s} {target} {stream} N={N} {v:22s} predict {1e3 * t_pred:.3f} ms  update {1e3 * t_upd:.2f} ms  "
          f"rmse {np.sqrt(np.mean((P - data['y'][N:N + 1000]) ** 2)):.4f}  leaves {len(m.root.leaves)}{extra}", flush=True)
