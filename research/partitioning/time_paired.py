"""Two settings timed side by side, point by point, so that the speed of the machine cancels:

    python time_paired.py target stream N variant_a variant_b

Each point is given to both trees, in alternating order; then both predict
1000 points that follow in the stream, alternating too.
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
from variants import make_model                                 # noqa: E402

target, stream, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
names = sys.argv[4:6]
data = bench.make_data(target, stream, N + 1000, 1)
X, y_obs, sigma = data['X'], data['y_obs'], data['sigma']
models, states = [], []
for name in names:
    np.random.seed(1)
    models.append(make_model(name, data['d'], 1))
    states.append(np.random.get_state())
t_upd, t_pred = [0.0, 0.0], [0.0, 0.0]
clock = time.perf_counter
for i in range(N):
    for k in ((0, 1) if i % 2 == 0 else (1, 0)):
        np.random.set_state(states[k])          # each tree with its own random numbers, whatever the order
        t0 = clock()
        models[k].update_tree(X[i:i + 1], y_obs[i:i + 1, None], sigma[i:i + 1, None])
        t_upd[k] += clock() - t0
        states[k] = np.random.get_state()
predictions = [np.empty(1000), np.empty(1000)]
for i in range(1000):
    for k in ((0, 1) if i % 2 == 0 else (1, 0)):
        t0 = clock()
        predictions[k][i] = models[k].predict(X[N + i:N + i + 1])[0][0, 0]
        t_pred[k] += clock() - t0
for k in (0, 1):
    m = models[k]
    leaves = m.root.leaves
    print(f"{target} {stream} N={N} {names[k]:28s} update {1e3 * t_upd[k] / N:.3f} ms  predict {t_pred[k]:.3f} ms  memory {m.memory_mb:5.1f} MB  "
          f"leaves {len(leaves)} asleep {sum(leaf.asleep for leaf in leaves)}  rmse {np.sqrt(np.mean((predictions[k] - data['y'][N:N + 1000]) ** 2)):.5f}", flush=True)
print(f"   {names[1]} / {names[0]}: update {t_upd[1] / t_upd[0]:.3f}  predict {t_pred[1] / t_pred[0]:.3f}", flush=True)
