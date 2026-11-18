"""Two settings timed side by side, point by point, so that the speed of the machine cancels:

    python time_paired.py target stream N variant_a variant_b

Each point is given to both trees, in alternating order; then both predict
1000 points that follow in the stream and 1000 uniform ones, alternating
too. Counted for the predictions: how many needed the own GP of a leaf, and
how many of those leaves were asleep.
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
from pygptreeo.gpnode import GPNode                             # noqa: E402

counts = {}         # per neighbour search: [own GPs asked, of them asleep]
_predict_own = GPNode._predict_own


def _counted(self, x, return_std=True):
    count = counts.setdefault(id(self.neighbours), [0, 0])
    count[0] += 1
    count[1] += bool(self.asleep)
    return _predict_own(self, x, return_std)


GPNode._predict_own = _counted
queries = {'stream': (X[N:N + 1000], data['y'][N:N + 1000]), 'uniform': (data['X_uni'][:1000], data['y_uni'][:1000])}
report = [[], []]
for kind, (Q, truth) in queries.items():
    counts.clear()
    t_pred, predictions = [0.0, 0.0], [np.empty(len(Q)), np.empty(len(Q))]
    for i in range(len(Q)):
        for k in ((0, 1) if i % 2 == 0 else (1, 0)):
            t0 = clock()
            predictions[k][i] = models[k].predict(Q[i:i + 1])[0][0, 0]
            t_pred[k] += clock() - t0
    for k in (0, 1):
        own, asleep = counts.get(id(models[k].neighbours), (0, 0))
        report[k].append(f"{kind}: {1e3 * t_pred[k] / len(Q):.3f} ms, own GP in {100 * own / len(Q):.1f}% (asleep {100 * asleep / len(Q):.1f}%), "
                         f"rmse {np.sqrt(np.mean((predictions[k] - truth) ** 2)):.5f}")
    report[0].append(None)
    report[1].append(t_pred[1] / t_pred[0])
GPNode._predict_own = _predict_own
for k in (0, 1):
    m = models[k]
    leaves = m.root.leaves
    print(f"{target} {stream} N={N} {names[k]:28s} update {1e3 * t_upd[k] / N:.3f} ms  memory {m.memory_mb:5.1f} MB  "
          f"leaves {len(leaves)} asleep {sum(leaf.asleep for leaf in leaves)}  predictions " + "; ".join(report[k][0::2]), flush=True)
print(f"   {names[1]} / {names[0]}: update {t_upd[1] / t_upd[0]:.3f}  predict stream {report[1][1]:.3f}  uniform {report[1][3]:.3f}", flush=True)
