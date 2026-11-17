"""Leaves that sleep by default with predictions from neighbours.

A leaf's GPs keep their kernel and points but release the factor of their
kernel matrix (``GPNode.sleep``) unless the leaf is among the ``keep`` leaves
that last fitted, or were last woken, for a point or for a prediction from
their own points.

    python sleepy.py target stream N [keep values separated by ,]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # before numpy is loaded
import sys
import time
import warnings
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ['PYGPTREEO_LIVE'] = '1'
sys.path.insert(0, HERE)
warnings.simplefilter('ignore')
import numpy as np                                              # noqa: E402
import bench                                                    # noqa: E402
from pygptreeo import GPTree                                    # noqa: E402
from pygptreeo.gpnode import GPNode                             # noqa: E402

BASE = dict(Nbar=100, theta=1e-4, retrain_every_n_points=25, splitting_strategy='gradual')
WAKES = [0]


class SleepyTree(GPTree):
    keep = None             # number of leaves left awake; None for all

    def _trim(self):
        while self.keep is not None and len(self._awake) > self.keep:
            leaf, _ = self._awake.popitem(last=False)
            if leaf.is_leaf:
                leaf.sleep()

    def update_tree(self, *args, **kwargs):
        super().update_tree(*args, **kwargs)
        self._trim()

    def predict(self, *args, **kwargs):
        out = super().predict(*args, **kwargs)
        self._trim()
        return out


def _used(node):
    tree = getattr(node.neighbours, 'tree', None)
    if isinstance(tree, SleepyTree):
        tree._awake[node] = None
        tree._awake.move_to_end(node)


_wake, _fit = GPNode.wake, GPNode.fit_my_GPR


def wake(self):
    WAKES[0] += self.asleep
    _wake(self)
    _used(self)


def fit_my_GPR(self, *args, **kwargs):
    out = _fit(self, *args, **kwargs)
    _used(self)
    return out


GPNode.wake, GPNode.fit_my_GPR = wake, fit_my_GPR

if __name__ == '__main__':
    target, stream, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
    keeps = [None if k == 'all' else int(k) for k in (sys.argv[4].split(',') if len(sys.argv) > 4 else ['all', '1', '8', '32'])]
    data = bench.make_data(target, stream, N + 1000, 1)
    X, y_obs, sigma = data['X'], data['y_obs'], data['sigma']
    for keep in keeps:
        np.random.seed(1)
        m = GPTree(predict_from='neighbours', **BASE)
        m.__class__ = SleepyTree
        m.keep, m._awake = keep, OrderedDict()
        WAKES[0] = 0
        t0 = time.perf_counter()
        for i in range(N):
            m.update_tree(X[i:i + 1], y_obs[i:i + 1, None], sigma[i:i + 1, None])
        t_upd, wakes_upd = (time.perf_counter() - t0) / N, WAKES[0]
        leaves = m.root.leaves
        memory, asleep = m.memory_mb, sum(leaf.asleep for leaf in leaves)
        fitted = sum(leaf.nbytes_fitted() for leaf in leaves) / 1e6
        own = np.mean([leaf.prefers_own for leaf in leaves])
        WAKES[0] = 0
        t0 = time.perf_counter()
        P = np.array([m.predict(X[i:i + 1])[0][0, 0] for i in range(N, N + 1000)])
        t_pred = (time.perf_counter() - t0) / 1000
        print(f"{target} {stream} N={N} awake {'all' if keep is None else keep:>3}  update {1e3 * t_upd:.2f} ms  predict {1e3 * t_pred:.3f} ms  "
              f"rmse {np.sqrt(np.mean((P - data['y'][N:N + 1000]) ** 2)):.5f}  memory {memory:6.1f} MB (GPs {fitted:6.1f})  leaves {len(leaves)} asleep {asleep}  "
              f"wakes per 1000 updates {1e3 * wakes_upd / N:.0f}, per 1000 predictions {WAKES[0]}  leaves that prefer their own GP {100 * own:.0f}%", flush=True)
