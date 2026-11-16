"""Where the time of a neighbour search goes as a tree grows, with one process:

    python profile_search.py live|snapshot target stream N

Builds the default tree and, at checkpoints, times the parts of a search for
the neighbours of 1000 points that follow in the stream.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # before numpy is loaded
import heapq
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if sys.argv[1] == 'live':
    os.environ['PYGPTREEO_LIVE'] = '1'
sys.path.insert(0, HERE)
import numpy as np                                              # noqa: E402
import bench                                                    # noqa: E402
from variants import make_model                                 # noqa: E402
from pygptreeo import leaf_index                                # noqa: E402
from pygptreeo.neighbours import NeighbourSearch                # noqa: E402


class CountingHeap:
    """heapq with a count of the pops."""
    pops = 0
    heappush = staticmethod(heapq.heappush)

    @classmethod
    def heappop(cls, heap):
        cls.pops += 1
        return heapq.heappop(heap)


def clock(f, n=1):
    t0 = time.perf_counter()
    for _ in range(n):
        out = f()
    return (time.perf_counter() - t0) / n, out


target, stream, N = sys.argv[2], sys.argv[3], int(sys.argv[4])
data = bench.make_data(target, stream, N + 1000, 1)
X, y_obs, sigma = data['X'], data['y_obs'], data['sigma']
m = make_model('base', data['d'], 1)
np.random.seed(1)
search = NeighbourSearch(m, n_neighbours=100, max_leaves=16)
for i in range(N):
    m.update_tree(X[i:i + 1], y_obs[i:i + 1, None], sigma[i:i + 1, None])
    if i + 1 in (5000, 20000, 50000, 100000, 200000):
        index = m.leaf_index
        Q = X[i + 1:i + 1001]
        leaves = [index.locate(q) for q in Q]
        keep = [k for k, leaf in enumerate(leaves) if leaf is not None and leaf.fit_std is not None]
        Q, W = Q[keep], [leaves[k].fit_std for k in keep]
        t_locate, _ = clock(lambda: [index.locate(q) for q in Q])
        leaf_index.heapq = CountingHeap
        CountingHeap.pops = 0
        found = [index.nearest(q, w, 16) for q, w in zip(Q, W)]
        groups = CountingHeap.pops / len(Q)
        leaf_index.heapq = heapq
        t_nearest, _ = clock(lambda: [index.nearest(q, w, 16) for q, w in zip(Q, W)])
        t_find, points = clock(lambda: [search.find(q, w) for q, w in zip(Q, W)])
        n_points = np.mean([sum(leaf.n_points for leaf in f) for f in found])
        zero = np.mean([sum(1 for leaf in f if np.all(index.data_box(leaf)[0] <= q) and np.all(q <= index.data_box(leaf)[1]))
                        for f, q in zip(found, Q)])
        t_predict, _ = clock(lambda: [m.predict(q[None, :]) for q in Q])
        print(f"{sys.argv[1]:8s} {target} {stream} N={i + 1:7d} leaves {len(index):5d} depth {index.depth()}  groups opened {groups:5.1f}  "
              f"boxes holding the query {zero:4.1f}  points of the 16 leaves {n_points:6.0f}  locate {1e3 * t_locate / len(Q):.3f} ms  "
              f"nearest {1e3 * t_nearest / len(Q):.3f} ms  find {1e3 * t_find / len(Q):.3f} ms  leaf predict {1e3 * t_predict / len(Q):.3f} ms", flush=True)
