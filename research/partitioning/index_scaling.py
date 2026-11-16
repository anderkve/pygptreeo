"""The search for the nearest leaves as the number of leaves grows, without GPs:

    python index_scaling.py d N [uniform|walker]

Leaves of 100 points are cut at the median along a random input. Prints, per
checkpoint, the time of ``LeafIndex.nearest(q, w, 16)`` and the groups it opens.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # before numpy is loaded
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.environ.get('PACKAGE', os.path.join('..', '..'))))    # or a folder with a copy
from pygptreeo import leaf_index                                # noqa: E402

NBAR = 100


class Leaf:
    def __init__(self, d, X=None):
        self.n_features = d
        self.X = np.empty((NBAR, d))
        self.n_points = 0
        if X is not None:
            self.n_points = len(X)
            self.X[:len(X)] = X

    @property
    def my_X_data(self):
        return self.X[:self.n_points]


class CountingHeap:
    """heapq with a count of the pops."""
    pops = 0

    @staticmethod
    def heappush(heap, item):
        import heapq
        heapq.heappush(heap, item)

    @classmethod
    def heappop(cls, heap):
        import heapq
        cls.pops += 1
        return heapq.heappop(heap)


d, N = int(sys.argv[1]), int(sys.argv[2])
kind = sys.argv[3] if len(sys.argv) > 3 else 'uniform'
if len(sys.argv) > 4:                   # a module with another way to divide a group
    import importlib.util
    spec = importlib.util.spec_from_file_location('alt', sys.argv[4])
    alt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(alt)
    leaf_index.LeafIndex._divide = alt.divide_min_overlap
rng = np.random.RandomState(1)
if kind == 'uniform':
    X = rng.rand(N, d)
else:                                   # a random walk reflected into the unit cube
    X = np.cumsum(0.01 * rng.randn(N, d), axis=0) + 0.5
    X = np.abs(np.mod(X, 2.0) - 1.0)
X = X * np.logspace(0, float(os.environ.get('DECADES', 0)), d)    # inputs in different units
index = leaf_index.LeafIndex(d)
root = Leaf(d)
index.add(root)
checkpoints = [n for n in (5000, 20000, 100000, 400000, 1000000) if n <= N]
for i in range(N):
    x = X[i]
    leaf = index.locate(x)
    leaf.X[leaf.n_points] = x
    leaf.n_points += 1
    index.grow(leaf, x)
    if leaf.n_points == NBAR:
        j = rng.randint(d)
        position = np.median(leaf.X[:, j])
        low = leaf.X[:, j] <= position
        left, right = Leaf(d, leaf.X[low]), Leaf(d, leaf.X[~low])
        index.split(leaf, left, right, j, position, 0.0)
    if i + 1 in checkpoints:
        Q = X[max(0, i - 20000):i + 1][rng.randint(0, min(i + 1, 20000), 500)] + 0.002 * rng.randn(500, d) * np.logspace(0, float(os.environ.get('DECADES', 0)), d)
        w = np.array([index.locate(q).my_X_data.std(axis=0) + 1e-12 if index.locate(q) is not None else np.ones(d) for q in Q])
        leaf_index.heapq = CountingHeap
        CountingHeap.pops = 0
        t0 = time.perf_counter()
        for q, wq in zip(Q, w):
            index.nearest(q, wq, 16)
        t = (time.perf_counter() - t0) / len(Q)
        import heapq
        leaf_index.heapq = heapq
        t0 = time.perf_counter()
        for q, wq in zip(Q, w):
            index.nearest(q, wq, 16)
        t = (time.perf_counter() - t0) / len(Q)
        t0 = time.perf_counter()
        for q in Q:
            index.locate(q)
        t_loc = (time.perf_counter() - t0) / len(Q)
        print(f"d={d:2d} {kind:8s} N={i + 1:8d}  leaves {len(index):6d}  index depth {index.depth()}  "
              f"groups opened {CountingHeap.pops / len(Q):6.1f}  nearest {1e3 * t:.3f} ms  locate {1e3 * t_loc:.3f} ms", flush=True)
