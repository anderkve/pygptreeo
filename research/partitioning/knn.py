"""A tree whose leaves supply the kernel and the calibration, while every query is
predicted from its own nearest stored points instead of from its leaf's points."""
import heapq

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular

from pygptreeo import GPTree
from pygptreeo.gpnode import GPNode


class PointStore:
    """Every point of the stream, in arrays that double in size when full."""

    def __init__(self):
        self.n = 0
        self.X = self.y = self.sigma = None

    def add(self, x, y, sigma):
        x = np.asarray(x, dtype=float).reshape(-1)
        if self.X is None:
            self.X = np.empty((1024, x.size))
            self.y = np.empty(1024)
            self.sigma = np.empty(1024)
        elif self.n == len(self.y):
            self.X = np.vstack((self.X, np.empty_like(self.X)))
            self.y = np.concatenate((self.y, np.empty_like(self.y)))
            self.sigma = np.concatenate((self.sigma, np.empty_like(self.sigma)))
        self.X[self.n] = x
        self.y[self.n] = float(np.asarray(y).reshape(-1)[0])
        self.sigma[self.n] = float(np.asarray(sigma).reshape(-1)[0])
        self.n += 1


def tree_neighbours(root, q, w, k, max_leaves=None):
    """The k stored points nearest to q, found through the tree: (X, y, sigma, leaves visited).

    Distances are Euclidean in the inputs divided by ``w``. Subtrees are visited
    nearest first and skipped once they cannot hold a nearer point than the k-th
    found; with ``max_leaves`` the search stops after that many leaves.
    """
    heap = [(0.0, 0, root)]
    count = 1
    best = np.full(k, np.inf)
    found = []
    visited = 0
    while heap:
        bound, _, node = heapq.heappop(heap)
        if bound >= best[-1]:
            break
        while not node.is_leaf:
            direction = getattr(node, 'split_direction', None)
            if direction is None:
                j = node.split_index
                t = (q[j] - node.split_position) / w[j]
            else:
                t = (q @ direction - node.split_position) / np.sqrt(((direction * w) ** 2).sum())
            near, far = (node.right, node.left) if t > 0 else (node.left, node.right)
            far_bound = max(bound, t * t)
            if far_bound < best[-1]:
                heapq.heappush(heap, (far_bound, count, far))
                count += 1
            node = near
        if node.n_points > 0:
            d2 = (((node.my_X_data - q) / w) ** 2).sum(axis=1)
            found.append((d2, node))
            best = np.sort(np.concatenate((best, d2)))[:k]
            visited += 1
            if max_leaves is not None and visited >= max_leaves:
                break
    radius = best[-1]
    X, y, sigma = [], [], []
    for d2, node in found:
        keep = d2 <= radius
        if keep.any():
            X.append(node.my_X_data[keep]); y.append(node.my_y_data[keep, 0]); sigma.append(node.my_sigma_data[keep, 0])
    return np.vstack(X)[:k], np.concatenate(y)[:k], np.concatenate(sigma)[:k], visited


class KNNNode(GPNode):
    """Class attributes:
        store: the tree's PointStore ('brute' search)
        tree: the GPTree ('tree' search)
        k: number of neighbours a prediction is conditioned on
        metric: 'ard' (inputs in units of the leaf's fitted length scales) or
            'std' (in units of the spread of the leaf's points)
        far: number of the k neighbours taken at geometrically spaced ranks beyond the nearest ('brute' only)
        ls_cap: length scales are capped at this many times the shortest one in the metric
        search: 'brute' (all stored points) or 'tree' (through the tree's cuts)
        max_leaves: with 'tree', stop the search after this many leaves (None: exact)
    """
    store = None
    tree = None
    k = 100
    metric = 'ard'
    far = 0
    ls_cap = 10.0
    search = 'brute'
    max_leaves = None
    _registering = False
    stats = None

    def _conditioning_set(self, q, w):
        if self.search == 'tree':
            k = self.k + 1 if self._registering else self.k
            X, y, sigma, visited = tree_neighbours(self.tree.root, q, w, k, self.max_leaves)
            if self._registering:           # the point itself is already stored in its leaf
                keep = np.any(X != q, axis=1)
                X, y, sigma = X[keep][:self.k], y[keep][:self.k], sigma[keep][:self.k]
            self.stats[0] += 1
            self.stats[1] += visited
            return X, y, sigma
        st = self.store
        d2 = (((st.X[:st.n] - q) / w) ** 2).sum(axis=1)
        k = min(self.k, st.n)
        if self.far <= 0 or st.n <= 2 * k:
            idx = np.argpartition(d2, k - 1)[:k]
        else:
            near = k - self.far
            order = np.argsort(d2)
            ranks = np.unique(np.round(np.geomspace(near, st.n - 1, self.far)).astype(int))
            idx = np.concatenate((order[:near], order[ranks]))
        return st.X[idx], st.y[idx], st.sigma[idx]

    def register_pred_perf(self, x, y, only_if_fitted=False):
        self._registering = True
        try:
            return super().register_pred_perf(x, y, only_if_fitted=only_if_fitted)
        finally:
            self._registering = False

    def predict(self, x, return_std=True, use_calibrated_sigma=False):
        st = self.store
        if (st.n < 10 or not self._is_fitted() or self.X_scaler is None or self.asleep or self.n_outputs != 1):
            return super().predict(x, return_std=return_std, use_calibrated_sigma=use_calibrated_sigma)
        kernel = self.my_GPRs[0].sklearn_gpr.kernel_
        xm, xs = self.X_scaler.mean_, self.X_scaler.scale_
        ym, ysc = self.y_scalers[0].mean_[0], self.y_scalers[0].scale_[0]
        w = xs
        if self.metric == 'ard':
            ls = self.my_GPRs[0].get_length_scales(x.shape[1])
            if ls is not None:
                w = xs * np.minimum(ls / ls.min(), self.ls_cap)
        x = np.asarray(x, dtype=float)
        mu = np.empty((x.shape[0], 1))
        sd = np.empty((x.shape[0], 1))
        for i in range(x.shape[0]):
            Xn, yn, sn = self._conditioning_set(x[i], w)
            Xn = (Xn - xm) / xs
            qs = ((x[i] - xm) / xs)[None, :]
            K = kernel(Xn)
            K[np.diag_indices_from(K)] += (sn / ysc) ** 2
            try:
                L = cholesky(K, lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                K[np.diag_indices_from(K)] += 1e-8
                L = cholesky(K, lower=True, check_finite=False)
            ks = kernel(Xn, qs)[:, 0]
            mu[i, 0] = ks @ cho_solve((L, True), (yn - ym) / ysc, check_finite=False) * ysc + ym
            v = solve_triangular(L, ks, lower=True, check_finite=False)
            sd[i, 0] = np.sqrt(max(kernel.diag(qs)[0] - v @ v, 0.0)) * ysc
        if use_calibrated_sigma:
            sd = sd * self.sigma_scaler
        return mu, sd


class KNNTree(GPTree):
    def update_tree(self, x, y, sigma, allow_training=True):
        super().update_tree(x, y, sigma, allow_training=allow_training)
        self.root.store.add(x, y, sigma)

    def extra_record(self):
        calls, visited = self.root.stats
        return dict(knn_leaves=np.array(visited / max(calls, 1)))


def knn_tree(k=100, metric='ard', far=0, search='brute', max_leaves=None, node_base=KNNNode, **tree_kwargs):
    tree_kwargs.setdefault('append_points_between_fits', False)
    gpt = KNNTree(**tree_kwargs)
    node_class = type('KNNNode_', (node_base,), dict(store=PointStore(), tree=gpt, k=k, metric=metric, far=far,
                                                     search=search, max_leaves=max_leaves, stats=[0, 0]))
    gpt.root.__class__ = node_class
    return gpt


class Reservoir:
    """A uniform sample of at most ``size`` of the points offered to it."""

    def __init__(self, size, d):
        self.size = size
        self.X = np.empty((size, d))
        self.y = np.empty(size)
        self.sigma = np.empty(size)
        self.n = 0          # points held
        self.seen = 0       # points offered

    def add(self, x, y, sigma):
        self.seen += 1
        if self.n < self.size:
            i = self.n
            self.n += 1
        else:
            i = np.random.randint(self.seen)
            if i >= self.size:
                return
        self.X[i], self.y[i], self.sigma[i] = x, y, sigma


class FisheyeNode(KNNNode):
    """Conditions a prediction on the nearest stored points and, for a view of the
    wider region, on a few points from each subtree that branches off the path
    from the query's leaf to the root.

    Class attributes:
        per_level: points taken from each such subtree (those nearest the query)
        reservoir_size: size of the uniform sample that every node keeps of its subtree
        levels: number of levels above the leaf that contribute (None: all)
    """
    per_level = 4
    reservoir_size = 32
    levels = None
    reservoir = None

    def _far_points(self, q, w):
        X, y, sigma = [], [], []
        node, level = self, 0
        while node.parent is not None and (self.levels is None or level < self.levels):
            sibling = node.parent.right if node.is_left else node.parent.left
            res = sibling.reservoir
            if res is not None and res.n > 0:
                m = min(self.per_level, res.n)
                d2 = (((res.X[:res.n] - q) / w) ** 2).sum(axis=1)
                idx = np.argpartition(d2, m - 1)[:m]
                X.append(res.X[idx]); y.append(res.y[idx]); sigma.append(res.sigma[idx])
            node = node.parent
            level += 1
        if not X:
            return None
        return np.vstack(X), np.concatenate(y), np.concatenate(sigma)

    def _conditioning_set(self, q, w):
        far = self._far_points(q, w)
        if far is None:
            return super()._conditioning_set(q, w)
        k_total = self.k
        type(self).k = max(k_total - len(far[1]), 10)
        try:
            Xn, yn, sn = super()._conditioning_set(q, w)
        finally:
            type(self).k = k_total
        # a far point that is also among the nearest is used once
        new = ~(far[0][:, None, :] == Xn[None, :, :]).all(axis=2).any(axis=1)
        return np.vstack((Xn, far[0][new])), np.concatenate((yn, far[1][new])), np.concatenate((sn, far[2][new]))

    def split_training_data(self):
        super().split_training_data()
        for child in self.children:
            child.reservoir = Reservoir(self.reservoir_size, self.my_X_data.shape[1])
            for x, y, s in zip(child.my_X_data, child.my_y_data[:, 0], child.my_sigma_data[:, 0]):
                child.reservoir.add(x, y, s)


class FisheyeTree(KNNTree):
    def update_tree(self, x, y, sigma, allow_training=True):
        super().update_tree(x, y, sigma, allow_training=allow_training)
        # every node on the point's path offers it to its sample
        xq = np.asarray(x, dtype=float).reshape(1, -1)
        yv, sv = float(np.asarray(y).reshape(-1)[0]), float(np.asarray(sigma).reshape(-1)[0])
        node = self.root
        while True:
            if node.reservoir is None:
                node.reservoir = Reservoir(node.reservoir_size, xq.shape[1])
            node.reservoir.add(xq[0], yv, sv)
            if node.is_leaf:
                break
            node = node.children[int(node.prob_func(xq)[0, 0] > 0.5)]


def fisheye_tree(k=100, per_level=4, reservoir_size=32, levels=None, metric='ard', search='brute', max_leaves=None,
                 **tree_kwargs):
    tree_kwargs.setdefault('append_points_between_fits', False)
    gpt = FisheyeTree(**tree_kwargs)
    node_class = type('FisheyeNode_', (FisheyeNode,), dict(
        store=PointStore(), tree=gpt, k=k, metric=metric, search=search, max_leaves=max_leaves, stats=[0, 0],
        per_level=per_level, reservoir_size=reservoir_size, levels=levels))
    gpt.root.__class__ = node_class
    return gpt
