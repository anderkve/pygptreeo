"""The nearest neighbours together with the rest of the leaf's own points.

A prototype: a query is conditioned on its `n_neighbours` nearest points and
on every other point of its leaf. The leaf's points count in full, the other
neighbours fade out as in the package. Where the kernels of several leaves are
mixed, the points of each leaf count by its weight against the largest one.
"""
import numpy as np
from pygptreeo import GPTree
from pygptreeo.neighbours import NeighbourSearch


def _rows(a):
    """One comparable item per row."""
    a = np.ascontiguousarray(a, dtype=float)
    return a.view([('', a.dtype)] * a.shape[1]).ravel()


class WithLeafSearch(NeighbourSearch):
    _blend = None           # the leaves and weights of the mixed prediction under way

    def predict_mixed(self, x, leaves, weights):
        self._blend = (leaves, weights)
        try:
            return super().predict_mixed(x, leaves, weights)
        finally:
            self._blend = None

    def find(self, q, w, leaf=None):
        found = super().find(q, w, leaf=leaf)
        if found is None:
            return None
        if self._blend is not None:
            leaves, weights = self._blend
        else:
            if leaf is None and self.tree.leaf_index is not None:
                leaf = self.tree.leaf_index.locate(q)
            if leaf is None:
                return found
            leaves, weights = [leaf], [1.0]
        X, y, sigma, growth = found
        growth = np.zeros(len(X)) if growth is None else growth.copy()
        reach = (((X - q) / w) ** 2).sum(axis=1).max()
        parts, top = [(X, y, sigma, growth)], max(weights)
        for near, weight in zip(leaves, weights):
            share = weight / top
            if near.n_points == 0 or share < 1e-3:
                continue
            fade = (1.0 - share) ** 2 / share ** 2
            points = near.my_X_data
            own = np.isin(_rows(X), _rows(points))
            growth[own] = np.minimum(growth[own], fade)
            beyond = (((points - q) / w) ** 2).sum(axis=1) > reach
            if beyond.any():
                parts.append((points[beyond], near.my_y_data[beyond], near.my_sigma_data[beyond], np.full(beyond.sum(), fade)))
        return tuple(np.concatenate([part[i] for part in parts]) for i in range(4))


def with_leaf(tree):
    """Give the tree's neighbour search the leaf's own points as well."""
    tree.neighbours.__class__ = WithLeafSearch
    return tree


def with_leaf_tree(**kw):
    return with_leaf(GPTree(predict_from='neighbours', **kw))
