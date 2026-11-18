"""Neighbours that fade out towards the edge of their set: the prototype of
the package's NEIGHBOUR_FADE, for a copy of the package from before it (``NeighbourSearch.find`` returned three arrays then).

The noise of a neighbour grows with its distance from the query, relative to
the distance of the first stored point that is not a neighbour, and is
infinite there: a point enters or leaves the set of neighbours with no
weight, so the prediction is continuous in the query while its leaf stays
the same.
"""
import numpy as np

from pygptreeo.neighbours import NeighbourSearch


class TaperedSearch(NeighbourSearch):
    start = 0.8         # relative distance from which the noise grows
    strength = 1.0      # the added noise variance at the middle of that range, in units of the leaf's squared error

    def find(self, q, w, leaf=None):
        k = self.n_neighbours
        self.n_neighbours = k + 1
        try:
            found = super().find(q, w, leaf)
        finally:
            self.n_neighbours = k
        if found is None or leaf is None or len(found[0]) <= k:
            return found
        X, y, sigma = found
        distance = np.sqrt((((X - q) / w) ** 2).sum(axis=1))
        order = np.argsort(distance)
        edge = distance[order[k]]
        order = order[:k]
        X, y, sigma, distance = X[order], y[order], sigma[order], distance[order]
        if not edge > 0:
            return X, y, sigma
        t = np.clip((distance / edge - self.start) / (1.0 - self.start), 0.0, 1.0)
        grown = self.strength * t ** 2 / np.maximum(1.0 - t, 1e-9) ** 2
        return X, y, np.sqrt(sigma ** 2 + grown[:, None] * self.squared_error(leaf, y)[None, :])

    @staticmethod
    def squared_error(leaf, y):
        """Per output, the typical squared error of the leaf's recent predictions
        (robust to a few large ones), or a hundredth of the neighbours' variance
        while there are few."""
        out = np.empty(y.shape[1])
        for i in range(y.shape[1]):
            residuals = leaf.residuals_list[i]
            out[i] = np.median(residuals ** 2) / 0.455 if len(residuals) >= 10 else 0.01 * y[:, i].var()
        return np.maximum(out, 1e-300)


def tapered_tree(start=0.8, strength=1.0, **tree_kwargs):
    from pygptreeo import GPTree
    gpt = GPTree(predict_from='neighbours', **tree_kwargs)
    gpt.neighbours.__class__ = TaperedSearch
    gpt.neighbours.start = start
    gpt.neighbours.strength = strength
    return gpt
