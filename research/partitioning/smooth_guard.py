"""A gradual change from the neighbours' prediction to the leaf's own.

The prototype of the package's NEIGHBOUR_TRANSITION and NEIGHBOUR_CONSISTENCY_OUTSIDE:
the two predictions are mixed with a weight for the neighbours
that falls from 1 to 0 as their consistency with the kernel rises from
`limit / width` to `limit * width`, linearly in its logarithm. The limit
itself slides from the package's NEIGHBOUR_CONSISTENCY to `far` as the centre of the
neighbours moves from `inside` to `outside` of the distance of the furthest
one. Where the kernels of several leaves are mixed, the other prediction is
the mixture of the leaves' predictions.

Importing the module wraps two methods of the package; trees without a
`SmoothSearch` are not affected.
"""
import numpy as np
from pygptreeo import GPTree
from pygptreeo.gpnode import GPNode
from pygptreeo.neighbours import NeighbourSearch
from variants import neighbour_tree


class _Seen:
    """In place of the limit of the test: lets every consistency pass and keeps it."""
    __array_ufunc__ = None

    def __init__(self):
        self.values = []

    def __ge__(self, consistency):
        self.values.append(float(consistency))
        return True


class SmoothSearch(NeighbourSearch):
    inside, outside, far, width = 0.5, 0.8, 3.0, 2.0
    _offset = 0.0           # centre of the last neighbours from the query, over the distance of the furthest
    _seen = None            # the consistencies of the conditioning under way
    _skip_mixed = False
    _mixed = False          # whether the last mixed prediction was made

    def find(self, q, w, leaf=None):
        found = super().find(q, w, leaf=leaf)
        self._offset = 0.0
        if found is not None:
            offsets = (found[0] - q) / w
            furthest = np.sqrt((offsets ** 2).sum(axis=1).max())
            if furthest > 0:
                self._offset = np.linalg.norm(offsets.mean(axis=0)) / furthest
        return found

    def predict_mixed(self, x, leaves, weights):
        if self._skip_mixed:
            return None
        result = super().predict_mixed(x, leaves, weights)
        self._mixed = result is not None
        return result

    @property
    def consistency_limit(self):
        return self._seen if self._seen is not None else self.__dict__['consistency_limit']

    @consistency_limit.setter
    def consistency_limit(self, value):
        self.__dict__['consistency_limit'] = value

    def prototype_share(self, seen):
        """The weight of the neighbours' prediction after the last search."""
        if not seen.values:
            return 0.0
        limit = self.__dict__['consistency_limit']
        slide = np.clip((self._offset - self.inside) / (self.outside - self.inside), 0.0, 1.0)
        limit = limit * (self.far / limit) ** slide
        consistency = max(max(seen.values), 1e-300)
        return float(np.clip(np.log(limit * self.width / consistency) / (2.0 * np.log(self.width)), 0.0, 1.0))


def _mix(share, mu, sigma, mu_other, sigma_other):
    mean = share * mu + (1.0 - share) * mu_other
    variance = share * (sigma ** 2 + (mu - mean) ** 2) + (1.0 - share) * (sigma_other ** 2 + (mu_other - mean) ** 2)
    return mean, np.sqrt(variance)


_node_predict = GPNode._predict_from_neighbours
_tree_predict = GPTree._predict_blended


def _predict_from_neighbours(self, x):
    search = self.neighbours
    if not isinstance(search, SmoothSearch):
        return _node_predict(self, x)
    n = x.shape[0]
    mu, sigma, share = np.full((n, self.n_outputs), np.nan), np.full((n, self.n_outputs), np.nan), np.zeros(n)
    for r in range(n):
        found = search.find(x[r], self.fit_std, leaf=self)
        if found is not None:
            outer, search._seen = search._seen, _Seen()
            try:
                self._condition_on(found, x[r:r + 1], mu[r], sigma[r])
                if not np.isnan(mu[r]).any():
                    share[r] = search.prototype_share(search._seen)
            finally:
                search._seen = outer
    rows = share < 1.0
    if rows.any():
        mu_own, sigma_own = self._predict_own(x[rows])
        weight = share[rows, None]
        mu[rows], sigma[rows] = _mix(weight, np.where(weight > 0, mu[rows], mu_own), np.where(weight > 0, sigma[rows], sigma_own),
                                     mu_own, sigma_own)
    return mu, sigma


def _predict_blended(self, x, leaves, weights, calibrated=True):
    search = self.neighbours
    if not isinstance(search, SmoothSearch):
        return _tree_predict(self, x, leaves, weights, calibrated)
    seen = search._seen = _Seen()
    search._mixed = False
    try:
        mu, sigma = _tree_predict(self, x, leaves, weights, calibrated)
    finally:
        search._seen = None
    if not search._mixed:
        return mu, sigma            # the mixture of the leaves' predictions already
    share = search.prototype_share(seen)
    if share >= 1.0:
        return mu, sigma
    search._skip_mixed = True
    try:
        mu_other, sigma_other = _tree_predict(self, x, leaves, weights, calibrated)
    finally:
        search._skip_mixed = False
    return _mix(share, mu, sigma, mu_other, sigma_other)


GPNode._predict_from_neighbours = _predict_from_neighbours
GPTree._predict_blended = _predict_blended


def smooth(tree, **settings):
    """Give the tree's neighbour search the gradual change."""
    tree.neighbours.__class__ = SmoothSearch
    tree.neighbours.transition = tree.neighbours.outside_limit = None      # the package's own version is off
    for name, value in settings.items():
        setattr(tree.neighbours, name, value)
    return tree


def smooth_tree(inside=0.5, outside=0.8, far=3.0, width=2.0, **kw):
    return smooth(neighbour_tree(**kw), inside=inside, outside=outside, far=far, width=width)
