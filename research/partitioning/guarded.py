"""Neighbours with a stricter test against the kernel for a query outside them.

A prototype: where a query lies outside its neighbours, they predict only if
their consistency with the kernel is at most `limit`. A query is outside by
`measure`:

  'centroid'  the distance of the neighbours' centre from the query exceeds
              `far` of the distance of the furthest neighbour
  'nearest'   the nearest neighbour is further than `far` of the leaf's spreads
"""
import numpy as np
from pygptreeo import GPTree
from pygptreeo.neighbours import NeighbourSearch


class GuardedSearch(NeighbourSearch):
    measure = 'centroid'
    far = 0.65
    far_limit = 3.0
    _outside = 0.0          # how far the last query lay outside its neighbours

    def find(self, q, w, leaf=None):
        found = super().find(q, w, leaf=leaf)
        self._outside = 0.0
        if found is not None:
            offsets = (found[0] - q) / w
            distance = np.sqrt((offsets ** 2).sum(axis=1))
            if self.measure == 'nearest':
                self._outside = distance.min()
            elif distance.max() > 0:
                self._outside = np.linalg.norm(offsets.mean(axis=0)) / distance.max()
        return found

    @property
    def consistency_limit(self):
        limit = self.__dict__['consistency_limit']
        return min(limit, self.far_limit) if self._outside > self.far else limit

    @consistency_limit.setter
    def consistency_limit(self, value):
        self.__dict__['consistency_limit'] = value


def guard(tree, measure='centroid', far=0.65, limit=3.0):
    """Give the tree's neighbour search the stricter test."""
    search = tree.neighbours
    search.__class__ = GuardedSearch
    search.measure, search.far, search.far_limit = measure, far, limit
    return tree


def guarded_tree(measure='centroid', far=0.65, limit=3.0, **kw):
    return guard(GPTree(predict_from='neighbours', **kw), measure=measure, far=far, limit=limit)
