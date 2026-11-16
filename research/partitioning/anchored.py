"""Neighbours of a small cell in place of neighbours of the query.

Each leaf is divided into ``2 ** depth`` cells by median cuts of its stored
points, and a prediction is conditioned on the stored points nearest to the
mean of the points of the query's cell. All queries of a cell share their
neighbours, so the prediction is smooth within a cell.
"""
import numpy as np

from pygptreeo.neighbours import NeighbourSearch


class AnchoredSearch(NeighbourSearch):
    depth = 2

    def anchor(self, q, w, leaf):
        """The mean of the points of the cell of ``leaf`` that holds ``q``."""
        P = leaf.my_X_data
        for _ in range(self.depth):
            if len(P) < 4:
                break
            j = int(np.argmax(np.ptp(P / w, axis=0)))
            cut = np.median(P[:, j])
            side = P[:, j] > cut if q[j] > cut else P[:, j] <= cut
            if not side.any():
                break
            P = P[side]
        return P.mean(axis=0)

    def find(self, q, w, leaf=None):
        if leaf is None or leaf.n_points < 4:
            return super().find(q, w, leaf)
        return super().find(self.anchor(q, w, leaf), w, leaf)


def anchored_tree(depth=2, **tree_kwargs):
    from pygptreeo import GPTree
    gpt = GPTree(predict_from='neighbours', **tree_kwargs)
    gpt.neighbours.__class__ = AnchoredSearch
    gpt.neighbours.depth = depth
    return gpt
