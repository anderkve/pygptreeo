"""The stored points nearest to a query.

Used by ``GPTree(predict_from='neighbours')``: a leaf then supplies the kernel
of a prediction, and the points the prediction is conditioned on are the
query's nearest neighbours among all the points the tree holds. The leaves
that can hold them are found through the tree's index of the leaves
(``LeafIndex``).
"""

from typing import Optional

import numpy as np


class NeighbourSearch:
    """Nearest-neighbour search over the points stored in the leaves of a tree.

    One instance is shared by reference between a tree and all of its nodes.

    Attributes:
        tree: The GPTree whose leaves hold the points.
        n_neighbours (int): Number of points a prediction is conditioned on.
        consistency_limit (float): A leaf predicts from its own points where
            the neighbours fit its kernel worse than this (see
            ``GPRegressorInterface.condition``).
        max_leaves (Optional[int]): The search stops after this many leaves;
            None for an exact search.
        n_searches (int): Number of searches made so far.
        n_leaves_searched (int): Number of leaves whose points they looked at.
    """

    def __init__(self, tree, n_neighbours: int, consistency_limit: float = 10.0, max_leaves: Optional[int] = None):
        if n_neighbours < 1:
            raise ValueError("n_neighbours must be at least 1")
        if max_leaves is not None and max_leaves < 1:
            raise ValueError("neighbour_leaves must be at least 1")
        self.tree = tree
        self.n_neighbours = int(n_neighbours)
        self.consistency_limit = float(consistency_limit)
        self.max_leaves = max_leaves
        self.n_searches = 0
        self.n_leaves_searched = 0

    def find(self, q: np.ndarray, w: np.ndarray, leaf=None):
        """The ``n_neighbours`` stored points nearest to ``q``.

        Distances are Euclidean in the inputs divided by ``w``. The leaf of
        the query, with the leaves next to it while they hold too few points,
        bounds the distance of the points looked for. The index then gives
        every leaf whose points can lie within that distance. With
        ``max_leaves`` the points are those of the leaves whose points' boxes
        are nearest to the query.

        Args:
            q: The query, shape (n_features,).
            w: The scale of each input, shape (n_features,).
            leaf: A leaf near the query, by default the one whose cell holds it.

        Returns:
            ``(X, y, sigma)`` of the points found, in no particular order, or
            None if the tree holds no point.
        """
        index = self.tree.leaf_index
        k = self.n_neighbours
        if index is None:
            # a tree without an index: every leaf
            leaves = [near for near in self.tree.root.leaves if near.n_points > 0]
        elif self.max_leaves is not None:
            leaves = index.nearest(q, w, self.max_leaves)
        else:
            if leaf is None or leaf not in index:
                leaf = index.locate(q)
            radius = np.inf             # squared distance within which the k nearest points lie
            if leaf is not None:
                first, n_points = [], 0
                for near in [leaf] + [other for other in index.siblings(leaf, q, w) if other is not leaf]:
                    if near.n_points > 0:
                        first.append(near.my_X_data)
                        n_points += near.n_points
                        if n_points >= k:
                            X = np.concatenate(first) if len(first) > 1 else first[0]
                            radius = np.partition((((X - q) / w) ** 2).sum(axis=1), k - 1)[k - 1]
                            break
            leaves = [near for near in index.within(q, w, radius)[0] if near.n_points > 0]
        self.n_searches += 1
        self.n_leaves_searched += len(leaves)
        if not leaves:
            return None
        X = np.concatenate([near.my_X_data for near in leaves])
        y = np.concatenate([near.my_y_data for near in leaves])
        sigma = np.concatenate([near.my_sigma_data for near in leaves])
        if len(X) > k:
            nearest = np.argpartition((((X - q) / w) ** 2).sum(axis=1), k - 1)[:k]
            return X[nearest], y[nearest], sigma[nearest]
        return X, y, sigma
