"""The stored points nearest to a query, found through the tree's cuts.

Used by ``GPTree(predict_from='neighbours')``: a leaf then supplies the kernel
of a prediction, and the points the prediction is conditioned on are the
query's nearest neighbours among all the points the tree holds.
"""

import heapq
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

    def find(self, q: np.ndarray, w: np.ndarray):
        """The ``n_neighbours`` stored points nearest to ``q``.

        Distances are Euclidean in the inputs divided by ``w``. The leaves
        nearest to ``q`` are taken until they hold enough points, which bounds
        the distance of the points looked for. Every other leaf that the cuts
        do not put beyond that distance is then added, nearest first.

        Args:
            q: The query, shape (n_features,).
            w: The scale of each input, shape (n_features,).

        Returns:
            ``(X, y, sigma)`` of the points found, in no particular order, or
            None if the tree holds no point.
        """
        k = self.n_neighbours
        # entries: lower bound of the squared distance to a subtree, a tie-breaker,
        # the subtree, and the offset from q to its region along each input
        heap = [(0.0, 0, self.tree.root, None)]
        pushed = 1
        radius = np.inf                 # squared distance within which the k nearest points lie
        leaves, n_points, d2 = [], 0, None
        while heap:
            bound, _, node, offsets = heapq.heappop(heap)
            if bound >= radius:
                break
            while not node.is_leaf:
                t, margin, j = node.cut_distance(q, w)
                near, far = (node.right, node.left) if t > 0 else (node.left, node.right)
                t = max(abs(t) - margin, 0.0)
                if j is None:
                    far_bound, far_offsets = max(bound, t * t), offsets
                else:
                    # the offsets along different inputs add up
                    far_offsets = np.zeros(len(q)) if offsets is None else offsets.copy()
                    far_bound = bound - far_offsets[j] ** 2 + t * t
                    far_offsets[j] = t
                if far_bound < radius:
                    heapq.heappush(heap, (far_bound, pushed, far, far_offsets))
                    pushed += 1
                node = near
            if node.n_points == 0:
                continue
            leaves.append(node)
            n_points += node.n_points
            if d2 is None and n_points >= k:
                # enough points to bound the distance of the k nearest
                X = np.concatenate([leaf.my_X_data for leaf in leaves]) if len(leaves) > 1 else node.my_X_data
                d2 = (((X - q) / w) ** 2).sum(axis=1)
                radius = np.partition(d2, k - 1)[k - 1]
                first = len(leaves)
            if self.max_leaves is not None and len(leaves) >= self.max_leaves:
                break
        self.n_searches += 1
        self.n_leaves_searched += len(leaves)
        if not leaves:
            return None
        if d2 is None:                  # fewer than k points among the leaves searched
            first = 0
        if len(leaves) > first:
            more = np.concatenate([leaf.my_X_data for leaf in leaves[first:]])
            more_d2 = (((more - q) / w) ** 2).sum(axis=1)
            X, d2 = (more, more_d2) if d2 is None else (np.concatenate((X, more)), np.concatenate((d2, more_d2)))
        y = np.concatenate([leaf.my_y_data for leaf in leaves])
        sigma = np.concatenate([leaf.my_sigma_data for leaf in leaves])
        if len(d2) > k:
            nearest = np.argpartition(d2, k - 1)[:k]
            return X[nearest], y[nearest], sigma[nearest]
        return X, y, sigma
