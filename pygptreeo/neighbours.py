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

    def find(self, q: np.ndarray, w: np.ndarray):
        """The ``n_neighbours`` stored points nearest to ``q``.

        Distances are Euclidean in the inputs divided by ``w``. Leaves are
        visited nearest first, and a subtree is skipped when no point beyond
        its cuts can be nearer than the points already found.

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
        best = np.empty(0)              # the smallest squared distances found, at most k
        radius = np.inf                 # the k-th smallest of them
        leaves, dists = [], []
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
            d2 = (((node.my_X_data - q) / w) ** 2).sum(axis=1)
            leaves.append(node)
            dists.append(d2)
            best = np.concatenate((best, d2[d2 < radius]))
            if len(best) >= k:
                best = np.partition(best, k - 1)[:k]
                radius = best[k - 1]
            if self.max_leaves is not None and len(leaves) >= self.max_leaves:
                break
        if not leaves:
            return None
        X, y, sigma = [], [], []
        for node, d2 in zip(leaves, dists):
            keep = d2 <= radius
            if keep.any():
                X.append(node.my_X_data[keep])
                y.append(node.my_y_data[keep])
                sigma.append(node.my_sigma_data[keep])
        return np.concatenate(X)[:k], np.concatenate(y)[:k], np.concatenate(sigma)[:k]
