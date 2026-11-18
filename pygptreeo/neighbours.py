"""The stored points nearest to a query.

Used by ``GPTree(predict_from='neighbours')``: a leaf then supplies the kernel
of a prediction, and the points the prediction is conditioned on are the
query's nearest neighbours among all the points the tree holds. The leaves
that can hold them are found through the tree's index of the leaves
(``LeafIndex``).
"""

from collections import OrderedDict
from typing import Optional

import numpy as np
from scipy.linalg.lapack import dpotrf, dtrtrs


# The settings of a tree that predicts from neighbours (the arguments of
# NeighbourSearch that GPTree gives it); None turns a mechanism off.
NEIGHBOUR_LEAVES = 16                   # leaves whose points are searched for the neighbours
NEIGHBOUR_FADE = 0.9                    # fraction of the neighbours' reach from which their noise grows
NEIGHBOUR_BLEND = 0.1                   # share of a leaf's width over which kernels are mixed at a cut
NEIGHBOUR_AWAKE = 8                     # leaves whose GPs stay awake
NEIGHBOUR_CONSISTENCY = 10.0            # limit on how badly the neighbours may fit the kernel
NEIGHBOUR_CONSISTENCY_OUTSIDE = 3.0     # that limit for a query outside its neighbours
NEIGHBOUR_TRANSITION = 2.0              # factor either side of the limit over which the leaf's own GP takes over

# Of the distance of the furthest neighbour: between these distances of the
# neighbours' centre from the query, the limit on their consistency passes
# from that for a query among its neighbours to that for one outside them.
OUTSIDE_FROM, OUTSIDE_TO = 0.5, 0.8


class NeighbourSearch:
    """Nearest-neighbour search over the points stored in the leaves of a tree.

    One instance is shared by reference between a tree and all of its nodes.

    Attributes:
        tree: The GPTree whose leaves hold the points.
        n_neighbours (int): Number of points a prediction is conditioned on.
        consistency_limit (float): A leaf predicts from its own points where
            the neighbours fit its kernel worse than this (see
            ``GPRegressorInterface.condition``).
        outside_limit (Optional[float]): That limit for a query outside its
            neighbours (``outside``); None for one limit everywhere.
        transition (Optional[float]): The prediction passes from the
            neighbours' to the leaf's own between the limit divided by this
            and the limit times this (``share``); None for a switch at the
            limit.
        max_leaves (Optional[int]): The search stops after this many leaves;
            None for an exact search.
        fade (Optional[float]): The noise of a neighbour grows from this
            fraction of the distance of the first stored point beyond the
            neighbours, and is infinite at that distance; None for no growth.
        blend (Optional[float]): The kernels of the leaves on both sides of a
            cut are mixed within a zone of this share of a leaf's width; None
            for the kernel of the query's leaf alone.
        awake (Optional[int]): Number of leaves whose GPs stay awake, those
            used last; None for all of them.
        in_use (OrderedDict): The leaves whose GPs were fitted or woken, the
            last one used last. The tree puts those beyond ``awake`` to sleep.
        n_searches (int): Number of searches made so far.
        n_leaves_searched (int): Number of leaves whose points they looked at.
    """
    # for a search saved before these existed
    fade = None
    blend = None
    awake = None
    outside_limit = None
    transition = None


    def __init__(self, tree, n_neighbours: int, consistency_limit: float = 10.0, max_leaves: Optional[int] = None,
                 fade: Optional[float] = None, blend: Optional[float] = None, awake: Optional[int] = None,
                 outside_limit: Optional[float] = None, transition: Optional[float] = None):
        if n_neighbours < 1:
            raise ValueError("n_neighbours must be at least 1")
        if max_leaves is not None and max_leaves < 1:
            raise ValueError("max_leaves must be at least 1, or None")
        if fade is not None and not 0.0 <= fade < 1.0:
            raise ValueError("fade must be at least 0 and below 1, or None")
        if blend is not None and not 0.0 < blend <= 1.0:
            raise ValueError("blend must be above 0 and at most 1, or None")
        if awake is not None and awake < 1:
            raise ValueError("awake must be at least 1, or None")
        if outside_limit is not None and not outside_limit > 0.0:
            raise ValueError("outside_limit must be above 0, or None")
        if transition is not None and not transition > 1.0:
            raise ValueError("transition must be above 1, or None")
        self.tree = tree
        self.n_neighbours = int(n_neighbours)
        self.consistency_limit = float(consistency_limit)
        self.max_leaves = max_leaves
        self.fade = fade
        self.blend = blend
        self.awake = awake
        self.outside_limit = outside_limit
        self.transition = transition
        self.in_use = OrderedDict()
        self.n_searches = 0
        self.n_leaves_searched = 0

    def used(self, leaf):
        """Note that the GPs of a leaf were fitted or woken just now."""
        if self.awake is not None:
            self.in_use[leaf] = None
            self.in_use.move_to_end(leaf)

    def outside(self, X: np.ndarray, q: np.ndarray, w: np.ndarray, growth=None) -> float:
        """How far the query ``q`` lies outside its neighbours ``X``: the
        distance of their centre from it over that of the furthest of them,
        in the inputs divided by ``w``. Near 0 with neighbours on all sides,
        towards 1 with all of them on one side. A neighbour that fades out
        (``growth``, as ``find`` returns it) counts by what is left of it,
        so that the measure is continuous in the query."""
        offsets = (X - q) / w
        furthest = np.sqrt((offsets ** 2).sum(axis=1).max())
        if not furthest > 0.0:
            return 0.0
        if growth is None:
            centre = offsets.mean(axis=0)
        else:
            left = 1.0 / (1.0 + np.sqrt(growth))
            centre = left @ offsets / left.sum()
        return float(np.sqrt(centre @ centre) / furthest)

    def share(self, consistency: float, outside: float = 0.0) -> float:
        """The weight of the neighbours' prediction against that of the
        leaf's own GP, for neighbours of the given ``consistency`` with the
        kernel and a query that far ``outside`` them.

        The limit on the consistency is ``consistency_limit`` up to
        OUTSIDE_FROM, ``outside_limit`` from OUTSIDE_TO, and passes from one
        to the other in between. The weight is 1 up to the limit divided by
        ``transition``, 0 from the limit times it, and linear in the
        logarithm of the consistency in between."""
        limit = self.consistency_limit
        if self.outside_limit is not None and outside > OUTSIDE_FROM and 0.0 < limit < np.inf:
            limit *= (self.outside_limit / limit) ** min((outside - OUTSIDE_FROM) / (OUTSIDE_TO - OUTSIDE_FROM), 1.0)
        if self.transition is None:
            return 1.0 if consistency <= limit else 0.0
        if consistency <= limit / self.transition:
            return 1.0
        if not consistency < limit * self.transition:
            return 0.0
        return float(np.log(limit * self.transition / consistency) / (2.0 * np.log(self.transition)))

    def weight(self, consistency: float, X: np.ndarray, q: np.ndarray, w: np.ndarray, growth=None) -> float:
        """``share`` for the neighbours ``X`` of the query ``q``. How far the
        query lies outside them is worked out only where the weight depends
        on it."""
        among = self.share(consistency)
        if among == self.share(consistency, 1.0):
            return among
        return self.share(consistency, self.outside(X, q, w, growth))

    @staticmethod
    def mix(share, mu, sigma, mu_other, sigma_other):
        """Mean and sigma of the mixture of two predictions, the first with
        the weight ``share``. Entries of the first are not read where its
        weight is 0."""
        share = np.broadcast_to(share, np.shape(mu_other))
        first, second = share >= 1.0, share <= 0.0
        mu, sigma = np.where(second, mu_other, mu), np.where(second, sigma_other, sigma)
        mean = share * mu + (1.0 - share) * mu_other
        variance = share * sigma ** 2 + (1.0 - share) * sigma_other ** 2 + share * (1.0 - share) * (mu - mu_other) ** 2
        return (np.where(first, mu, np.where(second, mu_other, mean)),
                np.where(first, sigma, np.where(second, sigma_other, np.sqrt(variance))))

    def find(self, q: np.ndarray, w: np.ndarray, leaf=None):
        """The ``n_neighbours`` stored points nearest to ``q``.

        Distances are Euclidean in the inputs divided by ``w``. The leaf of
        the query, with the leaves next to it while they hold too few points,
        bounds the distance of the points looked for. The index then gives
        every leaf whose points can lie within that distance. With
        ``max_leaves`` the points are those of the leaves whose points' boxes
        are nearest to the query.

        With ``fade`` the noise variance of a neighbour is to grow by the
        factor ``t^2 / (1 - t)^2`` of itself plus the variance of a
        prediction, where ``t`` rises from 0 at the fraction ``fade`` to 1 at
        the full distance of the nearest stored point that is not a
        neighbour. A point then enters and leaves the neighbours with no
        weight.

        Args:
            q: The query, shape (n_features,).
            w: The scale of each input, shape (n_features,).
            leaf: A leaf near the query, by default the one whose cell holds it.

        Returns:
            ``(X, y, sigma, growth)`` of the points found, in no particular
            order, or None if the tree holds no point. ``growth`` is that
            factor for each point, or None without ``fade`` or where no
            stored point lies beyond the neighbours.
        """
        index = self.tree.leaf_index
        k = self.n_neighbours
        n_wanted = k if self.fade is None else k + 1
        if index is None:
            # a tree without an index: every leaf
            leaves = [near for near in self.tree.root.leaves if near.n_points > 0]
        elif self.max_leaves is not None:
            leaves = index.nearest(q, w, self.max_leaves)
        else:
            if leaf is None or leaf not in index:
                leaf = index.locate(q)
            radius = np.inf             # squared distance within which the points looked for lie
            if leaf is not None:
                first, n_points = [], 0
                for near in [leaf] + [other for other in index.siblings(leaf, q, w) if other is not leaf]:
                    if near.n_points > 0:
                        first.append(near.my_X_data)
                        n_points += near.n_points
                        if n_points >= n_wanted:
                            X = np.concatenate(first) if len(first) > 1 else first[0]
                            radius = np.partition((((X - q) / w) ** 2).sum(axis=1), n_wanted - 1)[n_wanted - 1]
                            break
            leaves = [near for near in index.within(q, w, radius)[0] if near.n_points > 0]
        self.n_searches += 1
        self.n_leaves_searched += len(leaves)
        if not leaves:
            return None
        X = np.concatenate([near.my_X_data for near in leaves])
        y = np.concatenate([near.my_y_data for near in leaves])
        sigma = np.concatenate([near.my_sigma_data for near in leaves])
        if len(X) <= k:
            return X, y, sigma, None
        distance = (((X - q) / w) ** 2).sum(axis=1)
        if self.fade is None:
            nearest = np.argpartition(distance, k - 1)[:k]
            return X[nearest], y[nearest], sigma[nearest], None
        order = np.argpartition(distance, (k - 1, k))
        nearest, beyond = order[:k], distance[order[k]]
        growth = None
        if beyond > 0.0:
            t = np.clip((np.sqrt(distance[nearest] / beyond) - self.fade) / (1.0 - self.fade), 0.0, 1.0)
            growth = t ** 2 / np.maximum(1.0 - t, 1e-9) ** 2
        return X[nearest], y[nearest], sigma[nearest], growth

    def predict_mixed(self, x: np.ndarray, leaves, weights):
        """The prediction at the single query ``x`` (one row) from its
        neighbours, with the kernels of several leaves mixed.

        The kernel of an output is the sum of the leaves' kernels with the
        ``weights``, in the units of the inputs and targets. The metric of
        the distances, the mean of the targets with the leaves' snapshots of
        the global model and its error, and the unit of the fading are the
        leaves' with the same weights.

        Returns the mean and sigma per output, before calibration, and the
        weight of this prediction against the mixture of the leaves' own
        predictions (``share``, the smallest over the outputs). None where a
        leaf cannot give its kernel or predicts from its own points, or
        where that weight is 0.
        """
        for leaf in leaves:
            leaf._refresh_global()
        if any(not leaf._can_condition() or leaf.prefers_own or leaf.output_model != 'independent' for leaf in leaves):
            return None
        spread = sum(weight * leaf.fit_std for weight, leaf in zip(weights, leaves))
        found = self.find(x[0], spread, leaf=leaves[int(np.argmax(weights))])
        if found is None:
            return None
        X, y, noise_std, growth = found
        n, n_outputs = y.shape
        Z = np.vstack((X, x))
        covariances = np.zeros((n_outputs, n + 1, n + 1))
        mean = np.zeros((n + 1, n_outputs))
        scale, unit, without_unit = np.zeros(n_outputs), np.zeros(n_outputs), 0.0
        global_error = np.zeros(n_outputs)      # squared error of the global model, from the leaves that use it
        for weight, leaf in zip(weights, leaves):
            prior = leaf._prior(Z)
            if prior is None:
                return None
            for i in range(n_outputs):
                covariances[i] += weight * prior[0][i]
            mean += weight * prior[1]
            scale += weight * prior[2]
            if leaf._fitted_global is not None:
                mean += weight * leaf._fitted_global.predict(Z)
                error = getattr(leaf.global_mean, 'error_scale', None)
                if error is not None:
                    global_error += weight * np.asarray(error, dtype=float).reshape(-1) ** 2
            variance = leaf._predicted_variance()
            if variance is None:
                without_unit += weight
            else:
                unit += weight * variance
        residual = y - mean[:n]
        noise = noise_std ** 2
        if growth is not None:
            noise = noise + growth[:, None] * (noise + unit + without_unit * 0.01 * residual.var(axis=0))
        noise = noise * scale
        mu, sigma, share = np.empty(n_outputs), np.empty(n_outputs), 1.0
        for i in range(n_outputs):
            K = covariances[i, :n, :n].copy()
            K.flat[::n + 1] += noise[:, i]
            L, info = dpotrf(K, lower=True, overwrite_a=True)
            if info != 0:
                return None
            # L^-1 applied to the targets and to the covariances with the query at once
            solved, info = dtrtrs(L, np.column_stack((residual[:, i], covariances[i, :n, n])), lower=True)
            if info != 0:
                return None
            v, u = solved[:, 0], solved[:, 1]
            share = min(share, self.weight(v @ v / n, X, x[0], spread, growth))
            if share <= 0.0:
                return None
            mu[i] = mean[n, i] + u @ v
            sigma[i] = np.sqrt(max(covariances[i, n, n] - u @ u, 0.0) + global_error[i])
        return mu, sigma, share
