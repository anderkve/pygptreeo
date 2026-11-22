"""The stored points nearest to a query.

Used by ``GPTree(predict_from='neighbours')``: a leaf then supplies the kernel
of a prediction, and the points the prediction is conditioned on are the
query's nearest neighbours among the points of the leaves nearest to it.
Those leaves are found through the tree's index of the leaves (``LeafIndex``).
"""

from collections import OrderedDict

import numpy as np
from scipy.linalg.lapack import dpotrf, dtrtrs


# The settings of a tree that predicts from neighbours, the defaults of
# NeighbourSearch. None turns a mechanism off; the number of leaves and the
# limit on the consistency are always numbers.
NEIGHBOUR_LEAVES = 16                  # leaves whose points are searched for the neighbours
NEIGHBOUR_FADE = 0.9                    # fraction of the neighbours' reach from which their noise grows
NEIGHBOUR_BLEND = 0.1                   # share of a leaf's width over which kernels are mixed at a cut
NEIGHBOUR_AWAKE = 8                     # leaves whose GPs stay awake
NEIGHBOUR_CONSISTENCY = 10.0            # limit on how badly the neighbours may fit the kernel
NEIGHBOUR_CONSISTENCY_OUTSIDE = 3.0     # that limit for a query outside its neighbours
NEIGHBOUR_TRANSITION = 2.0              # factor either side of the limit over which the leaf's own GP takes over

# The limit passes from NEIGHBOUR_CONSISTENCY to NEIGHBOUR_CONSISTENCY_OUTSIDE
# between these values of ``NeighbourSearch.outside``.
OUTSIDE_FROM, OUTSIDE_TO = 0.5, 0.8


class NeighbourSearch:
    """Nearest-neighbour search over the points stored in the leaves of a
    tree, with the settings of the predictions made from them. One instance
    is shared by reference between a tree and all of its nodes.

    Attributes:
        tree: The GPTree whose leaves hold the points.
        n_neighbours (int): Number of points a prediction is conditioned on.
        consistency_limit, outside_limit, transition, max_leaves, fade,
            blend, awake: The settings, as the constants above.
        in_use (OrderedDict): The leaves whose GPs were fitted or woken, the
            last one used last. The tree puts those beyond ``awake`` to sleep.
    """

    def __init__(self, tree, n_neighbours, consistency_limit=NEIGHBOUR_CONSISTENCY, max_leaves=NEIGHBOUR_LEAVES,
                 fade=NEIGHBOUR_FADE, blend=NEIGHBOUR_BLEND, awake=NEIGHBOUR_AWAKE,
                 outside_limit=NEIGHBOUR_CONSISTENCY_OUTSIDE, transition=NEIGHBOUR_TRANSITION):
        self.tree = tree
        self.n_neighbours = n_neighbours
        self.consistency_limit = consistency_limit
        self.max_leaves = max_leaves
        self.fade = fade
        self.blend = blend
        self.awake = awake
        self.outside_limit = outside_limit
        self.transition = transition
        self.in_use = OrderedDict()

    def used(self, leaf):
        """Note that the GPs of a leaf were fitted or woken just now."""
        if self.awake is not None:
            self.in_use[leaf] = None
            self.in_use.move_to_end(leaf)

    def outside(self, X: np.ndarray, q: np.ndarray, w: np.ndarray, growth=None) -> float:
        """How far the query ``q`` lies outside its neighbours ``X``: the
        distance of their centre from it over that of the furthest of them,
        in the inputs divided by ``w``. A neighbour that fades out
        (``growth`` of ``find``) counts by what is left of it, so that the
        measure is continuous in the query."""
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
        leaf's own GP: 1 up to the limit on the ``consistency`` divided by
        ``transition``, 0 from the limit times it, and linear in the
        logarithm in between. The limit passes from ``consistency_limit`` to
        ``outside_limit`` as ``outside`` goes from OUTSIDE_FROM to OUTSIDE_TO."""
        limit = self.consistency_limit
        if self.outside_limit is not None and outside > OUTSIDE_FROM:
            limit *= (self.outside_limit / limit) ** min((outside - OUTSIDE_FROM) / (OUTSIDE_TO - OUTSIDE_FROM), 1.0)
        if self.transition is None:
            return 1.0 if consistency <= limit else 0.0
        if consistency <= limit / self.transition:
            return 1.0
        if not consistency < limit * self.transition:
            return 0.0
        return float(np.log(limit * self.transition / consistency) / (2.0 * np.log(self.transition)))

    def weight(self, consistency: float, X: np.ndarray, q: np.ndarray, w: np.ndarray, growth=None) -> float:
        """``share`` for the neighbours ``X`` of the query ``q``, with
        ``outside`` worked out only where the weight depends on it."""
        among = self.share(consistency)
        if among == self.share(consistency, 1.0):
            return among
        return self.share(consistency, self.outside(X, q, w, growth))

    @staticmethod
    def mix(share, mu, sigma, mu_other, sigma_other):
        """Mean and sigma of the mixture of two predictions, the first with
        the weight ``share``."""
        mean = share * mu + (1.0 - share) * mu_other
        variance = share * sigma ** 2 + (1.0 - share) * sigma_other ** 2 + share * (1.0 - share) * (mu - mu_other) ** 2
        return mean, np.sqrt(variance)

    def find(self, q: np.ndarray, w: np.ndarray):
        """The ``n_neighbours`` stored points nearest to the query ``q``, by
        the Euclidean distance in the inputs divided by ``w``, among the
        points of the ``max_leaves`` leaves whose points' boxes are nearest
        to the query.

        Returns ``(X, y, sigma, growth)`` in no particular order, or None if
        the tree holds no point. With ``fade``, the noise variance of a
        neighbour is to grow by the factor ``growth = t^2 / (1 - t)^2`` of
        itself plus the variance of a prediction, where ``t`` rises from 0 at
        the fraction ``fade`` to 1 at the distance of the nearest stored
        point that is not a neighbour; ``growth`` is None where there is no
        such point, or no ``fade``.
        """
        index = self.tree.leaf_index
        k = self.n_neighbours
        if index is None:
            # a tree without an index: every leaf
            leaves = [near for near in self.tree.root.leaves if near.n_points > 0]
        else:
            leaves = index.nearest(q, w, self.max_leaves)
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
        neighbours, with the kernels of several leaves mixed: the kernel of
        an output is the sum of the leaves' kernels with the ``weights``, in
        the units of the inputs and targets, and the metric, the mean, the
        global-model snapshots and the unit of the fading are the leaves'
        with the same weights.

        Returns the mean and sigma per output, before calibration, and the
        weight of this prediction against the mixture of the leaves'
        predictions (``share``, the smallest over the outputs). None where a
        leaf cannot give its kernel or predicts from its own points, or
        where that weight is 0.
        """
        for leaf in leaves:
            leaf._refresh_global()
        if any(not leaf._can_condition() or leaf.prefers_own or leaf.output_model != 'independent' for leaf in leaves):
            return None
        spread = sum(weight * leaf.fit_std for weight, leaf in zip(weights, leaves))
        X, y, noise_std, growth = self.find(x[0], spread)
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
                error = leaf.global_mean.error_scale
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
