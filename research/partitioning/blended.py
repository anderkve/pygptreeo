"""Predictions blended across the cuts: the prototypes of the package's NEIGHBOUR_BLEND.

Within a zone around each cut, whose width is the share ``blend`` of the
spread of the node's points along the cut, a query is predicted with the
hyperparameters of the leaves on both sides, each from the query's
neighbours, and the predictions are mixed with weights that change linearly
across the zone. Where a point is stored does not change.
"""
import numpy as np

from pygptreeo import GPTree
from variants import neighbour_tree


class BlendedTree(GPTree):
    blend = 0.1
    n_queries = 0
    n_leaves_used = 0

    def _leaves_of(self, x):
        leaves, weights = [], []
        stack = [(self.root, 1.0)]
        while stack:
            node, weight = stack.pop()
            if node.is_leaf:
                leaves.append(node)
                weights.append(weight)
                continue
            # the spread of the node's points along the cut when it was made
            spread = node.overlap / self.theta
            distance = node.split_coordinate(x)[0] - node.split_position
            p = float(np.clip(distance / (self.blend * spread) + 0.5, 0.0, 1.0)) if spread > 0 else float(distance > 0)
            if weight * p > 0:
                stack.append((node.right, weight * p))
            if weight * (1 - p) > 0:
                stack.append((node.left, weight * (1 - p)))
        self.n_queries += 1
        self.n_leaves_used += len(leaves)
        return leaves, weights


def blended_tree(blend=0.1, **tree_kwargs):
    # the package's own zones are off: these are the prototypes of them
    gpt = neighbour_tree(neighbour_blend=None, **tree_kwargs)
    gpt.__class__ = BlendedTree
    gpt.blend = blend
    return gpt


class MixedTree(BlendedTree):
    """One prediction per query within a zone: the neighbours are conditioned
    on with the mixture of the kernels of the leaves on both sides, and the
    metric of the distances, the mean, the unit of the fading and the
    calibration of sigma are mixed with the same weights. Single output, no
    global model; anything else falls back to the mixture of predictions."""
    n_mixed = 0

    def predict(self, X_test, **kwargs):
        if self.first_point:
            return super().predict(X_test, **kwargs)
        X_test = np.atleast_2d(X_test)
        mean, std = np.empty((len(X_test), 1)), np.empty((len(X_test), 1))
        for r in range(len(X_test)):
            x = X_test[r:r + 1]
            leaves, weights = self._leaves_of(x)
            out = self._one_kernel(x, leaves, np.array(weights)) if len(leaves) > 1 else None
            if out is None:
                mu, sd = np.array([[v[0, 0] for v in self._predict_leaf(leaf, x)] for leaf in leaves]).T
                w = np.array(weights)
                m = (w * mu).sum()
                out = m, np.sqrt((w * (sd ** 2 + (mu - m) ** 2)).sum())
            mean[r, 0], std[r, 0] = out
        return mean, std

    def _one_kernel(self, x, leaves, w):
        from scipy.linalg import cho_factor, cho_solve, LinAlgError
        if any(not leaf._can_condition() or leaf.prefers_own or leaf._fitted_global is not None for leaf in leaves):
            return None
        metric = sum(wi * leaf.fit_std for wi, leaf in zip(w, leaves))
        found = self.neighbours.find(x[0], metric, leaf=leaves[int(np.argmax(w))])
        if found is None:
            return None
        X, y, noise_std, growth = found
        n = len(X)
        Z = np.vstack((X, x))
        K, mean0, unit, scaler = np.zeros((n + 1, n + 1)), 0.0, 0.0, 0.0
        for wi, leaf in zip(w, leaves):
            xs, ys = leaf.X_scaler, leaf.y_scalers[0]
            K += wi * ys.scale_[0] ** 2 * leaf.my_GPRs[0].sklearn_gpr.kernel_((Z - xs.mean_) / xs.scale_)
            mean0 += wi * ys.mean_[0]
            variance = leaf._predicted_variance()
            unit += wi * (variance[0] if variance is not None else 0.01 * y[:, 0].var())
            scaler += wi * leaf.sigma_scalers[0]
        noise = noise_std[:, 0] ** 2
        if growth is not None:
            noise = noise + growth * (noise + unit)
        try:
            factor = cho_factor(K[:n, :n] + np.diag(noise), lower=True)
        except LinAlgError:
            return None
        centred = y[:, 0] - mean0
        alpha = cho_solve(factor, centred)
        if not centred @ alpha / n <= self.neighbours.consistency_limit:
            return None
        self.n_mixed += 1
        cross = K[:n, n]
        variance = max(K[n, n] - cross @ cho_solve(factor, cross), 0.0)
        return mean0 + cross @ alpha, np.sqrt(variance) * (scaler if self.use_calibrated_sigma else 1.0)


def mixed_tree(blend=0.1, **tree_kwargs):
    gpt = blended_tree(blend=blend, **tree_kwargs)
    gpt.__class__ = MixedTree
    return gpt
