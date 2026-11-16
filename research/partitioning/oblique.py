"""Trees whose cuts are hyperplanes with a free normal instead of a coordinate axis."""
import numpy as np

from pygptreeo import GPTree
from pygptreeo.gpnode import GPNode


class Whitener:
    """Centred, decorrelated inputs: ``(x - mean) @ W``, with ``W`` the symmetric
    inverse square root of the covariance of the points (the whitening closest to
    a scaling of each input). The correlation matrix is shrunk towards the identity."""

    def __init__(self, X, shrink=0.1):
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.where(std > 0, std, 1.0)
        Z = (X - self.mean_) / self.scale_
        C = (1 - shrink) * (Z.T @ Z) / len(Z) + shrink * np.eye(X.shape[1])
        evals, evecs = np.linalg.eigh(C)
        self.W = (evecs / np.sqrt(evals)) @ evecs.T / self.scale_[:, None]
        self.W_inv = np.linalg.inv(self.W)

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.W


class ObliqueNode(GPNode):
    """A node cut by the hyperplane ``x @ split_direction = split_position``.

    Class attributes (set per configuration by ``oblique_tree``):
        direction: how the normal is chosen
            'axis'     the package's own axis-aligned choice
            'pca'      the direction of largest variance of the node's points
            'grad'     the direction along which the GP mean changes most (leading
                       eigenvector of the mean outer product of its gradients)
            'curv'     the direction along which the gradient of the GP mean changes
                       most (leading eigenvector of the covariance of its gradients)
            'twomeans' the line between the centroids of a 2-means clustering
            'random'   a random direction
            'ard'      the axis of the node's scaled inputs with the shortest fitted length scale
        position: 'median', 'midpoint' (between the two centroids, 'twomeans' only) or
            'gsplit' (where the gradient of the GP mean along the normal changes most)
        min_fraction: smallest share of the node's points on either side of a cut
        whiten: whether the GPs see decorrelated inputs (``Whitener``) instead of inputs scaled one by one
        scaled: whether 'pca', 'twomeans' and 'random' work in the node's standardised inputs
    """
    direction = 'pca'
    position = 'median'
    scaled = False
    min_fraction = 0.25
    whiten = False
    split_direction = None

    def _fit_scalers(self, X, y):
        super()._fit_scalers(X, y)
        if self.whiten and X.shape[0] > X.shape[1] + 1:
            self.X_scaler = Whitener(X)

    def _scaled_inputs(self, x):
        if isinstance(self.X_scaler, Whitener):
            return self.X_scaler.transform(x)
        return super()._scaled_inputs(x)

    def _raw_normal(self, ws):
        """The normal in the raw inputs of the hyperplane with normal ``ws`` in the scaled inputs."""
        if isinstance(self.X_scaler, Whitener):
            return self.X_scaler.W @ ws
        return ws / self._scale()

    def _scaled_normal(self, w):
        if isinstance(self.X_scaler, Whitener):
            return self.X_scaler.W_inv @ w
        return w * self._scale()

    def split_coordinate(self, x):
        if self.split_direction is None:
            return super().split_coordinate(x)
        return x @ self.split_direction

    def _scale(self):
        if self.use_standard_scaling and self.X_scaler is not None:
            return self.X_scaler.scale_
        std = self.my_X_data.std(axis=0)
        return np.where(std > 0, std, 1.0)

    def _mean_gradients(self):
        """Gradients of the GP means at the node's points, in standardised inputs:
        an array (n_gps * n_points, n_features), each GP's rows scaled to unit mean norm."""
        if not self._is_fitted() or self.output_model == 'pca' and self._fitted_basis is None:
            return None
        Xs = self._scaled_inputs(self.my_X_data)
        n, d = Xs.shape
        h = 1e-3
        Xp = np.repeat(Xs[None], 2 * d, axis=0)
        for j in range(d):
            Xp[2 * j, :, j] += h
            Xp[2 * j + 1, :, j] -= h
        grads = []
        for gpr in self.my_GPRs:
            if not gpr.is_trained():
                continue
            m = np.asarray(gpr.predict(Xp.reshape(-1, d), return_std=False), dtype=float).reshape(2 * d, n, -1)
            for o in range(m.shape[2]):
                G = (m[0::2, :, o] - m[1::2, :, o]).T / (2 * h)
                norm = np.sqrt(np.mean(np.sum(G ** 2, axis=1)))
                if norm > 0 and np.all(np.isfinite(G)):
                    grads.append(G / norm)
        return np.vstack(grads) if grads else None

    def _choose_direction(self):
        """The cut's normal in the raw input coordinates, or None for an axis-aligned cut."""
        X = self.my_X_data
        d = X.shape[1]
        scale = self._scale()
        if self.direction in ('grad', 'curv'):
            G = self._mean_gradients()
            if G is None:
                return None
            C = (G.T @ G) / len(G) if self.direction == 'grad' else np.cov(G.T)
            w = np.linalg.eigh(np.atleast_2d(C))[1][:, -1]
            return self._raw_normal(w)
        if self.direction == 'ard':
            # the axis of the scaled inputs with the shortest fitted length scale
            ls = self._pooled_length_scales() if self._is_fitted() else None
            if ls is None:
                return None
            ws = np.zeros(d)
            ws[int(np.argmin(ls))] = 1.0
            return self._raw_normal(ws)
        Z = (X - X.mean(axis=0)) / scale if self.scaled else X - X.mean(axis=0)
        if self.direction == 'pca':
            w = np.linalg.eigh(np.atleast_2d(np.cov(Z.T)))[1][:, -1]
        elif self.direction == 'random':
            w = np.random.randn(d)
        elif self.direction == 'twomeans':
            # start from the two ends of the principal axis
            t = Z @ np.linalg.eigh(np.atleast_2d(np.cov(Z.T)))[1][:, -1]
            lab = t > np.median(t)
            for _ in range(10):
                c0, c1 = Z[~lab].mean(axis=0), Z[lab].mean(axis=0)
                new = ((Z - c1) ** 2).sum(axis=1) < ((Z - c0) ** 2).sum(axis=1)
                if new.all() or not new.any() or np.array_equal(new, lab):
                    break
                lab = new
            c0, c1 = Z[~lab].mean(axis=0), Z[lab].mean(axis=0)
            w = c1 - c0
            self._midpoint = 0.5 * (c0 + c1)
        else:
            raise ValueError(self.direction)
        return w / scale if self.scaled else w

    def _gradient_split(self, w):
        """The position along the normal ``w`` that best separates the node's points
        into two groups of similar GP-mean gradient along ``w`` (a regression-tree
        split of the directional derivative), within the central half of the points."""
        G = self._mean_gradients()
        if G is None:
            return None
        X = self.my_X_data
        n = X.shape[0]
        ws = self._scaled_normal(w)
        ws = ws / np.linalg.norm(ws)
        g = (G @ ws).reshape(-1, n)                    # one row per GP
        t = X @ w
        order = np.argsort(t)
        g = g[:, order]
        lo, hi = int(np.ceil(self.min_fraction * n)), int(np.floor((1 - self.min_fraction) * n))
        if hi <= lo:
            return None
        c1 = np.cumsum(g, axis=1)
        c2 = np.cumsum(g ** 2, axis=1)
        i = np.arange(lo, hi + 1)                       # sizes of the lower group
        sse = (c2[:, i - 1] - c1[:, i - 1] ** 2 / i) + ((c2[:, -1:] - c2[:, i - 1]) - (c1[:, -1:] - c1[:, i - 1]) ** 2 / (n - i))
        best = i[np.argmin(sse.sum(axis=0))]
        return 0.5 * (t[order[best - 1]] + t[order[best]])

    def compute_split_position_and_overlap(self, theta):
        self.split_direction = None
        w = None if self.direction == 'axis' or self.my_X_data.shape[0] < 3 else self._choose_direction()
        if w is not None:
            norm = np.linalg.norm(w)
            w = w / norm if norm > 0 and np.all(np.isfinite(w)) else None
        if w is None or np.ptp(self.my_X_data @ w) == 0:
            super().compute_split_position_and_overlap(theta)
            if self.position == 'gsplit':
                w = np.zeros(self.my_X_data.shape[1])
                w[self.split_index] = 1.0
                s = self._gradient_split(w)
                if s is not None:
                    self.split_position = float(s)
            return
        self.split_direction = w
        t = self.my_X_data @ w
        self.split_position = float(np.median(t))
        if self.position == 'midpoint' and self.direction == 'twomeans':
            X = self.my_X_data
            mid = X.mean(axis=0) + (self._midpoint * self._scale() if self.scaled else self._midpoint)
            # keep both children populated
            lo, hi = np.quantile(t, [self.min_fraction, 1 - self.min_fraction])
            self.split_position = float(np.clip(mid @ w, lo, hi))
        elif self.position == 'gsplit':
            s = self._gradient_split(w)
            if s is not None:
                self.split_position = float(s)
        self.overlap = theta * np.ptp(t)


def oblique_tree(direction='pca', position='median', scaled=False, min_fraction=0.25, whiten=False, **tree_kwargs):
    """A GPTree whose nodes are ObliqueNodes with the given configuration."""
    node_class = type('ObliqueNode_' + direction, (ObliqueNode,),
                      dict(direction=direction, position=position, scaled=scaled, min_fraction=min_fraction,
                           whiten=whiten))
    gpt = GPTree(**tree_kwargs)
    gpt.root.__class__ = node_class
    return gpt
