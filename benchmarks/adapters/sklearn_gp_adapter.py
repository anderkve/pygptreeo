"""Full-refit sklearn GP baseline.

A standard ``sklearn.gaussian_process.GaussianProcessRegressor`` cannot be
updated online. For a fair benchmark we buffer every observation and refit the
GP every ``retrain_every`` new points. To keep the O(N^3) cost manageable the
adapter caps the training-set size at ``max_train_points`` using a uniform
reservoir-style subsample of the buffer. This mirrors what a practitioner would
do if they wanted a single global GP on a stream.
"""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from .base import OnlineRegressor


class SklearnGPAdapter(OnlineRegressor):
    name = "sklearn_gp"
    supports_uncertainty = True

    def __init__(self, n_dims: int, retrain_every: int = 200,
                 max_train_points: int = 2000, random_state: int = 0):
        self.n_dims = n_dims
        self.retrain_every = retrain_every
        self.max_train_points = max_train_points
        self.random_state = random_state
        self._X_buf: list[np.ndarray] = []
        self._y_buf: list[np.ndarray] = []
        self._steps_since_refit = 0
        self._model: GaussianProcessRegressor | None = None
        self._trained = False

    def _make_model(self):
        kernel = ConstantKernel(1.0, (1e-3, 1e8)) * Matern(
            nu=1.5,
            length_scale=[1.0] * self.n_dims,
            length_scale_bounds=[(1e-5, 1e5)] * self.n_dims,
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=self.random_state,
        )

    def _refit(self):
        n = len(self._X_buf)
        if n < 3:
            return
        X = np.vstack(self._X_buf)
        y = np.vstack(self._y_buf).ravel()
        if n > self.max_train_points:
            rng = np.random.default_rng(self.random_state)
            # Keep the most recent 25% plus a random sample for coverage.
            n_recent = self.max_train_points // 4
            n_random = self.max_train_points - n_recent
            idx_random = rng.choice(n - n_recent, size=n_random, replace=False)
            idx = np.concatenate([idx_random, np.arange(n - n_recent, n)])
            X = X[idx]
            y = y[idx]
        self._model = self._make_model()
        self._model.fit(X, y)
        self._trained = True

    def predict(self, X):
        if not self._trained:
            return (np.zeros((X.shape[0], 1)),
                    np.full((X.shape[0], 1), np.nan))
        mean, std = self._model.predict(X, return_std=True)
        return mean.reshape(-1, 1), std.reshape(-1, 1)

    def update(self, x, y):
        self._X_buf.append(x.reshape(1, -1).copy())
        self._y_buf.append(y.reshape(1, 1).copy())
        self._steps_since_refit += 1
        if self._steps_since_refit >= self.retrain_every:
            self._refit()
            self._steps_since_refit = 0
