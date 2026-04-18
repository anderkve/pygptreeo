"""Random Forest baseline with periodic retraining.

Uses the variance across trees as a (pseudo-)predictive standard deviation. The
RF is retrained from scratch every ``retrain_every`` observations on the full
accumulated buffer (capped at ``max_train_points``). This is a common,
fast-to-fit, non-GP baseline for expensive-function emulation.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base import OnlineRegressor


class RandomForestAdapter(OnlineRegressor):
    name = "random_forest"
    supports_uncertainty = True

    def __init__(self, n_dims: int, retrain_every: int = 200,
                 n_estimators: int = 300, max_train_points: int = 20000,
                 random_state: int = 0):
        self.n_dims = n_dims
        self.retrain_every = retrain_every
        self.n_estimators = n_estimators
        self.max_train_points = max_train_points
        self.random_state = random_state
        self._X_buf: list[np.ndarray] = []
        self._y_buf: list[np.ndarray] = []
        self._steps_since_refit = 0
        self._n_refits = 0
        self._model: RandomForestRegressor | None = None
        self._trained = False

    def _refit(self):
        n = len(self._X_buf)
        if n < 3:
            return
        X = np.vstack(self._X_buf)
        y = np.vstack(self._y_buf).ravel()
        if n > self.max_train_points:
            rng = np.random.default_rng(self.random_state + self._n_refits)
            idx = rng.choice(n, size=self.max_train_points, replace=False)
            X = X[idx]
            y = y[idx]
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            n_jobs=1,
            random_state=self.random_state + self._n_refits,
        )
        self._model.fit(X, y)
        self._trained = True
        self._n_refits += 1

    def predict(self, X):
        if not self._trained:
            return (np.zeros((X.shape[0], 1)),
                    np.full((X.shape[0], 1), np.nan))
        preds = np.stack([t.predict(X) for t in self._model.estimators_], axis=0)
        mean = preds.mean(axis=0).reshape(-1, 1)
        std = preds.std(axis=0).reshape(-1, 1)
        return mean, std

    def update(self, x, y):
        self._X_buf.append(x.reshape(1, -1).copy())
        self._y_buf.append(y.reshape(1, 1).copy())
        self._steps_since_refit += 1
        if self._steps_since_refit >= self.retrain_every:
            self._refit()
            self._steps_since_refit = 0
