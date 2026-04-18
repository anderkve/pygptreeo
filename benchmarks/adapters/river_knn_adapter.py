"""Truly online kNN regressor via the `river` library.

River is designed for instance-by-instance learning. Its ``KNNRegressor`` has
no native uncertainty estimate; we report the standard deviation of the k
nearest-neighbour targets that the model would use for a prediction. This is a
weak but well-defined baseline for online predictive variance.
"""

from __future__ import annotations

import numpy as np
from river.neighbors import KNNRegressor, LazySearch

from .base import OnlineRegressor


class RiverKNNAdapter(OnlineRegressor):
    name = "river_knn"
    supports_uncertainty = True

    def __init__(self, n_dims: int, n_neighbors: int = 8,
                 window_size: int = 4000, std_floor_abs: float = 1e-6):
        self.n_dims = n_dims
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.std_floor_abs = std_floor_abs
        self._engine = LazySearch(window_size=window_size)
        self._model = KNNRegressor(
            n_neighbors=n_neighbors,
            engine=self._engine,
        )
        # Running y-min / y-max so we can report a physically meaningful
        # std floor even when only one neighbour is available.
        self._y_min = float("inf")
        self._y_max = float("-inf")
        self._seen = 0

    def _x_to_dict(self, x: np.ndarray) -> dict:
        x = x.ravel()
        return {i: float(x[i]) for i in range(x.size)}

    def predict(self, X):
        if self._seen == 0:
            return (np.zeros((X.shape[0], 1)),
                    np.full((X.shape[0], 1), np.nan))
        means = np.empty(X.shape[0])
        stds = np.empty(X.shape[0])
        # Physical floor: 0.1 % of observed y-range, or a small absolute value
        # if we haven't seen enough data yet.
        y_range = max(self._y_max - self._y_min, 0.0)
        floor = max(1e-3 * y_range, self.std_floor_abs)
        for i, x in enumerate(X):
            xd = self._x_to_dict(x)
            mean = self._model.predict_one(xd)
            means[i] = mean if mean is not None else 0.0
            # Uncertainty proxy: std of the k-nn target values.
            try:
                neighbors, _dists = self._engine.search(
                    (xd, None), self.n_neighbors
                )
                y_vals = [n[1] for n in neighbors if n[1] is not None]
                s = float(np.std(y_vals)) if len(y_vals) > 1 else 0.0
            except Exception:
                s = 0.0
            stds[i] = max(s, floor)
        return means.reshape(-1, 1), stds.reshape(-1, 1)

    def update(self, x, y):
        xd = self._x_to_dict(x)
        yv = float(y.ravel()[0])
        self._model.learn_one(xd, yv)
        self._y_min = min(self._y_min, yv)
        self._y_max = max(self._y_max, yv)
        self._seen += 1
