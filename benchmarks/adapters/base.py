"""Common interface for online regression methods in the benchmark.

Each adapter must implement:
    - predict(X): returns (mean, std) of shape (N, 1). std may be all-NaN if
      uncertainty is not supported.
    - update(x, y): process a single new (x, y) observation.

This lets the benchmark harness drive every method through the same loop:

    for x, y in stream:
        mean, std = method.predict(x)        # prediction BEFORE seeing (x, y)
        t0 = time()
        method.update(x, y)                  # incorporate the new point
        update_time = time() - t0

Methods that are inherently batch-only (like a standard sklearn GP) are wrapped
with a periodic-retrain strategy: the adapter buffers new points and refits the
underlying model every `retrain_every` observations.
"""

from __future__ import annotations

import abc
from typing import Optional, Tuple

import numpy as np


class OnlineRegressor(abc.ABC):
    """Abstract base class for adapters in the continual-emulation benchmark."""

    #: Short human-readable name used for plots and saved files.
    name: str = "base"

    #: Whether this adapter exposes a predictive standard deviation.
    supports_uncertainty: bool = True

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) with shape (N, 1) for a batch of test inputs."""

    @abc.abstractmethod
    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Ingest a single observation. `x` shape (1, d), `y` shape (1, 1)."""

    def close(self) -> None:
        """Optional cleanup hook."""
        return None
