"""Incremental (rank-1 Cholesky) Gaussian process regressor.

This module provides :class:`IncrementalGP`, a GP regressor designed for the
streaming setting of pygptreeo. A standard GP refit is O(n^3) because it
re-factorizes the kernel matrix from scratch. In an online tree, however, points
arrive one at a time, and most of the time the kernel hyperparameters do not need
to change between consecutive points.

:class:`IncrementalGP` exploits this:

* :meth:`fit` performs a full (expensive) fit: it optimizes the kernel
  hyperparameters and builds the Cholesky factor of the kernel matrix. This is
  the "re-optimization" step and is meant to run only periodically.
* :meth:`add_observation` incorporates a single new point with the hyperparameters
  held fixed, using an exact rank-1 update of the Cholesky factor. This costs
  O(n^2) instead of O(n^3) and -- crucially -- lets the posterior reflect every
  new point immediately, rather than ignoring recent points until the next full
  refit.

The hyperparameter optimization in :meth:`fit` is delegated to scikit-learn's
``GaussianProcessRegressor`` (keeping behaviour consistent with the rest of the
package); the resulting factorization is then mirrored into this object so that
subsequent rank-1 updates and predictions are self-contained.
"""

from copy import deepcopy
from typing import Optional, Tuple, Union

import numpy as np
from scipy.linalg import cho_solve, solve_triangular

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from pygptreeo.gp_interface import GPRegressorInterface


class IncrementalGP(GPRegressorInterface):
    """A GP regressor supporting exact rank-1 incremental updates.

    Parameters
    ----------
    kernel : sklearn kernel, optional
        The covariance kernel. Defaults to ``ConstantKernel() * Matern(nu=1.5)``.
    optimizer : str or callable, default='fmin_l_bfgs_b'
        Hyperparameter optimizer passed to the internal scikit-learn GP used by
        :meth:`fit`.
    n_restarts_optimizer : int, default=0
        Number of optimizer restarts for the full fit.
    jitter : float, default=1e-10
        Small value added to the kernel diagonal for numerical stability (in
        addition to the per-observation noise).
    """

    def __init__(self,
                 kernel=None,
                 optimizer: str = "fmin_l_bfgs_b",
                 n_restarts_optimizer: int = 0,
                 jitter: float = 1e-10):
        self.kernel = kernel if kernel is not None else ConstantKernel() * Matern(nu=1.5)
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.jitter = jitter

        # Observation noise (variance). Set by the caller before fit(), either as
        # a scalar or a per-observation array. Mirrors sklearn's `alpha`.
        self.alpha = 1e-10

        # Trained state (None until the first full fit).
        self.kernel_ = None          # optimized kernel, held fixed across updates
        self.X_train_ = None         # (n, d)
        self.y_train_ = None         # (n,)
        self.alpha_train_ = None     # (n,) per-observation noise variance
        self.L_ = None               # (n, n) lower Cholesky of K + noise
        self.alpha_vec_ = None       # (n,) = K^{-1} y

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _noise_array(self, n: int) -> np.ndarray:
        """Return the per-observation noise variance as an array of length n."""
        alpha = self.alpha
        if np.isscalar(alpha):
            return np.full(n, float(alpha))
        arr = np.asarray(alpha, dtype=float).reshape(-1)
        if arr.shape[0] == n:
            return arr
        # Fall back to a constant if the length does not match.
        return np.full(n, float(arr.ravel()[0]))

    # ------------------------------------------------------------------ #
    # Interface implementation
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "IncrementalGP":
        """Full fit: optimize hyperparameters and build the Cholesky factor."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n = X.shape[0]
        alpha_arr = self._noise_array(n)

        # Delegate hyperparameter optimization to scikit-learn, then mirror its
        # factorization into our own state so incremental updates are self-contained.
        gpr = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=alpha_arr,
            optimizer=self.optimizer,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=False,
            copy_X_train=True,
        )
        gpr.fit(X, y)

        self.kernel_ = gpr.kernel_
        self.L_ = np.array(gpr.L_, dtype=float)             # lower Cholesky of K + noise
        self.alpha_vec_ = np.asarray(gpr.alpha_, dtype=float).reshape(-1)
        self.X_train_ = np.array(gpr.X_train_, dtype=float)
        self.y_train_ = np.asarray(gpr.y_train_, dtype=float).reshape(-1)
        self.alpha_train_ = alpha_arr.copy()
        return self

    def add_observation(self, x: np.ndarray, y: float, alpha: float) -> None:
        """Incorporate one new point via a rank-1 Cholesky update (hyperparameters fixed).

        Parameters
        ----------
        x : np.ndarray
            New input point, shape (1, d) or (d,).
        y : float
            New target value.
        alpha : float
            Observation noise *variance* for the new point.
        """
        if self.L_ is None:
            # Not yet fitted; nothing to update incrementally.
            return

        x = np.asarray(x, dtype=float).reshape(1, -1)
        y = float(np.asarray(y).reshape(-1)[0])
        a = float(alpha)

        # Cross-covariance with existing points and the new self-covariance.
        k_vec = self.kernel_(self.X_train_, x).reshape(-1)        # (n,)
        k_ss = float(self.kernel_(x)[0, 0]) + a + self.jitter     # scalar

        # Rank-1 Cholesky extension:  L_new = [[L, 0], [l^T, l_star]]
        l = solve_triangular(self.L_, k_vec, lower=True)          # (n,)
        d2 = k_ss - l @ l
        if d2 <= 0.0:
            d2 = self.jitter  # numerical safeguard for the Schur complement
        l_star = np.sqrt(d2)

        n = self.L_.shape[0]
        L_new = np.zeros((n + 1, n + 1), dtype=float)
        L_new[:n, :n] = self.L_
        L_new[n, :n] = l
        L_new[n, n] = l_star
        self.L_ = L_new

        self.X_train_ = np.vstack([self.X_train_, x])
        self.y_train_ = np.append(self.y_train_, y)
        self.alpha_train_ = np.append(self.alpha_train_, a)

        # Refresh alpha_vec_ = K^{-1} y using the updated factor (O(n^2)).
        self.alpha_vec_ = cho_solve((self.L_, True), self.y_train_)

    def predict(self, X: np.ndarray, return_std: bool = False
                ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict the posterior mean (and optionally std) at the points X."""
        X = np.asarray(X, dtype=float)

        if self.L_ is None:
            # Prior predictive: zero mean, prior std from the (initial) kernel.
            mean = np.zeros(X.shape[0])
            if return_std:
                std = np.sqrt(np.maximum(self.kernel.diag(X), 0.0))
                return mean, std
            return mean

        K_trans = self.kernel_(X, self.X_train_)   # (m, n)
        mean = K_trans @ self.alpha_vec_
        if not return_std:
            return mean

        V = solve_triangular(self.L_, K_trans.T, lower=True)   # (n, m)
        var = self.kernel_.diag(X) - np.einsum("ij,ij->j", V, V)
        var = np.maximum(var, 0.0)
        return mean, np.sqrt(var)

    def is_trained(self) -> bool:
        return self.L_ is not None

    def supports_incremental_update(self) -> bool:
        return True

    def set_observation_noise(self, alpha: Union[float, np.ndarray]) -> None:
        self.alpha = alpha

    def get_kernel_covariance(self, X: np.ndarray) -> np.ndarray:
        kernel = self.kernel_ if self.kernel_ is not None else self.kernel
        return kernel(X)

    def clone(self) -> "IncrementalGP":
        # Clone the configuration only; trained state is not carried over
        # (child nodes train their own GP).
        return IncrementalGP(
            kernel=deepcopy(self.kernel),
            optimizer=self.optimizer,
            n_restarts_optimizer=self.n_restarts_optimizer,
            jitter=self.jitter,
        )

    def get_kernel(self):
        return self.kernel_ if self.kernel_ is not None else self.kernel

    def set_kernel(self, kernel) -> None:
        self.kernel = kernel

    def __repr__(self) -> str:
        return (f"IncrementalGP(kernel={self.kernel}, "
                f"n_restarts_optimizer={self.n_restarts_optimizer})")
