"""
Adapter for scikit-learn Gaussian Process Regressor.

This module provides an adapter that wraps scikit-learn's GaussianProcessRegressor
to conform to pygptreeo's GPRegressorInterface.
"""

import warnings
from typing import Union, Tuple
import numpy as np
from copy import deepcopy

from scipy.linalg import cho_solve, cholesky, solve_triangular
from sklearn.gaussian_process import GaussianProcessRegressor

from ..gp_interface import GPRegressorInterface


class SklearnGPAdapter(GPRegressorInterface):
    """
    Adapter for scikit-learn's GaussianProcessRegressor.

    This class wraps a scikit-learn GaussianProcessRegressor to provide
    the interface required by pygptreeo, allowing seamless integration
    with the existing codebase.

    Parameters
    ----------
    gpr : GaussianProcessRegressor
        A scikit-learn GaussianProcessRegressor instance to wrap.

    Attributes
    ----------
    _gpr : GaussianProcessRegressor
        The underlying scikit-learn GP regressor.

    Examples
    --------
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    >>> from pygptreeo.adapters import SklearnGPAdapter
    >>>
    >>> kernel = ConstantKernel(1.0) * RBF(1.0)
    >>> sklearn_gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-10)
    >>> adapter = SklearnGPAdapter(sklearn_gpr)
    >>>
    >>> # Now use adapter with pygptreeo
    >>> from pygptreeo import GPTree
    >>> gpt = GPTree(GPR=adapter)
    """

    def __init__(self, gpr: GaussianProcessRegressor, kernel_factory=None):
        """
        Initialize the adapter with a scikit-learn GaussianProcessRegressor.

        Parameters
        ----------
        gpr : GaussianProcessRegressor
            The scikit-learn GP regressor to wrap.
        kernel_factory : callable, optional
            ``kernel_factory(n_features) -> Kernel``. If given, the wrapped GPR's
            kernel is replaced by ``kernel_factory(X.shape[1])`` at the first
            ``fit`` call, so that kernels needing
            the input dimension (e.g. one length scale per dimension) can be
            built lazily. It is dropped once used, or if ``set_kernel`` is called.
        """
        if not isinstance(gpr, GaussianProcessRegressor):
            raise TypeError(
                f"Expected GaussianProcessRegressor, got {type(gpr).__name__}"
            )
        self._gpr = gpr
        self._kernel_factory = kernel_factory

    def _ensure_kernel(self, n_features: int) -> None:
        """Build the kernel from the factory (once) now that n_features is known."""
        if self._kernel_factory is not None:
            self._gpr.kernel = self._kernel_factory(int(n_features))
            self._kernel_factory = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SklearnGPAdapter':
        """
        Train the GP on the provided data.

        Parameters
        ----------
        X : np.ndarray
            Training input data of shape (n_samples, n_features).
        y : np.ndarray
            Training target values of shape (n_samples,) or (n_samples, 1).

        Returns
        -------
        self : SklearnGPAdapter
            The fitted adapter instance.
        """
        self._ensure_kernel(np.asarray(X).shape[1])
        self._gpr.fit(X, y)
        return self

    def supports_multitarget(self) -> bool:
        """scikit-learn's GPR fits a 2-D ``y`` with one shared kernel (multi-target)."""
        return True

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions at test points.

        Parameters
        ----------
        X : np.ndarray
            Test input data of shape (n_samples, n_features).
        return_std : bool, default=False
            If True, return both mean predictions and standard deviations.

        Returns
        -------
        y_mean : np.ndarray
            Mean predictions of shape (n_samples,) or (n_samples, 1).
        y_std : np.ndarray, optional
            Standard deviations of shape (n_samples,) or (n_samples, 1).
            Only returned if return_std=True.
        """
        sk = self._gpr
        X = np.asarray(X)
        fitted = all(hasattr(sk, a) for a in ('kernel_', 'L_', 'alpha_', 'X_train_', '_y_train_mean', '_y_train_std'))
        if not fitted or X.ndim != 2 or X.dtype != np.float64:
            if return_std:
                y_mean, y_std = sk.predict(X, return_std=True)
                return y_mean, y_std
            return sk.predict(X, return_std=False)
        # The posterior of GaussianProcessRegressor.predict from its fitted attributes,
        # without its input validation: the same operations, so the same numbers.
        K_trans = sk.kernel_(X, sk.X_train_)
        y_mean = sk._y_train_std * (K_trans @ sk.alpha_) + sk._y_train_mean
        if y_mean.ndim > 1 and y_mean.shape[1] == 1:
            y_mean = np.squeeze(y_mean, axis=1)
        if not return_std:
            return y_mean
        V = solve_triangular(sk.L_, K_trans.T, lower=True, check_finite=False)
        y_var = sk.kernel_.diag(X).copy()
        y_var -= np.einsum("ij,ji->i", V.T, V)
        negative = y_var < 0
        if np.any(negative):
            warnings.warn("Predicted variances smaller than 0. Setting those variances to 0.")
            y_var[negative] = 0.0
        y_var = np.outer(y_var, sk._y_train_std ** 2).reshape(*y_var.shape, -1)
        if y_var.shape[1] == 1:
            y_var = np.squeeze(y_var, axis=1)
        return y_mean, np.sqrt(y_var)

    def is_trained(self) -> bool:
        """
        Check whether the GP has been trained on data.

        For scikit-learn, we check for the presence of the 'kernel_' attribute,
        which is set after fitting.

        Returns
        -------
        bool
            True if the GP has been fitted, False otherwise.
        """
        return hasattr(self._gpr, 'kernel_')

    def set_observation_noise(self, alpha: Union[float, np.ndarray]) -> None:
        """
        Set the observation noise levels for training data.

        In scikit-learn, this is done by setting the 'alpha' attribute.

        Parameters
        ----------
        alpha : float or np.ndarray
            If float: a single noise level applied to all observations.
            If array: per-observation noise levels of shape (n_samples,).
        """
        if isinstance(alpha, np.ndarray):
            self._gpr.alpha = alpha.flatten()
        else:
            self._gpr.alpha = alpha

    def clone(self) -> 'SklearnGPAdapter':
        """
        Create a deep copy of this GP regressor.

        Returns
        -------
        SklearnGPAdapter
            A deep copy of this adapter instance.
        """
        return SklearnGPAdapter(deepcopy(self._gpr), kernel_factory=self._kernel_factory)

    def get_kernel(self):
        """
        Get the kernel object.

        For an untrained GP, this returns the initial kernel.
        For a trained GP, this returns the fitted kernel with optimized hyperparameters.

        Returns
        -------
        kernel
            The kernel object (sklearn.gaussian_process.kernels.Kernel).
        """
        # If trained, return the fitted kernel with optimized hyperparameters
        if self.is_trained():
            return self._gpr.kernel_
        # Otherwise, return the initial kernel
        return self._gpr.kernel

    def set_kernel(self, kernel) -> None:
        """
        Set the kernel object.

        This is used for hyperparameter inheritance when creating child nodes.

        Parameters
        ----------
        kernel
            The kernel object to set (sklearn.gaussian_process.kernels.Kernel).
        """
        self._gpr.kernel = kernel
        self._kernel_factory = None

    def get_length_scales(self, n_features: int):
        """Per-dimension length scales from the fitted kernel, or None.

        Collects every anisotropic (length ``n_features``) 'length_scale'
        hyperparameter of the (possibly composite) kernel. When several
        components contribute, the per-dimension minimum is taken (the shortest
        length scale governs the resolution). Isotropic (scalar) length scales
        carry no per-dimension information and are ignored, so a kernel with
        only isotropic components returns None and the caller falls back to a
        data-based criterion.
        """
        kernel = self.get_kernel()
        if kernel is None:
            return None
        try:
            params = kernel.get_params(deep=True)
        except Exception:
            return None

        ls_arrays = []
        for key, val in params.items():
            if not key.endswith('length_scale'):  # excludes 'length_scale_bounds'
                continue
            arr = np.atleast_1d(np.asarray(val, dtype=float)).ravel()
            if arr.size == n_features and n_features > 1:
                ls_arrays.append(arr)

        if not ls_arrays:
            return None
        return np.min(np.vstack(ls_arrays), axis=0)

    def refit_fixed(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Fit with the fitted kernel kept (no hyperparameter optimisation)."""
        sk = self._gpr
        if not hasattr(sk, 'kernel_'):
            return False
        optimizer, kernel = sk.optimizer, sk.kernel
        sk.optimizer, sk.kernel = None, sk.kernel_
        try:
            sk.fit(X, y)
        finally:
            sk.optimizer, sk.kernel = optimizer, kernel
        return True

    def append(self, x: np.ndarray, y, noise: float) -> bool:
        """One more row for the Cholesky factor: the factor of the kernel matrix
        with the new point is the old one with a row ``[l, lam]`` below it."""
        sk = self._gpr
        if not hasattr(sk, 'L_'):
            return False
        n = sk.X_train_.shape[0]
        x = np.asarray(x, dtype=float).reshape(1, -1)
        l = solve_triangular(sk.L_, sk.kernel_(sk.X_train_, x)[:, 0], lower=True, check_finite=False)
        lam2 = sk.kernel_.diag(x)[0] + noise - l @ l
        if not lam2 > 0.0:
            return False                  # the factor cannot tell the point from those it has
        L = np.zeros((n + 1, n + 1))
        L[:n, :n] = sk.L_
        L[n, :n] = l
        L[n, n] = np.sqrt(lam2)
        y = np.asarray(y, dtype=float).reshape((1,) + sk.y_train_.shape[1:])
        if sk.normalize_y:
            y = (y - sk._y_train_mean) / sk._y_train_std
        sk.X_train_ = np.vstack((sk.X_train_, x))
        sk.y_train_ = np.concatenate((sk.y_train_, y))
        sk.alpha = np.append(np.broadcast_to(sk.alpha, (n,)), noise)
        sk.L_ = L
        sk.alpha_ = cho_solve((L, True), sk.y_train_, check_finite=False)
        return True

    def nbytes(self) -> int:
        """Size of the training data and the Cholesky factor that a fit stores."""
        return sum(getattr(self._gpr, name).nbytes for name in ('X_train_', 'y_train_', 'L_', 'alpha_')
                   if hasattr(self._gpr, name))

    def release(self) -> bool:
        """Free the Cholesky factor and the weights. The training data, the noise
        and the fitted kernel stay, and determine them."""
        if not hasattr(self._gpr, 'L_'):
            return False
        del self._gpr.L_, self._gpr.alpha_
        return True

    def restore(self) -> bool:
        """The factor and the weights as scikit-learn's ``fit`` computes them."""
        sk = self._gpr
        if hasattr(sk, 'L_') or not hasattr(sk, 'X_train_'):
            return False
        K = sk.kernel_(sk.X_train_)
        K[np.diag_indices_from(K)] += sk.alpha
        sk.L_ = cholesky(K, lower=True, check_finite=False)
        sk.alpha_ = cho_solve((sk.L_, True), sk.y_train_, check_finite=False)
        return True

    # Provide access to the underlying scikit-learn GPR for advanced users
    @property
    def sklearn_gpr(self) -> GaussianProcessRegressor:
        """
        Access the underlying scikit-learn GaussianProcessRegressor.

        This is provided for advanced users who need direct access to
        scikit-learn specific features.

        Returns
        -------
        GaussianProcessRegressor
            The underlying scikit-learn GP regressor.
        """
        return self._gpr

    def __repr__(self) -> str:
        """String representation of the adapter."""
        if self._kernel_factory is not None:
            return f"SklearnGPAdapter({self._gpr}, kernel_factory={self._kernel_factory!r})"
        return f"SklearnGPAdapter({self._gpr})"
