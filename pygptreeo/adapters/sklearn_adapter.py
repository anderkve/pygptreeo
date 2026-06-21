"""
Adapter for scikit-learn Gaussian Process Regressor.

This module provides an adapter that wraps scikit-learn's GaussianProcessRegressor
to conform to pygptreeo's GPRegressorInterface.
"""

from typing import Union, Tuple
import numpy as np
from copy import deepcopy

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

    def __init__(self, gpr: GaussianProcessRegressor):
        """
        Initialize the adapter with a scikit-learn GaussianProcessRegressor.

        Parameters
        ----------
        gpr : GaussianProcessRegressor
            The scikit-learn GP regressor to wrap.
        """
        if not isinstance(gpr, GaussianProcessRegressor):
            raise TypeError(
                f"Expected GaussianProcessRegressor, got {type(gpr).__name__}"
            )
        self._gpr = gpr

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
        self._gpr.fit(X, y)
        return self

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
        if return_std:
            y_mean, y_std = self._gpr.predict(X, return_std=True)
            return y_mean, y_std
        else:
            return self._gpr.predict(X, return_std=False)

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

    def get_kernel_covariance(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the kernel covariance matrix K(X, X).

        This uses scikit-learn's kernel object to compute the covariance.

        Parameters
        ----------
        X : np.ndarray
            Input points of shape (n_samples, n_features).

        Returns
        -------
        K : np.ndarray
            Covariance matrix of shape (n_samples, n_samples).
        """
        return self._gpr.kernel(X)

    def clone(self) -> 'SklearnGPAdapter':
        """
        Create a deep copy of this GP regressor.

        Returns
        -------
        SklearnGPAdapter
            A deep copy of this adapter instance.
        """
        return SklearnGPAdapter(deepcopy(self._gpr))

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

    def get_length_scales(self, n_features: int):
        """
        Extract per-dimension length scales from the (fitted) sklearn kernel.

        Walks all 'length_scale' hyperparameters in the (possibly composite)
        kernel and combines them into a single per-dimension length scale. For a
        trained GP the optimized length scales are used; otherwise the initial
        ones. Anisotropic length-scale vectors (one value per dimension) are used
        as-is; isotropic (scalar) length scales are broadcast across dimensions.

        When several kernel components each contribute length scales (e.g. a sum
        of an anisotropic RQ and a Matern kernel), the per-dimension minimum is
        taken, since the shortest length scale is what governs the resolution
        needed in that dimension.

        Parameters
        ----------
        n_features : int
            Number of input dimensions.

        Returns
        -------
        Optional[np.ndarray]
            Array of shape (n_features,) of effective length scales, or None if
            the kernel exposes no usable length-scale hyperparameter.
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
            # Match 'length_scale' but not 'length_scale_bounds'
            if not key.endswith('length_scale'):
                continue
            arr = np.atleast_1d(np.asarray(val, dtype=float)).ravel()
            if arr.size == n_features:
                ls_arrays.append(arr)
            elif arr.size == 1:
                ls_arrays.append(np.full(n_features, arr.item()))
            # Mismatched sizes are ignored.

        if not ls_arrays:
            return None

        # Combine components: the shortest length scale dominates resolution.
        return np.min(np.vstack(ls_arrays), axis=0)

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
        return f"SklearnGPAdapter({self._gpr})"
