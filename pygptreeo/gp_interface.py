"""
Abstract interface for Gaussian Process regressors.

This module provides an abstract base class that defines the interface
required by pygptreeo for GP regressors. Different GP implementations
(scikit-learn, GPyTorch, etc.) can be integrated by creating adapter
classes that implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union
import numpy as np


class GPRegressorInterface(ABC):
    """
    Abstract base class defining the interface for Gaussian Process regressors
    used in pygptreeo.

    This interface allows pygptreeo to work with different GP implementations
    while keeping the main DLGP algorithm agnostic to the specific GP package.

    Any GP implementation must provide the following core operations:
    - Training (fit)
    - Prediction with uncertainty (predict)
    - Training state checking (is_trained)
    - Observation noise configuration (set_observation_noise)
    - Kernel covariance computation (get_kernel_covariance)
    - Deep copying (clone)
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GPRegressorInterface':
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
        self : GPRegressorInterface
            The fitted GP regressor instance.
        """
        pass

    @abstractmethod
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
            If False, return only mean predictions.

        Returns
        -------
        y_mean : np.ndarray
            Mean predictions of shape (n_samples,) or (n_samples, 1).
        y_std : np.ndarray, optional
            Standard deviations of shape (n_samples,) or (n_samples, 1).
            Only returned if return_std=True.
        """
        pass

    @abstractmethod
    def is_trained(self) -> bool:
        """
        Check whether the GP has been trained on data.

        Returns
        -------
        bool
            True if the GP has been fitted, False otherwise.
        """
        pass

    @abstractmethod
    def set_observation_noise(self, alpha: Union[float, np.ndarray]) -> None:
        """
        Set the observation noise levels for training data.

        This is used to specify per-observation noise variances or a single
        global noise level. The noise is applied during the next fit() call.

        Parameters
        ----------
        alpha : float or np.ndarray
            If float: a single noise level applied to all observations.
            If array: per-observation noise levels of shape (n_samples,).
        """
        pass

    @abstractmethod
    def get_kernel_covariance(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the kernel covariance matrix K(X, X).

        This is used by GPForest to compute prior variances.

        Parameters
        ----------
        X : np.ndarray
            Input points of shape (n_samples, n_features).

        Returns
        -------
        K : np.ndarray
            Covariance matrix of shape (n_samples, n_samples).
        """
        pass

    @abstractmethod
    def clone(self) -> 'GPRegressorInterface':
        """
        Create a deep copy of this GP regressor.

        This is used when creating child nodes in the tree, which need
        independent copies of the parent's GP configuration.

        Returns
        -------
        GPRegressorInterface
            A deep copy of this GP regressor instance.
        """
        pass

    @abstractmethod
    def get_kernel(self):
        """
        Get the kernel object or trained kernel parameters.

        This is used for hyperparameter inheritance when creating child nodes.

        Returns
        -------
        kernel
            The kernel object. The exact type depends on the GP implementation.
        """
        pass

    @abstractmethod
    def set_kernel(self, kernel) -> None:
        """
        Set the kernel object or trained kernel parameters.

        This is used for hyperparameter inheritance when creating child nodes.

        Parameters
        ----------
        kernel
            The kernel object to set. The exact type depends on the GP implementation.
        """
        pass

    def reset_training(self) -> None:
        """Discard any fitted state, returning the GP to an untrained condition.

        After this call :meth:`is_trained` must return False and the next
        :meth:`predict` falls back to the prior until :meth:`fit` is called
        again. Used by interaction pruning: a child built from a (possibly
        trained) template must forget the template's fit before being refit on
        its own data with a pruned kernel. The default is a no-op for backends
        that have no separate fitted state.
        """
        pass

    def log_marginal_likelihood(self) -> float:
        """Log marginal likelihood (model evidence) at the fitted hyperparameters.

        Used to compare different kernels on the same data for automatic kernel
        selection (see :class:`pygptreeo.auto_select.AutoSelectGPR`). Returns
        ``-inf`` when unavailable or untrained. Backends that cannot expose it
        keep the default.
        """
        return float("-inf")

    def get_length_scales(self, n_features: int) -> Optional[np.ndarray]:
        """Return the fitted per-dimension ARD length scales, or None.

        Optional capability used by the 'min_lengthscale' split criterion.
        Backends that cannot expose length scales return None (the tree then
        falls back to data spread). The returned array has shape (n_features,).
        """
        return None
