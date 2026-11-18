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

    def supports_multitarget(self) -> bool:
        """Whether ``fit`` accepts a 2-D ``y`` of shape (n_samples, n_targets) and
        ``predict`` then returns (n_samples, n_targets) means and standard
        deviations, all targets sharing one kernel. Required by
        ``GPTree(output_model='shared')``. Backends default to False.
        """
        return False

    def get_length_scales(self, n_features: int) -> Optional[np.ndarray]:
        """Return the fitted per-dimension ARD length scales, or None.

        Optional capability used by the 'min_lengthscale' split criterion.
        Backends that cannot expose length scales return None (the tree then
        falls back to data spread). The returned array has shape (n_features,).
        """
        return None

    def get_metric(self, n_features: int) -> Optional[np.ndarray]:
        """The matrix ``M`` (n_features, n_features) of a fitted kernel whose
        distance ``(x - x') M (x - x')^T`` is not along the inputs, or None.

        Optional capability used by the 'min_lengthscale' split criterion: such
        a leaf is cut across the direction of its shortest length scale.
        """
        return None

    def refit_fixed(self, X: np.ndarray, y: np.ndarray, hyperparameters: Optional[np.ndarray] = None) -> bool:
        """Fit to ``(X, y)`` without optimising: with ``hyperparameters`` (as
        ``get_hyperparameters`` returns them), or with those of the last fit.

        Optional capability used by global models with ``cheap_leaf_refresh``
        and by ``GPTree.apply_fit_result``. Returns False if the backend
        cannot, in which case the caller does a full fit.
        """
        return False

    def get_hyperparameters(self) -> Optional[np.ndarray]:
        """The hyperparameters of the last fit as a 1-D array that
        ``refit_fixed`` takes, or None without a fit or if the backend cannot.

        Optional capability used by ``GPTree.update_tree(defer_fits=True)``.
        """
        return None

    def unfitted_copy(self) -> Optional['GPRegressorInterface']:
        """A copy with the settings of this GP and nothing of its fits, to be
        fitted in another process, or None if the backend cannot.

        Optional capability used by ``GPTree.update_tree(defer_fits=True)``. A
        backend without it has its leaves fitted in place.
        """
        return None

    def append(self, x: np.ndarray, y, noise: float) -> bool:
        """Condition the fitted GP on one more point, with its hyperparameters kept.

        ``x`` has shape (1, n_features), ``y`` is the point's target (one row of
        targets for a multi-target fit) and ``noise`` its noise variance, all as
        ``fit`` takes them. The result is the GP that a fit with fixed
        hyperparameters to the points of the last fit and this one would give,
        at a cost quadratic in the number of points. Optional capability used
        by ``GPTree(append_points_between_fits=True)``. Returns False if there
        is no fit to extend or the backend cannot.
        """
        return False

    def mean_weights(self, X: np.ndarray) -> Optional[np.ndarray]:
        """The weights of the training targets in the predicted mean at ``X``, of
        shape (n_samples, n_train): the mean is ``weights @ y_train``.

        Optional capability used by ``GPTree(output_model='aligned')``. Returns
        None without a fit or if the backend cannot.
        """
        return None

    def leave_one_out(self):
        """The predictions of each training point from the others, with the
        hyperparameters kept: ``(weights, variance)``. Row ``i`` of the weights
        (n_train, n_train) gives the mean at training point ``i`` as
        ``weights[i] @ y_train``, and ``variance[i]`` is the variance of that
        prediction of the observed target, both in the units of ``fit``.

        Optional capability used by the 'shared' and 'aligned' output models.
        Returns None without a fit or if the backend cannot.
        """
        return None

    def nbytes(self) -> int:
        """Approximate size in bytes of what a fit has stored, 0 without a fit.

        Used by ``GPTree(max_memory_mb=...)``. A backend that does not report
        it is not counted.
        """
        return 0

    def release(self) -> bool:
        """Free the part of the fitted state that ``restore`` can compute again.

        Optional capability used to put a leaf to sleep under
        ``GPTree(max_memory_mb=...)``. Returns False if there is nothing to
        free or the backend cannot; its leaves are then never put to sleep.
        """
        return False

    def restore(self) -> bool:
        """Compute again what ``release`` freed, so that the GP predicts as before
        (to numerical precision after ``append``). Returns False if the backend
        cannot."""
        return False
