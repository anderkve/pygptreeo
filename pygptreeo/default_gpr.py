"""Default Gaussian Process Regressor configuration.

This module provides a factory function that creates a default GPRegressorInterface
instance suitable for use with GPTreeO. It sets up sensible defaults for kernel
configuration and hyperparameters commonly used in online regression tasks.
"""

# Standard library imports
from typing import Optional

# Third-party imports
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

# Local imports
from pygptreeo.adapters import SklearnGPAdapter


class DefaultKernelFactory:
    """Builds the default anisotropic (ARD) leaf kernel once the input dimension is known.

    ``ConstantKernel() * Matern(nu, length_scale=[1.0] * n_features)``. Used by
    :func:`Default_GPR` when no kernel is given: the input dimensionality is only
    known at the first ``fit``, so the kernel is created lazily. A class (rather
    than a lambda) keeps the adapter picklable for ``GPTree.save``.
    """

    def __init__(self, nu: float = 1.5):
        self.nu = nu

    def __call__(self, n_features: int):
        return ConstantKernel() * Matern(nu=self.nu, length_scale=np.ones(int(n_features)))

    def __repr__(self) -> str:
        return f"DefaultKernelFactory(nu={self.nu})"


def Default_GPR(
    kernel=None,
    *,
    alpha=1e-10,
    optimizer='fmin_l_bfgs_b',
    n_restarts_optimizer=0,
    normalize_y=False,
    copy_X_train=True,
    n_targets=None,
    random_state=None,
    nu=1.5,
):
    """Create a default GaussianProcessRegressor adapter for GPTreeO.

    This function creates and returns a SklearnGPAdapter wrapping a
    scikit-learn GaussianProcessRegressor with sensible default settings
    for use within the GPTreeO framework.

    Parameters
    ----------
    kernel : Kernel object, optional
        The kernel specifying the covariance function of the GP.
        If None is passed, an anisotropic (ARD) ``ConstantKernel() * Matern(nu)``
        with one length scale per input dimension is built lazily at the first
        ``fit`` call, when the input dimensionality becomes known. Per-dimension
        length scales are what make the ``'min_lengthscale'`` split criterion and
        the GP-aware split evaluation work; an isotropic kernel gives them nothing
        to work with. Note that the kernel's hyperparameters are optimized during fitting.
    alpha : float or ndarray of shape (n_samples,), default=1e-10
        Value added to the diagonal of the kernel matrix during fitting.
        This can represent the expected amount of noise in the observations.
        Larger values correspond to increased noise level. Note that GPTree
        replaces this with the per-point observation noise ``sigma**2`` passed
        to ``update_tree`` / ``fit`` whenever a leaf is trained.
    optimizer : "fmin_l_bfgs_b" or callable, default="fmin_l_bfgs_b"
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable.
    n_restarts_optimizer : int, default=0
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values.
    normalize_y : bool, default=False
        Whether or not to normalize the target values y by removing the mean
        and scaling to unit-variance.
    copy_X_train : bool, default=True
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.
    n_targets : int, optional
        The number of dimensions of the target values. If None, then it is
        inferred from y during fit.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation used to initialize the centers.
    nu : float, default=1.5
        Smoothness of the default Matern kernel (only used when ``kernel`` is None).

    Returns
    -------
    SklearnGPAdapter
        An adapter wrapping a configured scikit-learn GaussianProcessRegressor.

    Examples
    --------
    >>> from pygptreeo import GPTree, Default_GPR
    >>>
    >>> # Use default configuration
    >>> gpt = GPTree(GPR=Default_GPR())
    >>>
    >>> # Use custom kernel
    >>> from sklearn.gaussian_process.kernels import RBF
    >>> gpt = GPTree(GPR=Default_GPR(kernel=RBF()))
    """
    kernel_factory = None
    if kernel is None:
        # Placeholder until the input dimension is known at the first fit; the
        # factory then replaces it with the per-dimension (ARD) version.
        kernel_factory = DefaultKernelFactory(nu=nu)
        kernel = ConstantKernel() * Matern(nu=nu)

    # Create the underlying sklearn GPR
    sklearn_gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        optimizer=optimizer,
        n_restarts_optimizer=n_restarts_optimizer,
        normalize_y=normalize_y,
        copy_X_train=copy_X_train,
        n_targets=n_targets,
        random_state=random_state
    )

    # Wrap in adapter and return
    return SklearnGPAdapter(sklearn_gpr, kernel_factory=kernel_factory)
