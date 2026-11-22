"""Default Gaussian Process Regressor configuration.

This module provides a factory function that creates a default GPRegressorInterface
instance suitable for use with GPTreeO. It sets up sensible defaults for kernel
configuration and hyperparameters commonly used in online regression tasks.
"""

# Third-party imports
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

# Local imports
from pygptreeo.adapters import SklearnGPAdapter
from pygptreeo.kernels import ObliqueMatern


class DefaultKernelFactory:
    """Builds the default anisotropic (ARD) leaf kernel once the input dimension is known.

    ``ConstantKernel() * Matern(nu, length_scale=[1.0] * n_features)``. Used by
    :func:`Default_GPR` when no kernel is given: the input dimensionality is only
    known at the first ``fit``, so the kernel is created lazily. A class (rather
    than a lambda) keeps the adapter picklable for ``GPTree.save``.

    With ``oblique`` and more than one input the Matern kernel is an
    :class:`~pygptreeo.kernels.ObliqueMatern`.
    """

    def __init__(self, nu: float = 1.5, oblique: bool = False):
        self.nu = nu
        self.oblique = oblique

    def __call__(self, n_features: int):
        if self.oblique and n_features > 1:
            return ConstantKernel() * ObliqueMatern(nu=self.nu, length_scale=np.ones(int(n_features)))
        return ConstantKernel() * Matern(nu=self.nu, length_scale=np.ones(int(n_features)))

    def __repr__(self) -> str:
        return f"DefaultKernelFactory(nu={self.nu}, oblique={self.oblique})"


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
    oblique=False,
):
    """A scikit-learn GaussianProcessRegressor in a SklearnGPAdapter, with the
    default settings of pygptreeo.

    Parameters
    ----------
    kernel : Kernel, optional
        The kernel of the GP. By default ``ConstantKernel() * Matern(nu)`` with
        one length scale per input, built at the first fit, when the number of
        inputs is known. The ``'min_lengthscale'`` split criterion uses these
        length scales.
    alpha, optimizer, n_restarts_optimizer, normalize_y, copy_X_train, n_targets, random_state
        Passed to ``GaussianProcessRegressor``. GPTree replaces ``alpha`` by the
        noise variances ``sigma**2`` of a leaf's points at each fit.
    nu : float, default=1.5
        Smoothness of the default Matern kernel (only used when ``kernel`` is None).
    oblique : bool, default=False
        Let the length scales of the default kernel lie along combinations of
        the inputs (``ObliqueMatern``; only used when ``kernel`` is None). A fit
        keeps such a kernel only if it lowers the leave-one-out error.

    Examples
    --------
    >>> from pygptreeo import GPTree, Default_GPR
    >>> from sklearn.gaussian_process.kernels import RBF
    >>> gpt = GPTree(GPR=Default_GPR(kernel=RBF()))
    """
    kernel_factory = None
    if kernel is None:
        # Placeholder kernel; the factory replaces it with the per-dimension (ARD)
        # version at the first fit, when the input dimension is known.
        kernel_factory = DefaultKernelFactory(nu=nu, oblique=oblique)
        kernel = ConstantKernel() * Matern(nu=nu)

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

    return SklearnGPAdapter(sklearn_gpr, kernel_factory=kernel_factory)
