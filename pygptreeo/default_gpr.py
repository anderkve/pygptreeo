"""Default Gaussian Process Regressor configuration.

This module provides a factory function that creates a default GPRegressorInterface
instance suitable for use with GPTreeO. It sets up sensible defaults for kernel
configuration and hyperparameters commonly used in online regression tasks.
"""

# Standard library imports
from typing import Optional

# Third-party imports
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, RationalQuadratic

# Local imports
from pygptreeo.adapters import SklearnGPAdapter


def Default_GPR(
    kernel=None,
    *,
    alpha=1e-10,
    optimizer='fmin_l_bfgs_b',
    n_restarts_optimizer=0,
    normalize_y=False,
    copy_X_train=True,
    n_targets=None,
    random_state=None
):
    """Create a default GaussianProcessRegressor adapter for GPTreeO.

    This function creates and returns a SklearnGPAdapter wrapping a
    scikit-learn GaussianProcessRegressor with sensible default settings
    for use within the GPTreeO framework.

    Parameters
    ----------
    kernel : Kernel object, optional
        The kernel specifying the covariance function of the GP.
        If None is passed, the kernel ConstantKernel() * Matern(nu=1.5) is used.
        Note that the kernel's hyperparameters are optimized during fitting.
    alpha : float or ndarray of shape (n_samples,), default=1e-10
        Value added to the diagonal of the kernel matrix during fitting.
        This can represent the expected amount of noise in the observations.
        Larger values correspond to increased noise level.
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
    # Set default kernel if none provided
    if kernel is None:
        kernel = ConstantKernel() * Matern(nu=1.5)

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

    # Set the min_length_scale attribute (used by pygptreeo)
    sklearn_gpr.min_length_scale = 0.001

    # Wrap in adapter and return
    return SklearnGPAdapter(sklearn_gpr)


def get_kernel_by_index(kernel_idx: int):
    """Get a kernel by its index from the predefined list.

    Kernel types:
        0: Const*(RBF + Matern(nu=1.5))
        1: Const*(RQ + Matern(nu=1.5))
        2: Const*(RQ + RBF)
        3: Const*RQ
        4: Const*Matern(nu=1.5)
        5: Const*RBF

    Args:
        kernel_idx (int): Index of the kernel type (0-5)

    Returns:
        Kernel object: The kernel corresponding to the index

    Raises:
        ValueError: If kernel_idx is not in range 0-5
    """
    if kernel_idx == 0:
        # Const*(RBF + Matern(nu=1.5))
        return ConstantKernel() * (RBF() + Matern(nu=1.5))
    elif kernel_idx == 1:
        # Const*(RQ + Matern(nu=1.5))
        return ConstantKernel() * (RationalQuadratic() + Matern(nu=1.5))
    elif kernel_idx == 2:
        # Const*(RQ + RBF)
        return ConstantKernel() * (RationalQuadratic() + RBF())
    elif kernel_idx == 3:
        # Const*RQ
        return ConstantKernel() * RationalQuadratic()
    elif kernel_idx == 4:
        # Const*Matern(nu=1.5)
        return ConstantKernel() * Matern(nu=1.5)
    elif kernel_idx == 5:
        # Const*RBF
        return ConstantKernel() * RBF()
    else:
        raise ValueError(f"Invalid kernel_idx: {kernel_idx}. Must be in range 0-5.")
