"""Default Gaussian Process Regressor configuration.

This module provides a pre-configured GaussianProcessRegressor class suitable
for use with GPTreeO. It sets up sensible defaults for kernel configuration
and hyperparameters commonly used in online regression tasks.
"""

# Standard library imports
from typing import Callable, Optional, Type, Union

# Third-party imports
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF

class Default_GPR(GaussianProcessRegressor):
        """A default GaussianProcessRegressor configuration for GPTreeO.

        This class extends sklearn.gaussian_process.GaussianProcessRegressor
        to provide a default setup suitable for use within the GPTreeO framework.
        It initializes with a predefined kernel and a minimum length scale for
        kernel parameters.

        Attributes:
            kernel: The kernel to be used. Defaults to ConstantKernel * Matern(nu=1.5).
            min_length_scale: The minimum bound for the kernel's length scale
                hyperparameters. This is used to prevent the optimizer from
                choosing excessively small length scales. Defaults to 0.001.
            alpha: Value added to the diagonal of the kernel matrix during fitting.
                Larger values correspond to increased noise level in the observations.
                Passed to GaussianProcessRegressor.
            optimizer: The optimizer to use for fitting the kernel's hyperparameters.
                Passed to GaussianProcessRegressor.
            n_restarts_optimizer: The number of times the optimizer is restarted.
                Passed to GaussianProcessRegressor.
            normalize_y: Whether the target values y are normalized before fitting.
                Passed to GaussianProcessRegressor.
            copy_X_train: If True, a persistent copy of the training data is stored.
                Passed to GaussianProcessRegressor.
            n_targets: The number of dimensions of the target values. Used to
                reshape y if it is a 1D array. Passed to GaussianProcessRegressor.
            random_state: Controls the randomness of the initialization.
                Passed to GaussianProcessRegressor.
        """
        def __init__(self, kernel=None, *, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, n_targets=None, random_state=None):
            if kernel is None:
                self.kernel = ConstantKernel() * Matern(nu=1.5)
            else:
                self.kernel = kernel
            self.min_length_scale = 0.001
            self.alpha = alpha
            self.optimizer = optimizer
            self.n_restarts_optimizer = n_restarts_optimizer
            self.normalize_y = normalize_y
            self.copy_X_train = copy_X_train
            self.n_targets = n_targets
            self.random_state = random_state
    