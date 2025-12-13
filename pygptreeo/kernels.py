"""Custom kernel implementations for pygptreeo.

This module provides custom Gaussian process kernels that extend the
functionality available in scikit-learn, particularly anisotropic variants
of kernels that only have isotropic versions in sklearn.
"""

import numpy as np
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter, StationaryKernelMixin, NormalizedKernelMixin
from sklearn.gaussian_process.kernels import _check_length_scale


class AnisotropicRationalQuadratic(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Anisotropic Rational Quadratic kernel.

    The anisotropic RationalQuadratic kernel is a generalization of the standard
    (isotropic) RationalQuadratic kernel that allows for different length scales
    in each dimension. This is particularly useful for problems where different
    input dimensions have different characteristic scales of variation.

    The kernel is defined as:
        k(x, x') = (1 + d²(x, x') / (2 * alpha))^(-alpha)

    where the anisotropic squared distance is:
        d²(x, x') = sum_i ((x_i - x'_i) / length_scale_i)²

    The RationalQuadratic kernel can be seen as a scale mixture (infinite sum)
    of RBF kernels with different characteristic length scales. The parameter
    alpha determines the weighting of the different scales:
    - Small alpha: similar to a single RBF with fixed length scale
    - Large alpha: mixture includes wider range of length scales

    Parameters
    ----------
    length_scale : float or ndarray of shape (n_features,), default=1.0
        The length scale(s) of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        has its own length scale.

    alpha : float, default=1.0
        Scale mixture parameter that controls the relative weighting of
        different length scales. Must be positive.

    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on length_scale. If set to "fixed",
        length_scale cannot be changed during hyperparameter tuning.

    alpha_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on alpha. If set to "fixed",
        alpha cannot be changed during hyperparameter tuning.

    Attributes
    ----------
    anisotropic : bool
        Returns True if the kernel is configured with anisotropic length scales
        (different length scale per dimension), False otherwise.

    Examples
    --------
    >>> from pygptreeo.kernels import AnisotropicRationalQuadratic
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> import numpy as np
    >>>
    >>> # 3D example with different length scales per dimension
    >>> kernel = AnisotropicRationalQuadratic(
    ...     length_scale=[1.0, 0.5, 2.0],
    ...     alpha=1.0,
    ...     length_scale_bounds=(1e-3, 1e3)
    ... )
    >>> gpr = GaussianProcessRegressor(kernel=kernel)
    """

    def __init__(self, length_scale=1.0, alpha=1.0,
                 length_scale_bounds=(1e-5, 1e5),
                 alpha_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.alpha = alpha
        self.length_scale_bounds = length_scale_bounds
        self.alpha_bounds = alpha_bounds

    @property
    def anisotropic(self):
        """Returns True if the kernel is anisotropic (has multiple length scales)."""
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        """Returns the hyperparameter specification for length_scale."""
        if self.anisotropic:
            return Hyperparameter(
                "length_scale", "numeric", self.length_scale_bounds,
                len(self.length_scale)
            )
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_alpha(self):
        """Returns the hyperparameter specification for alpha."""
        return Hyperparameter("alpha", "numeric", self.alpha_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of the
            kernel hyperparameters is computed.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_Y, n_dims), optional
            The gradient of the kernel k(X, Y) with respect to the log of the
            hyperparameters of the kernel. Only returned when eval_gradient is True.
        """
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)

        if Y is None:
            Y = X
        else:
            Y = np.atleast_2d(Y)

        if eval_gradient:
            # Compute kernel and gradient
            K, K_gradient = self._compute_kernel_gradient(X, Y, length_scale)
            return K, K_gradient
        else:
            # Compute kernel only
            K = self._compute_kernel(X, Y, length_scale)
            return K

    def _compute_kernel(self, X, Y, length_scale):
        """Compute the kernel matrix K(X, Y).

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument
        Y : ndarray of shape (n_samples_Y, n_features)
            Right argument
        length_scale : float or ndarray of shape (n_features,)
            Length scale(s)

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel matrix
        """
        # Compute anisotropic squared distances
        # dists[i, j] = sum_k ((X[i, k] - Y[j, k]) / length_scale[k])^2
        dists = self._compute_squared_distances(X, Y, length_scale)

        # Rational quadratic kernel formula
        # k(d²) = (1 + d² / (2 * alpha))^(-alpha)
        K = (1.0 + dists / (2.0 * self.alpha)) ** (-self.alpha)

        return K

    def _compute_kernel_gradient(self, X, Y, length_scale):
        """Compute kernel matrix and gradient with respect to hyperparameters.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument
        Y : ndarray of shape (n_samples_Y, n_features)
            Right argument
        length_scale : float or ndarray of shape (n_features,)
            Length scale(s)

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel matrix
        K_gradient : ndarray of shape (n_samples_X, n_samples_Y, n_hyperparams)
            Gradient with respect to log of hyperparameters
        """
        # Compute squared distances and kernel
        dists = self._compute_squared_distances(X, Y, length_scale)
        base = 1.0 + dists / (2.0 * self.alpha)
        K = base ** (-self.alpha)

        # Number of hyperparameters
        if self.anisotropic:
            n_length_scale_params = length_scale.shape[0]
        else:
            n_length_scale_params = 1

        # Gradient has shape (n_samples_X, n_samples_Y, n_hyperparams)
        # IMPORTANT: sklearn orders hyperparameters alphabetically by property name
        # So the order is: [alpha, length_scale_0, length_scale_1, ..., length_scale_d]
        K_gradient = np.zeros(K.shape + (n_length_scale_params + 1,))

        # Gradient with respect to alpha (index 0 in theta)
        # d/d(log(alpha)) K = d/dalpha K * alpha
        # d/dalpha K = K * [-log(1 + d²/(2*alpha)) + alpha * d²/(2*alpha) / (1 + d²/(2*alpha))]
        #            = K * [alpha * d²/(2*alpha) / (1 + d²/(2*alpha)) - log(1 + d²/(2*alpha))]
        #            = K * [d²/(2*(1 + d²/(2*alpha))) - log(1 + d²/(2*alpha))]
        log_base = np.log(base)
        K_gradient[:, :, 0] = K * (dists / (2.0 * base) - log_base) * self.alpha

        # Gradient with respect to length_scale(s) (indices 1 to n_length_scale_params in theta)
        # d/d(log(l_i)) K = d/dl_i K * l_i  (chain rule for log)
        # d/dl_i K = K * alpha / (1 + d²/(2*alpha)) * d²/l_i / (2*alpha)
        #          = K * alpha * d²/l_i / (2*alpha * (1 + d²/(2*alpha)))
        #          = K * d²/l_i / (2 * (1 + d²/(2*alpha)))

        if self.anisotropic:
            # For anisotropic case, compute gradient for each length scale
            # d²_i/l_i = 2 * sum_j (X[j,i] - Y[k,i])² / l_i³
            for i in range(length_scale.shape[0]):
                # Squared differences in dimension i, scaled by length_scale
                diff_i = (X[:, i][:, np.newaxis] - Y[:, i][np.newaxis, :])
                squared_diff_i = diff_i ** 2 / (length_scale[i] ** 3)

                # Gradient: K * d²_i/l_i / (1 + d²/(2*alpha))
                # Note: we multiply by length_scale[i] for derivative w.r.t. log(length_scale[i])
                # Store at index i+1 because index 0 is alpha
                K_gradient[:, :, i+1] = K * squared_diff_i / base * length_scale[i]
        else:
            # For isotropic case
            # d²/l = 2 * d² / l  (since d² is proportional to 1/l²)
            # Store at index 1 because index 0 is alpha
            K_gradient[:, :, 1] = K * dists / base * length_scale

        return K, K_gradient

    def _compute_squared_distances(self, X, Y, length_scale):
        """Compute anisotropic squared distances between X and Y.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument
        Y : ndarray of shape (n_samples_Y, n_features)
            Right argument
        length_scale : float or ndarray of shape (n_features,)
            Length scale(s)

        Returns
        -------
        dists : ndarray of shape (n_samples_X, n_samples_Y)
            Squared distances: dists[i,j] = sum_k ((X[i,k] - Y[j,k]) / length_scale[k])^2
        """
        # Scale X and Y by length_scale
        X_scaled = X / length_scale
        Y_scaled = Y / length_scale

        # Compute squared Euclidean distances in scaled space
        # ||X_scaled - Y_scaled||² = sum_k ((X[i,k] - Y[j,k]) / length_scale[k])^2
        dists = np.sum(X_scaled ** 2, axis=1)[:, np.newaxis] + \
                np.sum(Y_scaled ** 2, axis=1)[np.newaxis, :] - \
                2.0 * np.dot(X_scaled, Y_scaled.T)

        # Numerical safety: ensure non-negative
        dists = np.maximum(dists, 0.0)

        return dists

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The diagonal of any stationary kernel is constant (all ones for normalized kernels).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Argument to the kernel

        Returns
        -------
        K_diag : ndarray of shape (n_samples,)
            Diagonal of K(X, X)
        """
        return np.ones(X.shape[0])

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return True

    def __repr__(self):
        if self.anisotropic:
            return (f"{self.__class__.__name__}(alpha={self.alpha}, "
                   f"length_scale=[{', '.join(f'{ls:.3g}' for ls in self.length_scale)}])")
        else:
            return (f"{self.__class__.__name__}(alpha={self.alpha}, "
                   f"length_scale={self.length_scale})")
