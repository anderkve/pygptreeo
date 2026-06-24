"""Custom kernel implementations for pygptreeo.

This module provides custom Gaussian process kernels that extend the
functionality available in scikit-learn, particularly anisotropic variants
of kernels that only have isotropic versions in sklearn, and additive kernels
for learning functions with low-dimensional structure.
"""

import numpy as np
from itertools import combinations
from math import comb
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter, StationaryKernelMixin, NormalizedKernelMixin
from sklearn.gaussian_process.kernels import _check_length_scale
from sklearn.gaussian_process.kernels import RBF, Matern


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


class AdditiveKernel(Kernel):
    """Additive kernel with configurable interaction depth.

    This kernel is designed for learning functions with low-dimensional structure
    by decomposing the kernel into a sum of terms with different interaction orders.
    For example, with interaction_depth=2 in 3D:

        k(x, x') = k₁(x₁, x'₁) + k₂(x₂, x'₂) + k₃(x₃, x'₃)           [order 1]
                 + k₁(x₁, x'₁) × k₂(x₂, x'₂)                          [order 2]
                 + k₁(x₁, x'₁) × k₃(x₃, x'₃)                          [order 2]
                 + k₂(x₂, x'₂) × k₃(x₃, x'₃)                          [order 2]

    This is particularly effective for:
    - Functions that are truly additive or have low-order interactions
    - High-dimensional problems with sparse data
    - Avoiding the curse of dimensionality in standard kernels
    - Interpretable models (see which dimensions/interactions matter)

    Parameters
    ----------
    input_dim : int
        Number of input dimensions

    interaction_depth : int, default=1
        Maximum order of interactions to include:
        - 1: Only main effects (purely additive)
        - 2: Main effects + pairwise interactions
        - 3: Up to 3-way interactions
        - etc.

    base_kernel : str, default='rbf'
        Type of 1D kernel to use for each dimension:
        - 'rbf': RBF kernel (smooth, infinitely differentiable)
        - 'matern': Matérn kernel with nu=1.5 (once differentiable)

    length_scale : float or array-like of shape (n_dims,), default=1.0
        Length scale for each dimension. If float, same length scale
        for all dimensions.

    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-3, 1e3)
        The lower and upper bound on length_scale parameters.

    Examples
    --------
    >>> from pygptreeo.kernels import AdditiveKernel
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> import numpy as np
    >>>
    >>> # 5D problem with main effects + pairwise interactions
    >>> kernel = AdditiveKernel(
    ...     input_dim=5,
    ...     interaction_depth=2,
    ...     base_kernel='rbf',
    ...     length_scale=1.0
    ... )
    >>> # This creates 5 + C(5,2) = 5 + 10 = 15 kernel terms
    >>>
    >>> gpr = GaussianProcessRegressor(kernel=kernel)
    """

    def __init__(self, input_dim, interaction_depth=1, base_kernel='rbf',
                 length_scale=1.0, length_scale_bounds=(1e-3, 1e3)):
        # Store constructor parameters as regular attributes (sklearn needs this)
        self.input_dim = input_dim
        self.interaction_depth = interaction_depth
        self.base_kernel = base_kernel
        self.length_scale_bounds = length_scale_bounds

        # Generate interaction term indices
        # For example, with input_dim=3, depth=2:
        # Order 1: [(0,), (1,), (2,)]
        # Order 2: [(0,1), (0,2), (1,2)]
        self._interaction_terms = []
        for order in range(1, min(interaction_depth + 1, input_dim + 1)):
            self._interaction_terms.extend(
                list(combinations(range(input_dim), order))
            )

        self._n_terms = len(self._interaction_terms)

        # Set length_scale (this can be an attribute as it's a hyperparameter)
        # IMPORTANT: Always store length_scale as an array of length input_dim
        # to ensure sklearn creates the right number of hyperparameters
        if np.ndim(length_scale) == 0:
            self.length_scale = np.full(input_dim, length_scale)
        else:
            self.length_scale = np.asarray(length_scale)
            if len(self.length_scale) != input_dim:
                raise ValueError(f"length_scale must have length {input_dim}, got {len(self.length_scale)}")

    @property
    def interaction_terms(self):
        """List of interaction term tuples."""
        return self._interaction_terms

    @property
    def n_terms(self):
        """Number of additive terms."""
        return self._n_terms

    @property
    def hyperparameter_length_scale(self):
        """Returns the hyperparameter specification for length_scale."""
        # Always return input_dim hyperparameters (one per dimension)
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds,
            self.input_dim
        )

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
            hyperparameters. Only returned when eval_gradient is True.
        """
        X = np.atleast_2d(X)

        # length_scale is already an array (initialized in __init__)
        length_scale = self.length_scale

        if Y is None:
            Y = X
        else:
            Y = np.atleast_2d(Y)

        if eval_gradient:
            K, K_gradient = self._compute_kernel_gradient(X, Y, length_scale)
            return K, K_gradient
        else:
            K = self._compute_kernel(X, Y, length_scale)
            return K

    def _compute_kernel(self, X, Y, length_scale):
        """Compute the additive kernel matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument
        Y : ndarray of shape (n_samples_Y, n_features)
            Right argument
        length_scale : ndarray of shape (n_dims,)
            Length scales for each dimension

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel matrix
        """
        K = np.zeros((X.shape[0], Y.shape[0]))

        # Sum over all interaction terms
        for term_dims in self.interaction_terms:
            # Compute product of 1D kernels for this term
            K_term = np.ones((X.shape[0], Y.shape[0]))
            for dim in term_dims:
                # Extract dimension and compute 1D kernel
                X_dim = X[:, dim:dim+1]
                Y_dim = Y[:, dim:dim+1]
                K_dim = self._compute_1d_kernel(X_dim, Y_dim, length_scale[dim])
                K_term *= K_dim

            K += K_term

        return K

    def _compute_kernel_gradient(self, X, Y, length_scale):
        """Compute kernel and gradient with respect to log(length_scale).

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument
        Y : ndarray of shape (n_samples_Y, n_features)
            Right argument
        length_scale : ndarray of shape (n_dims,)
            Length scales

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel matrix
        K_gradient : ndarray of shape (n_samples_X, n_samples_Y, n_dims)
            Gradient w.r.t. log(length_scale)
        """
        K = np.zeros((X.shape[0], Y.shape[0]))
        K_gradient = np.zeros((X.shape[0], Y.shape[0], self.input_dim))

        # For each interaction term
        for term_dims in self.interaction_terms:
            # Compute the kernel for this term and its gradient
            K_term = np.ones((X.shape[0], Y.shape[0]))
            K_1d_list = []  # Store individual 1D kernels
            grad_1d_list = []  # Store individual 1D gradients

            # Compute all 1D kernels in this product
            for dim in term_dims:
                X_dim = X[:, dim:dim+1]
                Y_dim = Y[:, dim:dim+1]
                K_1d, grad_1d = self._compute_1d_kernel_gradient(
                    X_dim, Y_dim, length_scale[dim]
                )
                K_1d_list.append(K_1d)
                grad_1d_list.append(grad_1d)
                K_term *= K_1d

            # Add this term to total kernel
            K += K_term

            # Compute gradient for each dimension in this term
            for i, dim in enumerate(term_dims):
                # Gradient via product rule:
                # d/d(log l_i) [k₁ × k₂ × ... × kₙ] = k₁ × ... × (dk_i/d(log l_i)) × ... × kₙ
                grad_term = grad_1d_list[i]
                for j in range(len(term_dims)):
                    if j != i:
                        grad_term = grad_term * K_1d_list[j]

                K_gradient[:, :, dim] += grad_term

        return K, K_gradient

    def _compute_1d_kernel(self, X, Y, length_scale):
        """Compute 1D kernel for a single dimension.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, 1)
            Left argument (single dimension)
        Y : ndarray of shape (n_samples_Y, 1)
            Right argument (single dimension)
        length_scale : float
            Length scale for this dimension

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            1D kernel matrix
        """
        # Compute squared distances
        dists_sq = np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=2) / (length_scale ** 2)

        if self.base_kernel == 'rbf':
            # RBF kernel: exp(-0.5 * r²)
            K = np.exp(-0.5 * dists_sq)
        elif self.base_kernel == 'matern':
            # Matérn kernel with nu=1.5: (1 + sqrt(3)*r) * exp(-sqrt(3)*r)
            r = np.sqrt(dists_sq)
            sqrt3_r = np.sqrt(3) * r
            K = (1.0 + sqrt3_r) * np.exp(-sqrt3_r)
        else:
            raise ValueError(f"Unknown base_kernel: {self.base_kernel}")

        return K

    def _compute_1d_kernel_gradient(self, X, Y, length_scale):
        """Compute 1D kernel and gradient w.r.t. log(length_scale).

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, 1)
            Left argument
        Y : ndarray of shape (n_samples_Y, 1)
            Right argument
        length_scale : float
            Length scale

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            1D kernel matrix
        grad : ndarray of shape (n_samples_X, n_samples_Y)
            Gradient w.r.t. log(length_scale)
        """
        # Compute squared distances
        dists_sq = np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=2) / (length_scale ** 2)

        if self.base_kernel == 'rbf':
            # RBF: k = exp(-0.5 * r²)
            # d/d(log l) k = d/dl k * l = (r²) * k * l
            K = np.exp(-0.5 * dists_sq)
            grad = dists_sq * K * length_scale

        elif self.base_kernel == 'matern':
            # Matérn 1.5: k = (1 + sqrt(3)*r) * exp(-sqrt(3)*r)
            # d/d(log l) k = d/dl k * l
            r = np.sqrt(dists_sq)
            sqrt3_r = np.sqrt(3) * r
            K = (1.0 + sqrt3_r) * np.exp(-sqrt3_r)

            # Derivative w.r.t. l (then multiply by l for log derivative)
            # dk/dl = dk/dr * dr/dl
            # dr/dl = -r/l
            # dk/dr = sqrt(3) * exp(-sqrt(3)*r) * (1 - (1 + sqrt(3)*r))
            #       = -3 * r * exp(-sqrt(3)*r)
            dk_dr = -3 * r * np.exp(-sqrt3_r)
            dr_dl = -r / length_scale
            grad = dk_dr * dr_dl * length_scale  # multiply by l for log derivative
            # Simplify: grad = 3 * r² * exp(-sqrt(3)*r)
            grad = 3 * dists_sq * np.exp(-sqrt3_r)

        else:
            raise ValueError(f"Unknown base_kernel: {self.base_kernel}")

        return K, grad

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        For additive kernels, the diagonal equals the number of terms,
        assuming each 1D kernel has diag = 1.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Argument to the kernel

        Returns
        -------
        K_diag : ndarray of shape (n_samples,)
            Diagonal of K(X, X)
        """
        # Each 1D kernel contributes 1 on the diagonal
        # Total diagonal = number of terms
        return np.full(X.shape[0], self.n_terms)

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return True

    def __repr__(self):
        # Check if all length scales are the same
        if len(np.unique(self.length_scale)) == 1:
            ls_repr = f"{self.length_scale[0]:.3g}"
        else:
            ls_repr = f"[{', '.join(f'{ls:.3g}' for ls in self.length_scale)}]"
        return (f"{self.__class__.__name__}(n_dims={self.input_dim}, "
                f"interaction_depth={self.interaction_depth}, "
                f"base_kernel='{self.base_kernel}', "
                f"length_scale={ls_repr}, "
                f"n_terms={self.n_terms})")


class NewtonGirardAdditiveKernel(Kernel):
    """Additive RBF kernel with per-order variances, via Newton-Girard recursion.

    Models the target as a sum of interaction terms grouped by *order*::

        k(x, x') = sum_{q=1}^{Q} sigma_q^2 * e_q(z_1, ..., z_d)

    where ``z_i = exp(-(x_i - x'_i)^2 / (2 l_i^2))`` is a per-dimension RBF and ``e_q``
    is the elementary symmetric polynomial of order q in ``(z_1, ..., z_d)``. The
    ``e_q`` are computed from power sums via the Newton-Girard identities in
    ``O(d * Q)`` -- never by enumerating the ``C(d, q)`` interaction terms -- so the
    kernel has only ``d`` length scales + ``Q`` order-variances (``O(d)``
    hyperparameters) and costs ``O(d * Q * n^2)`` to evaluate.

    This is the data-efficient way to exploit low interaction order: an additive
    (order-1) or low-order model needs far fewer samples than the fully-interacting
    product kernel (the standard ARD RBF), which this strictly generalises -- the
    order-``d`` term *is* the product kernel. Truncating to small ``Q`` (e.g. 1 or 2),
    optionally plus a Matern/RBF catch-all for higher orders, keeps it cheap while
    letting marginal-likelihood optimisation choose how much of each order to use.

    Unlike :class:`AdditiveKernel` (which enumerates ``combinations`` and so scales
    as ``C(d, q)``), this kernel stays linear in ``d``.

    Parameters
    ----------
    length_scale : array-like of shape (n_features,)
        Per-dimension length scales of the base RBF.
    order_std : array-like of shape (max_order,)
        Standard deviation (sqrt variance) for each interaction order ``1..Q``;
        ``len(order_std)`` sets the maximum interaction order ``Q``.
    length_scale_bounds, order_std_bounds : pair of floats or "fixed"
        Bounds for the respective hyperparameters.

    Examples
    --------
    >>> import numpy as np
    >>> from pygptreeo.kernels import NewtonGirardAdditiveKernel
    >>> from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    >>> # order-2 additive component + a Matern catch-all, in 5-D
    >>> k = (NewtonGirardAdditiveKernel(length_scale=[1.0] * 5, order_std=[1.0, 1.0])
    ...      + ConstantKernel(1.0) * Matern(length_scale=[1.0] * 5, nu=1.5))
    """

    def __init__(self, length_scale, order_std,
                 length_scale_bounds=(1e-2, 1e2), order_std_bounds=(1e-3, 1e3)):
        self.length_scale = length_scale
        self.order_std = order_std
        self.length_scale_bounds = length_scale_bounds
        self.order_std_bounds = order_std_bounds

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds,
                              len(np.atleast_1d(self.length_scale)))

    @property
    def hyperparameter_order_std(self):
        return Hyperparameter("order_std", "numeric", self.order_std_bounds,
                              len(np.atleast_1d(self.order_std)))

    def _components(self, X, Y):
        """Per-dimension RBF matrices Z_i and squared distances D_i."""
        ls = np.atleast_1d(self.length_scale)
        Zs, Ds = [], []
        for i in range(X.shape[1]):
            Di = (X[:, i][:, None] - Y[:, i][None, :]) ** 2
            Zs.append(np.exp(-Di / (2.0 * ls[i] ** 2)))
            Ds.append(Di)
        return Zs, Ds, ls

    @staticmethod
    def _elem_sym(Zs, Q):
        """Elementary symmetric polynomials E_0..E_Q via Newton-Girard (O(d*Q))."""
        shape = Zs[0].shape
        P = [None] + [sum(Z ** k for Z in Zs) for k in range(1, Q + 1)]   # power sums
        E = [np.ones(shape)]                                              # E_0
        for q in range(1, Q + 1):
            acc = np.zeros(shape)
            for k in range(1, q + 1):
                acc = acc + ((-1) ** (k - 1)) * E[q - k] * P[k]
            E.append(acc / q)
        return E

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        Y = X if Y is None else np.atleast_2d(Y)
        sig = np.atleast_1d(self.order_std)
        d, Q = X.shape[1], len(sig)
        Zs, Ds, ls = self._components(X, Y)
        E = self._elem_sym(Zs, Q)
        K = sum(sig[q - 1] ** 2 * E[q] for q in range(1, Q + 1))
        if not eval_gradient:
            return K

        # Gradient w.r.t. log-hyperparameters, ordered as sklearn expects
        # (alphabetical by name): length_scale (d) then order_std (Q). Fixed
        # hyperparameters are omitted.
        grads = []
        if not self.hyperparameter_length_scale.fixed:
            for i in range(d):
                Gi = Zs[i] * (Ds[i] / ls[i] ** 2)            # dZ_i / dlog l_i
                Eli = [np.ones(Zs[0].shape)]                 # leave-i-out E^{(\i)}_0
                for r in range(1, Q):
                    Eli.append(E[r] - Zs[i] * Eli[r - 1])    # E^{(\i)}_r
                S = sum(sig[q - 1] ** 2 * Eli[q - 1] for q in range(1, Q + 1))
                grads.append(Gi * S)                         # dE_q/dZ_i = E^{(\i)}_{q-1}
        if not self.hyperparameter_order_std.fixed:
            for q in range(1, Q + 1):
                grads.append(2.0 * sig[q - 1] ** 2 * E[q])   # dK / dlog sigma_q
        K_grad = (np.stack(grads, axis=2) if grads
                  else np.empty((X.shape[0], Y.shape[0], 0)))
        return K, K_grad

    def diag(self, X):
        # z_i(x, x) = 1, so e_q(1, ..., 1) = C(d, q): the diagonal is constant.
        Xd = np.atleast_2d(X)
        d = Xd.shape[1]
        sig = np.atleast_1d(self.order_std)
        val = sum(sig[q - 1] ** 2 * comb(d, q) for q in range(1, len(sig) + 1))
        return np.full(Xd.shape[0], float(val))

    def is_stationary(self):
        return True

    def __repr__(self):
        ls = np.atleast_1d(self.length_scale)
        sig = np.atleast_1d(self.order_std)
        return (f"{self.__class__.__name__}(d={len(ls)}, max_order={len(sig)}, "
                f"order_std=[{', '.join(f'{s:.3g}' for s in sig)}])")
