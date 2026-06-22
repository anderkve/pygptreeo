"""Custom kernel implementations for pygptreeo.

This module provides custom Gaussian process kernels that extend the
functionality available in scikit-learn, particularly anisotropic variants
of kernels that only have isotropic versions in sklearn, and additive kernels
for learning functions with low-dimensional structure.
"""

import numpy as np
from itertools import combinations
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter, StationaryKernelMixin, NormalizedKernelMixin
from sklearn.gaussian_process.kernels import _check_length_scale
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel


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

    def _per_dim_1d(self, X, Y, length_scale, eval_gradient=False):
        """Compute the 1-D base kernel (and gradient) for every input dimension.

        Each dimension's 1-D kernel matrix is computed exactly once here and then
        reused across all interaction terms that contain that dimension. This is
        the key to the vectorized assembly: a depth-``p`` kernel over ``d`` inputs
        contains terms whose sizes sum to O(d^p), so the old per-term-per-dim
        evaluation recomputed each dimension's (expensive, transcendental) 1-D
        kernel many times; computing them once is an O(d^{p-1}) reduction in those
        evaluations.

        The per-dimension formulas are identical to ``_compute_1d_kernel`` /
        ``_compute_1d_kernel_gradient`` so results are numerically unchanged.

        Returns
        -------
        K1d : list of ndarray
            ``K1d[d]`` is the (n_X, n_Y) 1-D kernel for dimension ``d``.
        g1d : list of ndarray or None
            ``g1d[d]`` is the (n_X, n_Y) derivative w.r.t. ``log(length_scale[d])``,
            or None when ``eval_gradient`` is False.
        """
        sqrt3 = np.sqrt(3)
        K1d = []
        g1d = [] if eval_gradient else None
        for d in range(self.input_dim):
            diff = X[:, d][:, None] - Y[:, d][None, :]
            dists_sq = (diff * diff) / (length_scale[d] ** 2)
            if self.base_kernel == 'rbf':
                K = np.exp(-0.5 * dists_sq)
                if eval_gradient:
                    # d/dlog(l) exp(-0.5 r^2/l^2) = (r^2/l^2) * K = dists_sq * K.
                    # (The previous implementation carried a spurious extra factor
                    # of length_scale here; verified against finite differences.)
                    g1d.append(dists_sq * K)
            elif self.base_kernel == 'matern':
                sqrt3_r = sqrt3 * np.sqrt(dists_sq)
                exp_term = np.exp(-sqrt3_r)
                K = (1.0 + sqrt3_r) * exp_term
                if eval_gradient:
                    g1d.append(3.0 * dists_sq * exp_term)
            else:
                raise ValueError(f"Unknown base_kernel: {self.base_kernel}")
            K1d.append(K)
        return K1d, g1d

    def _compute_kernel(self, X, Y, length_scale):
        """Compute the additive kernel matrix (vectorized over dimensions).

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
        K1d, _ = self._per_dim_1d(X, Y, length_scale, eval_gradient=False)

        K = np.zeros((X.shape[0], Y.shape[0]))
        for term_dims in self.interaction_terms:
            K_term = K1d[term_dims[0]].copy()
            for dim in term_dims[1:]:
                K_term *= K1d[dim]
            K += K_term

        return K

    def _compute_kernel_gradient(self, X, Y, length_scale):
        """Compute kernel and gradient w.r.t. log(length_scale) (vectorized).

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
        K1d, g1d = self._per_dim_1d(X, Y, length_scale, eval_gradient=True)

        nx, ny = X.shape[0], Y.shape[0]
        K = np.zeros((nx, ny))
        K_gradient = np.zeros((nx, ny, self.input_dim))

        for term_dims in self.interaction_terms:
            k = len(term_dims)
            # Fast paths for the common main-effect and pairwise terms; these
            # cover interaction_depth <= 2 with no division or temporaries.
            if k == 1:
                d0 = term_dims[0]
                K += K1d[d0]
                K_gradient[:, :, d0] += g1d[d0]
            elif k == 2:
                a, b = term_dims
                Ka, Kb = K1d[a], K1d[b]
                K += Ka * Kb
                # Product rule: d/dlog l_a [k_a k_b] = (dk_a/dlog l_a) k_b, etc.
                K_gradient[:, :, a] += g1d[a] * Kb
                K_gradient[:, :, b] += Ka * g1d[b]
            else:
                # General case: product of the others via prefix/suffix products
                # (numerically stable, avoids dividing by a possibly-tiny kernel).
                factors = [K1d[d] for d in term_dims]
                K_term = factors[0].copy()
                for f in factors[1:]:
                    K_term *= f
                K += K_term

                prefix = [None] * k
                pre = np.ones((nx, ny))
                for i in range(k):
                    prefix[i] = pre
                    pre = pre * factors[i]
                suf = np.ones((nx, ny))
                for i in range(k - 1, -1, -1):
                    others = prefix[i] * suf
                    K_gradient[:, :, term_dims[i]] += g1d[term_dims[i]] * others
                    suf = suf * factors[i]

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
            # RBF: k = exp(-0.5 * r²),  r² = dists_sq = Δ²/l²
            # d/d(log l) k = (Δ²/l²) * k = dists_sq * k
            K = np.exp(-0.5 * dists_sq)
            grad = dists_sq * K

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


def make_additive_kernel(n_features,
                         interaction_depth=2,
                         base_kernel='matern',
                         length_scale=1.0,
                         length_scale_bounds=(1e-4, 1e4),
                         amplitude=1.0,
                         amplitude_bounds=(1e-3, 1e8),
                         rescue=True,
                         rescue_nu=1.5,
                         rescue_length_scale_bounds=(1e-5, 1e5)):
    """Build a low-order additive GP kernel with a full-dimensional rescue term.

    The returned kernel is

        c1 * AdditiveKernel(interaction_depth)            [additive structure]
      + c2 * Matern(ARD over all n_features dimensions)   [full-D rescue]

    where ``c1`` and ``c2`` are independent, learnable ``ConstantKernel``
    amplitudes (the second term is omitted when ``rescue=False``).

    Why this helps sample efficiency
    --------------------------------
    A leaf GP in a GPTree sees only ``~Nbar`` points. A full-dimensional kernel
    must estimate a function over all of that leaf's box from those few points,
    so its sample complexity grows steeply with dimension. An additive kernel
    instead models the function as a sum of one- and two-dimensional pieces, so
    its effective dimensionality is the largest interaction order (1 or 2), not
    ``n_features``. Every point then constrains every low-order piece, which is
    exactly the regime where the curse of dimensionality is mild. Many real
    targets (and all of this repository's benchmark functions) are sums of 1-D
    or adjacent-pair terms, so the additive terms capture the structure with far
    fewer points than a full-D kernel.

    The rescue term is what makes this safe. On a genuinely high-order /
    non-additive target the additive terms cannot fit the function; the
    marginal-likelihood optimization then grows ``c2`` and shrinks ``c1``, so the
    model falls back to the ordinary full-D Matern (the current default) and does
    not degrade. On an additive target the opposite happens: ``c2`` shrinks and
    the cheaper additive representation dominates. The mix is chosen per leaf
    from data, with no manual tuning.

    Parameters
    ----------
    n_features : int
        Number of input dimensions.
    interaction_depth : int, default=2
        Maximum interaction order of the additive terms. 1 = purely additive
        (sum of 1-D effects); 2 = main effects + all pairwise interactions.
        Depth 2 is the safe default: it subsumes depth 1 and also captures
        pairwise-coupled targets (e.g. Rosenbrock) that depth 1 cannot represent.
    base_kernel : {'matern', 'rbf'}, default='matern'
        1-D base kernel used inside each additive term. 'matern' (nu=1.5)
        matches the library default.
    length_scale, length_scale_bounds : float/array, pair
        Initial length scale(s) and optimization bounds for the additive terms.
    amplitude, amplitude_bounds : float, pair
        Initial value and bounds for each ``ConstantKernel`` amplitude.
    rescue : bool, default=True
        Whether to add the full-D Matern rescue term. Strongly recommended; it
        guarantees the additive kernel never does worse than the plain Matern on
        non-additive targets.
    rescue_nu : float, default=1.5
        ``nu`` for the rescue Matern.
    rescue_length_scale_bounds : pair, default=(1e-5, 1e5)
        Per-dimension length-scale bounds for the rescue Matern.

    Returns
    -------
    sklearn.gaussian_process.kernels.Kernel
        A composed kernel ready to hand to a ``GaussianProcessRegressor`` (e.g.
        via ``Default_GPR(kernel=...)`` or a ``SklearnGPAdapter``).

    Examples
    --------
    >>> from pygptreeo.kernels import make_additive_kernel
    >>> from pygptreeo import GPTree, Default_GPR
    >>> kernel = make_additive_kernel(n_features=6, interaction_depth=2)
    >>> gpt = GPTree(GPR=Default_GPR(kernel=kernel, alpha=1e-6))
    """
    additive = ConstantKernel(amplitude, amplitude_bounds) * AdditiveKernel(
        input_dim=n_features,
        interaction_depth=interaction_depth,
        base_kernel=base_kernel,
        length_scale=length_scale,
        length_scale_bounds=length_scale_bounds,
    )

    if not rescue:
        return additive

    if np.ndim(length_scale) == 0:
        rescue_ls = [float(length_scale)] * n_features
    else:
        rescue_ls = list(np.asarray(length_scale, dtype=float))

    rescue_term = ConstantKernel(amplitude, amplitude_bounds) * Matern(
        nu=rescue_nu,
        length_scale=rescue_ls,
        length_scale_bounds=[rescue_length_scale_bounds] * n_features,
    )

    return additive + rescue_term
