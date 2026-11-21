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
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel


class AnisotropicRationalQuadratic(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Rational quadratic kernel with one length scale per input.

        k(x, x') = (1 + d^2 / (2 alpha))^(-alpha),   d^2 = sum_i ((x_i - x'_i) / length_scale_i)^2

    A mixture of RBF kernels of different length scales. The smaller ``alpha``,
    the wider their range; for a large ``alpha`` it is the RBF kernel.

    Parameters
    ----------
    length_scale : float or ndarray of shape (n_features,), default=1.0
        One length scale for all inputs, or one per input.
    alpha : float, default=1.0
        Positive.
    length_scale_bounds, alpha_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        Bounds of the hyperparameters.
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
        """Return the kernel k(X, Y) and, with ``eval_gradient``, its gradient with
        respect to the log of the hyperparameters."""
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)

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
        """The kernel matrix K(X, Y)."""
        dists = self._compute_squared_distances(X, Y, length_scale)

        K = (1.0 + dists / (2.0 * self.alpha)) ** (-self.alpha)

        return K

    def _compute_kernel_gradient(self, X, Y, length_scale):
        """The kernel matrix K(X, Y) and its gradient with respect to the log of the
        hyperparameters that are not fixed: alpha, then the length scales, which
        is the order of scikit-learn (by name)."""
        dists = self._compute_squared_distances(X, Y, length_scale)
        base = 1.0 + dists / (2.0 * self.alpha)
        K = base ** (-self.alpha)

        gradients = []
        if not self.hyperparameter_alpha.fixed:
            # dK/d(log alpha) = K * (d² / (2 base) - alpha log(base)), with base = 1 + d² / (2 alpha)
            gradients.append(K * (dists / (2.0 * base) - self.alpha * np.log(base)))
        if not self.hyperparameter_length_scale.fixed:
            # dK/d(log l_i) = K * ((x_i - x'_i) / l_i)² / base
            if self.anisotropic:
                for i in range(length_scale.shape[0]):
                    diff_i = (X[:, i][:, np.newaxis] - Y[:, i][np.newaxis, :]) / length_scale[i]
                    gradients.append(K * diff_i ** 2 / base)
            else:
                # one length scale for all inputs: the sum over them, which is d²
                gradients.append(K * dists / base)

        K_gradient = np.stack(gradients, axis=2) if gradients else np.empty(K.shape + (0,))
        return K, K_gradient

    def _compute_squared_distances(self, X, Y, length_scale):
        """Squared distances between the rows of X and of Y, with each input divided
        by its length scale."""
        X_scaled = X / length_scale
        Y_scaled = Y / length_scale

        dists = np.sum(X_scaled ** 2, axis=1)[:, np.newaxis] + \
                np.sum(Y_scaled ** 2, axis=1)[np.newaxis, :] - \
                2.0 * np.dot(X_scaled, Y_scaled.T)

        # Numerical safety: ensure non-negative
        dists = np.maximum(dists, 0.0)

        return dists

    def diag(self, X):
        """The diagonal of k(X, X): ones."""
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
    """Additive kernel: a sum of products of one-dimensional kernels.

    With ``interaction_depth=2`` in three dimensions:

        k(x, x') = k1 + k2 + k3 + k1 k2 + k1 k3 + k2 k3,   ki = ki(x_i, x'_i)

    Every term is evaluated, so the cost grows with their number.
    ``NewtonGirardAdditiveKernel`` has the same terms, with a variance per
    order, at a cost linear in the number of inputs.

    Parameters
    ----------
    input_dim : int
        Number of inputs.
    interaction_depth : int, default=1
        Largest number of inputs in a term: 1 for main effects, 2 adds pairs.
    base_kernel : {'rbf', 'matern'}, default='rbf'
        The one-dimensional kernel: RBF, or Matern with nu=1.5.
    length_scale : float or array-like of shape (input_dim,), default=1.0
        One length scale per input. A float is used for all of them.
    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-3, 1e3)
        Bounds of the length scales.
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
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds,
            self.input_dim
        )

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and, with ``eval_gradient``, its gradient with
        respect to the log of the hyperparameters."""
        X = np.atleast_2d(X)

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
        """The kernel matrix K(X, Y)."""
        K = np.zeros((X.shape[0], Y.shape[0]))

        for term_dims in self.interaction_terms:
            K_term = np.ones((X.shape[0], Y.shape[0]))
            for dim in term_dims:
                X_dim = X[:, dim:dim+1]
                Y_dim = Y[:, dim:dim+1]
                K_dim = self._compute_1d_kernel(X_dim, Y_dim, length_scale[dim])
                K_term *= K_dim

            K += K_term

        return K

    def _compute_kernel_gradient(self, X, Y, length_scale):
        """The kernel matrix K(X, Y) and its gradient with respect to
        log(length_scale), which has no entries if the length scales are fixed."""
        if self.hyperparameter_length_scale.fixed:
            return self._compute_kernel(X, Y, length_scale), np.empty((X.shape[0], Y.shape[0], 0))

        K = np.zeros((X.shape[0], Y.shape[0]))
        K_gradient = np.zeros((X.shape[0], Y.shape[0], self.input_dim))

        for term_dims in self.interaction_terms:
            K_term = np.ones((X.shape[0], Y.shape[0]))
            K_1d_list = []  # Store individual 1D kernels
            grad_1d_list = []  # Store individual 1D gradients

            for dim in term_dims:
                X_dim = X[:, dim:dim+1]
                Y_dim = Y[:, dim:dim+1]
                K_1d, grad_1d = self._compute_1d_kernel_gradient(
                    X_dim, Y_dim, length_scale[dim]
                )
                K_1d_list.append(K_1d)
                grad_1d_list.append(grad_1d)
                K_term *= K_1d

            K += K_term

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
        """The kernel of one input, for X and Y of shape (n, 1)."""
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
        """The kernel of one input, for X and Y of shape (n, 1), and its gradient
        with respect to log(length_scale)."""
        dists_sq = np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=2) / (length_scale ** 2)

        if self.base_kernel == 'rbf':
            # k = exp(-r² / 2), dk/d(log l) = r² k
            K = np.exp(-0.5 * dists_sq)
            grad = dists_sq * K

        elif self.base_kernel == 'matern':
            # k = (1 + sqrt(3) r) exp(-sqrt(3) r), dk/d(log l) = 3 r² exp(-sqrt(3) r)
            sqrt3_r = np.sqrt(3) * np.sqrt(dists_sq)
            decay = np.exp(-sqrt3_r)
            K = (1.0 + sqrt3_r) * decay
            grad = 3 * dists_sq * decay

        else:
            raise ValueError(f"Unknown base_kernel: {self.base_kernel}")

        return K, grad

    def diag(self, X):
        """The diagonal of k(X, X): the number of terms, as each one-dimensional
        kernel is 1 there."""
        return np.full(X.shape[0], self.n_terms)

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return True

    def __repr__(self):
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
    """Additive kernel with a variance per interaction order.

        k(x, x') = sum_{q=1}^{Q} order_std_q^2 * e_q(z_1, ..., z_d)

    ``z_i`` is a one-dimensional kernel of input ``i`` and ``e_q`` the
    elementary symmetric polynomial of order ``q``: the sum of the products of
    every ``q`` of the ``z_i``. The ``e_q`` are computed from power sums
    (Newton-Girard identities), at a cost linear in the number of inputs. See
    ``docs/design.md``.

    Parameters
    ----------
    length_scale : array-like of shape (n_features,)
        Length scale of each input.
    order_std : array-like of shape (max_order,)
        Standard deviation of each interaction order ``1..Q``. Its length sets ``Q``.
    length_scale_bounds, order_std_bounds : pair of floats or "fixed"
        Bounds of the hyperparameters.
    base : {"rbf", "matern"}, default="rbf"
        The one-dimensional kernel: RBF, or Matern with nu=1.5 for a rougher target.
    """

    def __init__(self, length_scale, order_std,
                 length_scale_bounds=(1e-2, 1e2), order_std_bounds=(1e-3, 1e3),
                 base="rbf"):
        self.length_scale = length_scale
        self.order_std = order_std
        self.length_scale_bounds = length_scale_bounds
        self.order_std_bounds = order_std_bounds
        self.base = base

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds,
                              len(np.atleast_1d(self.length_scale)))

    @property
    def hyperparameter_order_std(self):
        return Hyperparameter("order_std", "numeric", self.order_std_bounds,
                              len(np.atleast_1d(self.order_std)))

    def _components(self, X, Y):
        """The one-dimensional kernels Z_i of the inputs, their derivatives G_i with
        respect to log(length_scale_i), and the length scales."""
        ls = np.atleast_1d(self.length_scale)
        Zs, Gs = [], []
        for i in range(X.shape[1]):
            Di = (X[:, i][:, None] - Y[:, i][None, :]) ** 2
            if self.base == "rbf":
                Zi = np.exp(-Di / (2.0 * ls[i] ** 2))
                Zs.append(Zi)
                Gs.append(Zi * (Di / ls[i] ** 2))               # dZ_i/dlog l_i (RBF)
            elif self.base == "matern":
                s3 = np.sqrt(3.0)
                ri = np.sqrt(Di) / ls[i]
                e = np.exp(-s3 * ri)
                Zs.append((1.0 + s3 * ri) * e)                  # Matern-3/2
                Gs.append(3.0 * (Di / ls[i] ** 2) * e)          # dZ_i/dlog l_i (Matern)
            else:
                raise ValueError(f"Unknown base {self.base!r}; use 'rbf' or 'matern'")
        return Zs, Gs, ls

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
        Zs, Gs, ls = self._components(X, Y)
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
                Gi = Gs[i]                                   # dZ_i / dlog l_i
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
                f"base='{self.base}', "
                f"order_std=[{', '.join(f'{s:.3g}' for s in sig)}])")


class AdditivePeriodicKernel(Kernel):
    """Sum of one periodic kernel per input.

        k(x, x') = sum_{i=1}^{d} exp(-2 sin^2(pi (x_i - x'_i) / period_i) / length_scale_i^2)

    Each input has its own period and length scale. See ``docs/design.md``.

    Parameters
    ----------
    length_scale : array-like of shape (n_features,)
        Length scale of each input within one period: small for sharp peaks,
        large for a nearly sinusoidal term.
    period : array-like of shape (n_features,)
        Period of each input.
    length_scale_bounds, period_bounds : pair of floats >= 0 or "fixed", default=(1e-1, 1e1)
        Bounds of the hyperparameters, in the units of the inputs that the GP
        is fitted to (standardised ones in a leaf).
    """

    def __init__(self, length_scale, period,
                 length_scale_bounds=(1e-1, 1e1), period_bounds=(1e-1, 1e1)):
        self.length_scale = length_scale
        self.period = period
        self.length_scale_bounds = length_scale_bounds
        self.period_bounds = period_bounds

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds,
                              len(np.atleast_1d(self.length_scale)))

    @property
    def hyperparameter_period(self):
        return Hyperparameter("period", "numeric", self.period_bounds,
                              len(np.atleast_1d(self.period)))

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel ``k(X, Y)`` and optionally its gradient.

        The gradient (when requested) is taken with respect to the log
        hyperparameters in scikit-learn's canonical (alphabetical) order:
        ``length_scale`` (``d`` entries) then ``period`` (``d`` entries).
        """
        X = np.atleast_2d(X)
        if eval_gradient and Y is not None:
            raise ValueError("Gradient can only be evaluated when Y is None.")
        Yv = X if Y is None else np.atleast_2d(Y)
        ls = np.atleast_1d(self.length_scale).astype(float)
        pr = np.atleast_1d(self.period).astype(float)
        d = X.shape[1]

        K = np.zeros((X.shape[0], Yv.shape[0]))
        Zs, Ss, Cs, Ds = [], [], [], []        # z_i, sin(a_i), cos(a_i), Delta_i
        for i in range(d):
            Di = X[:, i][:, None] - Yv[:, i][None, :]
            ai = np.pi * Di / pr[i]
            si = np.sin(ai)
            zi = np.exp(-2.0 * si ** 2 / ls[i] ** 2)
            K += zi
            Zs.append(zi); Ss.append(si); Cs.append(np.cos(ai)); Ds.append(Di)

        if not eval_gradient:
            return K

        grads = []
        if not self.hyperparameter_length_scale.fixed:
            # dz_i / dlog l_i = z_i * 4 sin^2(a_i) / l_i^2
            for i in range(d):
                grads.append(Zs[i] * (4.0 * Ss[i] ** 2 / ls[i] ** 2))
        if not self.hyperparameter_period.fixed:
            # dz_i / dlog p_i = z_i * 4 pi sin(a_i) cos(a_i) Delta_i / (l_i^2 p_i)
            for i in range(d):
                grads.append(Zs[i] * (4.0 * np.pi * Ss[i] * Cs[i] * Ds[i]
                                      / (ls[i] ** 2 * pr[i])))
        K_grad = (np.stack(grads, axis=2) if grads
                  else np.empty((X.shape[0], Yv.shape[0], 0)))
        return K, K_grad

    def diag(self, X):
        # z_i(x, x) = 1, so the diagonal is the constant d (one per dimension).
        X = np.atleast_2d(X)
        return np.full(X.shape[0], float(X.shape[1]))

    def is_stationary(self):
        return True

    def __repr__(self):
        ls = np.atleast_1d(self.length_scale)
        pr = np.atleast_1d(self.period)
        return (f"{self.__class__.__name__}(d={len(ls)}, "
                f"period=[{', '.join(f'{p:.3g}' for p in pr)}])")


def _broadcast_length(value, n, name):
    """Return ``value`` as a 1-D float array of length ``n``.

    A scalar is broadcast to every entry; an array-like must already have
    length ``n`` (a clear error is raised otherwise).
    """
    arr = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    if arr.size == 1:
        return np.full(n, arr.item())
    if arr.size != n:
        raise ValueError(
            f"{name} must be a scalar or an array of length {n}, "
            f"got length {arr.size}"
        )
    return arr


def AdditiveMaternKernel(
    d,
    *,
    order=2,
    nu=1.5,
    base="rbf",
    length_scale=1.0,
    order_std=1.0,
    constant_value=1.0,
    matern_length_scale=1.0,
    length_scale_bounds=(1e-2, 1e2),
    order_std_bounds=(1e-3, 1e3),
    constant_value_bounds=(1e-5, 1e5),
    matern_length_scale_bounds=(1e-5, 1e5),
):
    """An additive kernel plus a Matern kernel:

        NewtonGirardAdditiveKernel(order) + ConstantKernel() * Matern(nu)

    The additive part models main effects and interactions of up to ``order``
    inputs. The Matern part has its own length scales and models the rest.
    See ``docs/design.md``.

    Parameters
    ----------
    d : int
        Number of inputs.
    order : int, default=2
        Largest interaction order of the additive part, from 1 to ``d``.
    nu : float, default=1.5
        Smoothness of the Matern part.
    base : {"rbf", "matern"}, default="rbf"
        One-dimensional kernel of the additive part (see ``NewtonGirardAdditiveKernel``).
    length_scale : float or array-like of shape (d,), default=1.0
        Initial length scales of the additive part.
    order_std : float or array-like of shape (order,), default=1.0
        Initial standard deviation of each order of the additive part.
    constant_value : float, default=1.0
        Initial variance of the Matern part.
    matern_length_scale : float or array-like of shape (d,), default=1.0
        Initial length scales of the Matern part.
    length_scale_bounds, order_std_bounds, constant_value_bounds, matern_length_scale_bounds : pair of floats or "fixed"
        Bounds of the hyperparameters.

    Returns
    -------
    sklearn.gaussian_process.kernels.Sum

    Examples
    --------
    >>> from pygptreeo import GPTree, Default_GPR, AdditiveMaternKernel
    >>> kernel = AdditiveMaternKernel(d=4, order=2)
    >>> gpt = GPTree(GPR=Default_GPR(kernel=kernel, alpha=1e-6), Nbar=100)
    """
    # ---- Validate the structural parameters --------------------------------
    if int(d) != d or d < 1:
        raise ValueError(f"d must be a positive integer, got {d!r}")
    d = int(d)
    if int(order) != order or order < 1:
        raise ValueError(f"order must be a positive integer, got {order!r}")
    order = int(order)
    if order > d:
        raise ValueError(
            f"order (Q={order}) cannot exceed the input dimensionality d={d}; "
            f"the elementary symmetric polynomial of order q vanishes for q > d"
        )

    # ---- Build the per-dimension / per-order initial values ----------------
    additive_ls = _broadcast_length(length_scale, d, "length_scale")
    matern_ls = _broadcast_length(matern_length_scale, d, "matern_length_scale")
    order_sigma = _broadcast_length(order_std, order, "order_std")

    # ---- Assemble the composite kernel -------------------------------------
    additive = NewtonGirardAdditiveKernel(
        length_scale=additive_ls,
        order_std=order_sigma,
        length_scale_bounds=length_scale_bounds,
        order_std_bounds=order_std_bounds,
        base=base,
    )
    catch_all = (
        ConstantKernel(constant_value, constant_value_bounds)
        * Matern(length_scale=matern_ls, nu=nu,
                 length_scale_bounds=matern_length_scale_bounds)
    )
    return additive + catch_all


def AdditivePeriodicMaternKernel(
    d,
    *,
    nu=1.5,
    catch_all="matern",
    length_scale=1.0,
    period=1.0,
    periodic_constant_value=1.0,
    constant_value=1.0,
    catch_all_length_scale=1.0,
    length_scale_bounds=(1e-1, 1e1),
    period_bounds=(1e-1, 1e1),
    periodic_constant_value_bounds=(1e-3, 1e3),
    constant_value_bounds=(1e-5, 1e5),
    catch_all_length_scale_bounds=(1e-5, 1e5),
):
    """A periodic kernel per input plus a kernel that is not periodic:

        ConstantKernel() * AdditivePeriodicKernel() + ConstantKernel() * Matern(nu)

    For a target that is periodic in each input, with periods short enough to
    show within a leaf. The second part models the rest. See ``docs/design.md``.

    Parameters
    ----------
    d : int
        Number of inputs.
    nu : float, default=1.5
        Smoothness of the Matern part (not used with ``catch_all="rbf"``).
    catch_all : {"matern", "rbf"}, default="matern"
        The part that is not periodic: a Matern kernel, or an RBF kernel for a
        smooth one.
    length_scale, period : float or array-like of shape (d,), default=1.0
        Initial length scales and periods of the periodic part.
    periodic_constant_value, constant_value : float, default=1.0
        Initial variances of the periodic part and of the other part.
    catch_all_length_scale : float or array-like of shape (d,), default=1.0
        Initial length scales of the part that is not periodic.
    length_scale_bounds, period_bounds, periodic_constant_value_bounds, constant_value_bounds, catch_all_length_scale_bounds : pair of floats or "fixed"
        Bounds of the hyperparameters.

    Returns
    -------
    sklearn.gaussian_process.kernels.Sum
    """
    if int(d) != d or d < 1:
        raise ValueError(f"d must be a positive integer, got {d!r}")
    d = int(d)
    if catch_all not in ("matern", "rbf"):
        raise ValueError(f"catch_all must be 'matern' or 'rbf', got {catch_all!r}")

    periodic_ls = _broadcast_length(length_scale, d, "length_scale")
    periodic_p = _broadcast_length(period, d, "period")
    catch_ls = _broadcast_length(catch_all_length_scale, d, "catch_all_length_scale")

    periodic = (
        ConstantKernel(periodic_constant_value, periodic_constant_value_bounds)
        * AdditivePeriodicKernel(
            length_scale=periodic_ls, period=periodic_p,
            length_scale_bounds=length_scale_bounds, period_bounds=period_bounds)
    )
    if catch_all == "matern":
        base = Matern(length_scale=catch_ls, nu=nu,
                      length_scale_bounds=catch_all_length_scale_bounds)
    else:
        base = RBF(length_scale=catch_ls,
                   length_scale_bounds=catch_all_length_scale_bounds)
    catch = ConstantKernel(constant_value, constant_value_bounds) * base
    return periodic + catch


class ObliqueMatern(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Matern kernel whose length scales need not lie along the inputs.

    The distance is ``r^2 = sum_k w_k^2`` with ``w = (x - x') @ L / length_scale``
    and ``L`` unit lower triangular: ``w_k`` is input ``k`` plus ``shear`` times
    the inputs after it. With ``shear = 0`` this is the anisotropic Matern kernel.
    The metric ``L diag(length_scale^-2) L^T`` can be any positive definite one.

    ``theta`` holds the logarithms of the length scales, then the shear as it is
    (the entries of ``L`` below its diagonal, row by row).

    Parameters
    ----------
    length_scale : ndarray of shape (n_features,)
    shear : ndarray of shape (n_features (n_features - 1) / 2,), default zeros
    nu : 0.5, 1.5, 2.5 or inf
    """

    def __init__(self, length_scale, shear=None, nu=1.5, length_scale_bounds=(1e-5, 1e5), shear_bounds=(-10.0, 10.0)):
        self.length_scale = length_scale
        self.shear = shear
        self.nu = nu
        self.length_scale_bounds = length_scale_bounds
        self.shear_bounds = shear_bounds

    @property
    def n_features(self):
        return np.size(self.length_scale)

    @property
    def n_shear(self):
        return self.n_features * (self.n_features - 1) // 2

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds, self.n_features)

    @property
    def hyperparameter_shear(self):
        return Hyperparameter("shear", "numeric", self.shear_bounds, self.n_shear)

    def _shear(self):
        return np.zeros(self.n_shear) if self.shear is None else np.asarray(self.shear, dtype=float)

    def _free(self):
        """Whether the length scales and the shear are free."""
        return not self.hyperparameter_length_scale.fixed, not self.hyperparameter_shear.fixed and self.n_shear > 0

    @property
    def theta(self):
        scales, shear = self._free()
        return np.concatenate([np.log(np.atleast_1d(self.length_scale).astype(float)) if scales else [], self._shear() if shear else []])

    @theta.setter
    def theta(self, theta):
        scales, shear = self._free()
        n = self.n_features if scales else 0
        if len(theta) != n + (self.n_shear if shear else 0):
            raise ValueError(f"theta has {len(theta)} entries, the kernel {n + (self.n_shear if shear else 0)}")
        if scales:
            self.length_scale = np.exp(theta[:n])
        if shear:
            self.shear = np.array(theta[n:], dtype=float)

    @property
    def bounds(self):
        scales, shear = self._free()
        rows = []
        if scales:
            rows.append(np.log(self.hyperparameter_length_scale.bounds))
        if shear:
            rows.append(self.hyperparameter_shear.bounds)
        return np.vstack(rows) if rows else np.empty((0, 2))

    def mixing(self):
        """The unit lower triangular ``L``."""
        L = np.eye(self.n_features)
        L[np.tril_indices(self.n_features, -1)] = self._shear()
        return L

    def metric(self):
        """``M`` with ``r^2 = (x - x') M (x - x')^T``."""
        scaled = self.mixing() / np.atleast_1d(self.length_scale)
        return scaled @ scaled.T

    def axis_length_scales(self):
        """The length scale along each input: the distance over which ``r`` grows
        by one when only that input changes."""
        scaled = self.mixing() / np.atleast_1d(self.length_scale)
        return 1.0 / np.sqrt(np.sum(scaled ** 2, axis=1))

    def _correlation(self, r2):
        """The kernel and its derivative with respect to ``r^2``."""
        r = np.sqrt(r2)
        if self.nu == 0.5:
            K = np.exp(-r)
            return K, -0.5 * K / np.where(r > 0.0, r, np.inf)
        if self.nu == 1.5:
            e = np.exp(-np.sqrt(3.0) * r)
            return (1.0 + np.sqrt(3.0) * r) * e, -1.5 * e
        if self.nu == 2.5:
            e = np.exp(-np.sqrt(5.0) * r)
            return (1.0 + np.sqrt(5.0) * r + 5.0 / 3.0 * r2) * e, -5.0 / 6.0 * (1.0 + np.sqrt(5.0) * r) * e
        if self.nu == np.inf:
            K = np.exp(-0.5 * r2)
            return K, -0.5 * K
        raise ValueError(f"ObliqueMatern has nu = 0.5, 1.5, 2.5 or inf, not {self.nu!r}")

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if X.shape[1] != self.n_features:
            raise ValueError(f"The kernel has {self.n_features} length scales and X {X.shape[1]} features")
        L, scale = self.mixing(), np.atleast_1d(self.length_scale).astype(float)
        W = (X @ L / scale).T                                                # (n_features, n)
        if Y is None:
            D = W[:, :, None] - W[:, None, :]
        elif eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")
        else:
            D = W[:, :, None] - (np.atleast_2d(Y) @ L / scale).T[:, None, :]
        K, dK = self._correlation(np.einsum("kij,kij->ij", D, D))
        if not eval_gradient:
            return K
        scales, shear = self._free()
        gradient = np.empty((self.n_features * scales + self.n_shear * shear,) + K.shape)
        if scales:
            np.multiply(D, D, out=gradient[:self.n_features])
            gradient[:self.n_features] *= -2.0 * dK                         # d r^2 / d log(length_scale_k)
        if shear:
            raw = X.T[:, :, None] - X.T[:, None, :]
            D *= 2.0 * dK / scale[:, None, None]
            for m, (j, k) in enumerate(zip(*np.tril_indices(self.n_features, -1))):
                np.multiply(D[k], raw[j], out=gradient[m - self.n_shear])   # d r^2 / d L_jk
        return K, np.moveaxis(gradient, 0, -1)

    def __repr__(self):
        return (f"{self.__class__.__name__}(length_scale=[{', '.join(f'{v:.3g}' for v in np.atleast_1d(self.length_scale))}], "
                f"shear=[{', '.join(f'{v:.3g}' for v in self._shear())}], nu={self.nu:.3g})")


def shear_dims(kernel):
    """The positions in ``kernel.theta`` of the shear of its ObliqueMatern parts."""
    if isinstance(kernel, ObliqueMatern):
        scales, shear = kernel._free()
        first = kernel.n_features if scales else 0
        return np.arange(first, first + (kernel.n_shear if shear else 0))
    if hasattr(kernel, "k1") and hasattr(kernel, "k2"):
        return np.concatenate([shear_dims(kernel.k1), kernel.k1.n_dims + shear_dims(kernel.k2)]).astype(int)
    if hasattr(kernel, "kernel"):
        return shear_dims(kernel.kernel)
    return np.arange(0)
