"""Prototype: per-order additive kernel via Newton-Girard (variant "A").

This is a *speed experiment*, kept separate from the production ``AdditiveKernel``
in ``kernels.py`` and deliberately not wired into the tree.

Where the production kernel enumerates every interaction term explicitly -- a
depth-``D`` kernel over ``d`` inputs assembles ``sum_{n<=D} C(d,n)`` product
matrices on every evaluation, i.e. ``O(d^D)`` work -- this kernel uses the
classical additive-GP construction (Duvenaud et al. 2011; the same backbone as
OAK, Lu et al. 2022). The order-``n`` contribution is the ``n``-th elementary
symmetric polynomial ``e_n`` of the ``d`` one-dimensional kernels, and all of
``e_1..e_D`` are obtained from the power sums by the Newton-Girard recursion in
``O(d*D)`` elementwise matrix operations -- independent of ``C(d,n)``.

Parameterization (this is variant "A", per-*order*, NOT per-term):
    k(x, x') = sum_{n=1..D}  v_n * e_n( z_1, ..., z_d )
where ``z_i = k_i(x_i, x'_i)`` is the 1-D base kernel for dimension ``i`` (its own
length scale ``l_i``) and ``v_n`` is a single variance shared by *all* order-``n``
terms. Hyperparameters: ``d`` length scales + ``D`` order variances (``d + D``),
independent of the number of terms.

Gradients are analytic (verified against finite differences in
``examples/benchmark_oak_kernel.py``):
    dk/dlog v_n   = v_n * e_n
    dk/dlog l_i   = ( sum_n v_n * e_{n-1}^{\\i} ) * dz_i/dlog l_i
where ``e_m^{\\i}`` is the ``m``-th elementary symmetric polynomial of every
dimension *except* ``i``, built by ``e_m^{\\i} = e_m - z_i * e_{m-1}^{\\i}``.
"""

import numpy as np
from sklearn.gaussian_process.kernels import (Kernel, Hyperparameter,
                                              StationaryKernelMixin,
                                              ConstantKernel, Matern)


class OrderAdditiveKernel(StationaryKernelMixin, Kernel):
    """Additive kernel with per-order variances, assembled via Newton-Girard.

    Parameters
    ----------
    input_dim : int
        Number of input dimensions ``d``.
    max_order : int, default=2
        Maximum interaction order ``D`` (1 = main effects only, 2 = + pairwise).
    base_kernel : {'matern', 'rbf'}, default='matern'
        1-D base kernel (Matern uses nu=1.5).
    length_scale : float or ndarray of shape (input_dim,), default=1.0
        Per-dimension length scale(s).
    length_scale_bounds : pair, default=(1e-4, 1e4)
    order_variance : float or ndarray of shape (max_order,), default=1.0
        Variance ``v_n`` for each interaction order.
    order_variance_bounds : pair, default=(1e-6, 1e8)
    """

    def __init__(self, input_dim, max_order=2, base_kernel='matern',
                 length_scale=1.0, length_scale_bounds=(1e-4, 1e4),
                 order_variance=1.0, order_variance_bounds=(1e-6, 1e8)):
        self.input_dim = input_dim
        self.max_order = min(max_order, input_dim)
        self.base_kernel = base_kernel
        self.length_scale_bounds = length_scale_bounds
        self.order_variance_bounds = order_variance_bounds

        if np.ndim(length_scale) == 0:
            self.length_scale = np.full(input_dim, float(length_scale))
        else:
            self.length_scale = np.asarray(length_scale, dtype=float)
            if len(self.length_scale) != input_dim:
                raise ValueError(f"length_scale must have length {input_dim}")

        if np.ndim(order_variance) == 0:
            self.order_variance = np.full(self.max_order, float(order_variance))
        else:
            self.order_variance = np.asarray(order_variance, dtype=float)
            if len(self.order_variance) != self.max_order:
                raise ValueError(f"order_variance must have length {self.max_order}")

    # sklearn collects hyperparameters from `hyperparameter_*` attributes in
    # alphabetical order: 'length_scale' < 'order_variance', so theta (and the
    # gradient columns) are [length scales (d), order variances (D)].
    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric",
                              self.length_scale_bounds, self.input_dim)

    @property
    def hyperparameter_order_variance(self):
        return Hyperparameter("order_variance", "numeric",
                              self.order_variance_bounds, self.max_order)

    def _per_dim(self, X, Y, eval_gradient):
        """1-D base kernels z_i and their d/dlog(l_i) derivatives, per dimension."""
        ls = self.length_scale
        sqrt3 = np.sqrt(3)
        z, g = [], ([] if eval_gradient else None)
        for i in range(self.input_dim):
            diff = X[:, i][:, None] - Y[:, i][None, :]
            dists_sq = (diff * diff) / (ls[i] ** 2)
            if self.base_kernel == 'rbf':
                zi = np.exp(-0.5 * dists_sq)
                if eval_gradient:
                    g.append(dists_sq * zi)
            elif self.base_kernel == 'matern':
                sqrt3_r = sqrt3 * np.sqrt(dists_sq)
                exp_term = np.exp(-sqrt3_r)
                zi = (1.0 + sqrt3_r) * exp_term
                if eval_gradient:
                    g.append(3.0 * dists_sq * exp_term)
            else:
                raise ValueError(f"Unknown base_kernel: {self.base_kernel}")
            z.append(zi)
        return z, g

    def _elementary(self, z, nx, ny):
        """Power sums p_k and elementary symmetric polynomials e_0..e_D."""
        D = self.max_order
        p = [None] * (D + 1)
        for k in range(1, D + 1):
            acc = np.zeros((nx, ny))
            for zi in z:
                acc += zi ** k
            p[k] = acc
        e = [None] * (D + 1)
        e[0] = np.ones((nx, ny))
        for n in range(1, D + 1):
            s = np.zeros((nx, ny))
            for k in range(1, n + 1):
                s += ((-1) ** (k - 1)) * e[n - k] * p[k]
            e[n] = s / n
        return e

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        Y = X if Y is None else np.atleast_2d(Y)
        nx, ny = X.shape[0], Y.shape[0]
        D = self.max_order
        v = self.order_variance

        z, g = self._per_dim(X, Y, eval_gradient)
        e = self._elementary(z, nx, ny)

        K = np.zeros((nx, ny))
        for n in range(1, D + 1):
            K += v[n - 1] * e[n]

        if not eval_gradient:
            return K

        n_hyper = self.input_dim + D
        grad = np.zeros((nx, ny, n_hyper))

        # d k / dlog l_i = ( sum_n v_n e_{n-1}^{\i} ) * dz_i/dlog l_i
        for i in range(self.input_dim):
            ei = [None] * D                 # e_0^{\i} .. e_{D-1}^{\i}
            ei[0] = np.ones((nx, ny))
            for m in range(1, D):
                ei[m] = e[m] - z[i] * ei[m - 1]
            dK_dzi = np.zeros((nx, ny))
            for n in range(1, D + 1):
                dK_dzi += v[n - 1] * ei[n - 1]
            grad[:, :, i] = dK_dzi * g[i]

        # d k / dlog v_n = v_n e_n
        for n in range(1, D + 1):
            grad[:, :, self.input_dim + (n - 1)] = v[n - 1] * e[n]

        return K, grad

    def diag(self, X):
        # z_i(x, x) = 1, so e_n(1,...,1) = C(d, n); diag is the constant sum.
        from math import comb
        d = self.input_dim
        val = sum(self.order_variance[n - 1] * comb(d, n)
                  for n in range(1, self.max_order + 1))
        return np.full(X.shape[0], float(val))

    def is_stationary(self):
        return True

    def __repr__(self):
        return (f"OrderAdditiveKernel(d={self.input_dim}, max_order={self.max_order}, "
                f"base='{self.base_kernel}', "
                f"order_variance={np.array2string(self.order_variance, precision=3)})")


def make_order_additive_kernel(n_features, max_order=2, base_kernel='matern',
                               length_scale=1.0, length_scale_bounds=(1e-4, 1e4),
                               order_variance=1.0, order_variance_bounds=(1e-6, 1e8),
                               rescue=True, rescue_nu=1.5,
                               rescue_length_scale_bounds=(1e-5, 1e5)):
    """Variant-A counterpart of ``make_additive_kernel``: per-order additive
    kernel (Newton-Girard) plus the same full-D Matern rescue term.

    Note the per-order variances ``v_n`` already provide the additive amplitudes,
    so no extra ``ConstantKernel`` is needed in front of the additive block.
    """
    additive = OrderAdditiveKernel(
        input_dim=n_features, max_order=max_order, base_kernel=base_kernel,
        length_scale=length_scale, length_scale_bounds=length_scale_bounds,
        order_variance=order_variance, order_variance_bounds=order_variance_bounds)

    if not rescue:
        return additive

    if np.ndim(length_scale) == 0:
        rescue_ls = [float(length_scale)] * n_features
    else:
        rescue_ls = list(np.asarray(length_scale, dtype=float))
    rescue_term = ConstantKernel(1.0, (1e-3, 1e8)) * Matern(
        nu=rescue_nu, length_scale=rescue_ls,
        length_scale_bounds=[rescue_length_scale_bounds] * n_features)
    return additive + rescue_term
