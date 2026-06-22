"""Tests for the per-order additive kernel (OrderAdditiveKernel, Newton-Girard).

See pygptreeo/kernels.py. The gradient-vs-finite-difference test is the
important one: the kernel ships analytic gradients to sklearn's optimizer.
"""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor

from pygptreeo.kernels import OrderAdditiveKernel, make_order_additive_kernel


@pytest.mark.parametrize("d,D,base", [(4, 2, "matern"), (6, 3, "matern"),
                                      (5, 2, "rbf"), (4, 4, "rbf")])
def test_gradient_matches_finite_difference(d, D, base):
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 1, (7, d))
    k = OrderAdditiveKernel(d, max_order=D, base_kernel=base,
                            length_scale=rng.uniform(0.5, 1.5, d),
                            order_variance=rng.uniform(0.5, 1.5, min(D, d)))
    _, g = k(X, eval_gradient=True)

    theta = k.theta.copy()
    eps = 1e-6
    g_fd = np.zeros_like(g)
    for i in range(len(theta)):
        tp = theta.copy(); tp[i] += eps
        tm = theta.copy(); tm[i] -= eps
        k.theta = tp; Kp = k(X)
        k.theta = tm; Km = k(X)
        g_fd[:, :, i] = (Kp - Km) / (2 * eps)
    k.theta = theta
    assert np.max(np.abs(g - g_fd)) < 1e-5


def test_diag_matches_full_kernel():
    rng = np.random.RandomState(1)
    X = rng.uniform(0, 1, (10, 6))
    k = OrderAdditiveKernel(6, max_order=2, order_variance=[1.3, 0.7])
    assert np.allclose(k.diag(X), np.diag(k(X)))


def test_main_effects_only_is_sum_of_1d():
    """max_order=1 -> kernel is just v_1 * sum_i k_i (the e_1 term)."""
    rng = np.random.RandomState(2)
    X = rng.uniform(0, 1, (8, 4))
    k = OrderAdditiveKernel(4, max_order=1, base_kernel="rbf",
                            length_scale=[0.5, 1.0, 1.5, 2.0], order_variance=2.0)
    ls = k.length_scale
    expected = np.zeros((8, 8))
    for i in range(4):
        diff = X[:, i][:, None] - X[:, i][None, :]
        expected += np.exp(-0.5 * (diff ** 2) / ls[i] ** 2)
    assert np.allclose(k(X), 2.0 * expected)


def test_sklearn_clone_roundtrip():
    k = OrderAdditiveKernel(5, max_order=2, order_variance=[1.1, 0.4])
    ck = clone(k)
    assert ck.input_dim == 5 and ck.max_order == 2
    assert np.allclose(ck.order_variance, [1.1, 0.4])


def test_fits_and_predicts_in_gpr():
    rng = np.random.RandomState(3)
    X = rng.uniform(0, 1, (40, 6))
    y = np.sin(3 * X[:, 0]) + X[:, 1] * X[:, 2] + X[:, 3]
    g = GaussianProcessRegressor(
        kernel=make_order_additive_kernel(6, max_order=2, rescue=True),
        alpha=1e-6, n_restarts_optimizer=0)
    g.fit(X, (y - y.mean()) / y.std())
    mu, sd = g.predict(X, return_std=True)
    assert mu.shape == (40,) and np.all(np.isfinite(mu)) and np.all(sd >= 0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
