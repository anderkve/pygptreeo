"""Tests for NewtonGirardAdditiveKernel."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pygptreeo.kernels import NewtonGirardAdditiveKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


def test_analytic_gradient_matches_numeric():
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 1, (7, 4))
    k = NewtonGirardAdditiveKernel([0.7, 1.2, 0.4, 0.9], [0.9, 0.5, 1.1])
    K, G = k(X, eval_gradient=True)
    assert G.shape == (7, 7, 4 + 3)              # d length scales + Q order stds
    th = k.theta.copy()
    eps = 1e-6
    Gn = np.zeros_like(G)
    for j in range(len(th)):
        tp = th.copy(); tp[j] += eps
        tm = th.copy(); tm[j] -= eps
        Gn[:, :, j] = (k.clone_with_theta(tp)(X) - k.clone_with_theta(tm)(X)) / (2 * eps)
    assert np.max(np.abs(G - Gn)) < 1e-5


def test_psd_symmetric_and_diag():
    rng = np.random.RandomState(1)
    X = rng.uniform(0, 1, (15, 5))
    k = NewtonGirardAdditiveKernel([1.0] * 5, [1.0, 1.0])
    K = k(X)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(K) >= -1e-8)
    # diag is constant = sum_q sigma_q^2 * C(d, q) = 5 + 10 = 15
    np.testing.assert_allclose(k.diag(X), np.diag(K), atol=1e-10)
    np.testing.assert_allclose(k.diag(X), 15.0, atol=1e-10)


def test_shapes_and_cross():
    rng = np.random.RandomState(2)
    X = rng.uniform(0, 1, (8, 3)); Y = rng.uniform(0, 1, (5, 3))
    k = NewtonGirardAdditiveKernel([1.0] * 3, [1.0, 1.0, 1.0])
    assert k(X, Y).shape == (8, 5)


def test_fixed_bounds_omit_gradient():
    rng = np.random.RandomState(3)
    X = rng.uniform(0, 1, (6, 3))
    k = NewtonGirardAdditiveKernel([1.0] * 3, [1.0, 1.0],
                                   length_scale_bounds="fixed")
    _, G = k(X, eval_gradient=True)
    assert G.shape == (6, 6, 2)                  # only the 2 order stds remain free


def test_gp_fit_predict():
    rng = np.random.RandomState(4)
    X = rng.uniform(0, 1, (60, 4))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + np.cos(4 * X[:, 2]) + X[:, 3]
    kernel = (NewtonGirardAdditiveKernel([1.0] * 4, [1.0, 1.0])
              + ConstantKernel(1.0) * Matern([1.0] * 4, nu=1.5))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=1, random_state=0)
    gp.fit(X, y)
    mu, sd = gp.predict(rng.uniform(0, 1, (20, 4)), return_std=True)
    assert np.all(np.isfinite(mu)) and np.all(sd > 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
