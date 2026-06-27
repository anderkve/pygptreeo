"""Tests for AdditivePeriodicKernel and the AdditivePeriodicMaternKernel shorthand."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pygptreeo.kernels import AdditivePeriodicKernel, AdditivePeriodicMaternKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF


def test_analytic_gradient_matches_numeric():
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 1, (8, 4))
    k = AdditivePeriodicKernel(length_scale=[0.7, 1.2, 0.4, 0.9],
                               period=[0.5, 1.3, 0.8, 2.0])
    K, G = k(X, eval_gradient=True)
    assert G.shape == (8, 8, 4 + 4)              # d length scales + d periods
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
    k = AdditivePeriodicKernel([1.0] * 5, [1.0] * 5)
    K = k(X)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(K) >= -1e-8)
    # diag is constant = d (each per-dimension term contributes 1 at zero separation)
    np.testing.assert_allclose(k.diag(X), np.diag(K), atol=1e-10)
    np.testing.assert_allclose(k.diag(X), 5.0, atol=1e-10)


def test_multidim_periodic_is_psd_unlike_isotropic_expsinesquared():
    """The whole point of the per-dimension build: PSD in >1D (ExpSineSquared is not)."""
    from sklearn.gaussian_process.kernels import ExpSineSquared
    rng = np.random.RandomState(7)
    X = rng.standard_normal((40, 4))                       # multidimensional, spread out
    # isotropic ExpSineSquared on Euclidean distance is NOT PSD in >1D
    assert np.linalg.eigvalsh(ExpSineSquared(1.0, 1.0)(X)).min() < -1e-3
    # the per-dimension additive periodic kernel stays PSD
    assert np.linalg.eigvalsh(AdditivePeriodicKernel([1.0] * 4, [1.0] * 4)(X)).min() >= -1e-8


def test_shapes_and_cross():
    rng = np.random.RandomState(2)
    X = rng.uniform(0, 1, (8, 3)); Y = rng.uniform(0, 1, (5, 3))
    k = AdditivePeriodicKernel([1.0] * 3, [1.0] * 3)
    assert k(X, Y).shape == (8, 5)
    with pytest.raises(ValueError):           # gradient only defined for Y is None
        k(X, Y, eval_gradient=True)


def test_fixed_bounds_omit_gradient():
    rng = np.random.RandomState(3)
    X = rng.uniform(0, 1, (6, 3))
    k = AdditivePeriodicKernel([1.0] * 3, [1.0] * 3, period_bounds="fixed")
    _, G = k(X, eval_gradient=True)
    assert G.shape == (6, 6, 3)                # only the 3 length scales remain free


def test_large_period_degrades_to_rbf_like():
    """As the period grows past the data span the term stops oscillating (RBF-like)."""
    rng = np.random.RandomState(5)
    X = rng.uniform(0, 1, (12, 4))
    K = AdditivePeriodicKernel([1.0] * 4, [1e3] * 4)(X)
    assert np.all(K > 0)                        # no sign oscillation when period >> span


def test_shorthand_matern_and_rbf_catch_all():
    rng = np.random.RandomState(4)
    X = rng.uniform(0, 1, (10, 4))
    km = AdditivePeriodicMaternKernel(d=4)                     # Matern catch-all
    kr = AdditivePeriodicMaternKernel(d=4, catch_all="rbf")    # RBF catch-all
    for k in (km, kr):
        K = k(X)
        assert np.all(np.isfinite(K))
        assert np.all(np.linalg.eigvalsh(K) >= -1e-8)
    # the periodic component is the left summand; the catch-all differs
    assert isinstance(km.k2.k2, Matern)
    assert isinstance(kr.k2.k2, RBF)
    with pytest.raises(ValueError):
        AdditivePeriodicMaternKernel(d=4, catch_all="bogus")


def test_shorthand_gp_fit_predict():
    rng = np.random.RandomState(6)
    X = rng.uniform(0, 1, (60, 3))
    # additive periodic target (per-dimension oscillation)
    y = (np.sin(2 * np.pi * 3 * X[:, 0]) + np.sin(2 * np.pi * 3 * X[:, 1])
         + np.sin(2 * np.pi * 3 * X[:, 2]))
    gp = GaussianProcessRegressor(kernel=AdditivePeriodicMaternKernel(d=3),
                                  normalize_y=True, n_restarts_optimizer=1,
                                  random_state=0)
    gp.fit(X, y)
    mu, sd = gp.predict(rng.uniform(0, 1, (20, 3)), return_std=True)
    assert np.all(np.isfinite(mu)) and np.all(sd > 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
