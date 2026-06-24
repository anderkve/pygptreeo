"""Tests for the AdditiveMaternKernel shorthand.

AdditiveMaternKernel is the convenience constructor for the README's recommended
leaf kernel: a low-order NewtonGirardAdditiveKernel summed with a separate
ConstantKernel * Matern catch-all. These tests check that the shorthand builds
exactly that composite, validates its arguments, and behaves correctly through
the scikit-learn (clone / optimise / length-scale extraction) and GPTreeO
(end-to-end fit / predict / split) machinery.
"""

import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, Product, Sum

from pygptreeo import (
    AdditiveMaternKernel,
    Default_GPR,
    GPTree,
    NewtonGirardAdditiveKernel,
)
from pygptreeo.adapters import SklearnGPAdapter


def _low_order_target(X):
    """A function with only main effects + one pairwise interaction."""
    return (np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + np.cos(4 * X[:, 2])
            + 0.5 * X[:, 0] * X[:, 1])


class TestAdditiveMaternStructure(unittest.TestCase):

    def test_returns_expected_composite(self):
        k = AdditiveMaternKernel(d=4, order=2, nu=1.5)
        # Sum(additive, Product(ConstantKernel, Matern))
        self.assertIsInstance(k, Sum)
        self.assertIsInstance(k.k1, NewtonGirardAdditiveKernel)
        self.assertIsInstance(k.k2, Product)
        self.assertIsInstance(k.k2.k1, ConstantKernel)
        self.assertIsInstance(k.k2.k2, Matern)
        self.assertEqual(k.k2.k2.nu, 1.5)

    def test_theta_size(self):
        # order-2, d=4: 4 additive ls + 2 order std + 1 const + 4 matern ls = 11
        self.assertEqual(AdditiveMaternKernel(d=4, order=2).n_dims, 11)
        # order-1, d=5: 5 + 1 + 1 + 5 = 12
        self.assertEqual(AdditiveMaternKernel(d=5, order=1).n_dims, 12)

    def test_additive_and_matern_have_d_length_scales_each(self):
        k = AdditiveMaternKernel(d=6, order=2)
        self.assertEqual(len(np.atleast_1d(k.k1.length_scale)), 6)
        self.assertEqual(len(np.atleast_1d(k.k2.k2.length_scale)), 6)
        # order_std has one entry per active order
        self.assertEqual(len(np.atleast_1d(k.k1.order_std)), 2)

    def test_matches_explicit_construction(self):
        """The shorthand reproduces the explicit README composite exactly."""
        d = 4
        k = AdditiveMaternKernel(d=d, order=2, nu=1.5)
        k_ref = (NewtonGirardAdditiveKernel(length_scale=[1.0] * d,
                                            order_std=[1.0, 1.0])
                 + ConstantKernel() * Matern(nu=1.5, length_scale=[1.0] * d))
        X = np.random.RandomState(0).uniform(0, 1, (7, d))
        np.testing.assert_allclose(k(X), k_ref(X))
        np.testing.assert_allclose(k.theta, k_ref.theta)
        np.testing.assert_allclose(k.bounds, k_ref.bounds)

    def test_psd_and_symmetric(self):
        k = AdditiveMaternKernel(d=4, order=2)
        X = np.random.RandomState(1).uniform(0, 1, (12, 4))
        K = k(X)
        np.testing.assert_allclose(K, K.T, atol=1e-12)
        self.assertTrue(np.all(np.linalg.eigvalsh(K) >= -1e-8))
        np.testing.assert_allclose(k.diag(X), np.diag(K), atol=1e-10)


class TestAdditiveMaternArguments(unittest.TestCase):

    def test_scalar_length_scale_is_broadcast(self):
        k = AdditiveMaternKernel(d=3, length_scale=0.5, matern_length_scale=2.0)
        np.testing.assert_allclose(np.atleast_1d(k.k1.length_scale), [0.5, 0.5, 0.5])
        np.testing.assert_allclose(np.atleast_1d(k.k2.k2.length_scale), [2.0, 2.0, 2.0])

    def test_array_length_scales_are_honoured_and_independent(self):
        k = AdditiveMaternKernel(d=3, length_scale=[0.1, 0.2, 0.3],
                                 matern_length_scale=[1.0, 2.0, 3.0],
                                 order_std=[0.5, 0.7])
        np.testing.assert_allclose(np.atleast_1d(k.k1.length_scale), [0.1, 0.2, 0.3])
        np.testing.assert_allclose(np.atleast_1d(k.k2.k2.length_scale), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(np.atleast_1d(k.k1.order_std), [0.5, 0.7])

    def test_custom_bounds_and_constant_value(self):
        k = AdditiveMaternKernel(d=2, constant_value=3.0,
                                 constant_value_bounds=(1e-2, 1e2),
                                 matern_length_scale_bounds=(1e-1, 1e1))
        self.assertEqual(k.k2.k1.constant_value, 3.0)
        self.assertEqual(tuple(k.k2.k1.constant_value_bounds), (1e-2, 1e2))
        self.assertEqual(tuple(np.atleast_2d(k.k2.k2.length_scale_bounds)[0]), (1e-1, 1e1))

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            AdditiveMaternKernel(d=0)
        with self.assertRaises(ValueError):
            AdditiveMaternKernel(d=3, order=0)
        with self.assertRaises(ValueError):
            AdditiveMaternKernel(d=3, order=4)            # order > d
        with self.assertRaises(ValueError):
            AdditiveMaternKernel(d=4, length_scale=[1.0, 2.0, 3.0])   # wrong length
        with self.assertRaises(ValueError):
            AdditiveMaternKernel(d=4, order=2, order_std=[1.0, 1.0, 1.0])  # != order


class TestAdditiveMaternSklearnIntegration(unittest.TestCase):

    def test_clone_preserves_structure_and_theta(self):
        k = AdditiveMaternKernel(d=4, order=2)
        ck = clone(k)
        self.assertIsInstance(ck, Sum)
        np.testing.assert_allclose(ck.theta, k.theta)
        np.testing.assert_allclose(ck.bounds, k.bounds)

    def test_clone_with_theta_sets_hyperparameters(self):
        k = AdditiveMaternKernel(d=3, order=2)
        new_theta = k.theta + 0.3
        k2 = k.clone_with_theta(new_theta)
        np.testing.assert_allclose(k2.theta, new_theta)
        # original is unchanged
        np.testing.assert_allclose(k.theta, AdditiveMaternKernel(d=3, order=2).theta)

    def test_deep_params_expose_length_scales(self):
        """get_params(deep=True) must expose length_scale keys for both parts.

        SklearnGPAdapter.get_length_scales (used by GPTreeO's split logic) relies
        on this to find per-dimension length scales in the composite kernel.
        """
        k = AdditiveMaternKernel(d=4, order=2)
        ls_keys = [key for key in k.get_params(deep=True)
                   if key.endswith("length_scale")]
        self.assertEqual(len(ls_keys), 2)   # additive + matern length scales

    def test_gp_fit_predict(self):
        rng = np.random.RandomState(4)
        X = rng.uniform(0, 1, (60, 3))
        y = _low_order_target(X)
        kernel = AdditiveMaternKernel(d=3, order=2)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                      n_restarts_optimizer=1, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X, y)
        mu, sd = gp.predict(rng.uniform(0, 1, (15, 3)), return_std=True)
        self.assertTrue(np.all(np.isfinite(mu)))
        self.assertTrue(np.all(sd > 0))

    def test_length_scales_recoverable_after_fit(self):
        rng = np.random.RandomState(5)
        X = rng.uniform(0, 1, (60, 3))
        adapter = SklearnGPAdapter(
            GaussianProcessRegressor(kernel=AdditiveMaternKernel(d=3, order=2),
                                     alpha=1e-6, normalize_y=True)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter.fit(X, _low_order_target(X).reshape(-1, 1))
        ls = adapter.get_length_scales(3)
        self.assertIsNotNone(ls)
        self.assertEqual(ls.shape, (3,))
        self.assertTrue(np.all(ls > 0))


class TestAdditiveMaternGPTreeIntegration(unittest.TestCase):

    def test_end_to_end_training_and_prediction(self):
        """The shorthand works as a GPTree leaf kernel, through node splits."""
        rng = np.random.RandomState(6)
        d = 3
        X = rng.uniform(0, 1, (80, d))
        y = _low_order_target(X)

        gpt = GPTree(
            GPR=Default_GPR(kernel=AdditiveMaternKernel(d=d, order=2), alpha=1e-6),
            Nbar=30,
            split_dimension_criteria="min_lengthscale",  # exercises get_length_scales
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(len(X)):
                gpt.update_tree(X[i:i + 1, :], float(y[i]), 1e-3)

            y_pred, y_std = gpt.predict(rng.uniform(0, 1, (10, d)))

        self.assertEqual(y_pred.shape[0], 10)
        self.assertTrue(np.all(np.isfinite(y_pred)))
        self.assertTrue(np.all(y_std >= 0))


if __name__ == "__main__":
    sys.exit(unittest.main())
