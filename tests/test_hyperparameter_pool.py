"""Tests for tree-global hyperparameter pooling."""

import io
import os
import sys
import contextlib
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo.gptree import GPTree
from pygptreeo.default_gpr import Default_GPR
from pygptreeo.hyperparameter_pool import HyperparameterPool
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

warnings.filterwarnings("ignore")


def _silent(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _stream(gpt, X, y, sigma):
    for i in range(len(X)):
        gpt.update_tree(X[i:i + 1], y[i:i + 1].reshape(1, 1),
                        sigma[i:i + 1].reshape(1, 1))


class TestHyperparameterPoolUnit(unittest.TestCase):
    """Tests for the HyperparameterPool helper class in isolation."""

    def test_disabled_by_default(self):
        pool = HyperparameterPool(n_outputs=1)
        self.assertFalse(pool.enabled)

    def test_update_estimate_remove(self):
        pool = HyperparameterPool(n_outputs=1, enabled=True)
        self.assertIsNone(pool.estimate(0))
        self.assertEqual(pool.n_contributors(0), 0)

        pool.update("a", 0, np.array([1.0, 3.0]))
        pool.update("b", 0, np.array([3.0, 5.0]))
        self.assertEqual(pool.n_contributors(0), 2)
        # Elementwise median of the two contributions.
        np.testing.assert_allclose(pool.estimate(0), [2.0, 4.0])

        # Re-updating the same node overwrites rather than duplicates.
        pool.update("a", 0, np.array([5.0, 7.0]))
        self.assertEqual(pool.n_contributors(0), 2)
        np.testing.assert_allclose(pool.estimate(0), [4.0, 6.0])

        # Removal drops the contribution.
        pool.remove("a")
        self.assertEqual(pool.n_contributors(0), 1)
        np.testing.assert_allclose(pool.estimate(0), [3.0, 5.0])

    def test_update_ignores_none(self):
        pool = HyperparameterPool(n_outputs=1, enabled=True)
        pool.update("a", 0, None)
        self.assertEqual(pool.n_contributors(0), 0)

    def test_inconsistent_shapes_return_none(self):
        pool = HyperparameterPool(n_outputs=1, enabled=True)
        pool.update("a", 0, np.array([1.0, 2.0]))
        pool.update("b", 0, np.array([1.0, 2.0, 3.0]))
        self.assertIsNone(pool.estimate(0))


class TestSklearnAdapterHyperparameters(unittest.TestCase):
    """Tests for the get/set_hyperparameters interface on the sklearn adapter."""

    def test_roundtrip(self):
        gpr = Default_GPR(kernel=ConstantKernel() * Matern(length_scale=1.0, nu=1.5))
        X = np.random.RandomState(0).rand(20, 1)
        y = np.sin(3 * X[:, 0])
        _silent(gpr.fit, X, y)

        theta = gpr.get_hyperparameters()
        self.assertIsNotNone(theta)
        self.assertEqual(theta.shape[0], 2)  # constant_value + length_scale

        # Setting a new theta changes the initial kernel for the next fit.
        new_theta = theta + 0.5
        gpr.set_hyperparameters(new_theta)
        np.testing.assert_allclose(gpr.sklearn_gpr.kernel.theta, new_theta, atol=1e-9)

    def test_set_wrong_length_is_ignored(self):
        gpr = Default_GPR(kernel=ConstantKernel() * Matern(length_scale=1.0, nu=1.5))
        before = np.copy(gpr.sklearn_gpr.kernel.theta)
        gpr.set_hyperparameters(np.array([0.0, 0.0, 0.0, 0.0]))  # wrong length
        np.testing.assert_allclose(gpr.sklearn_gpr.kernel.theta, before)


class TestPoolingIntegration(unittest.TestCase):
    """Tests that pooling is correctly wired into GPTree/GPNode."""

    def setUp(self):
        rng = np.random.RandomState(0)
        self.X = rng.rand(300, 2)
        self.y = np.sin(3 * self.X[:, 0]) + 0.5 * np.cos(5 * self.X[:, 1])
        self.sigma = np.full(len(self.X), 1e-3)

    def _build(self, pool):
        np.random.seed(0)
        gpt = GPTree(GPR=Default_GPR(), Nbar=60, theta=1e-4,
                     pool_hyperparameters=pool, retrain_every_n_points=15)
        _silent(_stream, gpt, self.X, self.y, self.sigma)
        return gpt

    def test_disabled_by_default(self):
        gpt = GPTree()
        self.assertFalse(gpt.hp_pool.enabled)
        self.assertFalse(gpt.root.hp_pool.enabled)

    def test_pool_shared_by_reference(self):
        gpt = self._build(pool=True)
        # Every node references the same pool object as the tree.
        for leaf in gpt.root.leaves:
            self.assertIs(leaf.hp_pool, gpt.hp_pool)

    def test_pool_populated_when_enabled(self):
        gpt = self._build(pool=True)
        self.assertTrue(gpt.hp_pool.enabled)
        # Some mature leaves should have contributed their hyperparameters.
        self.assertGreater(gpt.hp_pool.n_contributors(0), 0)
        self.assertIsNotNone(gpt.hp_pool.estimate(0))

    def test_pool_empty_when_disabled(self):
        gpt = self._build(pool=False)
        self.assertFalse(gpt.hp_pool.enabled)
        self.assertEqual(gpt.hp_pool.n_contributors(0), 0)

    def test_internal_nodes_not_in_pool(self):
        gpt = self._build(pool=True)
        # Only current leaves may contribute; split (internal) nodes are removed.
        leaf_names = {leaf.name for leaf in gpt.root.leaves}
        for name in gpt.hp_pool._thetas[0].keys():
            self.assertIn(name, leaf_names)

    def test_predictions_finite(self):
        gpt = self._build(pool=True)
        mu, sd = gpt.predict(self.X[:20])
        self.assertTrue(np.all(np.isfinite(mu)))
        self.assertTrue(np.all(np.isfinite(sd)))


if __name__ == "__main__":
    unittest.main()
