"""Tests for GPForest: how the trees' predictions are combined."""

import contextlib
import io
import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo import GPForest

warnings.filterwarnings("ignore")


def _target(X):
    return np.sin(3 * X[:, 0]) + 0.5 * np.cos(5 * X[:, 1])


def _fitted_forest(scale=1.0, seed=0):
    """A forest of three trees on 600 points of the target times ``scale``, and test data."""
    rng = np.random.RandomState(1)
    X = rng.rand(600, 2)
    y = scale * _target(X)[:, None]
    X_test = rng.rand(200, 2)
    np.random.seed(seed)
    forest = GPForest(Nbar=[50, 100, 200], theta=[1e-4, 1e-4, 1e-4])
    with contextlib.redirect_stdout(io.StringIO()):
        forest.fit(X, y, np.full_like(y, 1e-3 * scale))
    return forest, X_test, scale * _target(X_test)


class _FixedTree:
    """Stands in for a fitted tree: returns a given mean and sigma."""

    def __init__(self, mean, sigma):
        self.mean, self.sigma = np.asarray(mean, dtype=float), np.asarray(sigma, dtype=float)

    def predict(self, X):
        n = X.shape[0]
        return np.tile(self.mean, (n, 1)), np.tile(self.sigma, (n, 1))


def _combine(trees):
    forest = GPForest(Nbar=[100] * len(trees), theta=[1e-4] * len(trees))
    forest.GPTrees = trees
    mean, std = forest.predict(np.zeros((3, 2)))
    return mean[0, 0], std[0, 0]


class TestCombination(unittest.TestCase):

    def test_equal_sigmas(self):
        mean, std = _combine([_FixedTree(1.0, 0.2), _FixedTree(3.0, 0.2), _FixedTree(5.0, 0.2)])
        self.assertAlmostEqual(mean, 3.0)
        self.assertAlmostEqual(std, 0.2)

    def test_precision_weighted_mean(self):
        mean, std = _combine([_FixedTree(1.0, 0.1), _FixedTree(4.0, 0.2)])
        self.assertAlmostEqual(mean, (1.0 / 0.01 + 4.0 / 0.04) / (1 / 0.01 + 1 / 0.04))
        self.assertAlmostEqual(std, np.sqrt(2.0 / (1 / 0.01 + 1 / 0.04)))

    def test_confident_tree_dominates(self):
        mean, std = _combine([_FixedTree(1.0, 1e-6), _FixedTree(4.0, 1.0)])
        self.assertAlmostEqual(mean, 1.0, places=6)

    def test_zero_sigma(self):
        mean, std = _combine([_FixedTree(1.0, 0.0), _FixedTree(4.0, 0.5)])
        self.assertEqual(mean, 1.0)
        self.assertEqual(std, 0.0)
        mean, std = _combine([_FixedTree(1.0, 0.0), _FixedTree(4.0, 0.0)])
        self.assertEqual(mean, 2.5)
        self.assertEqual(std, 0.0)

    def test_sigma_in_large_units(self):
        """Sigmas on both sides of 1 give a positive sigma between them."""
        mean, std = _combine([_FixedTree(100.0, 0.5), _FixedTree(102.0, 5.0), _FixedTree(98.0, 50.0)])
        self.assertTrue(0.5 < std < 50.0)
        self.assertTrue(98.0 < mean < 102.0)


class TestFittedForest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.forest, cls.X_test, cls.y_test = _fitted_forest()
        cls.mean, cls.std = cls.forest.predict(cls.X_test)

    def test_shapes_and_positive_sigma(self):
        self.assertEqual(self.mean.shape, (200, 1))
        self.assertEqual(self.std.shape, (200, 1))
        self.assertTrue(np.all(np.isfinite(self.mean)))
        self.assertTrue(np.all(self.std > 0))

    def test_prediction_does_not_depend_on_the_batch(self):
        mean, std = self.forest.predict(self.X_test[:50])
        np.testing.assert_allclose(mean, self.mean[:50], rtol=1e-10)
        np.testing.assert_allclose(std, self.std[:50], rtol=1e-10)

    def test_sigma_within_the_range_of_the_trees(self):
        sigmas = np.stack([tree.predict(self.X_test)[1] for tree in self.forest.GPTrees])
        self.assertTrue(np.all(self.std >= sigmas.min(axis=0) * (1 - 1e-9)))
        self.assertTrue(np.all(self.std <= sigmas.max(axis=0) * (1 + 1e-9)))

    def test_accuracy_and_coverage_like_the_trees(self):
        tree_rmse, tree_coverage = [], []
        for tree in self.forest.GPTrees:
            mu, sd = tree.predict(self.X_test)
            tree_rmse.append(np.sqrt(np.mean((mu[:, 0] - self.y_test) ** 2)))
            tree_coverage.append(np.mean(np.abs(mu[:, 0] - self.y_test) <= sd[:, 0]))
        err = self.mean[:, 0] - self.y_test
        self.assertLess(np.sqrt(np.mean(err ** 2)), max(tree_rmse))
        coverage = np.mean(np.abs(err) <= self.std[:, 0])
        self.assertTrue(min(tree_coverage) - 0.05 <= coverage <= max(tree_coverage) + 0.05, (coverage, tree_coverage))

    def test_rescaling_the_target_rescales_the_prediction(self):
        forest, X_test, _ = _fitted_forest(scale=1000.0)
        mean, std = forest.predict(X_test)
        np.testing.assert_allclose(mean, 1000.0 * self.mean, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(std, 1000.0 * self.std, rtol=1e-3)


if __name__ == '__main__':
    unittest.main()
