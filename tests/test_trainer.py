"""Tests for ``Trainer``, which chooses the points a tree is trained on."""

import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo import GPTree, Trainer

warnings.filterwarnings("ignore")

BOUNDS = (np.zeros(2), np.ones(2))
RANGE = (-np.inf, 0.5)        # about a fifth of the unit square


def _target(X):
    """A bowl with ripples: the lowest values lie around the centre of the unit square."""
    return 8.0 * ((X[:, 0] - 0.5) ** 2 + (X[:, 1] - 0.5) ** 2) + 0.3 * np.sin(12.0 * X[:, 0]) * np.cos(9.0 * X[:, 1])


def _tree(seed=0, **kw):
    np.random.seed(seed)
    return GPTree(Nbar=40, retrain_every_n_points=15, **kw)


def _train(trainer, n, batch=20, target=_target):
    while trainer.tree.n_points_seen < n:
        X = trainer.suggest(batch)
        y = target(X)
        trainer.add(X, y, np.full(np.shape(y), 1e-3))


def _trained(n=300, seed=0, **kw):
    trainer = Trainer(_tree(seed), bounds=BOUNDS, seed=seed, **kw)
    _train(trainer, n)
    return trainer


class TestNeed(unittest.TestCase):

    def test_arguments(self):
        with self.assertRaises(ValueError):
            Trainer(_tree(), atol=0.1)
        with self.assertRaises(ValueError):
            Trainer(_tree(), bounds=BOUNDS, candidates=lambda n: np.zeros((n, 2)), atol=0.1)
        with self.assertRaises(ValueError):
            Trainer(_tree(), bounds=BOUNDS)

    def test_sigma_in_units_of_the_tolerance(self):
        trainer = _trained(atol=0.05, rtol=0.1)
        X = np.random.RandomState(1).rand(50, 2)
        mu, sigma = trainer.tree.predict(X)
        np.testing.assert_allclose(trainer.need(X), (sigma[:, 0] / (0.05 + 0.1 * np.abs(mu[:, 0]))) ** 2, rtol=1e-6)

    def test_range(self):
        trainer = _trained(atol=0.05)
        X = np.random.RandomState(1).rand(200, 2)
        everywhere = trainer.need(X)
        mu, sigma = trainer.tree.predict(X)

        trainer.target_range = (np.asarray(-np.inf), np.asarray(0.5))
        in_range = trainer.need(X)
        self.assertTrue(np.all(in_range <= everywhere))
        far_below, far_above = mu[:, 0] < 0.5 - 5 * sigma[:, 0], mu[:, 0] > 0.5 + 5 * sigma[:, 0]
        self.assertTrue(far_below.any() and far_above.any())
        np.testing.assert_allclose(in_range[far_below], everywhere[far_below], rtol=1e-5)
        np.testing.assert_allclose(in_range[far_above], 0.0, atol=1e-5 * everywhere.max())

    def test_several_outputs(self):
        np.random.seed(0)
        tree = GPTree(Nbar=40, retrain_every_n_points=15, n_outputs=2)
        trainer = Trainer(tree, bounds=BOUNDS, atol=[0.05, 0.5], target_range=([-np.inf, -np.inf], [0.5, np.inf]), seed=0)
        _train(trainer, 200, target=lambda X: np.column_stack((_target(X), 3.0 * _target(X))))
        X = np.random.RandomState(1).rand(50, 2)
        mu, sigma = tree.predict(X)
        need = trainer.need(X)
        self.assertEqual(need.shape, (50,))
        self.assertTrue(np.all(need <= ((sigma / [0.05, 0.5]) ** 2).max(axis=1) * (1 + 1e-6)))
        self.assertEqual(trainer.suggest(10).shape, (10, 2))
        self.assertEqual(np.shape(trainer.report()['rms_error']), (2,))


class TestSuggest(unittest.TestCase):

    def test_first_points_are_candidates(self):
        trainer = Trainer(_tree(), bounds=(np.array([-1.0, 2.0]), np.array([1.0, 5.0])), atol=0.1, seed=0)
        X = trainer.suggest(30)
        self.assertEqual(X.shape, (30, 2))
        self.assertTrue(np.all((X >= [-1.0, 2.0]) & (X <= [1.0, 5.0])))
        self.assertEqual(trainer.n_initial, 80)
        self.assertEqual(len(trainer._pending), 0)

    def test_points_where_the_need_is(self):
        trainer = _trained(atol=0.02, target_range=RANGE, random_share=0.0)
        X = trainer.suggest(200)
        self.assertEqual(X.shape, (200, 2))
        self.assertEqual(len(np.unique(X, axis=0)), 200)
        candidates = np.random.RandomState(1).rand(2000, 2)
        self.assertGreater(np.mean(trainer.need(X)), 3.0 * np.mean(trainer.need(candidates)))
        self.assertGreater(np.mean(_target(X) < 0.5), 2.0 * np.mean(_target(candidates) < 0.5))

    def test_candidates_from_a_function(self):
        grid = np.stack(np.meshgrid(np.linspace(0, 1, 60), np.linspace(0, 1, 60)), axis=-1).reshape(-1, 2)
        rng = np.random.RandomState(3)
        trainer = Trainer(_tree(), candidates=lambda n: grid[rng.choice(len(grid), n, replace=False)], atol=0.02, seed=0)
        _train(trainer, 200)
        X = trainer.suggest(20)
        on_grid = {row.tobytes() for row in grid}
        self.assertTrue(all(row.tobytes() in on_grid for row in X))

    def test_more_points_than_candidates(self):
        trainer = _trained(n=100, atol=0.02, n_candidates=50)
        self.assertEqual(trainer.suggest(40).shape, (40, 2))

    def test_no_need_anywhere(self):
        trainer = _trained(n=100, atol=0.02, target_range=(100.0, 200.0))
        self.assertEqual(trainer.suggest(40).shape, (40, 2))
        self.assertLess(trainer.report()['share_in_range'], 1e-6)

    def test_lower_error_in_the_range_than_with_random_points(self):
        X_test = np.random.RandomState(2).rand(4000, 2)
        X_test = X_test[_target(X_test) < 0.5]
        for seed in range(2):
            trainer = _trained(n=400, seed=seed, atol=0.01, target_range=RANGE)
            mu, _ = trainer.tree.predict(X_test)
            error = np.sqrt(np.mean((mu[:, 0] - _target(X_test)) ** 2))

            tree = _tree(seed)
            X = np.random.RandomState(seed).rand(400, 2)
            tree.batch_update_tree(X, _target(X), 1e-3)
            mu, _ = tree.predict(X_test)
            self.assertLess(error, 0.5 * np.sqrt(np.mean((mu[:, 0] - _target(X_test)) ** 2)))


class TestChecks(unittest.TestCase):

    def test_checks_are_predicted_before_they_are_added(self):
        trainer = _trained(n=200, atol=0.02, random_share=0.5)
        X = trainer.suggest(40)
        pending = [i for i, x in enumerate(X) if x.tobytes() in trainer._pending]
        self.assertGreater(len(pending), 5)
        mu, sigma = trainer.tree.predict(X[pending])
        y = _target(X)
        before = len(trainer._checks)
        order = np.random.RandomState(0).permutation(40)
        trainer.add(X[order[:25]], y[order[:25]], 1e-3)
        trainer.add(X[order[25:]], y[order[25:]], 1e-3)
        self.assertEqual(len(trainer._pending), 0)
        self.assertEqual(len(trainer._checks), before + len(pending))
        self.assertEqual(trainer.tree.n_points_seen, 240)
        # the first check of the first batch was predicted by the tree as it was
        first = next(i for i in order if i in pending)
        error, predicted_sigma, value, weight = trainer._checks[before]
        k = pending.index(first)
        np.testing.assert_allclose(error, y[first] - mu[k])
        np.testing.assert_allclose(predicted_sigma, sigma[k])
        self.assertEqual(weight, 1.0)

    def test_points_of_the_user(self):
        trainer = _trained(n=200, atol=0.02)
        X = np.random.RandomState(4).rand(30, 2)
        before = len(trainer._checks)
        trainer.add(X, _target(X), 1e-3)
        self.assertEqual(len(trainer._checks), before)
        self.assertEqual(trainer.tree.n_points_seen, 230)

    def test_report(self):
        trainer = _trained(n=600, atol=0.05)
        report = trainer.report()
        self.assertEqual(report['n_points'], 600)
        self.assertAlmostEqual(report['share_in_range'], 1.0)
        self.assertTrue(0.0 <= report['share_above_tolerance'] <= 1.0)
        self.assertEqual(report['n_checks'], len(trainer._checks))
        self.assertGreater(report['n_checks'], 20)
        self.assertTrue(0.3 < report['coverage'] <= 1.0)
        self.assertTrue(0.0 < report['share_within_tolerance'] <= 1.0)
        self.assertEqual(np.shape(report['rms_error']), (1,))

    def test_report_for_a_range(self):
        trainer = _trained(n=600, atol=0.05, target_range=RANGE, random_share=0.3)
        report = trainer.report()
        truth = np.mean(_target(np.random.RandomState(1).rand(20000, 2)) < 0.5)
        self.assertAlmostEqual(report['share_in_range'], truth, delta=0.05)
        values = np.array([check[2][0] for check in trainer._checks])
        self.assertEqual(report['n_checks'], np.sum(values <= 0.5))
        # more of the checks lie in the range than of uniform points, and those have the smaller weights
        self.assertGreater(np.mean(values <= 0.5), 1.5 * truth)
        weights = np.array([check[3] for check in trainer._checks])
        self.assertLess(weights[values <= 0.5].mean(), weights[values > 0.5].mean())

    def test_report_without_checks(self):
        trainer = _trained(n=200, atol=0.05, random_share=0.0)
        report = trainer.report()
        self.assertEqual(report['n_checks'], 0)
        self.assertTrue(np.isnan(report['coverage']))


if __name__ == '__main__':
    unittest.main()
