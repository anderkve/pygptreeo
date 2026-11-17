"""Tests for the leaf fits handed out as tasks (``update_tree(defer_fits=True)``,
``solve_fit_task`` and ``apply_fit_result``)."""

import os
import pickle
import sys
import unittest
import warnings
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo import GPTree, Default_GPR, GlobalMeanLearner, SklearnGPAdapter, solve_fit_task
from pygptreeo import gptree as gptree_module

warnings.filterwarnings("ignore")


def _target(X):
    return np.sin(3.0 * X[:, 0]) + X[:, 1] ** 2


def _data(n, seed=0, p=1):
    X = np.random.RandomState(seed).rand(n, 2)
    Y = np.column_stack([(j + 1) * _target(X) for j in range(p)])
    return X, Y


def _update(gpt, X, Y, i, **kw):
    return gpt.update_tree(X[i:i + 1], Y[i:i + 1], 1e-3, **kw)


def _solve(task):
    """What a fit worker returns for a task, both sent as pickles."""
    return pickle.loads(pickle.dumps(solve_fit_task(pickle.loads(pickle.dumps(task)))))


def _stream(gpt, X, Y, lag=None):
    """Feed the points to the tree and return its predictions of them, each made
    before the point is added. With ``lag``, the fits are handed out and their
    results applied that many points later."""
    predictions, results = [], []
    for i in range(X.shape[0]):
        while results and results[0][0] <= i:
            gpt.apply_fit_result(results.pop(0)[1])
        predictions.append(gpt.predict(X[i:i + 1])[0][0, 0])
        task = _update(gpt, X, Y, i, defer_fits=lag is not None)
        if task is not None:
            results.append((i + lag, _solve(task)))
    return np.array(predictions)


class Snapshot:
    def __init__(self, version, slope):
        self.version, self.slope = version, slope

    def predict(self, X):
        return self.slope * np.atleast_2d(X)[:, :1]


class SteppingLearner(GlobalMeanLearner):
    """A global model that publishes a new snapshot every 30 points."""

    def __init__(self):
        self.current, self.error_scale, self.n_seen = None, None, 0

    def observe(self, x, y, sigma):
        self.n_seen += 1
        if self.n_seen % 30 == 0:
            self.current = Snapshot(self.n_seen // 30, 0.1 * self.n_seen)
        return False


class TestBackend(unittest.TestCase):

    def test_sklearn_adapter(self):
        X, Y = _data(40)
        gpr = Default_GPR()
        self.assertIsNone(gpr.get_hyperparameters())
        gpr.set_observation_noise(np.full(40, 1e-6))
        copy = gpr.unfitted_copy()
        gpr.fit(X, Y)
        theta = gpr.get_hyperparameters()
        self.assertEqual(theta.shape, (3,))           # an amplitude and two length scales

        # An unfitted copy of a fitted GP has its settings and nothing of the fit
        for other in (copy, gpr.unfitted_copy()):
            self.assertFalse(other.is_trained())
            self.assertEqual(other.nbytes(), 0)
            other.fit(X, Y)
            np.testing.assert_array_equal(other.get_hyperparameters(), theta)

        # A GP that is given the hyperparameters is the fitted one
        given = gpr.unfitted_copy()
        self.assertFalse(given.refit_fixed(X, Y))     # no hyperparameters to keep
        self.assertTrue(given.refit_fixed(X, Y, theta))
        X_test = np.random.RandomState(1).rand(20, 2)
        for a, b in zip(given.predict(X_test, return_std=True), gpr.predict(X_test, return_std=True)):
            np.testing.assert_array_equal(a, b)

    def test_a_backend_that_cannot_is_fitted_in_place(self):
        X, Y = _data(30)
        gpt = GPTree(Nbar=100, retrain_every_n_points=10)
        with mock.patch.object(SklearnGPAdapter, 'unfitted_copy', lambda self: None):
            tasks = [_update(gpt, X, Y, i, defer_fits=True) for i in range(30)]
        self.assertEqual(tasks, [None] * 30)
        self.assertTrue(gpt.root.my_GPR.is_trained())
        self.assertEqual(gpt.root.n_points_since_retrain, 0)


class TestDeferredFits(unittest.TestCase):

    def _same_as_in_place(self, n, p=1, options=dict):
        """Until the first split, a result applied at once gives the tree fitted in
        place. Returns the number of fits handed out."""
        X, Y = _data(n, p=p)
        X_test = np.random.RandomState(1).rand(30, 2)
        in_place = GPTree(Nbar=100, retrain_every_n_points=20, n_outputs=p, **options())
        deferred = GPTree(Nbar=100, retrain_every_n_points=20, n_outputs=p, **options())
        n_tasks = 0
        for i in range(n):
            self.assertIsNone(_update(in_place, X, Y, i))
            task = _update(deferred, X, Y, i, defer_fits=True)
            if task is not None:
                n_tasks += 1
                self.assertEqual(len(task['gprs']), p)
                self.assertTrue(deferred.apply_fit_result(_solve(task)))
            for a, b in zip(deferred.predict(X_test), in_place.predict(X_test)):
                np.testing.assert_array_equal(a, b)
        self.assertEqual(deferred._fit_tasks, {})
        return n_tasks

    def test_same_as_in_place(self):
        self.assertEqual(self._same_as_in_place(90), 4)

    def test_same_as_in_place_with_two_outputs(self):
        self.assertEqual(self._same_as_in_place(60, p=2), 3)

    def test_same_as_in_place_without_scaling(self):
        self.assertEqual(self._same_as_in_place(60, options=lambda: dict(use_standard_scaling=False)), 3)

    def test_same_as_in_place_with_a_global_model(self):
        """The task holds the residual of the snapshot, and the result is applied to that of the current one."""
        self.assertGreater(self._same_as_in_place(90, options=lambda: dict(global_mean=SteppingLearner())), 0)

    def test_a_task_is_small_and_leaves_the_leaf_as_it_is(self):
        X, Y = _data(60)
        gpt = GPTree(Nbar=100, retrain_every_n_points=20)
        tasks = [task for task in (_update(gpt, X, Y, i, defer_fits=True) for i in range(20)) if task is not None]
        self.assertEqual(len(tasks), 1)
        gpt.apply_fit_result(_solve(tasks[0]))
        X_test = np.random.RandomState(1).rand(30, 2)
        for i in range(20, 39):
            self.assertIsNone(_update(gpt, X, Y, i, defer_fits=True))
        before = gpt.predict(X_test)
        task = _update(gpt, X, Y, 39, defer_fits=True)
        self.assertEqual(task['X'].shape, (40, 2))
        self.assertFalse(task['gprs'][0].is_trained())
        self.assertLess(len(pickle.dumps(task)), 4000)
        self.assertLess(len(pickle.dumps(_solve(task))), 400)
        self.assertEqual(gpt.root.n_points_since_retrain, 0)
        # The leaf has taken the point as an appended one and keeps its hyperparameters
        self.assertEqual(gpt.root.my_GPR.sklearn_gpr.X_train_.shape[0], 40)
        after = gpt.predict(X_test)
        self.assertFalse(np.array_equal(before[0], after[0]))
        theta = gpt.root.my_GPR.get_hyperparameters()
        np.testing.assert_array_equal(theta, _solve(tasks[0])['hyperparameters'][0])

    def test_a_leaf_waits_for_its_result(self):
        X, Y = _data(95)
        gpt = GPTree(Nbar=100, retrain_every_n_points=20)
        tasks = [task for task in (_update(gpt, X, Y, i, defer_fits=True) for i in range(70)) if task is not None]
        self.assertEqual(len(tasks), 1)                # due at 20, 40 and 60 points
        self.assertFalse(gpt.root.my_GPR.is_trained())
        self.assertEqual(gpt.root.n_points_since_retrain, 50)
        self.assertTrue(gpt.apply_fit_result(_solve(tasks[0])))
        # Fitted to the 70 points it holds, with the hyperparameters found for the first 20
        self.assertEqual(gpt.root.my_GPR.sklearn_gpr.X_train_.shape[0], 70)
        self.assertFalse(gpt.apply_fit_result(_solve(tasks[0])))     # a result is taken once
        # It asks for the fit it is due with its next point
        task = _update(gpt, X, Y, 70, defer_fits=True)
        self.assertEqual(task['X'].shape[0], 71)
        self.assertEqual(task['id'], tasks[0]['id'] + 1)

    def test_a_failed_fit_lets_the_leaf_ask_again(self):
        X, Y = _data(40)
        gpt = GPTree(Nbar=100, retrain_every_n_points=20)
        tasks = [task for task in (_update(gpt, X, Y, i, defer_fits=True) for i in range(25)) if task is not None]
        self.assertFalse(gpt.apply_fit_result({'id': tasks[0]['id'], 'hyperparameters': None}))
        self.assertFalse(gpt.root.my_GPR.is_trained())
        # As after a fit in place that failed: with the next full retrain buffer
        tasks = [_update(gpt, X, Y, i, defer_fits=True) for i in range(25, 40)]
        self.assertEqual([task is not None for task in tasks], [False] * 14 + [True])

    def test_the_result_of_a_split_leaf_goes_to_the_leaves_under_it(self):
        X, Y = _data(200)
        gpt = GPTree(Nbar=40, retrain_every_n_points=15)
        task = None
        for i in range(40):
            if task is not None:
                gpt.apply_fit_result(_solve(task))
            task = _update(gpt, X, Y, i, defer_fits=True)
        # The root asked for a fit when it was full, and has split
        self.assertEqual(task['X'].shape[0], 40)
        left, right = gpt.root.children
        self.assertEqual((left.fit_task_id, right.fit_task_id), (task['id'], task['id']))
        old = left.my_GPR.get_hyperparameters()
        # Its children are handed no task before the result is there, also when they split
        for i in range(40, 100):
            self.assertIsNone(_update(gpt, X, Y, i, defer_fits=True))
        leaves = gpt.root.leaves
        self.assertGreater(len(leaves), 2)
        for leaf in leaves:
            np.testing.assert_array_equal(leaf.my_GPR.get_hyperparameters(), old)
        result = _solve(task)
        self.assertTrue(gpt.apply_fit_result(result))
        for leaf in leaves:
            np.testing.assert_array_equal(leaf.my_GPR.get_hyperparameters(), result['hyperparameters'][0])
            self.assertEqual(leaf.my_GPR.sklearn_gpr.X_train_.shape[0], leaf.n_points + leaf.n_shared_points)
        tasks = [task for task in (_update(gpt, X, Y, i, defer_fits=True) for i in range(100, 200)) if task is not None]
        self.assertGreater(len(tasks), 1)

    def test_fits_handed_out_are_not_saved(self):
        X, Y = _data(60)
        gpt = GPTree(Nbar=100, retrain_every_n_points=20)
        tasks = [task for task in (_update(gpt, X, Y, i, defer_fits=True) for i in range(30)) if task is not None]
        loaded = pickle.loads(pickle.dumps(gpt))
        result = _solve(tasks[0])
        self.assertFalse(loaded.apply_fit_result(result))
        self.assertFalse(loaded.root.my_GPR.is_trained())
        # The loaded tree asks again with its next full retrain buffer
        task = [_update(loaded, X, Y, i, defer_fits=True) for i in range(30, 40)][-1]
        self.assertGreater(task['id'], tasks[0]['id'])
        self.assertTrue(loaded.apply_fit_result(_solve(task)))
        self.assertTrue(gpt.apply_fit_result(result))           # the tree that was saved goes on

    def test_other_output_models_are_fitted_in_place(self):
        X, Y = _data(60, p=2)
        for output_model in ('shared', 'pca'):
            gpt = GPTree(Nbar=100, retrain_every_n_points=20, n_outputs=2, output_model=output_model,
                         output_basis_min_points=10)
            tasks = [_update(gpt, X, Y, i, defer_fits=True) for i in range(60)]
            self.assertEqual(tasks, [None] * 60)
            self.assertTrue(gpt.root.my_GPRs[0].is_trained())

    def test_late_results_on_a_stream(self):
        X, Y = _data(2000, seed=2)
        np.random.seed(0)
        in_place = GPTree(Nbar=60, retrain_every_n_points=15)
        reference = _stream(in_place, X, Y)
        np.random.seed(0)
        gpt = GPTree(Nbar=60, retrain_every_n_points=15)
        fit_in_place = gptree_module.GPNode.fit_my_GPR

        def no_fit(node, force_training=False):
            self.assertFalse(fit_in_place(node, force_training))
            return False

        with mock.patch.object(gptree_module.GPNode, 'fit_my_GPR', no_fit):
            predictions = _stream(gpt, X, Y, lag=25)
        rmse = lambda p: np.sqrt(np.mean((p[1000:] - Y[1000:, 0]) ** 2))
        self.assertLess(rmse(predictions), 1.5 * rmse(reference))
        self.assertLess(rmse(predictions), 0.01 * np.ptp(Y))
        self.assertAlmostEqual(len(gpt.root.leaves), len(in_place.root.leaves), delta=5)
        self.assertLessEqual(len(gpt._fit_tasks), len(gpt.root.leaves))
        self.assertTrue(all(leaf.my_GPR.is_trained() for leaf in gpt.root.leaves))

    def test_with_a_memory_limit(self):
        """Results reach leaves that sleep, and are dropped for leaves that were removed."""
        rng = np.random.RandomState(0)
        np.random.seed(0)
        X = 0.3 * rng.randn(2500, 2)
        X[:, 0] += np.linspace(0.0, 10.0, 2500)
        Y = _target(X)[:, None]
        gpt = GPTree(Nbar=60, retrain_every_n_points=15, max_memory_mb=0.6)
        applied, results = [], []
        for i in range(2500):
            while results and results[0][0] <= i:
                applied.append(gpt.apply_fit_result(results.pop(0)[1]))
                self.assertLessEqual(gpt.memory_mb, 0.6)
            gpt.predict(X[i:i + 1])
            task = _update(gpt, X, Y, i, defer_fits=True)
            if task is not None:
                # most results soon, some after the stream has left the leaf
                results.append((i + (900 if task['id'] % 4 == 0 else 20), _solve(task)))
                results.sort(key=lambda item: item[0])
            self.assertLessEqual(gpt.memory_mb, 0.6)
        self.assertIn(False, applied)
        self.assertGreater(sum(applied), 20)
        leaves = gpt.root.leaves
        self.assertGreater(sum(leaf.asleep for leaf in leaves), 0)
        self.assertEqual(set(map(id, gpt._leaves)), set(map(id, leaves)))
        overhead = gptree_module.NODE_OVERHEAD_BYTES + gptree_module.GPR_OVERHEAD_BYTES
        self.assertEqual(gpt._leaf_bytes, sum(leaf.nbytes_points() + leaf.nbytes_fitted() + overhead for leaf in leaves))
        self.assertEqual(gpt._fitted_bytes, sum(leaf.nbytes_fitted() for leaf in leaves if not leaf.asleep))


if __name__ == '__main__':
    unittest.main()
