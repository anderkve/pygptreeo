"""Tests for ``GPTree.batch_update_tree``, and for ``fit`` on a tree that holds points."""

import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo import GPTree, solve_fit_task

warnings.filterwarnings("ignore")


def _target(X):
    return np.sin(3.0 * X[:, 0]) + X[:, 1] ** 2


def _data(n, seed=0, p=1):
    X = np.random.RandomState(seed).rand(n, 2)
    Y = np.column_stack([(j + 1) * _target(X) for j in range(p)])
    return X, Y


def _point_by_point(gpt, X, Y, sigma):
    for i in range(X.shape[0]):
        gpt.update_tree(X[i:i + 1], Y[i:i + 1], sigma[i:i + 1])


def _tree(seed=0, **kw):
    np.random.seed(seed)
    return GPTree(Nbar=40, retrain_every_n_points=15, **kw)


class TestBatchUpdateTree(unittest.TestCase):

    def assertSameTree(self, a, b):
        X_test = np.random.RandomState(1).rand(50, 2)
        self.assertEqual([leaf.name for leaf in a.root.leaves], [leaf.name for leaf in b.root.leaves])
        for u, v in zip(a.predict(X_test), b.predict(X_test)):
            np.testing.assert_array_equal(u, v)

    def test_same_as_point_by_point(self):
        """Into a tree without points and into one with points, for each shape of y and sigma."""
        X, Y = _data(300)
        S = np.random.RandomState(2).uniform(1e-4, 1e-2, (300, 1))
        reference = _tree()
        _point_by_point(reference, X, Y, S)
        self.assertGreater(len(reference.root.leaves), 4)
        for y, sigma in ((Y, S), (Y[:, 0], S[:, 0]), (Y.tolist(), S.tolist())):
            gpt = _tree()
            self.assertEqual(gpt.batch_update_tree(X[:120], y[:120], sigma[:120]), [])
            self.assertEqual(gpt.batch_update_tree(X[120:], y[120:], sigma[120:]), [])
            self.assertSameTree(gpt, reference)

    def test_one_sigma_for_all_points(self):
        X, Y = _data(150)
        reference = _tree()
        _point_by_point(reference, X, Y, np.full((150, 1), 1e-3))
        gpt = _tree()
        gpt.batch_update_tree(X, Y[:, 0], 1e-3)
        self.assertSameTree(gpt, reference)

    def test_several_outputs(self):
        X, Y = _data(150, p=2)
        S = np.random.RandomState(2).uniform(1e-4, 1e-2, (150, 2))
        for sigma, per_output in ((S, S), (S[:, 0], np.repeat(S[:, :1], 2, axis=1))):
            reference = _tree(n_outputs=2)
            _point_by_point(reference, X, Y, per_output)
            gpt = _tree(n_outputs=2)
            gpt.batch_update_tree(X, Y, sigma)
            self.assertSameTree(gpt, reference)

    def test_fits_handed_out(self):
        X, Y = _data(200)
        gpt = _tree()
        tasks = gpt.batch_update_tree(X, Y, 1e-3, defer_fits=True)
        self.assertGreater(len(tasks), 0)
        self.assertEqual([task['id'] for task in tasks], sorted(gpt._fit_tasks))
        self.assertFalse(gpt.root.leaves[0].my_GPRs[0].is_trained())
        for task in tasks:
            self.assertTrue(gpt.apply_fit_result(solve_fit_task(task)))
        self.assertTrue(all(leaf.my_GPRs[0].is_trained() for leaf in gpt.root.leaves))

    def test_the_arrays_stay_with_the_caller(self):
        X, Y = _data(150)
        S = np.full(150, 1e-3)
        X0, Y0, S0 = X.copy(), Y.copy(), S.copy()
        gpt = _tree()
        gpt.batch_update_tree(X, Y, S)
        for kept, given in ((X0, X), (Y0, Y), (S0, S)):
            np.testing.assert_array_equal(kept, given)
        X_test = np.random.RandomState(1).rand(20, 2)
        before = gpt.predict(X_test)
        X[:], Y[:], S[:] = 0.0, 0.0, 1.0
        for u, v in zip(gpt.predict(X_test), before):
            np.testing.assert_array_equal(u, v)

    def test_no_points(self):
        gpt = _tree()
        self.assertEqual(gpt.batch_update_tree(np.empty((0, 2)), np.empty(0), 1e-3), [])
        self.assertTrue(gpt.first_point)
        X, Y = _data(50)
        gpt.batch_update_tree(X, Y, 1e-3)
        self.assertEqual(gpt.batch_update_tree(np.empty((0, 2)), np.empty((0, 1)), np.empty(0)), [])
        self.assertEqual(sum(leaf.n_points for leaf in gpt.root.leaves), 50)

    def test_shapes_that_do_not_fit(self):
        X, Y = _data(30)
        gpt = _tree()
        gpt.batch_update_tree(X[:10], Y[:10], 1e-3)
        for args in ((X[10], Y[10], 1e-3),                      # a single point as a row
                     (X[10:, :1], Y[10:], 1e-3),                # another number of features
                     (X[10:], Y[10:25], 1e-3),                  # fewer targets than points
                     (X[10:], np.column_stack((Y[10:], Y[10:])), 1e-3),
                     (X[10:], Y[10:], np.full(19, 1e-3)),
                     (X[10:], Y[10:], np.full((20, 2), 1e-3))):
            with self.assertRaises(ValueError):
                gpt.batch_update_tree(*args)
        self.assertEqual(gpt.root.n_points, 10)                 # nothing was added


class TestFitOnATreeWithPoints(unittest.TestCase):

    def test_fit_is_for_a_tree_without_points(self):
        X, Y = _data(120)
        S = np.full(120, 1e-3)
        first = [lambda gpt: gpt.update_tree(X[:1], Y[:1], 1e-3),
                 lambda gpt: gpt.batch_update_tree(X[:30], Y[:30], S[:30]),
                 lambda gpt: gpt.batch_update_tree(X[:90], Y[:90], S[:90]),    # the root has split
                 lambda gpt: gpt.fit(X[:90], Y[:90], S[:90])]
        for add in first:
            gpt = _tree()
            add(gpt)
            n_points = sum(leaf.n_points for leaf in gpt.root.leaves)
            with self.assertRaisesRegex(ValueError, "batch_update_tree"):
                gpt.fit(X[90:], Y[90:], S[90:])
            self.assertEqual(sum(leaf.n_points for leaf in gpt.root.leaves), n_points)
            gpt.batch_update_tree(X[90:], Y[90:], S[90:])
            self.assertEqual(sum(leaf.n_points for leaf in gpt.root.leaves), n_points + 30)

    def test_fit_of_a_new_tree(self):
        X, Y = _data(120)
        gpt = _tree()
        gpt.fit(X, Y, np.full(120, 1e-3))
        self.assertEqual(sum(leaf.n_points for leaf in gpt.root.leaves), 120)
        self.assertTrue(all(leaf.my_GPRs[0].is_trained() for leaf in gpt.root.leaves))


if __name__ == '__main__':
    unittest.main()
