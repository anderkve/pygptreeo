"""Tests for the sigma that grows with the distance of a query from the fitted points
of its leaf (``widen_sigma_with_distance``)."""

import os
import sys
import unittest
import warnings

import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo import GPTree, Default_GPR
from pygptreeo import gptree as gptree_module

warnings.filterwarnings("ignore")

START = gptree_module.SIGMA_DISTANCE_START


def _tree(**kw):
    """A tree whose leaf GPs have fixed hyperparameters, so a fit is one factorisation."""
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=[1.0, 1.0], length_scale_bounds="fixed")
    kw.setdefault('Nbar', 30)
    return GPTree(GPR=Default_GPR(kernel=kernel), **kw)


def _target(X):
    return np.sin(3.0 * X[:, 0]) + X[:, 1] ** 2


def _moving(n=1500, seed=0):
    """A stream that moves along the first input, so that its leaves predict beyond their points."""
    rng = np.random.RandomState(seed)
    return np.column_stack((np.linspace(0.0, 6.0, n) + 0.2 * rng.randn(n), rng.rand(n)))


def _run(X, **kw):
    """The prequential predictions of a tree along the stream, and the tree."""
    np.random.seed(0)
    gpt = GPTree(Nbar=50, retrain_every_n_points=10, **kw)
    y = _target(X)
    mu, sd = np.empty(len(X)), np.empty(len(X))
    for i in range(len(X)):
        m, s = gpt.predict(X[i:i + 1])
        mu[i], sd[i] = m[0, 0], s[0, 0]
        gpt.update_tree(X[i:i + 1], y[i:i + 1, None], 1e-3)
    return mu, sd, gpt


def _owner(gpt, x):
    node = gpt.root
    while not node.is_leaf:
        node = node.children[int(node.prob_func(x)[0, 0] >= 0.5)]
    return node


class TestDistance(unittest.TestCase):

    def test_distance_from_the_points_of_the_last_fit(self):
        gpt = _tree()
        x = np.array([[1.0, 2.0], [3.0, -1.0]])
        np.testing.assert_array_equal(gpt.root.distance_from_fit(x), [0.0, 0.0])         # no fit yet
        X = np.random.RandomState(5).rand(20, 2) * [1.0, 10.0]
        for i in range(20):
            gpt.update_tree(X[i:i + 1], _target(X[i:i + 1])[:, None], 1e-3)
        gpt.root.fit_my_GPR(force_training=True)
        z = np.abs((x - X.mean(axis=0)) / X.std(axis=0))
        np.testing.assert_allclose(gpt.root.distance_from_fit(x), z.max(axis=1))

    def test_children_start_with_the_statistics_of_the_fit_they_inherit(self):
        gpt = _tree(retrain_every_n_points=1000)
        X = 4.0 * np.random.RandomState(6).rand(30, 2)
        for i in range(30):
            gpt.update_tree(X[i:i + 1], _target(X[i:i + 1])[:, None], 1e-3)
        left, right = gpt.root.children
        self.assertIsNotNone(left.fit_mean)
        np.testing.assert_array_equal(left.fit_mean, right.fit_mean)
        np.testing.assert_array_equal(left.fit_std, right.fit_std)


class TestSigma(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X = _moving()
        cls.on = _run(cls.X)
        cls.off = _run(cls.X, widen_sigma_with_distance=False)

    def test_default_setting(self):
        self.assertTrue(GPTree().widen_sigma_with_distance)

    def test_a_slope_is_learned_on_a_moving_stream(self):
        gpt = self.on[2]
        self.assertGreaterEqual(len(gpt._distance_errors), gptree_module.SIGMA_DISTANCE_MIN)
        self.assertGreater(gpt._sigma_slope, 0.0)
        self.assertEqual(self.off[2]._sigma_slope, 0.0)
        self.assertEqual(len(self.off[2]._distance_errors), 0)

    def test_the_predictions_do_not_change_and_sigma_does_not_shrink(self):
        np.testing.assert_array_equal(self.on[0], self.off[0])
        self.assertTrue(np.all(self.on[1] >= self.off[1]))
        self.assertGreater(np.mean(self.on[1] > self.off[1]), 0.02)

    def test_fewer_large_errors_without_a_warning(self):
        error = np.abs(self.on[0] - _target(self.X))[300:]
        missed_on = np.mean(error > 5.0 * self.on[1][300:])
        missed_off = np.mean(error > 5.0 * self.off[1][300:])
        self.assertGreater(missed_off, 0.01)
        self.assertLess(missed_on, 0.5 * missed_off)

    def test_sigma_of_a_query_by_its_distance(self):
        gpt = self.on[2]
        rng = np.random.RandomState(1)
        X_test = np.column_stack((rng.uniform(0.0, 9.0, 200), rng.rand(200)))        # also beyond the end of the stream
        _, sd = gpt.predict(X_test)
        far = 0
        for i in range(len(X_test)):
            x = X_test[i:i + 1]
            leaf = _owner(gpt, x)
            if leaf.marg_prob(x)[0, 0] != 1.0:
                continue
            beyond = max(leaf.distance_from_fit(x)[0] - START, 0.0)
            far += beyond > 0.0
            expected = leaf.predict(x, use_calibrated_sigma=True)[1][0, 0] * (1.0 + gpt._sigma_slope * beyond)
            np.testing.assert_allclose(sd[i, 0], expected, rtol=1e-8)
        self.assertGreater(far, 10)

    def test_predict_modes_agree(self):
        gpt = self.on[2]
        rng = np.random.RandomState(2)
        X_test = np.column_stack((rng.uniform(0.0, 9.0, 100), rng.rand(100)))
        mu, sd = gpt.predict(X_test)
        mu_loop, sd_loop = gpt.predict(X_test, mode='loop')
        np.testing.assert_allclose(mu, mu_loop, atol=1e-8)
        np.testing.assert_allclose(sd, sd_loop, rtol=1e-5)

    def test_little_changes_on_a_stream_that_stays(self):
        X = np.random.RandomState(3).rand(1500, 2) * [6.0, 1.0]
        on, off = _run(X), _run(X, widen_sigma_with_distance=False)
        np.testing.assert_array_equal(on[0], off[0])
        self.assertLess(np.mean(on[1][300:] > off[1][300:]), 0.05)

    def test_without_calibrated_sigma(self):
        mu, sd, gpt = _run(self.X[:600], use_calibrated_sigma=False)
        self.assertEqual(gpt._sigma_slope, 0.0)
        self.assertEqual(len(gpt._distance_errors), 0)

    def test_several_outputs(self):
        np.random.seed(0)
        gpt = GPTree(Nbar=50, retrain_every_n_points=10, n_outputs=2)
        X = self.X[:800]
        Y = np.column_stack((_target(X), np.cos(2.0 * X[:, 0])))
        for i in range(len(X)):
            gpt.update_tree(X[i:i + 1], Y[i:i + 1], 1e-3)
        self.assertGreater(gpt._sigma_slope, 0.0)
        mu, sd = gpt.predict(np.array([[7.5, 0.5]]))
        self.assertEqual(sd.shape, (1, 2))
        leaf = _owner(gpt, np.array([[7.5, 0.5]]))
        self.assertTrue(np.all(sd > leaf.predict(np.array([[7.5, 0.5]]), use_calibrated_sigma=True)[1]))


if __name__ == '__main__':
    unittest.main()
