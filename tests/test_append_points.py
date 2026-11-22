"""Tests for the points added to a leaf's fitted GPs between two fits
(``append_points_between_fits``)."""

import os
import sys
import unittest
import warnings
from unittest import mock

import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo import GPTree, Default_GPR
from pygptreeo import gptree as gptree_module
from pygptreeo.gpnode import GPNode

warnings.filterwarnings("ignore")


def _tree(d=2, **kw):
    """A tree whose leaf GPs have fixed hyperparameters, so a fit is one factorisation."""
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=[1.0] * d, length_scale_bounds="fixed")
    kw.setdefault('Nbar', 30)
    return GPTree(GPR=Default_GPR(kernel=kernel), **kw)


def _target(X):
    return np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2


def _stream(gpt, X):
    y = _target(X)
    for i in range(X.shape[0]):
        gpt.update_tree(X[i:i + 1], y[i:i + 1, None], 1e-3)


def _grown_tree(n=600, seed=0, **kw):
    np.random.seed(seed)
    gpt = _tree(**kw)
    _stream(gpt, 4.0 * np.random.RandomState(seed).rand(n, 2))
    return gpt


class TestAppend(unittest.TestCase):

    def test_sklearn_adapter_matches_a_factorisation_with_the_point(self):
        rng = np.random.RandomState(0)
        X = rng.rand(41, 3); X_test = rng.rand(20, 3)
        noise = rng.uniform(1e-6, 1e-3, 41)
        for y in (np.sin(3 * X[:, :1]) + X[:, 1:2], np.column_stack((np.sin(3 * X[:, 0]), X[:, 1] ** 2))):
            gpr = Default_GPR()
            self.assertFalse(gpr.append(X[40:], y[40:], noise[40]))            # no fit to extend
            gpr.set_observation_noise(noise[:40])
            gpr.fit(X[:40], y[:40])
            reference = gpr.clone()
            reference.set_observation_noise(noise)
            self.assertTrue(reference.refit_fixed(X, y))
            size = gpr.nbytes()
            self.assertTrue(gpr.append(X[40:], y[40:], noise[40]))
            self.assertEqual(gpr.nbytes() - size, 8 * (41 * 41 - 40 * 40 + 3 + 2 * y.shape[1]))
            for a, b in zip(gpr.predict(X_test, return_std=True), reference.predict(X_test, return_std=True)):
                np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-9)
            before = gpr.predict(X_test, return_std=True)
            self.assertTrue(gpr.release() and gpr.restore())                   # a sleeping GP keeps the point
            for a, b in zip(gpr.predict(X_test, return_std=True), before):
                np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-9)

    def _check_node(self, n_outputs=1, **kw):
        np.random.seed(1)
        rng = np.random.RandomState(1)
        X = 4.0 * rng.rand(200, 2)
        Y = np.column_stack([_target(X) * (1.0 + 0.3 * k) + 0.1 * k * X[:, 0] for k in range(n_outputs)])
        gpt = GPTree(Nbar=60, retrain_every_n_points=20, n_outputs=n_outputs, **kw)
        for i in range(X.shape[0]):
            gpt.update_tree(X[i:i + 1], Y[i:i + 1], 1e-3)
        checked = 0
        for leaf in gpt.root.leaves:
            if not leaf._is_fitted() or leaf.n_points < 10:
                continue
            x = leaf.my_X_data[:3].mean(axis=0, keepdims=True)
            mu, _ = leaf.predict(x)
            y = mu + 0.5                                              # not what the leaf expects there
            n_points = leaf.n_points
            self.assertTrue(leaf.append_point(x, y, np.full(y.shape, 1e-3)))
            self.assertLess(np.abs(leaf.predict(x)[0] - y).max(), 0.25)         # most of the way to the new point
            self.assertEqual((leaf.n_points, leaf.my_X_data.shape[0]), (n_points, n_points))    # the node's points are the tree's to store
            self.assertTrue(leaf.fit_my_GPR(force_training=True))
            checked += 1
        self.assertGreater(checked, 2)

    def test_node_one_output(self):
        self._check_node()

    def test_node_independent_outputs(self):
        self._check_node(n_outputs=3)

    def test_node_shared_kernel(self):
        self._check_node(n_outputs=3, output_model='shared')

    def test_node_output_basis(self):
        self._check_node(n_outputs=5, output_model='pca', output_basis_min_points=30)

    def test_node_on_the_residual_of_a_global_model(self):
        # a global model of few points, so that the leaves have a residual to model
        self._check_node(global_mean='additive_gp', global_mean_kwargs=dict(reservoir_size=8, min_points=8, n_restarts_optimizer=0))

    def test_node_without_standard_scaling(self):
        self._check_node(use_standard_scaling=False)

    def test_node_without_a_fit_or_asleep(self):
        gpt = _tree()
        _stream(gpt, 4.0 * np.random.RandomState(2).rand(5, 2))
        x = np.array([[1.0, 1.0]])
        self.assertFalse(gpt.root.append_point(x, np.array([[1.0]]), np.array([[1e-3]])))
        gpt = _grown_tree(n=40, Nbar=60, retrain_every_n_points=10)
        self.assertTrue(gpt.root.sleep())
        self.assertFalse(gpt.root.append_point(x, np.array([[1.0]]), np.array([[1e-3]])))

    def _appended(self, **kw):
        """The points appended along a stream, and the error of the tree at each point right after it was given."""
        rng = np.random.RandomState(3)
        X = 4.0 * rng.rand(400, 2); y = _target(X)
        calls, errors = [], []
        real = GPNode.append_point

        def append_point(node, x, *a):
            done = real(node, x, *a)
            calls.append(done)
            return done

        np.random.seed(3)
        gpt = _tree(retrain_every_n_points=10, **kw)
        with mock.patch.object(GPNode, 'append_point', append_point):
            for i in range(X.shape[0]):
                gpt.update_tree(X[i:i + 1], y[i:i + 1, None], 1e-3)
                errors.append(abs(gpt.predict(X[i:i + 1])[0][0, 0] - y[i]))
        return calls, np.array(errors[50:])

    def test_tree_appends_every_point_by_default(self):
        calls, errors = self._appended()
        self.assertGreater(len(calls), 250)                           # all but the points that came with a fit
        self.assertTrue(all(calls[9:]))                               # the root is fitted with its tenth point
        self.assertLess(np.mean(errors), 0.002)                       # the tree knows a point as soon as it has it

    def test_tree_appends_no_point_when_switched_off(self):
        calls, errors = self._appended(append_points_between_fits=False)
        self.assertEqual(calls, [])
        self.assertGreater(np.mean(errors), 0.01)

    def test_default_setting(self):
        self.assertTrue(GPTree().append_points_between_fits)

    def test_under_a_memory_limit(self):
        np.random.seed(4)
        gpt = _tree(Nbar=60, retrain_every_n_points=10, max_memory_mb=0.5)
        X = 4.0 * np.random.RandomState(4).rand(1500, 2); y = _target(X)
        for i in range(X.shape[0]):
            gpt.predict(X[i:i + 1])
            gpt.update_tree(X[i:i + 1], y[i:i + 1, None], 1e-3)
            self.assertLessEqual(gpt.memory_mb, 0.5)
        leaves = gpt.root.leaves
        overhead = gptree_module.NODE_OVERHEAD_BYTES + gptree_module.GPR_OVERHEAD_BYTES
        self.assertEqual(gpt._leaf_bytes, sum(leaf.nbytes_points() + leaf.nbytes_fitted() + overhead for leaf in leaves))
        self.assertGreater(sum(leaf.asleep for leaf in leaves), 0)


if __name__ == '__main__':
    unittest.main()
