"""Tests for the ObliqueMatern kernel, its fit in the scikit-learn adapter
(``Default_GPR(oblique=True)``) and the splits of leaves that hold such a kernel."""

import os
import pickle
import sys
import tempfile
import unittest
import warnings

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo import GPTree, Default_GPR, ObliqueMatern, solve_fit_task
from pygptreeo.kernels import shear_dims

warnings.filterwarnings("ignore")

DIRECTION = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)


def _along_direction(X, p=1):
    """A function of one combination of the three inputs, with ``p`` outputs."""
    return np.sin(5.0 * X @ DIRECTION[:, None] + np.linspace(0.0, 1.0, p)[None, :])


def _along_axis(X, p=1):
    return np.sin(5.0 * X[:, :1] + np.linspace(0.0, 1.0, p)[None, :])


def _error(gpr, f, p, X_test):
    return float(np.sqrt(np.mean((gpr.predict(X_test).reshape(len(X_test), -1) - f(X_test, p)) ** 2)))


class TestObliqueMatern(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(0)
        self.X, self.Y = rng.randn(7, 3), rng.randn(4, 3)
        self.length_scale = np.array([0.5, 1.3, 2.0])
        self.shear = np.array([0.3, -0.7, 1.1])

    def test_without_shear_it_is_the_matern_kernel(self):
        for nu in (0.5, 1.5, 2.5, np.inf):
            kernel, reference = ObliqueMatern(self.length_scale, nu=nu), Matern(self.length_scale, nu=nu)
            np.testing.assert_allclose(kernel(self.X), reference(self.X), atol=1e-12)
            np.testing.assert_allclose(kernel(self.X, self.Y), reference(self.X, self.Y), atol=1e-12)
            np.testing.assert_allclose(kernel(self.X, eval_gradient=True)[1][:, :, :3], reference(self.X, eval_gradient=True)[1], atol=1e-12)

    def test_gradient(self):
        for nu in (0.5, 1.5, 2.5, np.inf):
            kernel = ObliqueMatern(self.length_scale, shear=self.shear, nu=nu)
            gradient = kernel(self.X, eval_gradient=True)[1]
            self.assertEqual(gradient.shape, (7, 7, 6))
            for i in range(6):
                up, down = kernel.theta.copy(), kernel.theta.copy()
                up[i] += 1e-6
                down[i] -= 1e-6
                numeric = (kernel.clone_with_theta(up)(self.X) - kernel.clone_with_theta(down)(self.X)) / 2e-6
                np.testing.assert_allclose(gradient[:, :, i], numeric, atol=1e-6)

    def test_is_a_correlation(self):
        K = ObliqueMatern(self.length_scale, shear=self.shear)(self.X)
        np.testing.assert_allclose(np.diag(K), 1.0)
        self.assertGreater(np.linalg.eigvalsh(K).min(), 0.0)

    def test_theta_and_bounds(self):
        kernel = ConstantKernel(2.0) * ObliqueMatern(self.length_scale, shear=self.shear)
        np.testing.assert_allclose(kernel.theta, np.concatenate([[np.log(2.0)], np.log(self.length_scale), self.shear]))
        self.assertEqual(kernel.bounds.shape, (7, 2))
        np.testing.assert_array_equal(shear_dims(kernel), [4, 5, 6])
        moved = kernel.clone_with_theta(kernel.theta + 0.1)
        np.testing.assert_allclose(moved.k2.shear, self.shear + 0.1)
        np.testing.assert_allclose(moved.k2.length_scale, self.length_scale * np.exp(0.1))
        np.testing.assert_allclose(clone(kernel).theta, kernel.theta)
        np.testing.assert_allclose(pickle.loads(pickle.dumps(kernel))(self.X), kernel(self.X))

    def test_one_input_has_no_shear(self):
        kernel = ObliqueMatern(np.array([2.0]))
        self.assertEqual(kernel.theta.shape, (1,))
        self.assertEqual(shear_dims(ConstantKernel() * kernel).size, 0)

    def test_metric_and_axis_length_scales(self):
        kernel = ObliqueMatern(self.length_scale, shear=self.shear, nu=np.inf)
        difference = self.X[0] - self.X[1]
        r2 = difference @ kernel.metric() @ difference
        self.assertAlmostEqual(kernel(self.X[:1], self.X[1:2])[0, 0], np.exp(-0.5 * r2))
        np.testing.assert_allclose(kernel.axis_length_scales(), 1.0 / np.sqrt(np.diag(kernel.metric())))
        np.testing.assert_allclose(ObliqueMatern(self.length_scale).axis_length_scales(), self.length_scale)

    def test_other_smoothness_is_refused(self):
        with self.assertRaises(ValueError):
            ObliqueMatern(self.length_scale, nu=2.0)(self.X)


class TestObliqueFit(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(1)
        self.X, self.X_test = rng.rand(80, 3), rng.rand(300, 3)

    def test_keeps_the_shear_for_a_function_of_a_combination_of_inputs(self):
        for p in (1, 20):
            y = _along_direction(self.X, p)
            plain = Default_GPR(alpha=1e-8).fit(self.X, y)
            oblique = Default_GPR(oblique=True, alpha=1e-8).fit(self.X, y)
            self.assertTrue(np.any(oblique.get_kernel().k2.shear != 0.0))
            self.assertLess(2.0 * _error(oblique, _along_direction, p, self.X_test), _error(plain, _along_direction, p, self.X_test))
            # the shortest length scale is along the direction of the function
            fastest = np.linalg.eigh(oblique.get_metric(3))[1][:, -1]
            self.assertGreater(abs(fastest @ DIRECTION), 0.99)

    def test_declines_the_shear_for_a_function_of_one_input(self):
        for p in (1, 20):
            y = _along_axis(self.X, p)
            plain = Default_GPR(alpha=1e-8).fit(self.X, y)
            oblique = Default_GPR(oblique=True, alpha=1e-8).fit(self.X, y)
            np.testing.assert_array_equal(oblique.get_kernel().k2.shear, 0.0)
            self.assertIsNone(oblique.get_metric(3))
            np.testing.assert_allclose(oblique.predict(self.X_test), plain.predict(self.X_test), atol=1e-6)
            np.testing.assert_allclose(oblique.get_length_scales(3), plain.get_length_scales(3), rtol=1e-3)

    def test_axis_length_scales_of_a_sheared_kernel(self):
        oblique = Default_GPR(oblique=True, alpha=1e-8).fit(self.X, _along_direction(self.X, 20))
        scales = oblique.get_length_scales(3)
        np.testing.assert_allclose(scales, oblique.get_kernel().k2.axis_length_scales())
        self.assertLess(scales.max() / scales.min(), 1.5)          # the direction has equal parts of the three inputs

    def test_fixed_refit_and_copies(self):
        y = _along_direction(self.X, 20)
        oblique = Default_GPR(oblique=True, alpha=1e-8).fit(self.X, y)
        again = oblique.unfitted_copy()
        self.assertTrue(again.refit_fixed(self.X, y, oblique.get_hyperparameters()))
        np.testing.assert_allclose(again.predict(self.X_test), oblique.predict(self.X_test), atol=1e-8)
        np.testing.assert_allclose(pickle.loads(pickle.dumps(oblique)).predict(self.X_test), oblique.predict(self.X_test))

    def test_one_input(self):
        X = self.X[:, :1]
        gpr = Default_GPR(oblique=True, alpha=1e-8).fit(X, np.sin(5.0 * X))
        self.assertIsInstance(gpr.get_kernel().k2, Matern)


class TestObliqueTree(unittest.TestCase):

    def _tree(self, oblique, p=1, **kw):
        return GPTree(GPR=Default_GPR(oblique=oblique), Nbar=60, retrain_every_n_points=20, n_outputs=p,
                      output_model='shared' if p > 1 else 'independent', **kw)

    def _nodes(self, gpt):
        nodes, queue = [], [gpt.root]
        while queue:
            node = queue.pop()
            nodes.append(node)
            queue += list(node.children or [])
        return nodes

    def test_splits_across_the_direction(self):
        rng = np.random.RandomState(2)
        X, X_test = rng.rand(400, 3), rng.rand(300, 3)
        for p in (1, 16):
            errors = {}
            for oblique in (False, True):
                np.random.seed(0)
                gpt = self._tree(oblique, p)
                gpt.batch_update_tree(X, _along_direction(X, p), np.full((400, p), 1e-4))
                errors[oblique] = float(np.sqrt(np.mean((gpt.predict(X_test)[0].reshape(300, -1) - _along_direction(X_test, p)) ** 2)))
                directions = [node.split_direction for node in self._nodes(gpt) if not node.is_leaf]
                self.assertGreater(len(directions), 0)
                if oblique:
                    self.assertTrue(all(d is not None and abs(d @ DIRECTION) > 0.95 for d in directions))
                    self.assertIsNone(gpt.leaf_index)            # the cells are no boxes
                else:
                    self.assertTrue(all(d is None for d in directions))
                    self.assertIsNotNone(gpt.leaf_index)
            self.assertLess(2.0 * errors[True], errors[False])

    def test_stream_save_and_load(self):
        rng = np.random.RandomState(3)
        X, X_test = rng.rand(200, 3), rng.rand(50, 3)
        Y = _along_direction(X, 8)
        gpt = self._tree(True, 8)
        for i in range(200):
            gpt.update_tree(X[i:i + 1], Y[i:i + 1], np.full((1, 8), 1e-4))
        self.assertTrue(any(node.split_direction is not None for node in self._nodes(gpt)))
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "tree.joblib")
            gpt.save(path)
            loaded = joblib.load(path)
        np.testing.assert_allclose(loaded.predict(X_test)[0], gpt.predict(X_test)[0])

    def test_predictions_from_neighbours(self):
        rng = np.random.RandomState(2)
        X, X_test = rng.rand(300, 3), rng.rand(200, 3)
        np.random.seed(0)
        gpt = self._tree(True, predict_from='neighbours')
        gpt.batch_update_tree(X, _along_direction(X), np.full((300, 1), 1e-4))
        self.assertTrue(any(node.split_direction is not None for node in self._nodes(gpt)))
        mean, sigma = gpt.predict(X_test)
        self.assertLess(np.sqrt(np.mean((mean.ravel() - _along_direction(X_test).ravel()) ** 2)), 0.01)
        self.assertTrue(np.all(sigma > 0.0))

    def test_deferred_fits(self):
        rng = np.random.RandomState(4)
        X = rng.rand(150, 3)
        Y = _along_direction(X)
        gpt = self._tree(True)
        for i in range(150):
            task = gpt.update_tree(X[i:i + 1], Y[i:i + 1], 1e-4, defer_fits=True)
            if task is not None:
                gpt.apply_fit_result(pickle.loads(pickle.dumps(solve_fit_task(pickle.loads(pickle.dumps(task))))))
        X_test = rng.rand(100, 3)
        self.assertLess(np.sqrt(np.mean((gpt.predict(X_test)[0].ravel() - _along_direction(X_test).ravel()) ** 2)), 0.05)


if __name__ == '__main__':
    unittest.main()
