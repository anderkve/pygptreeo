"""Tests for length-scale-driven split direction (#1) and resolution-based
adaptive splitting (#2)."""

import unittest
import warnings

import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.exceptions import ConvergenceWarning

from pygptreeo.gpnode import GPNode
from pygptreeo.default_gpr import Default_GPR
from pygptreeo.adapters import SklearnGPAdapter


def _anisotropic_gpr(n_dims):
    kernel = ConstantKernel(1.0, (1e-3, 1e8)) * Matern(
        nu=1.5, length_scale=[1.0] * n_dims, length_scale_bounds=[(1e-3, 1e3)] * n_dims
    )
    return SklearnGPAdapter(
        GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=2)
    )


class TestLengthScaleExtraction(unittest.TestCase):

    def test_untrained_returns_initial_lengthscales(self):
        gpr = _anisotropic_gpr(3)
        ls = gpr.get_length_scales(3)
        self.assertEqual(ls.shape, (3,))
        np.testing.assert_allclose(ls, np.ones(3))

    def test_trained_lengthscales_track_variation(self):
        """The fastest-varying dimension should get the smallest length scale."""
        np.random.seed(0)
        nd = 3
        gpr = _anisotropic_gpr(nd)
        X = np.random.uniform(0, 1, (150, nd))
        # dim 1 varies fastest, dim 0 medium, dim 2 nearly flat
        y = (np.sin(2 * np.pi * 1.5 * X[:, 1])
             + 0.5 * np.sin(2 * np.pi * 0.6 * X[:, 0])
             + 0.05 * X[:, 2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gpr.fit(X, y.reshape(-1, 1))
        ls = gpr.get_length_scales(nd)
        self.assertEqual(int(np.argmin(ls)), 1)

    def test_isotropic_kernel_broadcasts(self):
        gpr = Default_GPR()  # ConstantKernel * Matern with scalar length_scale
        ls = gpr.get_length_scales(4)
        self.assertEqual(ls.shape, (4,))
        self.assertTrue(np.all(ls == ls[0]))


class TestMinLengthScaleSplit(unittest.TestCase):

    def test_split_index_picks_smallest_lengthscale_dim(self):
        np.random.seed(1)
        nd = 3
        node = GPNode(0, my_GPR=_anisotropic_gpr(nd), Nbar=200,
                      split_dimension_criteria='min_lengthscale',
                      retrain_every_n_points=1, use_standard_scaling=True)
        node.init_data_set(nd)
        X = np.random.uniform(0, 1, (150, nd))
        y = (np.sin(2 * np.pi * 1.5 * X[:, 1])
             + 0.5 * np.sin(2 * np.pi * 0.6 * X[:, 0])
             + 0.05 * X[:, 2])
        for xi, yi in zip(X, y):
            node.store_point(xi.reshape(1, -1), float(yi), 1e-3, increment_buffer=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            node.fit_my_GPR(force_training=True)
            node.compute_split_position_and_overlap(theta=1e-4)
        self.assertEqual(node.split_index, 1)

    def test_falls_back_when_untrained(self):
        """With no trained GP, min_lengthscale falls back to max_spread."""
        nd = 2
        node = GPNode(0, my_GPR=_anisotropic_gpr(nd), Nbar=200,
                      split_dimension_criteria='min_lengthscale',
                      use_standard_scaling=True)
        node.init_data_set(nd)
        # dim 0 has much larger spread than dim 1
        X = np.array([[0.0, 0.5], [10.0, 0.4], [5.0, 0.45], [3.0, 0.5]])
        for xi in X:
            node.store_point(xi.reshape(1, -1), 1.0, 1e-3, increment_buffer=False)
        node.compute_split_position_and_overlap(theta=1e-4)
        self.assertEqual(node.split_index, 0)


class TestObliqueSplit(unittest.TestCase):

    def _fit_diagonal_node(self, nd=2, freq=1.5):
        node = GPNode(0, my_GPR=_anisotropic_gpr(nd), Nbar=300,
                      split_dimension_criteria='oblique',
                      retrain_every_n_points=1, use_standard_scaling=True)
        node.init_data_set(nd)
        X = np.random.uniform(0, 1, (150, nd))
        # Diagonal ridge: varies along (1,1,...), flat in perpendicular directions
        y = np.sin(2 * np.pi * freq * X.sum(axis=1) / np.sqrt(nd))
        for xi, yi in zip(X, y):
            node.store_point(xi.reshape(1, -1), float(yi), 1e-3, increment_buffer=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            node.fit_my_GPR(force_training=True)
        return node

    def test_direction_recovers_diagonal(self):
        """The estimated oblique direction aligns with the (1,1) ridge."""
        np.random.seed(0)
        node = self._fit_diagonal_node(nd=2)
        w = node._compute_oblique_direction()
        self.assertIsNotNone(w)
        self.assertAlmostEqual(np.linalg.norm(w), 1.0, places=6)
        # Aligned with [1,1]/sqrt(2) up to sign: |cos angle| ~ 1
        cos = abs(w @ (np.ones(2) / np.sqrt(2)))
        self.assertGreater(cos, 0.95)

    def test_oblique_sets_split_direction_and_routes(self):
        """The 'oblique' criterion sets split_direction; opposite corners route
        to opposite children."""
        np.random.seed(1)
        node = self._fit_diagonal_node(nd=2)
        node.compute_split_position_and_overlap(theta=1e-4)
        self.assertIsNotNone(node.split_direction)
        p_lo = node.prob_func(np.array([[0.0, 0.0]]))[0, 0]
        p_hi = node.prob_func(np.array([[1.0, 1.0]]))[0, 0]
        self.assertNotAlmostEqual(p_lo, p_hi)
        self.assertTrue({round(p_lo), round(p_hi)} == {0, 1})

    def test_oblique_falls_back_when_untrained(self):
        """With no trained GP, 'oblique' leaves split_direction None and uses an
        axis-aligned (max_spread) split instead of crashing."""
        nd = 2
        node = GPNode(0, my_GPR=_anisotropic_gpr(nd),
                      split_dimension_criteria='oblique', use_standard_scaling=True)
        node.init_data_set(nd)
        X = np.array([[0.0, 0.5], [10.0, 0.4], [5.0, 0.45], [3.0, 0.5]])
        for xi in X:
            node.store_point(xi.reshape(1, -1), 1.0, 1e-3, increment_buffer=False)
        node.compute_split_position_and_overlap(theta=1e-4)
        self.assertIsNone(node.split_direction)
        self.assertEqual(node.split_index, 0)  # max_spread fallback -> widest dim

    def test_oblique_split_sets_orthonormal_child_rotation(self):
        """An oblique split produces an orthonormal child-rotation basis whose
        first column matches the split direction."""
        np.random.seed(3)
        node = self._fit_diagonal_node(nd=2)
        node.compute_split_position_and_overlap(theta=1e-4)
        R = node._child_rotation
        self.assertIsNotNone(R)
        self.assertEqual(R.shape, (2, 2))
        # Orthonormal: R^T R = I
        np.testing.assert_allclose(R.T @ R, np.eye(2), atol=1e-8)
        # First column is the split direction (up to sign)
        self.assertGreater(abs(R[:, 0] @ node.split_direction), 0.999)


class TestObliqueRotationEndToEnd(unittest.TestCase):

    def test_oblique_tree_fits_diagonal_function(self):
        """End-to-end: an oblique GPTree learns a diagonal plane wave, its leaves
        fit in a rotated frame, and it clearly beats an axis-aligned tree."""
        from pygptreeo import GPTree

        np.random.seed(7)
        nd = 2
        N = 1500
        X = np.random.uniform(0, 1, (N, nd))
        y = np.sin(2 * np.pi * 1.5 * X.sum(axis=1) / np.sqrt(nd))

        def build(criterion):
            gpt = GPTree(GPR=_anisotropic_gpr(nd), Nbar=150, theta=1e-4,
                         retrain_every_n_points=50, use_standard_scaling=True,
                         use_calibrated_sigma=True, splitting_strategy='gradual',
                         max_n_pred_leaves=3, aggregation='moe',
                         split_dimension_criteria=criterion)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for xi, yi in zip(X, y):
                    gpt.update_tree(xi.reshape(1, -1), np.array([[yi]]), 1e-3)
            return gpt

        gpt_obl = build('oblique')
        gpt_axis = build('max_spread')

        # At least one leaf should carry a rotation (an oblique split happened).
        rotated = [leaf for leaf in gpt_obl.root.leaves if leaf.rotation is not None]
        self.assertTrue(len(rotated) > 0)
        for leaf in rotated:
            np.testing.assert_allclose(leaf.rotation.T @ leaf.rotation,
                                       np.eye(nd), atol=1e-8)

        # Evaluate on a fresh grid; oblique should be clearly more accurate.
        Xt = np.random.uniform(0, 1, (400, nd))
        yt = np.sin(2 * np.pi * 1.5 * Xt.sum(axis=1) / np.sqrt(nd))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mo, _ = gpt_obl.predict(Xt)
            ma, _ = gpt_axis.predict(Xt)
        rmse_obl = np.sqrt(np.mean((mo[:, 0] - yt) ** 2))
        rmse_axis = np.sqrt(np.mean((ma[:, 0] - yt) ** 2))
        self.assertLess(rmse_obl, rmse_axis)


if __name__ == '__main__':
    unittest.main()
