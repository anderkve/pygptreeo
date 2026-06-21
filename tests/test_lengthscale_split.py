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


class TestResolutionSplit(unittest.TestCase):

    def test_should_split_on_nbar(self):
        node = GPNode(0, my_GPR=_anisotropic_gpr(2), Nbar=10)
        node.init_data_set(2)
        node.n_points = 10
        self.assertTrue(node.should_split())

    def test_no_resolution_split_when_disabled(self):
        node = GPNode(0, my_GPR=_anisotropic_gpr(2), Nbar=1000,
                      split_on_resolution=False)
        node.init_data_set(2)
        node.n_points = 50
        self.assertFalse(node.should_split())

    def _make_node(self, nd, budget):
        node = GPNode(0, my_GPR=_anisotropic_gpr(nd), Nbar=10000,
                      split_on_resolution=True, resolution_budget=budget,
                      min_points_for_resolution_split=30,
                      retrain_every_n_points=1, use_standard_scaling=True)
        node.init_data_set(nd)
        return node

    def _fit_node(self, node, X, y):
        for xi, yi in zip(X, y):
            node.store_point(xi.reshape(1, -1), float(yi), 1e-3, increment_buffer=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            node.fit_my_GPR(force_training=True)

    def test_resolution_split_triggers_for_rough_region(self):
        """A rough region (many length scales) triggers an early split,
        while a smooth region with identical layout does not."""
        np.random.seed(2)
        nd = 2
        X = np.random.uniform(0, 1, (80, nd))

        # Rough: high-frequency variation -> short length scale -> high ratio
        rough = self._make_node(nd, budget=3.0)
        self._fit_node(rough, X, np.sin(2 * np.pi * 4.0 * X[:, 0]) + 0.02 * X[:, 1])
        ratios = rough._resolution_ratios()
        self.assertIsNotNone(ratios)
        self.assertTrue(rough.should_split())

        # Smooth: gentle variation -> long length scale -> low ratio, no split
        smooth = self._make_node(nd, budget=3.0)
        self._fit_node(smooth, X, 0.3 * X[:, 0] + 0.2 * X[:, 1])
        self.assertFalse(smooth.should_split())

    def test_no_resolution_split_below_min_points(self):
        node = GPNode(0, my_GPR=_anisotropic_gpr(2), Nbar=10000,
                      split_on_resolution=True, resolution_budget=0.0,
                      min_points_for_resolution_split=50)
        node.init_data_set(2)
        node.n_points = 10
        # Below min points, even a zero budget must not trigger a split.
        self.assertFalse(node.should_split())


if __name__ == '__main__':
    unittest.main()
