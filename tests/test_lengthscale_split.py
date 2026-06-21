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


if __name__ == '__main__':
    unittest.main()
