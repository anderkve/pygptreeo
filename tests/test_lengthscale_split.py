"""Tests for the 'min_lengthscale' split-dimension criterion (#1)."""

import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from pygptreeo.gpnode import GPNode
from pygptreeo.adapters import SklearnGPAdapter


def _anisotropic_gpr(n_dims):
    kernel = ConstantKernel(1.0, (1e-3, 1e8)) * Matern(
        nu=1.5, length_scale=[1.0] * n_dims, length_scale_bounds=[(1e-3, 1e3)] * n_dims
    )
    return SklearnGPAdapter(
        GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=2)
    )


# A target whose dim 1 varies fastest, dim 0 medium, dim 2 nearly flat.
def _aniso_target(X):
    return (np.sin(2 * np.pi * 1.5 * X[:, 1])
            + 0.5 * np.sin(2 * np.pi * 0.6 * X[:, 0])
            + 0.05 * X[:, 2])


class TestMinLengthScaleSplit(unittest.TestCase):

    def test_extraction_tracks_variation(self):
        """The fastest-varying dimension gets the smallest fitted length scale."""
        np.random.seed(0)
        gpr = _anisotropic_gpr(3)
        X = np.random.uniform(0, 1, (150, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gpr.fit(X, _aniso_target(X).reshape(-1, 1))
        ls = gpr.get_length_scales(3)
        self.assertEqual(ls.shape, (3,))
        self.assertEqual(int(np.argmin(ls)), 1)

    def test_split_picks_smallest_lengthscale_dim(self):
        np.random.seed(1)
        nd = 3
        node = GPNode(0, my_GPR=_anisotropic_gpr(nd), Nbar=200,
                      split_dimension_criteria='min_lengthscale',
                      retrain_every_n_points=1, use_standard_scaling=True)
        node.init_data_set(nd)
        X = np.random.uniform(0, 1, (150, nd))
        for xi, yi in zip(X, _aniso_target(X)):
            node.store_point(xi.reshape(1, -1), float(yi), 1e-3, increment_buffer=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            node.fit_my_GPR(force_training=True)
            node.compute_split_position_and_overlap(theta=1e-4)
        self.assertEqual(node.split_index, 1)

    def test_scale_invariant_without_standardization(self):
        """Without standard scaling, length scales are normalised by per-dim std
        so dimensions with very different ranges stay comparable."""
        np.random.seed(2)
        nd = 2
        # dim 0 spans [0, 100] with 2.5 cycles; dim 1 spans [0, 1] with 1.2 cycles.
        # dim 0 varies faster *relative to its range*, so it should be split, but
        # its raw length scale is far larger than dim 1's -- a naive argmin on raw
        # length scales would wrongly pick dim 1.
        X = np.column_stack([np.random.uniform(0, 100, 200),
                             np.random.uniform(0, 1, 200)])
        y = np.sin(2 * np.pi * 2.5 * X[:, 0] / 100) + np.sin(2 * np.pi * 1.2 * X[:, 1])

        node = GPNode(0, my_GPR=_anisotropic_gpr(nd), Nbar=400,
                      split_dimension_criteria='min_lengthscale',
                      use_standard_scaling=False)
        node.init_data_set(nd)
        for xi, yi in zip(X, y):
            node.store_point(xi.reshape(1, -1), float(yi), 1e-3, increment_buffer=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            node.fit_my_GPR(force_training=True)
            # Raw length scales would (wrongly) point at dim 1...
            self.assertEqual(int(np.argmin(node.my_GPRs[0].get_length_scales(nd))), 1)
            # ...but the std-normalised criterion correctly picks dim 0.
            node.compute_split_position_and_overlap(theta=1e-4)
        self.assertEqual(node.split_index, 0)

    def test_falls_back_to_spread_when_untrained(self):
        """With no trained GP, min_lengthscale splits the widest dimension."""
        node = GPNode(0, my_GPR=_anisotropic_gpr(2), Nbar=200,
                      split_dimension_criteria='min_lengthscale',
                      use_standard_scaling=True)
        node.init_data_set(2)
        for xi in np.array([[0.0, 0.5], [10.0, 0.4], [5.0, 0.45], [3.0, 0.5]]):
            node.store_point(xi.reshape(1, -1), 1.0, 1e-3, increment_buffer=False)
        node.compute_split_position_and_overlap(theta=1e-4)
        self.assertEqual(node.split_index, 0)  # dim 0 is widest


if __name__ == '__main__':
    unittest.main()
