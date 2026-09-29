"""Tests for the default leaf kernel (lazily built ARD Matern) and the
'min_lengthscale' default / isotropic fallback."""

import io
import os
import pickle
import sys
import contextlib
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from pygptreeo.gptree import GPTree
from pygptreeo.gpnode import GPNode
from pygptreeo.default_gpr import Default_GPR, DefaultKernelFactory
from pygptreeo.adapters import SklearnGPAdapter

warnings.filterwarnings("ignore")


class TestDefaultKernel(unittest.TestCase):

    def test_default_kernel_becomes_ard_at_first_fit(self):
        gpr = Default_GPR()
        self.assertIsInstance(gpr._kernel_factory, DefaultKernelFactory)
        rng = np.random.RandomState(0)
        X = rng.rand(30, 3)
        gpr.fit(X, np.sin(4 * X[:, 1]))
        self.assertIsNone(gpr._kernel_factory)
        ls = gpr.sklearn_gpr.kernel.k2.length_scale
        self.assertEqual(np.shape(ls), (3,))
        self.assertEqual(gpr.sklearn_gpr.kernel.k2.nu, 1.5)
        # the fitted length scales are per dimension too, and the fastest dim is found
        fitted = gpr.get_length_scales(3)
        self.assertEqual(fitted.shape, (3,))
        self.assertEqual(int(np.argmin(fitted)), 1)

    def test_custom_kernel_is_left_alone(self):
        k = ConstantKernel() * Matern(nu=2.5)
        gpr = Default_GPR(kernel=k)
        self.assertIsNone(gpr._kernel_factory)
        self.assertEqual(str(gpr.sklearn_gpr.kernel), str(k))

    def test_nu_is_passed_through(self):
        gpr = Default_GPR(nu=2.5)
        gpr.fit(np.random.rand(10, 2), np.random.rand(10))
        self.assertEqual(gpr.sklearn_gpr.kernel.k2.nu, 2.5)

    def test_clone_keeps_factory_and_picklable(self):
        gpr = Default_GPR()
        c = gpr.clone()
        self.assertIsInstance(c._kernel_factory, DefaultKernelFactory)
        restored = pickle.loads(pickle.dumps(gpr))
        restored.fit(np.random.rand(10, 4), np.random.rand(10))
        self.assertEqual(np.shape(restored.sklearn_gpr.kernel.k2.length_scale), (4,))

    def test_set_kernel_drops_factory(self):
        gpr = Default_GPR()
        gpr.set_kernel(ConstantKernel() * Matern())
        self.assertIsNone(gpr._kernel_factory)

    def test_kernel_covariance_builds_kernel(self):
        gpr = Default_GPR()
        X = np.random.rand(5, 2)
        K = gpr.get_kernel_covariance(X)
        self.assertEqual(K.shape, (5, 5))
        self.assertEqual(np.shape(gpr.sklearn_gpr.kernel.k2.length_scale), (2,))


class TestIsotropicFallback(unittest.TestCase):

    def test_isotropic_kernel_gives_no_length_scales(self):
        gpr = Default_GPR(kernel=ConstantKernel() * Matern(nu=1.5))
        X = np.random.RandomState(0).rand(40, 3)
        gpr.fit(X, np.sin(3 * X[:, 1]))
        self.assertIsNone(gpr.get_length_scales(3))

    def test_min_lengthscale_falls_back_to_spread_for_isotropic_kernel(self):
        gpr = Default_GPR(kernel=ConstantKernel() * Matern(nu=1.5))
        node = GPNode(0, my_GPR=gpr, Nbar=200, split_dimension_criteria='min_lengthscale',
                      retrain_every_n_points=1)
        node.init_data_set(3)
        rng = np.random.RandomState(1)
        # dim 2 has by far the widest spread; an isotropic kernel must not pin dim 0
        X = np.column_stack([rng.rand(60), rng.rand(60), 10 * rng.rand(60)])
        for xi in X:
            node.store_point(xi.reshape(1, -1), float(np.sin(3 * xi[1])), 1e-3, increment_buffer=False)
        with contextlib.redirect_stdout(io.StringIO()):
            node.fit_my_GPR(force_training=True)
            node.compute_split_position_and_overlap(theta=1e-4)
        self.assertEqual(node.split_index, 2)


class TestTreeDefaults(unittest.TestCase):

    def test_default_split_criterion(self):
        gpt = GPTree()
        self.assertEqual(gpt.root.split_dimension_criteria, 'min_lengthscale')
        node = GPNode(0, my_GPR=Default_GPR())
        self.assertEqual(node.split_dimension_criteria, 'min_lengthscale')

    def test_default_tree_splits_on_fastest_dimension(self):
        """Out of the box (ARD default kernel + min_lengthscale), the first split
        is along the dimension the target varies fastest in, not the widest one."""
        rng = np.random.RandomState(2)
        n = 80
        X = np.column_stack([rng.uniform(0, 3, n), rng.uniform(0, 1, n)])  # dim 0 widest
        y = np.sin(2 * np.pi * 2.0 * X[:, 1]) + 0.1 * X[:, 0]             # dim 1 fastest
        gpt = GPTree(Nbar=n, use_calibrated_sigma=False)
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(n):
                gpt.update_tree(X[i:i+1], np.array([[y[i]]]), 1e-3)
        self.assertFalse(gpt.root.is_leaf)
        self.assertEqual(gpt.root.split_index, 1)


if __name__ == '__main__':
    unittest.main()
