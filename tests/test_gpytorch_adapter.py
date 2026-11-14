"""Tests for GPyTorchAdapter, the GPyTorch backend of the leaf GPs."""

import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import torch
    import gpytorch
    from pygptreeo.adapters import GPyTorchAdapter
    GPYTORCH_AVAILABLE = True
except ImportError:
    GPYTORCH_AVAILABLE = False

from pygptreeo import GPTree

warnings.filterwarnings("ignore")


def _target(X):
    return np.sin(2 * np.pi * X[:, 0])


def _adapter(**kwargs):
    kwargs.setdefault('training_iterations', 50)
    return GPyTorchAdapter(covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()), **kwargs)


@unittest.skipUnless(GPYTORCH_AVAILABLE, "GPyTorch is not installed")
class TestGPyTorchAdapter(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        rng = np.random.RandomState(0)
        self.X = rng.rand(40, 1)
        self.y = _target(self.X).reshape(-1, 1)
        self.X_test = np.linspace(0.05, 0.95, 25).reshape(-1, 1)
        self.y_test = _target(self.X_test)

    def _rmse(self, gpr):
        return np.sqrt(np.mean((gpr.predict(self.X_test)[:, 0] - self.y_test) ** 2))

    def test_untrained_prediction_is_the_prior(self):
        gpr = _adapter()
        self.assertFalse(gpr.is_trained())
        mean, std = gpr.predict(self.X_test, return_std=True)
        np.testing.assert_array_equal(mean, np.zeros((25, 1)))
        np.testing.assert_array_equal(std, np.ones((25, 1)))

    def test_fit_and_predict(self):
        gpr = _adapter()
        self.assertIs(gpr.fit(self.X, self.y), gpr)
        self.assertTrue(gpr.is_trained())
        mean, std = gpr.predict(self.X_test, return_std=True)
        self.assertEqual(mean.shape, (25, 1))
        self.assertEqual(std.shape, (25, 1))
        self.assertTrue(np.all(std > 0))
        self.assertLess(self._rmse(gpr), 0.1)
        np.testing.assert_array_equal(gpr.predict(self.X_test), mean)

    def test_lbfgs(self):
        gpr = _adapter(optimizer='lbfgs', learning_rate=1.0, training_iterations=5).fit(self.X, self.y)
        self.assertLess(self._rmse(gpr), 0.1)

    def test_restarts(self):
        gpr = _adapter(n_restarts_optimizer=2).fit(self.X, self.y)
        self.assertLess(self._rmse(gpr), 0.1)

    def test_unknown_optimizer(self):
        with self.assertRaises(ValueError):
            _adapter(optimizer='sgd').fit(self.X, self.y)

    def test_normalize_y(self):
        """A target far from zero is learned when it is normalised."""
        gpr = _adapter(normalize_y=True).fit(self.X, 100.0 + self.y)
        mean = gpr.predict(self.X_test)
        self.assertLess(np.sqrt(np.mean((mean[:, 0] - 100.0 - self.y_test) ** 2)), 0.1)

    def test_observation_noise_per_point_or_shared(self):
        for noise in (1e-2, np.full(40, 1e-2)):
            gpr = _adapter()
            gpr.set_observation_noise(noise)
            self.assertLess(self._rmse(gpr.fit(self.X, self.y)), 0.1)

    def test_clone_is_independent(self):
        gpr = _adapter().fit(self.X, self.y)
        mean = gpr.predict(self.X_test)
        clone = gpr.clone()
        self.assertTrue(clone.is_trained())
        np.testing.assert_allclose(clone.predict(self.X_test), mean)
        clone.fit(self.X, -self.y)
        np.testing.assert_array_equal(gpr.predict(self.X_test), mean)
        self.assertFalse(_adapter().clone().is_trained())

    def test_kernel_covariance(self):
        gpr = _adapter()
        with self.assertRaises(RuntimeError):
            gpr.get_kernel_covariance(self.X_test)
        K = gpr.fit(self.X, self.y).get_kernel_covariance(self.X_test)
        self.assertEqual(K.shape, (25, 25))
        np.testing.assert_allclose(K, K.T, atol=1e-6)
        self.assertTrue(np.all(np.diag(K) > 0))

    def test_get_and_set_kernel(self):
        gpr = _adapter().fit(self.X, self.y)
        self.assertIsInstance(gpr.get_kernel(), gpytorch.kernels.Kernel)
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        gpr.set_kernel(kernel)
        self.assertIs(gpr.get_kernel(), kernel)

    def test_optional_capabilities_are_absent(self):
        gpr = _adapter().fit(self.X, self.y)
        self.assertFalse(gpr.supports_multitarget())
        self.assertIsNone(gpr.get_length_scales(1))


@unittest.skipUnless(GPYTORCH_AVAILABLE, "GPyTorch is not installed")
class TestGPTreeWithGPyTorch(unittest.TestCase):

    def test_stream(self):
        """Leaves of GPyTorch models learn a stream and split with the default settings."""
        torch.manual_seed(1)
        np.random.seed(1)
        rng = np.random.RandomState(1)
        X = rng.rand(120, 1)
        y = _target(X)
        gpt = GPTree(GPR=_adapter(training_iterations=30), Nbar=40, retrain_every_n_points=10)
        for i in range(120):
            gpt.update_tree(X[i:i + 1], y[i:i + 1, None], np.full((1, 1), 1e-2))
        self.assertGreater(len(gpt.root.leaves), 1)
        for leaf in gpt.root.leaves:
            self.assertIsInstance(leaf.my_GPRs[0], GPyTorchAdapter)
        X_test = np.linspace(0.05, 0.95, 40).reshape(-1, 1)
        mean, std = gpt.predict(X_test)
        self.assertTrue(np.all(np.isfinite(mean)) and np.all(std > 0))
        self.assertLess(np.sqrt(np.mean((mean[:, 0] - _target(X_test)) ** 2)), 0.1)

    def test_shared_output_model_needs_another_backend(self):
        with self.assertRaises(ValueError):
            GPTree(GPR=_adapter(), Nbar=40, n_outputs=2, output_model='shared')


if __name__ == '__main__':
    unittest.main()
