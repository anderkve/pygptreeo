"""Tests for the gradients of AnisotropicRationalQuadratic and AdditiveKernel."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pygptreeo.kernels import AdditiveKernel, AnisotropicRationalQuadratic


def _numeric_gradient(kernel, X, eps=1e-6):
    """The gradient of the kernel matrix with respect to ``kernel.theta``, by central differences."""
    theta = kernel.theta
    G = np.zeros((X.shape[0], X.shape[0], theta.size))
    for j in range(theta.size):
        up, down = theta.copy(), theta.copy()
        up[j] += eps
        down[j] -= eps
        G[:, :, j] = (kernel.clone_with_theta(up)(X) - kernel.clone_with_theta(down)(X)) / (2 * eps)
    return G


class TestKernelGradients(unittest.TestCase):

    def setUp(self):
        self.X = np.random.RandomState(0).uniform(0, 1, (7, 3))

    def _check(self, kernel, n_free):
        """The analytic gradient has an entry per free hyperparameter and equals the numeric one."""
        K, G = kernel(self.X, eval_gradient=True)
        np.testing.assert_allclose(K, kernel(self.X), atol=1e-12)
        self.assertEqual(kernel.theta.size, n_free)
        self.assertEqual(G.shape, (7, 7, n_free))
        np.testing.assert_allclose(G, _numeric_gradient(kernel, self.X), atol=1e-6)

    def test_rational_quadratic(self):
        for length_scale, n_scales in ((1.0, 1), (2.0, 1), ([0.7, 1.3, 2.0], 3)):
            for alpha in (1.0, 0.4, 2.5):
                with self.subTest(length_scale=length_scale, alpha=alpha):
                    self._check(AnisotropicRationalQuadratic(length_scale=length_scale, alpha=alpha), 1 + n_scales)

    def test_rational_quadratic_with_fixed_hyperparameters(self):
        scales = [0.7, 1.3, 2.0]
        self._check(AnisotropicRationalQuadratic(length_scale=scales, alpha=1.5, alpha_bounds="fixed"), 3)
        self._check(AnisotropicRationalQuadratic(length_scale=scales, alpha=1.5, length_scale_bounds="fixed"), 1)
        self._check(AnisotropicRationalQuadratic(length_scale=2.0, alpha=1.5, alpha_bounds="fixed",
                                                 length_scale_bounds="fixed"), 0)

    def test_additive(self):
        for base_kernel in ('rbf', 'matern'):
            for depth in (1, 2, 3):
                for length_scale in (1.0, [0.7, 1.3, 2.0]):
                    with self.subTest(base_kernel=base_kernel, depth=depth, length_scale=length_scale):
                        self._check(AdditiveKernel(3, interaction_depth=depth, base_kernel=base_kernel,
                                                   length_scale=length_scale), 3)

    def test_additive_with_fixed_length_scales(self):
        self._check(AdditiveKernel(3, interaction_depth=2, length_scale=[0.7, 1.3, 2.0], length_scale_bounds="fixed"), 0)


if __name__ == "__main__":
    unittest.main()
