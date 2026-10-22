"""A pretrained, frozen global model (FixedGlobalMean): the leaves model its residual."""

import contextlib
import io
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from pygptreeo import GPTree, Default_GPR, FixedGlobalMean


def smooth_part(X):
    return 3.0 * np.sin(2 * np.pi * X[:, 0]) + 2.0 * X[:, 1] ** 2


def target(X):
    return smooth_part(X) + 0.1 * np.sin(12 * np.pi * X[:, 0] * X[:, 1])


class TestFixedGlobalMean(unittest.TestCase):

    def test_residual_tree(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(size=(600, 2)); y = target(X)
        Xe = rng.uniform(size=(300, 2))
        errs = {}
        for name, gm in [("plain", None), ("fixed", FixedGlobalMean(smooth_part))]:
            gpt = GPTree(GPR=Default_GPR(kernel=ConstantKernel(1.0) * RBF(length_scale=np.ones(2))), Nbar=100,
                         retrain_every_n_points=25, global_mean=gm)
            with contextlib.redirect_stdout(io.StringIO()):
                for i in range(len(X)):
                    gpt.update_tree(X[i:i + 1], y[i:i + 1].reshape(1, 1), 1e-4)
            mu, sd = gpt.predict(Xe)
            errs[name] = np.sqrt(np.mean((mu.ravel() - target(Xe)) ** 2))
            self.assertTrue(np.all(np.isfinite(sd)))
        self.assertLess(errs["fixed"], errs["plain"])
        gm = FixedGlobalMean(smooth_part, error_scale=0.05)
        np.testing.assert_allclose(gm.error_scale, [0.05])
        self.assertFalse(gm.observe(X[:1], y[:1], 1e-4))
        self.assertEqual(gm.current.predict(X[:5]).shape, (5, 1))


if __name__ == '__main__':
    unittest.main()
