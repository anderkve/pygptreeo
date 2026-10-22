"""The leaf fit retries with a noise jitter when the kernel matrix is numerically singular."""

import contextlib
import io
import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from pygptreeo import GPTree, Default_GPR


class TestFitJitter(unittest.TestCase):

    def test_duplicate_points_without_noise(self):
        """Identical inputs with zero noise give a singular kernel matrix; the fit must still succeed."""
        gpt = GPTree(GPR=Default_GPR(kernel=ConstantKernel(1.0) * RBF(length_scale=np.ones(2))), Nbar=50,
                     retrain_every_n_points=1, use_standard_scaling=False)
        X = np.array([[0.3, 0.6]] * 4 + [[0.5, 0.1]] * 4 + [[0.9, 0.9]] * 4)
        y = X[:, 0] + X[:, 1]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with contextlib.redirect_stdout(io.StringIO()):
                for i in range(len(X)):
                    gpt.update_tree(X[i:i + 1], y[i:i + 1].reshape(1, 1), 0.0)
        self.assertTrue(gpt.root.my_GPR.is_trained())
        self.assertTrue(any("jitter" in str(x.message) for x in w))
        mu, sd = gpt.predict(np.array([[0.3, 0.6]]))
        self.assertAlmostEqual(float(mu[0, 0]), 0.9, places=2)


if __name__ == '__main__':
    unittest.main()
