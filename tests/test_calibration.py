"""Tests for the closed-form ('quantile') uncertainty calibration vs legacy."""

import io
import os
import sys
import contextlib
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo.gptree import GPTree
from pygptreeo.gpnode import GPNode, DEFAULT_SIGMA_SCALER, TARGET_COVERAGE
from pygptreeo.default_gpr import Default_GPR

warnings.filterwarnings("ignore")


def _silent(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


class TestCalibrationUnit(unittest.TestCase):
    """Direct tests of update_sigma_scaler for the two methods."""

    def _scaler_for(self, method, res, sp):
        node = GPNode(0, my_GPR=Default_GPR(), calibration_method=method)
        node.n_points_pred_perf = len(res)
        node.residuals_list = [res.copy()]
        node.sigma_preds_list = [sp.copy()]
        node.sigma_scalers = [DEFAULT_SIGMA_SCALER]
        node.sigma_scaler_inits = [DEFAULT_SIGMA_SCALER]
        _silent(node.update_sigma_scaler)
        return node.sigma_scalers[0]

    def test_quantile_matches_rootfind_scaler(self):
        rng = np.random.RandomState(0)
        res = rng.normal(0.0, 1.0, 25)
        sp = np.abs(rng.normal(1.0, 0.2, 25))
        sq = self._scaler_for("quantile", res, sp)
        sr = self._scaler_for("rootfind", res, sp)
        # The closed-form quantile is the value the root finder searches for.
        self.assertAlmostEqual(sq, sr, delta=0.05)

    def test_quantile_hits_target_coverage_on_window(self):
        rng = np.random.RandomState(1)
        res = rng.normal(0.0, 1.0, 25)
        sp = np.abs(rng.normal(1.0, 0.2, 25))
        sq = self._scaler_for("quantile", res, sp)
        coverage = np.mean(np.abs(res) < sq * sp)
        # ~0.68 up to the granularity of 25 points.
        self.assertLessEqual(abs(coverage - TARGET_COVERAGE), 0.08)

    def test_quantile_no_rootfinding_failure_modes(self):
        # Degenerate inputs that can trip the bracketed root finder are handled
        # cleanly by the closed form (all residuals tiny -> small scaler).
        res = np.full(25, 1e-8)
        sp = np.ones(25)
        sq = self._scaler_for("quantile", res, sp)
        self.assertTrue(np.isfinite(sq) and sq > 0)


class TestCalibrationIntegration(unittest.TestCase):
    """Short streaming check that both methods calibrate comparably."""

    def setUp(self):
        def f(X):
            return np.sin(3 * X[:, 0]) + 0.5 * np.cos(5 * X[:, 1])
        rng = np.random.RandomState(3)
        self.f = f
        self.N = 1500
        self.X = rng.rand(self.N, 2)
        self.y = f(self.X)
        self.sigma = np.maximum(1e-3 * np.abs(self.y), 1e-6)

    def _prequential_coverage(self, method):
        np.random.seed(3)
        gpt = GPTree(GPR=Default_GPR(), Nbar=100, theta=1e-4,
                     retrain_every_n_points=100, calibration_method=method)
        covered = []
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(self.N):
                xi = self.X[i:i + 1]
                mu, sd = gpt.predict(xi)
                covered.append(abs(mu[0, 0] - self.y[i]) <= sd[0, 0])
                gpt.update_tree(xi, np.array([[self.y[i]]]),
                                np.array([[self.sigma[i]]]))
        # Coverage over the second half (after warm-up).
        return float(np.mean(covered[self.N // 2:]))

    def test_default_method_is_quantile(self):
        gpt = GPTree()
        self.assertEqual(gpt.root.calibration_method, "quantile")

    def test_both_methods_near_target_and_each_other(self):
        cov_q = self._prequential_coverage("quantile")
        cov_r = self._prequential_coverage("rootfind")
        # Both should land near the 0.68 target...
        self.assertLessEqual(abs(cov_q - TARGET_COVERAGE), 0.12)
        self.assertLessEqual(abs(cov_r - TARGET_COVERAGE), 0.12)
        # ...and close to each other.
        self.assertLessEqual(abs(cov_q - cov_r), 0.06)

    def test_invalid_method_raises(self):
        np.random.seed(3)
        gpt = GPTree(GPR=Default_GPR(), Nbar=100, calibration_method="bogus")
        with self.assertRaises(ValueError):
            with contextlib.redirect_stdout(io.StringIO()):
                # Need enough points for the calibration branch to be reached.
                for i in range(120):
                    xi = self.X[i:i + 1]
                    gpt.update_tree(xi, np.array([[self.y[i]]]),
                                    np.array([[self.sigma[i]]]))


if __name__ == "__main__":
    unittest.main()
