"""Tests for the uncertainty calibration (sigma scaler)."""

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
    """Direct tests of update_sigma_scaler."""

    def _scaler(self, res, sp):
        node = GPNode(0, my_GPR=Default_GPR())
        node.n_points_pred_perf = len(res)
        node.residuals_list = [res.copy()]
        node.sigma_preds_list = [sp.copy()]
        node.sigma_scalers = [DEFAULT_SIGMA_SCALER]
        _silent(node.update_sigma_scaler)
        return node.sigma_scalers[0]

    def test_scaler_equals_target_quantile(self):
        rng = np.random.RandomState(0)
        res = rng.normal(0.0, 1.0, 25)
        sp = np.abs(rng.normal(1.0, 0.2, 25))
        scaler = self._scaler(res, sp)
        expected = np.quantile(np.abs(res) / (sp + 1e-10), TARGET_COVERAGE)
        self.assertAlmostEqual(scaler, expected, places=9)

    def test_hits_target_coverage_on_window(self):
        rng = np.random.RandomState(1)
        res = rng.normal(0.0, 1.0, 25)
        sp = np.abs(rng.normal(1.0, 0.2, 25))
        scaler = self._scaler(res, sp)
        coverage = np.mean(np.abs(res) < scaler * sp)
        self.assertLessEqual(abs(coverage - TARGET_COVERAGE), 0.08)

    def test_degenerate_inputs_handled(self):
        # Near-zero residuals give a small, finite, positive scaler.
        scaler = self._scaler(np.full(25, 1e-8), np.ones(25))
        self.assertTrue(np.isfinite(scaler) and scaler > 0)


class TestCalibrationIntegration(unittest.TestCase):
    """Short streaming check that calibrated coverage tracks the target."""

    def setUp(self):
        def f(X):
            return np.sin(3 * X[:, 0]) + 0.5 * np.cos(5 * X[:, 1])
        rng = np.random.RandomState(3)
        self.N = 1500
        self.X = rng.rand(self.N, 2)
        self.y = f(self.X)
        self.sigma = np.maximum(1e-3 * np.abs(self.y), 1e-6)

    def test_prequential_coverage_near_target(self):
        np.random.seed(3)
        gpt = GPTree(GPR=Default_GPR(), Nbar=100, theta=1e-4,
                     retrain_every_n_points=100)
        covered = []
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(self.N):
                xi = self.X[i:i + 1]
                mu, sd = gpt.predict(xi)
                covered.append(abs(mu[0, 0] - self.y[i]) <= sd[0, 0])
                gpt.update_tree(xi, np.array([[self.y[i]]]),
                                np.array([[self.sigma[i]]]))
        # Coverage over the second half (after warm-up) near the 0.68 target.
        coverage = float(np.mean(covered[self.N // 2:]))
        self.assertLessEqual(abs(coverage - TARGET_COVERAGE), 0.12)


class TestBatchFit(unittest.TestCase):
    """After ``GPTree.fit`` no prediction performance is registered: sigma is the GPs' own."""

    def setUp(self):
        rng = np.random.RandomState(4)
        self.X = rng.rand(400, 2)
        self.y = (np.sin(3 * self.X[:, 0]) + 0.5 * np.cos(5 * self.X[:, 1]))[:, None]
        self.X_test = rng.rand(200, 2)

    def _fit(self, scale=1.0, **kwargs):
        np.random.seed(4)
        gpt = GPTree(Nbar=100, **kwargs)
        y = scale * self.y
        _silent(gpt.fit, self.X, y, np.full_like(y, 1e-3 * scale))
        return gpt

    def test_sigma_is_the_uncalibrated_sigma(self):
        gpt = self._fit()
        for leaf in gpt.root.leaves:
            self.assertEqual(leaf.sigma_scalers, [1.0])
            self.assertEqual(leaf.residuals_list[0].shape[0], 0)
        _, sd = gpt.predict(self.X_test)
        _, sd_uncalibrated = self._fit(use_calibrated_sigma=False).predict(self.X_test)
        np.testing.assert_array_equal(sd, sd_uncalibrated)

    def test_rescaling_the_target_rescales_sigma(self):
        mu, sd = self._fit().predict(self.X_test)
        mu_k, sd_k = self._fit(scale=1000.0).predict(self.X_test)
        np.testing.assert_allclose(mu_k, 1000.0 * mu, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(sd_k, 1000.0 * sd, rtol=1e-3)

    def test_with_a_global_model_the_fitted_leaves_calibrate(self):
        """Leaves fitted against a global-model version while the tree is built register
        their predictions; an untrained leaf's prior predictions are left out."""
        kwargs = dict(global_mean='additive_gp',
                      global_mean_kwargs=dict(reservoir_size=80, min_points=120, n_restarts_optimizer=0, random_state=0))
        gpt = self._fit(**kwargs)
        registered = [leaf.residuals_list[0].shape[0] for leaf in gpt.root.leaves]
        self.assertGreater(min(registered), 0)
        self.assertTrue(any(leaf.sigma_scalers != [1.0] for leaf in gpt.root.leaves))
        mu, sd = gpt.predict(self.X_test)
        mu_k, sd_k = self._fit(scale=1000.0, **kwargs).predict(self.X_test)
        np.testing.assert_allclose(mu_k, 1000.0 * mu, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(sd_k, 1000.0 * sd, rtol=1e-3)

    def test_updates_after_a_fit_calibrate(self):
        gpt = self._fit()
        rng = np.random.RandomState(5)
        X_new = rng.rand(60, 2)
        y_new = np.sin(3 * X_new[:, 0]) + 0.5 * np.cos(5 * X_new[:, 1])
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(60):
                gpt.update_tree(X_new[i:i + 1], np.array([[y_new[i]]]), np.array([[1e-3]]))
        registered = sum(leaf.residuals_list[0].shape[0] for leaf in gpt.root.leaves)
        self.assertGreater(registered, 0)
        _, sd = gpt.predict(self.X_test)
        self.assertTrue(np.all(np.isfinite(sd)) and np.all(sd > 0))


if __name__ == "__main__":
    unittest.main()
