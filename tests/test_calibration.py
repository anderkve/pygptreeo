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
from pygptreeo.gpnode import GPNode, DEFAULT_SIGMA_SCALER, TARGET_COVERAGE, SIGMA_SCALER_MIN_FRACTION
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


class TestCalibrationWithObservationNoise(unittest.TestCase):
    """The scaler applies to the latent sigma; observation noise is added in quadrature."""

    def _scaler(self, res, sp, so):
        node = GPNode(0, my_GPR=Default_GPR())
        node.n_points_pred_perf = len(res)
        node.residuals_list = [res.copy()]
        node.sigma_preds_list = [sp.copy()]
        node.sigma_obs_list = [so.copy()]
        node.sigma_scalers = [DEFAULT_SIGMA_SCALER]
        _silent(node.update_sigma_scaler)
        return node.sigma_scalers[0]

    def test_reduces_to_plain_rule_without_noise(self):
        rng = np.random.RandomState(0)
        res = rng.normal(0.0, 1.0, 25)
        sp = np.abs(rng.normal(1.0, 0.2, 25))
        scaler = self._scaler(res, sp, np.zeros(25))
        expected = np.quantile(np.abs(res) / (sp + 1e-10), TARGET_COVERAGE)
        self.assertAlmostEqual(scaler, expected, places=9)

    def test_noise_is_subtracted_in_quadrature(self):
        rng = np.random.RandomState(2)
        res = rng.normal(0.0, 1.0, 25)
        sp = np.abs(rng.normal(0.5, 0.1, 25))
        so = np.full(25, 0.8)
        scaler = self._scaler(res, sp, so)
        expected = np.quantile(np.sqrt(np.maximum(res ** 2 - so ** 2, 0.0)) / (sp + 1e-10), TARGET_COVERAGE)
        self.assertAlmostEqual(scaler, expected, places=9)
        self.assertLess(scaler, self._scaler(res, sp, np.zeros(25)))
        # Coverage with the combined width hits the target
        coverage = np.mean(np.abs(res) < np.sqrt((scaler * sp) ** 2 + so ** 2))
        self.assertLessEqual(abs(coverage - TARGET_COVERAGE), 0.08)

    def test_residuals_within_noise_floor_at_fraction_of_scatter(self):
        # All residuals within the labelled noise: the excess quantile is zero,
        # so the scaler falls back to a fraction of the noise-inclusive one.
        res = np.full(25, 0.3)
        scaler = self._scaler(res, np.ones(25), np.full(25, 0.5))
        self.assertAlmostEqual(scaler, SIGMA_SCALER_MIN_FRACTION * 0.3, places=9)
        self.assertLess(scaler, self._scaler(res, np.ones(25), np.zeros(25)))

    def test_latent_sigma_can_drop_below_noise_on_a_stream(self):
        # Exact function, noisy observations: the calibrated latent sigma of a
        # well-sampled leaf should end up below the observation noise.
        def f(X):
            return np.sin(3 * X[:, 0]) + 0.5 * np.cos(5 * X[:, 1])
        rng = np.random.RandomState(5)
        N, noise = 600, 0.1
        X = rng.rand(N, 2)
        y = f(X) + rng.normal(0.0, noise, N)
        np.random.seed(5)
        gpt = GPTree(GPR=Default_GPR(), Nbar=100, theta=1e-4, retrain_every_n_points=25)
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(N):
                gpt.update_tree(X[i:i + 1], np.array([[y[i]]]), noise)
            Xt = rng.rand(200, 2)
            mu, sd = gpt.predict(Xt)
        err = np.abs(mu[:, 0] - f(Xt))
        self.assertLess(np.median(sd), noise)
        # ... while the combined width covers the noisy observations at about the target rate
        yt = f(Xt) + rng.normal(0.0, noise, len(Xt))
        coverage = np.mean(np.abs(mu[:, 0] - yt) < np.sqrt(sd[:, 0] ** 2 + noise ** 2))
        self.assertLessEqual(abs(coverage - TARGET_COVERAGE), 0.12)


class TestCalibrationIntegration(unittest.TestCase):
    """Short streaming check that calibrated coverage tracks the target."""

    def setUp(self):
        def f(X):
            return np.sin(3 * X[:, 0]) + 0.5 * np.cos(5 * X[:, 1])
        rng = np.random.RandomState(3)
        self.N = 1500
        self.X = rng.rand(self.N, 2)
        y_true = f(self.X)
        # Observations carry the noise they are labelled with
        self.sigma = np.maximum(1e-3 * np.abs(y_true), 1e-6)
        self.y = y_true + rng.normal(0.0, 1.0, self.N) * self.sigma

    def test_prequential_coverage_near_target(self):
        np.random.seed(3)
        gpt = GPTree(GPR=Default_GPR(), Nbar=100, theta=1e-4,
                     retrain_every_n_points=100)
        covered = []
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(self.N):
                xi = self.X[i:i + 1]
                mu, sd = gpt.predict(xi)
                # The calibrated sigma is the latent one; a noisy observation is
                # covered by the combined width.
                width = np.sqrt(sd[0, 0] ** 2 + self.sigma[i] ** 2)
                covered.append(abs(mu[0, 0] - self.y[i]) <= width)
                gpt.update_tree(xi, np.array([[self.y[i]]]),
                                np.array([[self.sigma[i]]]))
        # Coverage over the second half (after warm-up) near the 0.68 target.
        coverage = float(np.mean(covered[self.N // 2:]))
        self.assertLessEqual(abs(coverage - TARGET_COVERAGE), 0.12)


if __name__ == "__main__":
    unittest.main()
