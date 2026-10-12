"""Tests for the sigma calibration with known observation noise."""

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


def _scaler(res, sp, so):
    node = GPNode(0, my_GPR=Default_GPR())
    node.n_points_pred_perf = len(res)
    node.residuals_list = [res.copy()]
    node.sigma_preds_list = [sp.copy()]
    node.sigma_obs_list = [so.copy()]
    node.sigma_scalers = [DEFAULT_SIGMA_SCALER]
    _silent(node.update_sigma_scaler)
    return node.sigma_scalers[0]


class TestNoiseAwareScalerUnit(unittest.TestCase):

    def test_zero_noise_is_the_plain_quantile(self):
        rng = np.random.RandomState(0)
        res = rng.normal(0.0, 1.0, 25)
        sp = np.abs(rng.normal(1.0, 0.2, 25))
        expected = np.quantile(np.abs(res) / (sp + 1e-10), TARGET_COVERAGE)
        self.assertAlmostEqual(_scaler(res, sp, np.zeros(25)), expected, places=9)

    def test_noise_dominated_window_keeps_the_scaler_near_one(self):
        # Residuals are pure observation noise, ten times the latent sigma: the old
        # rule would return a scaler of about 10; the window cannot resolve the
        # latent error, so the scaler stays near one, with the scatter 25
        # residuals allow.
        rng = np.random.RandomState(1)
        sp = np.full(25, 0.1)
        so = np.full(25, 1.0)
        scalers = np.array([_scaler(rng.normal(0.0, 1.0, 25), sp, so) for _ in range(200)])
        self.assertLess(np.mean(scalers), 1.3)
        self.assertLess(np.quantile(scalers, 0.9), 2.0)
        self.assertGreater(np.min(scalers), 0.5)

    def test_latent_error_above_the_noise_is_seen(self):
        # Residuals three times the latent sigma with a small noise: the scaler
        # should rise towards three, as without noise.
        rng = np.random.RandomState(2)
        sp = np.full(25, 1.0)
        so = np.full(25, 0.1)
        res = rng.normal(0.0, 3.0, 25)
        scaler = _scaler(res, sp, so)
        expected = np.quantile(np.abs(res) / sp, TARGET_COVERAGE)
        self.assertLess(abs(scaler - expected) / expected, 0.1)

    def test_register_stores_the_observation_sigma(self):
        node = GPNode(0, my_GPR=Default_GPR())
        node.init_data_set(1)
        X = np.linspace(0, 1, 30).reshape(-1, 1)
        for i in range(30):
            _silent(node.store_point, X[i:i + 1], np.sin(X[i, 0]), 1e-3)
        _silent(node.fit_my_GPR, force_training=True)
        _silent(node.register_pred_perf, X[:1], 0.0, 0.25)
        _silent(node.register_pred_perf, X[1:2], 0.0)
        np.testing.assert_allclose(node.sigma_obs_list[0], [0.0, 0.25])


class TestNoiseAwareCalibrationStream(unittest.TestCase):
    """Noisy observations, noise well above the latent error: the calibrated sigma
    must cover the noisy residuals at the target rate without inflating to the
    noise level."""

    def test_coverage_without_inflation(self):
        def f(X):
            return np.sin(3 * X[:, 0]) + 0.5 * np.cos(5 * X[:, 1])
        rng = np.random.RandomState(5)
        N = 1500
        X = rng.rand(N, 2)
        noise = 0.1
        y = f(X) + rng.normal(0.0, noise, N)
        np.random.seed(5)
        gpt = GPTree(GPR=Default_GPR(), Nbar=100, theta=1e-4, retrain_every_n_points=50)
        covered, sigmas = [], []
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(N):
                xi = X[i:i + 1]
                if i >= 600:
                    mu, sd = gpt.predict(xi)
                    r = y[i] - mu[0, 0]
                    covered.append(abs(r) < np.sqrt(sd[0, 0] ** 2 + noise ** 2))
                    sigmas.append(sd[0, 0])
                gpt.update_tree(xi, y[i], noise)
        coverage = np.mean(covered)
        self.assertLessEqual(abs(coverage - TARGET_COVERAGE), 0.08)
        # The calibrated sigma stays an uncertainty on the function, well below the noise
        self.assertLess(np.median(sigmas), 0.5 * noise)


if __name__ == '__main__':
    unittest.main()
