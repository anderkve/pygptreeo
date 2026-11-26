"""Tests for the particle-physics-style test targets of ``examples/target_functions_physics.py``."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples')))

from target_functions_physics import (PRECISION, TARGETS, _electroweakino_rates, electroweakino_spectrum,
                                      electroweakino_xsec_lo)


class TestParticlePhysicsTargets(unittest.TestCase):

    def test_values_are_finite_and_deterministic(self):
        for name, target in TARGETS.items():
            X = np.random.RandomState(0).rand(500, target.d)
            y = target(X)
            self.assertEqual(y.shape, (500,), name)
            self.assertTrue(np.all(np.isfinite(y)), name)
            self.assertGreater(y.std(), 0.0, name)
            np.testing.assert_array_equal(target(X), y, err_msg=name)
            np.testing.assert_allclose(target(X[7]), y[7:8], err_msg=name)   # one point, given as a vector

    def test_corners_and_faces_of_the_cube(self):
        for name, target in TARGETS.items():
            rng = np.random.RandomState(1)
            X = rng.rand(200, target.d)
            X[rng.rand(200, target.d) < 0.3] = 0.0; X[rng.rand(200, target.d) < 0.3] = 1.0
            self.assertTrue(np.all(np.isfinite(target(X))), name)

    def test_sample_without_noise_is_exact(self):
        for name, target in TARGETS.items():
            if target.noisy is not None:
                continue
            X = np.random.RandomState(2).rand(50, target.d)
            y, sigma, truth = target.sample(X, np.random.RandomState(3))
            np.testing.assert_array_equal(y, truth, err_msg=name)
            np.testing.assert_allclose(sigma, PRECISION[target.unit], err_msg=name)

    def test_monte_carlo_noise_is_declared(self):
        target = TARGETS['signal_yield_mc']
        X = np.random.RandomState(4).rand(4000, target.d)
        y, sigma, truth = target.sample(X, np.random.RandomState(5))
        np.testing.assert_array_equal(truth, target(X))
        self.assertTrue(np.all(sigma > 0))
        pull = (y - truth) / sigma
        self.assertLess(abs(np.median(pull)), 0.15)
        self.assertAlmostEqual(np.mean(np.abs(pull) < 1.0), 0.68, delta=0.12)
        y2, _, _ = target.sample(X, np.random.RandomState(5))
        np.testing.assert_array_equal(y, y2)                    # reproducible from the generator

    def test_electroweakino_spectrum(self):
        X = np.random.RandomState(6).rand(2000, 6)
        P = electroweakino_spectrum(X)
        self.assertEqual(P.shape, (2000, 8))
        self.assertTrue(np.all(P[:, 0] > 0) and np.all(P[:, 4] >= 0))
        self.assertTrue(0.1 < np.mean(P[:, 1] < 0) < 0.9)       # neutralino masses of both signs
        # A light wino pair above a bino, with heavy higgsinos: the couplings of a wino
        wino = electroweakino_spectrum(np.array([[0.0, 0.0, 1.0, 0.5, 0.5, 0.5]]))[0]
        np.testing.assert_allclose(wino[:2], [150.0, 150.0], rtol=0.05)
        np.testing.assert_allclose(wino[4:], [1.0, 1.0, 0.5, -0.5], atol=0.05)

    def test_electroweakino_rates(self):
        X = np.random.RandomState(7).rand(2000, 6)
        P = electroweakino_spectrum(X)
        rates, k = _electroweakino_rates(P)
        self.assertTrue(np.all(rates[:, :2] > 0) and np.all(rates.sum(axis=1) > 0))
        self.assertGreater(np.mean(rates[:, 2] < 0), 0.9)       # the two exchanges interfere destructively
        # A quadratic form in the couplings
        scaled = P.copy(); scaled[:, 4:] *= -3.0
        np.testing.assert_allclose(_electroweakino_rates(scaled)[0], 9.0 * rates, rtol=1e-10)
        # Squarks that are far heavier leave the W exchange
        heavy = P.copy(); heavy[:, 2] = 1e9
        decoupled, _ = _electroweakino_rates(heavy)
        np.testing.assert_allclose(decoupled[:, 0], rates[:, 0], rtol=1e-12)
        self.assertTrue(np.all(np.abs(decoupled[:, 1:]) < 1e-6 * decoupled[:, :1]))
        # The K-factor of the cross-section, with its tail where the tree-level rates cancel
        k_factor = 10 ** (TARGETS['electroweakino_xsec'](X) - electroweakino_xsec_lo(X))
        self.assertAlmostEqual(np.median(k_factor), 1.22, delta=0.02)
        self.assertTrue(np.all((k_factor > 1.0) & (k_factor < 4.0)))
        self.assertGreater(k_factor.max(), 1.5)


if __name__ == '__main__':
    unittest.main()
