"""Tests for the particle-physics-style test targets of ``examples/particle_physics_targets.py``."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples')))

from particle_physics_targets import PRECISION, TARGETS


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


if __name__ == '__main__':
    unittest.main()
