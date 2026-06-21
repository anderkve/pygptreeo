"""Tests for IncrementalGP and the tree's incremental-update path."""

import io
import os
import sys
import contextlib
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo.gptree import GPTree
from pygptreeo.incremental_gp import IncrementalGP
from pygptreeo.default_gpr import Default_GPR
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

warnings.filterwarnings("ignore")


def _silent(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _fixed_kernel():
    """A kernel with fixed hyperparameters (optimizer becomes a no-op)."""
    return (ConstantKernel(1.3, "fixed")
            * Matern(length_scale=0.4, length_scale_bounds="fixed", nu=1.5))


class TestIncrementalGPUnit(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(0)
        self.X = rng.rand(50, 2)
        self.y = np.sin(3 * self.X[:, 0]) + 0.5 * np.cos(4 * self.X[:, 1])
        self.noise = np.full(50, 1e-3)  # variance per point
        self.Xt = rng.rand(40, 2)

    def test_supports_incremental(self):
        self.assertTrue(IncrementalGP().supports_incremental_update())

    def test_untrained_prior(self):
        gp = IncrementalGP(kernel=_fixed_kernel(), optimizer=None)
        self.assertFalse(gp.is_trained())
        mu, sd = gp.predict(self.Xt, return_std=True)
        np.testing.assert_allclose(mu, 0.0)          # prior mean
        self.assertTrue(np.all(sd > 0))              # prior std

    def test_rank1_matches_full_fit(self):
        """Incremental updates must reproduce a from-scratch full fit exactly."""
        gp_inc = IncrementalGP(kernel=_fixed_kernel(), optimizer=None)
        gp_inc.alpha = self.noise[:15]
        _silent(gp_inc.fit, self.X[:15], self.y[:15])
        for i in range(15, 50):
            gp_inc.add_observation(self.X[i:i + 1], self.y[i], self.noise[i])

        gp_full = IncrementalGP(kernel=_fixed_kernel(), optimizer=None)
        gp_full.alpha = self.noise
        _silent(gp_full.fit, self.X, self.y)

        mu_i, sd_i = gp_inc.predict(self.Xt, return_std=True)
        mu_f, sd_f = gp_full.predict(self.Xt, return_std=True)
        np.testing.assert_allclose(mu_i, mu_f, atol=1e-7)
        np.testing.assert_allclose(sd_i, sd_f, atol=1e-7)

    def test_add_observation_grows_training_set(self):
        gp = IncrementalGP(kernel=_fixed_kernel(), optimizer=None)
        gp.alpha = self.noise[:10]
        _silent(gp.fit, self.X[:10], self.y[:10])
        self.assertEqual(gp.X_train_.shape[0], 10)
        gp.add_observation(self.X[10:11], self.y[10], self.noise[10])
        self.assertEqual(gp.X_train_.shape[0], 11)

    def test_add_observation_before_fit_is_noop(self):
        gp = IncrementalGP(kernel=_fixed_kernel(), optimizer=None)
        gp.add_observation(self.X[0:1], self.y[0], self.noise[0])  # must not raise
        self.assertFalse(gp.is_trained())

    def test_clone_preserves_trained_state(self):
        # clone() deep-copies the fitted state (warm start), mirroring the
        # sklearn adapter, so a freshly-split child can predict immediately.
        gp = IncrementalGP(kernel=_fixed_kernel(), optimizer=None)
        gp.alpha = self.noise[:10]
        _silent(gp.fit, self.X[:10], self.y[:10])
        clone = gp.clone()
        self.assertTrue(gp.is_trained())
        self.assertTrue(clone.is_trained())
        mu_a, sd_a = gp.predict(self.Xt, return_std=True)
        mu_b, sd_b = clone.predict(self.Xt, return_std=True)
        np.testing.assert_allclose(mu_a, mu_b)
        np.testing.assert_allclose(sd_a, sd_b)
        # The clone is independent: extending it does not affect the original.
        clone.add_observation(self.X[10:11], self.y[10], self.noise[10])
        self.assertEqual(gp.X_train_.shape[0], 10)
        self.assertEqual(clone.X_train_.shape[0], 11)


class TestDefaultBackendUnaffected(unittest.TestCase):
    """The default (sklearn) backend must not support incremental updates."""

    def test_sklearn_no_incremental(self):
        self.assertFalse(Default_GPR().supports_incremental_update())


class TestTreeIncrementalIntegration(unittest.TestCase):

    def setUp(self):
        def f(X):
            return np.sin(3 * X[:, 0]) + 0.5 * np.cos(5 * X[:, 1])
        rng = np.random.RandomState(2)
        self.X = rng.rand(300, 2)
        self.y = f(self.X)
        self.sigma = np.full(len(self.X), 1e-3)
        self.Xt = rng.rand(150, 2)
        self.yt = f(self.Xt)

    def _run(self, incremental, R):
        np.random.seed(2)
        gpt = GPTree(GPR=IncrementalGP(), Nbar=150, theta=1e-4,
                     retrain_every_n_points=R, incremental_updates=incremental)
        _silent(self._stream, gpt)
        mu, _ = gpt.predict(self.Xt)
        return gpt, float(np.sqrt(np.mean((mu[:, 0] - self.yt) ** 2)))

    def _stream(self, gpt):
        for i in range(len(self.X)):
            gpt.update_tree(self.X[i:i + 1], self.y[i:i + 1].reshape(1, 1),
                            self.sigma[i:i + 1].reshape(1, 1))

    def test_incremental_keeps_gp_in_sync_with_node(self):
        gpt, _ = self._run(incremental=True, R=40)
        checked = 0
        for leaf in gpt.root.leaves:
            # Once a leaf has been fit on its own data (so rank-1 updates are
            # active), every stored point should be in the GP's training set.
            # Leaves still on an inherited parent GP are excluded.
            if leaf._gp_fitted_on_own_data and leaf.n_shared_points == 0:
                self.assertEqual(leaf.my_GPRs[0].X_train_.shape[0], leaf.n_points)
                checked += 1
        self.assertGreater(checked, 0, "expected at least one own-data-fit leaf")

    def test_incremental_improves_accuracy_over_lazy_refit(self):
        _, rmse_on = self._run(incremental=True, R=40)
        _, rmse_off = self._run(incremental=False, R=40)
        # Incorporating recent points should not be worse than ignoring them.
        self.assertLessEqual(rmse_on, rmse_off + 1e-9)

    def test_incremental_matches_full_refit_accuracy(self):
        _, rmse_inc = self._run(incremental=True, R=40)
        _, rmse_gold = self._run(incremental=False, R=1)
        # Incremental (periodic re-opt + rank-1) should be close to refitting
        # every point, within a modest tolerance.
        self.assertLess(rmse_inc, 2.0 * rmse_gold + 1e-4)


if __name__ == "__main__":
    unittest.main()
