"""Tests for the feature network global model (``GPTree(global_mean='net')``)."""

import contextlib
import io
import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo.gptree import GPTree
from pygptreeo.feature_net import TORCH_AVAILABLE, FeatureNetLearner, NetGlobalMean

warnings.filterwarnings("ignore")


def _stream(gpt, X, Y, S):
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(X.shape[0]):
            gpt.update_tree(X[i:i + 1], Y[i:i + 1], S[i:i + 1])


def _target(X):
    """A smooth 3D test function on [0, 1]^3."""
    return np.sin(3 * X[:, 0]) * np.cos(2 * X[:, 1]) + 0.5 * X[:, 2] ** 2


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class TestFeatureNet(unittest.TestCase):

    def _learner(self, **kw):
        # small budgets, and no polish phase unless a test asks for one, so the step counting stays explicit
        kw.setdefault('steps', 300); kw.setdefault('min_points', 60); kw.setdefault('random_state', 0)
        kw.setdefault('polish_steps', 0)
        return FeatureNetLearner(**kw)

    def test_bounded_reservoir(self):
        rng = np.random.RandomState(7)
        X = rng.rand(500, 3); y = _target(X)
        learner = self._learner(min_points=50, reservoir_size=100)
        for i in range(500):
            learner.observe(X[i], y[i], 1e-3)
        self.assertEqual(learner.sample.n, 100)
        self.assertEqual(learner.n_seen, 500)
        self.assertGreaterEqual(learner.n_refits, 1)

    def test_fixed_refit_cost(self):
        """A refit runs a fixed number of steps on a bounded sample, so its cost does not grow with the stream."""
        import time
        rng = np.random.RandomState(8)
        times = []
        for n in (200, 1600):
            X = rng.rand(n, 3); y = _target(X)
            learner = self._learner(min_points=n, steps=100, reservoir_size=200)
            for i in range(n):
                learner.observe(X[i], y[i], 1e-3)
            times.append(learner.fit_seconds)
        self.assertLess(times[1], 3 * times[0] + 0.5)

    def test_amortised_refit_publishes_once_complete(self):
        rng = np.random.RandomState(9)
        X = rng.rand(600, 3); y = _target(X)
        learner = self._learner(min_points=100, steps=120, steps_per_update=4)
        for i in range(100):
            learner.observe(X[i], y[i], 1e-3)
        self.assertEqual(learner.version, 1)                  # the first fit runs at once
        self.assertFalse(learner.refit_in_progress)
        head_v1 = learner.head(X[:5]).copy()
        for i in range(100, 200):
            learner.observe(X[i], y[i], 1e-3)                 # a refit becomes due at 200
        self.assertTrue(learner.refit_in_progress)
        self.assertEqual(learner.version, 1)
        np.testing.assert_allclose(learner.head(X[:5]), head_v1)   # the old network until published
        for i in range(200, 230):                             # 30 x 4 = 120 steps: complete
            learner.observe(X[i], y[i], 1e-3)
        self.assertFalse(learner.refit_in_progress)
        self.assertEqual(learner.version, 2)
        self.assertFalse(np.allclose(learner.head(X[:5]), head_v1))

    def test_lbfgs_fits_and_amortises(self):
        """The L-BFGS phase alone reaches a fit as good as the Adam phase alone, and
        advances one iteration per observation when amortised."""
        rng = np.random.RandomState(11)
        X = rng.rand(600, 3); y = _target(X); Xt = rng.rand(300, 3); yt = _target(Xt)
        errs = {}
        for name, kw in (('adam', dict(steps=0, polish_steps=600)), ('lbfgs', dict(steps=60, polish_steps=0))):
            learner = self._learner(min_points=400, reservoir_size=400, **kw)
            for i in range(400):
                learner.observe(X[i], y[i], 1e-3)
            errs[name] = np.sqrt(np.mean((learner.head(Xt)[:, 0] - yt) ** 2)) / (yt.max() - yt.min())
        self.assertLess(errs['lbfgs'], 0.05)
        self.assertLess(errs['lbfgs'], 2 * errs['adam'] + 0.01)
        learner = self._learner(min_points=100, steps=20, steps_per_update=2)
        for i in range(200):
            learner.observe(X[i], y[i], 1e-3)                 # first fit at 100, a refit due at 200
        self.assertTrue(learner.refit_in_progress)
        for i in range(200, 210):                             # 10 x 2 = 20 iterations: complete
            learner.observe(X[i], y[i], 1e-3)
        self.assertFalse(learner.refit_in_progress)
        self.assertEqual(learner.version, 2)

    def test_sigma_weighted_loss_discounts_the_noisy_points(self):
        """Half the points carry noise of the function's own size; the sigma weighting recovers the function."""
        rng = np.random.RandomState(12)
        n = 400
        X = rng.rand(n, 3); y = _target(X)
        noisy = np.arange(n) % 2 == 0
        sig = np.where(noisy, 1.0, 1e-3)
        y_obs = y + sig * rng.randn(n)
        Xt = rng.rand(300, 3); yt = _target(Xt)
        learner = self._learner(min_points=n, steps=0, polish_steps=800)
        for i in range(n):
            learner.observe(X[i], y_obs[i], sig[i])
        err = np.sqrt(np.mean((learner.head(Xt)[:, 0] - yt) ** 2)) / (yt.max() - yt.min())
        self.assertLess(err, 0.06)                            # the noisy half barely moves the fit

    def test_polish_phase_runs_over_every_point_and_amortises(self):
        rng = np.random.RandomState(15)
        X = rng.rand(1200, 3); y = _target(X); Xt = rng.rand(300, 3); yt = _target(Xt)
        learner = self._learner(min_points=1200, steps=60, reservoir_size=200, polish_steps=400)
        for i in range(1200):
            learner.observe(X[i], y[i], 1e-3)
        self.assertEqual(learner.store.n, 1200)                   # every point is kept for the polish
        self.assertEqual(learner.sample.n, 200)
        self.assertLess(np.sqrt(np.mean((learner.head(Xt)[:, 0] - yt) ** 2)) / (yt.max() - yt.min()), 0.05)
        # amortised: 20 L-BFGS iterations one per update, then 80 Adam steps at 8 per update
        learner = self._learner(min_points=100, steps=20, reservoir_size=100, polish_steps=80, steps_per_update=1)
        for i in range(200):
            learner.observe(X[i], y[i], 1e-3)                     # a refit due at 200
        self.assertTrue(learner.refit_in_progress)
        # the polish starts in the update that ends the L-BFGS phase: 20 + 9 updates leave 8 steps to go
        for i in range(200, 228):
            learner.observe(X[i], y[i], 1e-3)
        self.assertTrue(learner.refit_in_progress)
        learner.observe(X[228], y[228], 1e-3)                     # the 10th polish update completes it
        self.assertFalse(learner.refit_in_progress)
        self.assertEqual(learner.version, 2)

    def test_hybrid_with_amortised_refits(self):
        from pygptreeo.default_gpr import Default_GPR
        rng = np.random.RandomState(10)
        N = 500
        X = rng.rand(N, 3); Y = _target(X)[:, None]; S = np.full((N, 1), 1e-3)
        gm = NetGlobalMean(steps=100, polish_steps=0, min_points=60, steps_per_update=1, random_state=0)
        gpt = GPTree(GPR=Default_GPR(), Nbar=60, retrain_every_n_points=20, global_mean=gm)
        _stream(gpt, X, Y, S)
        self.assertGreaterEqual(gm.learner.n_refits, 2)
        Xt = rng.rand(200, 3); yt = _target(Xt)
        P, _ = gpt.predict(Xt)
        self.assertLess(np.sqrt(np.mean((P[:, 0] - yt) ** 2)) / (yt.max() - yt.min()), 0.05)

    def test_hybrid_gp_leaves_on_the_network_residual(self):
        from pygptreeo.default_gpr import Default_GPR
        rng = np.random.RandomState(13)
        N = 500
        X = rng.rand(N, 3); Y = _target(X)[:, None]; S = np.full((N, 1), 1e-3)
        gpt = GPTree(GPR=Default_GPR(), Nbar=60, retrain_every_n_points=20, use_calibrated_sigma=True,
                     global_mean='net', global_mean_kwargs=dict(steps=300, min_points=60, random_state=0))
        self.assertIsInstance(gpt.global_mean, NetGlobalMean)
        self.assertTrue(gpt.root.use_standard_scaling)        # the leaves are ordinary GPs
        _stream(gpt, X, Y, S)
        gm = gpt.global_mean
        self.assertGreaterEqual(gm.learner.n_refits, 2)
        self.assertEqual(gm.current.version, gm.learner.version)
        self.assertIsNotNone(gm.error_scale)
        Xt = rng.rand(200, 3); yt = _target(Xt)
        P, Sd = gpt.predict(Xt)
        self.assertLess(np.sqrt(np.mean((P[:, 0] - yt) ** 2)) / (yt.max() - yt.min()), 0.05)
        self.assertTrue(np.all(Sd > 0))
        for leaf in gpt.root.leaves:                          # every leaf fitted against the current snapshot
            self.assertEqual(leaf._fitted_global.version, gm.current.version)


if __name__ == '__main__':
    unittest.main()
