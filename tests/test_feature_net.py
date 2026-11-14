"""Tests for the feature network global model (``GPTree(global_mean='net')``)."""

import contextlib
import io
import os
import sys
import unittest
import warnings
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo import feature_net
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

    def test_lbfgs_iterations_share_optimiser_calls(self):
        """The L-BFGS phase evaluates the loss about once per iteration and ends at
        the weights that one optimiser call per iteration gives."""
        rng = np.random.RandomState(3)
        X = rng.rand(300, 3); y = _target(X)
        loss, train = FeatureNetLearner._loss, FeatureNetLearner._train

        def fit(one_call_per_iteration):
            learner = self._learner(steps=100, min_points=10 ** 9)
            for i in range(X.shape[0]):
                learner.observe(X[i], y[i], 1e-3)
            calls = []
            patches = [mock.patch.object(FeatureNetLearner, '_loss', staticmethod(lambda *a: (calls.append(1), loss(*a))[1]))]
            if one_call_per_iteration:
                def one_by_one(self, net, ph, n_steps):
                    for _ in range(n_steps):
                        train(self, net, ph, 1)
                patches.append(mock.patch.object(FeatureNetLearner, '_train', one_by_one))
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                learner.fit()
            return learner, len(calls)

        shared, n_shared = fit(False)
        single, n_single = fit(True)
        self.assertLess(n_shared, 1.5 * shared.steps)
        self.assertGreaterEqual(n_single, 2 * single.steps)
        for a, b in zip(shared.net.parameters(), single.net.parameters()):
            self.assertTrue(feature_net.torch.equal(a, b))

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

    def test_holdout_argument(self):
        self.assertEqual(FeatureNetLearner().holdout, 0.2)
        X = np.random.RandomState(0).rand(50, 3)
        self.assertFalse(self._learner(holdout=0.0).held_out(X).any())
        for bad in (-0.1, 1.0):
            with self.assertRaises(ValueError):
                self._learner(holdout=bad)

    def test_held_out_points_depend_on_their_coordinates_only(self):
        rng = np.random.RandomState(11)
        X = rng.rand(2000, 3)
        mask = self._learner().held_out(X)
        self.assertAlmostEqual(mask.mean(), 0.2, delta=0.04)
        perm = rng.permutation(2000)
        np.testing.assert_array_equal(self._learner().held_out(X[perm]), mask[perm])

    def test_held_out_points_are_not_trained_on(self):
        rng = np.random.RandomState(12)
        X = rng.rand(300, 3); y = _target(X)
        learner = self._learner(polish_steps=200, reservoir_size=100)
        for i in range(300):
            learner.observe(X[i], y[i], 1e-3)
        n_held = int(learner.held_out(learner.sample.X).sum())
        ph, scal = learner._lbfgs_phase(learner.net)
        self.assertEqual(ph['Xt'].shape[0], learner.sample.X.shape[0] - n_held)
        self.assertEqual(ph['best'].Xh.shape[0], n_held)
        polish = learner._polish_phase(learner.net, scal, ph['best'])
        self.assertEqual(polish['Xt'].shape[0], 300 - int(learner.held_out(X).sum()))
        self.assertIs(polish['best'], ph['best'])
        everything = self._learner(holdout=0.0)
        for i in range(100):
            everything.observe(X[i], y[i], 1e-3)
        ph, _ = everything._lbfgs_phase(everything.net)
        self.assertEqual(ph['Xt'].shape[0], 100)
        self.assertIsNone(ph['best'])

    def test_refit_publishes_the_weights_with_the_lowest_held_out_loss(self):
        rng = np.random.RandomState(16)
        N = 300
        X = rng.rand(N, 3); f = _target(X); y = f + 0.5 * f.std() * rng.randn(N)
        seen = {'losses': []}
        check = feature_net._HeldOutBest.check

        def recording(best, net):
            check(best, net)
            with feature_net.torch.no_grad():
                seen['losses'].append(float((((net(best.Xh) - best.Yh) ** 2) * best.Wh).mean()))
            seen['best'] = best

        for steps_per_update in (None, 5):
            seen['losses'] = []
            with mock.patch.object(feature_net._HeldOutBest, 'check', recording):
                learner = self._learner(min_points=N, steps=100, polish_steps=600, steps_per_update=steps_per_update)
                for i in range(N):
                    learner.observe(X[i], y[i], 0.5 * f.std())
            self.assertEqual(learner.n_refits, 1)
            self.assertGreater(len(seen['losses']), 4)
            best = seen['best']
            with feature_net.torch.no_grad():
                published = float((((learner.net(best.Xh) - best.Yh) ** 2) * best.Wh).mean())
            self.assertAlmostEqual(published, min(seen['losses']), places=6)
            self.assertLess(published, seen['losses'][-1])        # the last weights are not the best ones here

    def test_holdout_lowers_the_error_on_noisy_targets(self):
        rng = np.random.RandomState(12)
        N = 600
        X = rng.rand(N, 3); f = _target(X); noise = 0.5 * f.std(); y = f + noise * rng.randn(N)
        Xt = rng.rand(500, 3); ft = _target(Xt)
        err = {}
        for holdout in (0.0, 0.2):
            learner = self._learner(min_points=150, steps=300, polish_steps=1500, holdout=holdout)
            for i in range(N):
                learner.observe(X[i], y[i], noise)
            err[holdout] = float(np.sqrt(np.mean((learner.head(Xt)[:, 0] - ft) ** 2)))
        self.assertLess(err[0.2], 0.8 * err[0.0])

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

    def test_hybrid_defaults(self):
        gm = GPTree(global_mean='net').global_mean
        self.assertTrue(gm.cheap_leaf_refresh)
        self.assertEqual(gm.learner.holdout, 0.2)
        self.assertEqual(GPTree(global_mean='net', global_mean_kwargs=dict(holdout=0.0)).global_mean.learner.holdout, 0.0)

    def test_leaves_take_up_a_new_version_with_their_hyperparameters_kept(self):
        from pygptreeo.default_gpr import Default_GPR
        rng = np.random.RandomState(17)
        N = 260
        X = rng.rand(N, 3); Y = _target(X)[:, None]; S = np.full((N, 1), 1e-3)
        counts = {}
        for cheap in (True, False):
            gm = NetGlobalMean(steps=100, polish_steps=0, min_points=60, random_state=0)
            gm.cheap_leaf_refresh = cheap
            gpt = GPTree(GPR=Default_GPR(), Nbar=1000, retrain_every_n_points=1000, global_mean=gm)
            _stream(gpt, X[:230], Y[:230], S[:230])
            leaf = gpt.root
            with contextlib.redirect_stdout(io.StringIO()):
                gpt.predict(X[:5])
            gpr = leaf.my_GPRs[0]; theta = gpr.get_kernel().theta.copy(); version = gm.current.version
            calls = []
            fit, refit_fixed = gpr.fit, gpr.refit_fixed
            gpr.fit = lambda *a: (calls.append('fit'), fit(*a))[1]
            gpr.refit_fixed = lambda *a: (calls.append('fixed'), refit_fixed(*a))[1]
            _stream(gpt, X[230:], Y[230:], S[230:])            # the count doubles at 240: a new version
            self.assertGreater(gm.current.version, version)
            with contextlib.redirect_stdout(io.StringIO()):
                P, _ = gpt.predict(X)
            self.assertEqual(leaf._fitted_global.version, gm.current.version)
            self.assertLess(np.abs(P[:, 0] - Y[:, 0]).max(), 0.05)
            counts[cheap] = calls
            if cheap:
                np.testing.assert_allclose(gpr.get_kernel().theta, theta)
        self.assertEqual(counts[True], ['fixed'])
        self.assertEqual(counts[False], ['fit'])


if __name__ == '__main__':
    unittest.main()
