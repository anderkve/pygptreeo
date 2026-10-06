"""Tests for the neural-linear leaves (``GPTree(GPR=NeuralLinearGPR(FeatureNetLearner()))``)."""

import contextlib
import io
import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo.gptree import GPTree
from pygptreeo.neural_linear import TORCH_AVAILABLE, FeatureNetLearner, NeuralLinearGPR, NetGlobalMean, UniformReservoir

warnings.filterwarnings("ignore")


def _stream(gpt, X, Y, S):
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(X.shape[0]):
            gpt.update_tree(X[i:i + 1], Y[i:i + 1], S[i:i + 1])


def _target(X):
    """A smooth 3D test function on [0, 1]^3."""
    return np.sin(3 * X[:, 0]) * np.cos(2 * X[:, 1]) + 0.5 * X[:, 2] ** 2


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class TestNeuralLinearGPR(unittest.TestCase):

    def _learner(self, **kw):
        kw.setdefault('steps', 300); kw.setdefault('min_points', 60); kw.setdefault('random_state', 0)
        return FeatureNetLearner(**kw)

    def test_fit_predict_shapes_and_evidence(self):
        rng = np.random.RandomState(1)
        X = rng.rand(80, 3); y = _target(X) + 0.1 * rng.randn(80)
        gpr = NeuralLinearGPR(self._learner(), residual=False)
        self.assertFalse(gpr.is_trained())
        m, s = gpr.predict(X[:5], return_std=True)          # untrained: prior
        self.assertEqual(m.shape, (5,)); self.assertEqual(s.shape, (5,))
        gpr.set_observation_noise(np.full(80, 1e-6))
        gpr.fit(X, y)
        self.assertTrue(gpr.is_trained())
        m, s = gpr.predict(X[:5], return_std=True)
        self.assertEqual(m.shape, (5,)); self.assertEqual(s.shape, (5,))
        self.assertTrue(np.all(s > 0))
        # The evidence must attribute the 0.1 noise to the extra noise term, since
        # alpha says 1e-6: s2 is in units of the leaf's target variance.
        tau2, s2 = gpr.get_kernel()
        self.assertGreater(s2 * np.var(y), 1e-4)
        self.assertEqual(gpr.predict(X[:3]).shape, (3,))

    def test_tree_learns_and_uses_the_network(self):
        rng = np.random.RandomState(2)
        N = 600
        X = rng.rand(N, 3); Y = _target(X)[:, None]; S = np.full((N, 1), 1e-3)
        learner = self._learner()
        gpt = GPTree(GPR=NeuralLinearGPR(learner), Nbar=60, retrain_every_n_points=20, use_calibrated_sigma=True)
        self.assertFalse(gpt.root.use_standard_scaling)      # switched off for this backend
        _stream(gpt, X, Y, S)
        self.assertEqual(learner.n_seen, N)                   # the tree fed the stream to the learner
        self.assertGreaterEqual(learner.n_refits, 2)          # first fit at 60, refits at doublings
        Xt = rng.rand(300, 3); yt = _target(Xt)
        P, Sd = gpt.predict(Xt)
        self.assertEqual(P.shape, (300, 1)); self.assertEqual(Sd.shape, (300, 1))
        nrmse = np.sqrt(np.mean((P[:, 0] - yt) ** 2)) / (yt.max() - yt.min())
        self.assertLess(nrmse, 0.05)
        self.assertTrue(np.all(np.isfinite(Sd)) and np.all(Sd > 0))
        for leaf in gpt.root.leaves:
            self.assertEqual(leaf.my_GPR.version, learner.version)   # every leaf on the current features

    def test_standard_scaling_request_is_rejected(self):
        with self.assertRaises(ValueError):
            GPTree(GPR=NeuralLinearGPR(self._learner()), Nbar=50, use_standard_scaling=True)

    def test_refresh_after_network_refit(self):
        rng = np.random.RandomState(3)
        X = rng.rand(120, 3); y = _target(X)
        learner = self._learner(min_points=50)
        for i in range(70):
            learner.observe(X[i], y[i], 1e-3)
        self.assertEqual(learner.version, 1)
        gpr = NeuralLinearGPR(learner)
        gpr.set_observation_noise(1e-6); gpr.fit(X[:60], y[:60])
        self.assertEqual(gpr.version, 1)
        learner.fit()                                         # a forced refit publishes version 2
        self.assertEqual(learner.version, 2)
        n_before = gpr.n_solves
        gpr.predict(X[:3], return_std=True)
        self.assertEqual(gpr.version, 2)                      # re-solved on the new features on first use
        self.assertEqual(gpr.n_solves, n_before + 1)
        gpr.predict(X[:3], return_std=True)
        self.assertEqual(gpr.n_solves, n_before + 1)          # and not again

    def test_clone_is_independent(self):
        rng = np.random.RandomState(4)
        X = rng.rand(70, 3); y = _target(X)
        learner = self._learner(min_points=50)
        for i in range(70):
            learner.observe(X[i], y[i], 1e-3)
        gpr = NeuralLinearGPR(learner); gpr.set_observation_noise(1e-6); gpr.fit(X, y)
        c = gpr.clone()
        self.assertIs(c.learner, learner)                     # the network is shared
        np.testing.assert_allclose(c.predict(X[:4]), gpr.predict(X[:4]))
        c.set_observation_noise(1e-6); c.fit(X[:30], y[:30])
        self.assertFalse(np.allclose(c.mu, gpr.mu))           # refitting the clone leaves the original alone

    def test_length_scales_pick_the_fast_dimension(self):
        rng = np.random.RandomState(5)
        X = rng.rand(400, 3)
        y = np.sin(12 * X[:, 0]) + 0.1 * X[:, 1]              # varies fastest along dimension 0
        learner = self._learner(min_points=100, steps=800)
        for i in range(400):
            learner.observe(X[i], y[i], 1e-3)
        gpr = NeuralLinearGPR(learner); gpr.set_observation_noise(1e-6); gpr.fit(X[:150], y[:150])
        ls = gpr.get_length_scales(3)
        self.assertEqual(ls.shape, (3,))
        self.assertEqual(int(np.argmin(ls)), 0)

    def test_residual_leaf_reverts_to_the_head_and_carries_its_error(self):
        rng = np.random.RandomState(6)
        X = rng.rand(300, 3); y = _target(X)
        learner = self._learner(min_points=100)
        for i in range(300):
            learner.observe(X[i], y[i], 1e-3)
        self.assertIsNotNone(learner.error_scale)
        gpr = NeuralLinearGPR(learner, residual=True); gpr.set_observation_noise(1e-6)
        gpr.fit(X[:40], y[:40])                               # a leaf with few points
        far = np.array([[0.95, 0.95, 0.95]])
        m, s = gpr.predict(far, return_std=True)
        h = learner.head(far)[0, 0]
        self.assertLess(abs(m[0] - h), 5 * (abs(h) + 0.1))    # a bounded correction of the head
        self.assertGreater(gpr.predict_floor(far)[0], s[0])    # far from the leaf's points the floor exceeds the model sigma

    def test_bounded_reservoirs(self):
        rng = np.random.RandomState(7)
        X = rng.rand(500, 3); y = _target(X)
        for kind in ('coverage', 'uniform'):
            learner = self._learner(min_points=50, reservoir_size=100, reservoir=kind)
            for i in range(500):
                learner.observe(X[i], y[i], 1e-3)
            self.assertEqual(learner.sample.n, 100)
            self.assertEqual(learner.n_seen, 500)
            self.assertGreaterEqual(learner.n_refits, 1)
        r = UniformReservoir(10, 3, 1, np.random.RandomState(0))
        for i in range(200):
            r.add(X[i], y[i], 1e-3)
        self.assertEqual(r.n, 10); self.assertEqual(r.n_offered, 200)

    def test_fixed_refit_cost(self):
        """A refit runs a fixed number of steps, so its cost does not grow with the sample."""
        import time
        rng = np.random.RandomState(8)
        times = []
        for n in (200, 1600):
            X = rng.rand(n, 3); y = _target(X)
            learner = self._learner(min_points=n, steps=400, batch_size=64)
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
        feats_v1 = learner.features(X[:5]).copy()
        for i in range(100, 200):
            learner.observe(X[i], y[i], 1e-3)                 # a refit becomes due at 200
        self.assertTrue(learner.refit_in_progress)
        self.assertEqual(learner.version, 1)
        np.testing.assert_allclose(learner.features(X[:5]), feats_v1)   # old features until published
        for i in range(200, 230):                             # 30 x 4 = 120 steps: complete
            learner.observe(X[i], y[i], 1e-3)
        self.assertFalse(learner.refit_in_progress)
        self.assertEqual(learner.version, 2)
        self.assertFalse(np.allclose(learner.features(X[:5]), feats_v1))
        gpr = NeuralLinearGPR(learner); gpr.set_observation_noise(1e-6); gpr.fit(X[:50], y[:50])
        self.assertEqual(gpr.version, 2)

    def test_lbfgs_fits_and_amortises(self):
        """L-BFGS reaches a fit as good as Adam's, and advances one iteration per observation when amortised."""
        rng = np.random.RandomState(11)
        X = rng.rand(600, 3); y = _target(X); Xt = rng.rand(300, 3); yt = _target(Xt)
        errs = {}
        for opt in ('adam', 'lbfgs'):
            learner = self._learner(min_points=400, optimizer=opt, steps=(600 if opt == 'adam' else 60),
                                    reservoir_size=400)
            for i in range(400):
                learner.observe(X[i], y[i], 1e-3)
            errs[opt] = np.sqrt(np.mean((learner.head(Xt)[:, 0] - yt) ** 2)) / (yt.max() - yt.min())
        self.assertLess(errs['lbfgs'], 0.05)
        self.assertLess(errs['lbfgs'], 2 * errs['adam'] + 0.01)
        learner = self._learner(min_points=100, optimizer='lbfgs', steps=20, steps_per_update=2)
        for i in range(200):
            learner.observe(X[i], y[i], 1e-3)                 # first fit at 100, a refit due at 200
        self.assertTrue(learner.refit_in_progress)
        for i in range(200, 210):                             # 10 x 2 = 20 iterations: complete
            learner.observe(X[i], y[i], 1e-3)
        self.assertFalse(learner.refit_in_progress)
        self.assertEqual(learner.version, 2)

    def test_sigma_weighted_loss_discounts_the_noisy_points(self):
        """Half the points carry noise of the function's own size; weighting by sigma recovers the function."""
        rng = np.random.RandomState(12)
        n = 400
        X = rng.rand(n, 3); y = _target(X)
        noisy = np.arange(n) % 2 == 0
        sig = np.where(noisy, 1.0, 1e-3)
        y_obs = y + sig * rng.randn(n)
        Xt = rng.rand(300, 3); yt = _target(Xt)
        errs = {}
        for weighted in (False, True):
            learner = self._learner(min_points=n, steps=800, weight_by_sigma=weighted)
            for i in range(n):
                learner.observe(X[i], y_obs[i], sig[i])
            errs[weighted] = np.sqrt(np.mean((learner.head(Xt)[:, 0] - yt) ** 2)) / (yt.max() - yt.min())
        self.assertLess(errs[True], 0.5 * errs[False])
        self.assertLess(errs[True], 0.06)

    def test_log_transform_predicts_in_original_units(self):
        """A positive, heavy-tailed target: the log fit returns original units and its relative error is better."""
        rng = np.random.RandomState(13)
        n = 500
        X = rng.rand(n, 2); y = np.exp(6.0 * X[:, 0] * X[:, 1]) + 0.5 * X[:, 1]    # range 0.5 to 400
        Xt = rng.rand(300, 2); yt = np.exp(6.0 * Xt[:, 0] * Xt[:, 1]) + 0.5 * Xt[:, 1]
        rel = {}
        for tr in (None, 'log'):
            learner = self._learner(min_points=n, steps=800, target_transform=tr)
            for i in range(n):
                learner.observe(X[i], y[i], 1e-3)
            h = learner.head(Xt)[:, 0]
            rel[tr] = np.median(np.abs(h - yt) / yt)
        self.assertIsNotNone(learner.y_shift)
        self.assertTrue(np.all(np.isfinite(learner.head(X))))
        self.assertLess(rel['log'], 0.5 * rel[None])
        # a target with negative values is shifted, not rejected
        learner = self._learner(min_points=100, steps=50, target_transform='log')
        for i in range(100):
            learner.observe(X[i], -y[i], 1e-3)
        self.assertTrue(np.all(np.isfinite(learner.head(X[:10]))))

    def test_tree_with_amortised_refits(self):
        rng = np.random.RandomState(10)
        N = 500
        X = rng.rand(N, 3); Y = _target(X)[:, None]; S = np.full((N, 1), 1e-3)
        learner = self._learner(steps_per_update=8)
        gpt = GPTree(GPR=NeuralLinearGPR(learner), Nbar=60, retrain_every_n_points=20)
        _stream(gpt, X, Y, S)
        self.assertGreaterEqual(learner.n_refits, 2)
        Xt = rng.rand(200, 3); yt = _target(Xt)
        P, _ = gpt.predict(Xt)
        self.assertLess(np.sqrt(np.mean((P[:, 0] - yt) ** 2)) / (yt.max() - yt.min()), 0.05)

    def test_distance_floor_rises_to_the_function_scale_away_from_the_leaf_points(self):
        rng = np.random.RandomState(11)
        X = rng.rand(300, 3); y = _target(X)
        learner = self._learner(min_points=100)
        for i in range(300):
            learner.observe(X[i], y[i], 1e-3)
        Xl = 0.5 * rng.rand(60, 3); yl = _target(Xl)          # a leaf whose points fill [0, 0.5]^3
        gpr = NeuralLinearGPR(learner); gpr.set_observation_noise(1e-6); gpr.fit(Xl, yl)
        self.assertEqual(gpr.loo.shape, (60,))
        near = Xl[:5]; far = np.array([[0.95, 0.95, 0.95], [0.9, 0.1, 0.95]])
        f_near = gpr.predict_floor(near); f_far = gpr.predict_floor(far)
        loo_rms = np.sqrt(np.mean(gpr.loo ** 2))
        self.assertTrue(np.all(f_near <= 3 * loo_rms))         # on the data: the local leave-one-out error
        self.assertTrue(np.all(f_near < 0.5 * learner.y_sd))
        self.assertTrue(np.all(f_far > 0.9 * learner.y_sd))   # far away: the function's scale
        self.assertTrue(np.all(f_far <= learner.y_sd + 1e-9))
        off = NeuralLinearGPR(learner, distance_floor=False); off.set_observation_noise(1e-6); off.fit(Xl, yl)
        self.assertTrue(np.all(off.predict_floor(far) == 0.0))
        np.testing.assert_allclose(gpr.clone().predict_floor(far), f_far)

    def test_tree_adds_the_floor_after_calibration(self):
        rng = np.random.RandomState(12)
        N = 500
        X = 0.5 * rng.rand(N, 3); Y = _target(X)[:, None]; S = np.full((N, 1), 1e-3)   # the stream fills [0, 0.5]^3
        learner = self._learner()
        gpt = GPTree(GPR=NeuralLinearGPR(learner), Nbar=60, retrain_every_n_points=20, use_calibrated_sigma=True)
        _stream(gpt, X, Y, S)
        far = np.array([[0.95, 0.95, 0.95]])
        _, s_far = gpt.predict(far)
        self.assertGreater(s_far[0, 0], 0.9 * learner.y_sd)   # far from every leaf's data: the function's scale
        _, s_near = gpt.predict(X[:20])
        self.assertTrue(np.all(s_near[:, 0] < learner.y_sd))

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
