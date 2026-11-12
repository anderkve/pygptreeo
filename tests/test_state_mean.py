"""Tests for the interaction-order state as global model (``GPTree(global_mean='state')``)."""

import contextlib
import io
import os
import pickle
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo.gptree import GPTree
from pygptreeo.default_gpr import Default_GPR
from pygptreeo.feature_net import TORCH_AVAILABLE
from pygptreeo.global_mean import GlobalMeanLearner, make_global_mean
from pygptreeo.state_mean import SineBasis, StateGlobalMean, StateSnapshot, fit_interaction_prior

warnings.filterwarnings("ignore")


def _stream(gpt, X, y, sigma=1e-3):
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(X.shape[0]):
            gpt.update_tree(X[i:i + 1], np.array([[y[i]]]), np.array([[sigma]]))


def _predict(gpt, X):
    with contextlib.redirect_stdout(io.StringIO()):
        mu, sd = gpt.predict(X, mode='loop')
    return mu[:, 0], sd[:, 0]


def _additive(X):
    return np.sin(5 * X[:, 0]) + X[:, 1] ** 2 + np.exp(X[:, 2]) + np.cos(3 * X[:, 3])


def _product(X):
    return np.sin(3 * X[:, 0]) * np.cos(3 * X[:, 1]) * (1 + X[:, 2]) * np.exp(-X[:, 3])


def _small(**kw):
    kw.setdefault('reservoir_size', 200); kw.setdefault('n_centres', 100); kw.setdefault('fit_evals', 30)
    kw.setdefault('random_state', 0)
    return StateGlobalMean(**kw)


def _tree(gm, **kw):
    np.random.seed(0)
    return GPTree(GPR=Default_GPR(n_restarts_optimizer=0), Nbar=60, retrain_every_n_points=20, global_mean=gm, **kw)


class FakeSnapshot:
    def __init__(self, version, fn):
        self.version, self.fn, self.n_outputs = version, fn, 1

    def predict(self, X):
        return np.asarray(self.fn(np.atleast_2d(X)), dtype=float).reshape(-1, 1)


class FakeLearner(GlobalMeanLearner):
    def __init__(self, cheap):
        self.current = None; self.error_scale = None; self.cheap_leaf_refresh = cheap

    def observe(self, x, y, sigma):
        return False


class TestPriorAndBasis(unittest.TestCase):

    def test_sine_basis_represents_the_additive_kernel(self):
        rng = np.random.RandomState(0)
        ls = np.array([0.3, 0.8]); s1 = 0.7
        basis = SineBasis(np.zeros(2), np.ones(2), ls)
        X = rng.rand(40, 2)
        F = basis.features(X)[:, 1:]                                   # without the constant
        K = (F * basis.prior_var(s1)[1:]) @ F.T
        exact = sum(np.exp(-0.5 * ((X[:, j, None] - X[None, :, j]) / ls[j]) ** 2) for j in range(2)) * s1 ** 2 / 2
        self.assertLess(np.abs(K - exact).max(), 0.02 * s1 ** 2 / 2)   # largest at the edge of the range

    def test_prior_finds_the_interaction_order(self):
        rng = np.random.RandomState(1)
        X = rng.rand(250, 4)
        for target, largest in ((_additive, 'additive'), (_product, 'full')):
            gm = _small(max_order=1, reservoir_size=250)
            _stream(_tree(gm), X, target(X))
            shares = gm.order_shares()
            self.assertEqual(max(shares, key=shares.get), largest)
            self.assertGreater(shares[largest], 0.9)

    def test_prior_gradient_reaches_a_stationary_point(self):
        rng = np.random.RandomState(2)
        X = rng.rand(120, 3); t = np.sin(4 * X[:, 0]) + X[:, 1] * X[:, 2]
        t = (t - t.mean()) / t.std()
        a = fit_interaction_prior(X, t, np.ones(3), max_order=2, evals=200)
        b = fit_interaction_prior(X, t, np.ones(3), max_order=2, starts=(), warm=a['p'], evals=200)
        self.assertLessEqual(b['nll'], a['nll'] + 1e-9)
        self.assertAlmostEqual(a['nll'], b['nll'], places=3)

    def test_arguments(self):
        with self.assertRaises(ValueError):
            StateGlobalMean(max_order=3)
        with self.assertRaises(ValueError):
            StateGlobalMean(refresh=1.0)
        self.assertIsInstance(make_global_mean('state', n_centres=50), StateGlobalMean)


class TestStateInTree(unittest.TestCase):

    def test_tree_attaches_the_learner(self):
        gpt = GPTree(global_mean='state', global_mean_kwargs=dict(n_centres=50))
        self.assertIs(gpt.global_mean.tree, gpt)
        gm = _small()
        gpt = _tree(gm)
        self.assertIs(gm.tree, gpt)

    def test_unattached_learner_raises(self):
        gm = _small(min_points=5)
        X = np.random.RandomState(0).rand(5, 2)
        with self.assertRaises(RuntimeError):
            for i in range(5):
                gm.observe(X[i], X[i].sum(), 1e-3)

    def test_single_output_only(self):
        gpt = GPTree(n_outputs=2, global_mean='state')
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            gpt.update_tree(np.array([[0.1, 0.2]]), np.array([[1.0, 2.0]]), np.array([[0.1, 0.1]]))

    def test_schedule(self):
        rng = np.random.RandomState(3)
        X = rng.rand(500, 4); y = _additive(X)
        gm = _small()
        gpt = _tree(gm); _stream(gpt, X, y)
        self.assertEqual(gm.n_refits + gm.n_skipped, 4)                # due at 60, 120, 240 and 480 points
        self.assertGreater(gm.n_states, 4)                             # and solved in between
        self.assertEqual(gm.current.version, gm.n_states)
        self.assertEqual(gm.error_scale, None)

    def test_state_learns_an_additive_target(self):
        rng = np.random.RandomState(4)
        X = rng.rand(600, 4); y = _additive(X)
        Xt = rng.rand(400, 4); yt = _additive(Xt)
        gm = _small()
        gpt = _tree(gm); _stream(gpt, X, y)
        plain = _tree(None); _stream(plain, X, y)
        rmse = lambda p: float(np.sqrt(np.mean((p - yt) ** 2)))
        e_state = rmse(gm.current.predict(Xt)[:, 0])
        e_tree, e_plain = rmse(_predict(gpt, Xt)[0]), rmse(_predict(plain, Xt)[0])
        self.assertLess(e_state, 0.05 * yt.std())                      # the state alone
        self.assertLess(e_tree, 0.3 * e_plain)
        self.assertGreater(gm.order_shares()['additive'], 0.9)

    def test_sigma_comes_from_the_state(self):
        rng = np.random.RandomState(5)
        X = rng.rand(400, 4); y = _product(X)
        gm = _small()
        gpt = _tree(gm); _stream(gpt, X, y)
        Xt = rng.rand(200, 4)
        mu, sd = _predict(gpt, Xt)
        self.assertTrue(np.all(np.isfinite(sd)) and np.all(sd > 0))
        # never below the leaf GP's own sigma
        leaf = gpt.root.leaves[0]
        x = leaf.my_X_data[:5] + 0.01
        own = leaf._predict_independent(x, True)[1]
        with contextlib.redirect_stdout(io.StringIO()):
            total = leaf.predict(x)[1]
        self.assertTrue(np.all(total >= own - 1e-12))
        # larger away from the data than among it
        far = _predict(gpt, 1.5 + rng.rand(50, 4))[1]
        self.assertGreater(far.mean(), 3 * sd.mean())
        # roughly calibrated on a noise-free smooth target
        z = np.abs(mu - _product(Xt)) / sd
        self.assertGreater(np.mean(z <= 2), 0.6)

    def test_new_snapshot_refits_leaves_with_hyperparameters_kept(self):
        rng = np.random.RandomState(6)
        X = rng.rand(60, 2); y = np.sin(4 * X[:, 0]) + X[:, 1]
        for cheap in (True, False):
            gm = FakeLearner(cheap)
            gm.current = FakeSnapshot(1, lambda Z: Z[:, 1])
            gpt = GPTree(GPR=Default_GPR(n_restarts_optimizer=0), Nbar=200, retrain_every_n_points=20, global_mean=gm)
            _stream(gpt, X, y)
            leaf = gpt.root; gpr = leaf.my_GPRs[0]
            theta = gpr.get_kernel().theta.copy(); pending = leaf.n_points_since_retrain
            calls = []
            fit, refit_fixed = gpr.fit, gpr.refit_fixed
            gpr.fit = lambda *a: (calls.append('fit'), fit(*a))[1]
            gpr.refit_fixed = lambda *a: (calls.append('fixed'), refit_fixed(*a))[1]
            gm.current = FakeSnapshot(2, lambda Z: Z[:, 1] + 0.5 * np.sin(4 * Z[:, 0]))
            mu, _ = _predict(gpt, X)
            self.assertEqual(leaf._fitted_global.version, 2)
            self.assertLess(np.abs(mu - y).max(), 1e-2)
            self.assertEqual(calls, ['fixed'] if cheap else ['fit'])
            if cheap:
                np.testing.assert_allclose(gpr.get_kernel().theta, theta)
                self.assertEqual(leaf.n_points_since_retrain, pending)

    def test_backend_capabilities(self):
        rng = np.random.RandomState(10)
        X = rng.rand(30, 2); y = np.sin(3 * X[:, :1]) + 2.0
        for normalize_y in (False, True):
            gpr = Default_GPR(n_restarts_optimizer=0, normalize_y=normalize_y)
            self.assertFalse(gpr.refit_fixed(X, y))                    # not fitted yet
            self.assertIsNone(gpr.mean_weights(X))
            gpr.set_observation_noise(np.full(30, 1e-4)); gpr.fit(X, y)
            np.testing.assert_allclose(gpr.get_observation_noise(), 1e-4)
            Xq = rng.rand(7, 2)
            A, X_train = gpr.mean_weights(Xq)
            np.testing.assert_allclose(A.T @ y[:, 0], np.ravel(gpr.predict(Xq)), atol=1e-8)
            np.testing.assert_allclose(X_train, X)
            theta = gpr.get_kernel().theta.copy()
            self.assertTrue(gpr.refit_fixed(X, 3 * y))
            np.testing.assert_allclose(gpr.get_kernel().theta, theta)
            np.testing.assert_allclose(np.ravel(gpr.predict(Xq)), 3 * (A.T @ y[:, 0]), atol=1e-7)

    def test_without_standard_scaling(self):
        rng = np.random.RandomState(7)
        X = rng.rand(300, 3); y = np.sin(4 * X[:, 0]) + X[:, 1] * X[:, 2]
        gm = _small()
        gpt = _tree(gm, use_standard_scaling=False); _stream(gpt, X, y)
        Xt = rng.rand(100, 3)
        mu, sd = _predict(gpt, Xt)
        self.assertTrue(np.all(np.isfinite(mu)) and np.all(np.isfinite(sd)) and np.all(sd > 0))
        self.assertLess(np.sqrt(np.mean((mu - (np.sin(4 * Xt[:, 0]) + Xt[:, 1] * Xt[:, 2])) ** 2)), 0.1)

    def test_pickle(self):
        rng = np.random.RandomState(8)
        X = rng.rand(200, 3); y = X[:, 0] + np.sin(4 * X[:, 1]) * X[:, 2]
        gpt = _tree(_small()); _stream(gpt, X, y)
        Xt = rng.rand(20, 3)
        mu, sd = _predict(gpt, Xt)
        clone = pickle.loads(pickle.dumps(gpt))
        mu2, sd2 = _predict(clone, Xt)
        np.testing.assert_allclose(mu2, mu); np.testing.assert_allclose(sd2, sd)
        self.assertIs(clone.global_mean.tree, clone)
        _stream(clone, X[:30] * 0.9, y[:30])                           # and it keeps learning


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class TestStateWithNetwork(unittest.TestCase):

    def test_network_features_join_the_state(self):
        rng = np.random.RandomState(9)
        target = lambda Z: np.sin(6 * Z[:, 0] * Z[:, 1]) + Z[:, 2]
        X = rng.rand(500, 3); y = target(X); Xt = rng.rand(300, 3)
        net_kwargs = dict(hidden=32, steps=100, polish_steps=300)
        gm = _small(net=True, net_min_points=120, net_kwargs=net_kwargs)
        gpt = _tree(gm); _stream(gpt, X, y)
        snap = gm.current
        self.assertIsInstance(snap, StateSnapshot)
        self.assertEqual(snap.features(Xt).shape[1], snap.P2 + 32)
        self.assertEqual(gm.net_learner.holdout, 0.2)                  # the default of the state's network
        self.assertEqual(gm.net_learner.n_refits, 3)                   # at 120, 240 and 480 points
        mu, sd = _predict(gpt, Xt)
        self.assertTrue(np.all(np.isfinite(mu)) and np.all(sd > 0))
        self.assertLess(np.sqrt(np.mean((mu - target(Xt)) ** 2)), 0.25 * target(Xt).std())
        clone = pickle.loads(pickle.dumps(gpt))
        np.testing.assert_allclose(_predict(clone, Xt)[0], mu)


if __name__ == '__main__':
    unittest.main()
