"""Tests for the additive global model from running sums (``GPTree(global_mean='additive')``)
and for the leaf hooks it uses."""

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
from pygptreeo.global_mean import GlobalMeanLearner, make_global_mean
from pygptreeo.additive_sums import AdditiveGlobalMean, AdditiveSnapshot, SineFeatures, evidence

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
    return np.sin(9 * X[:, 0]) + X[:, 1] ** 2 + np.exp(X[:, 2]) + np.cos(5 * X[:, 3])


def _product(X):
    return np.sin(3 * X[:, 0]) * np.cos(3 * X[:, 1]) * (1 + X[:, 2]) * np.exp(-X[:, 3])


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


class TestSumsAndEvidence(unittest.TestCase):

    def _problem(self):
        rng = np.random.RandomState(0); d = 4
        X = rng.rand(300, d); t = np.sin(6 * X[:, 0]) + X[:, 1] ** 2 + 0.3 * X[:, 2] * X[:, 3]
        t = (t - t.mean()) / t.std()
        basis = SineFeatures(X.min(axis=0), X.max(axis=0), 24)
        F = basis(X)
        theta = np.log([0.2, 0.5, 1.0, 0.3, 0.8, 0.2])
        return X, t, basis, F, theta

    def test_features_represent_the_rbf_kernel(self):
        """For length scales that are short against the margin of the interval."""
        rng = np.random.RandomState(0); ls = np.array([0.05, 0.12]); amp = 0.7
        basis = SineFeatures(np.zeros(2), np.ones(2), 64)
        X = rng.rand(40, 2); F = basis(X)[:, 1:]
        K = (F * np.exp(basis.log_prior_var(np.log(ls), np.log(amp))[0]).ravel()) @ F.T
        exact = sum(np.exp(-0.5 * ((X[:, j, None] - X[None, :, j]) / ls[j]) ** 2) for j in range(2)) * amp ** 2 / 2
        self.assertLess(np.abs(K - exact).max(), 0.02 * amp ** 2 / 2)

    def test_evidence_from_sums_equals_the_dense_one(self):
        X, t, basis, F, theta = self._problem(); n = len(t)
        value, _, _ = evidence(theta, basis, F.T @ F, F.T @ t, n)
        lam = np.concatenate([[10.0], np.exp(basis.log_prior_var(theta[:4], theta[4])[0]).ravel()])
        C = np.exp(2 * theta[5]) * np.eye(n) + (F * lam) @ F.T
        dense = 0.5 * (n * np.log(2 * np.pi) + np.linalg.slogdet(C)[1] + t @ np.linalg.solve(C, t)) / n
        self.assertAlmostEqual(value, dense, places=9)

    def test_evidence_gradient(self):
        X, t, basis, F, theta = self._problem(); G = F.T @ F; b = F.T @ t
        _, grad, _ = evidence(theta, basis, G, b, len(t))
        num = np.array([(evidence(theta + 1e-5 * e, basis, G, b, len(t))[0] - evidence(theta - 1e-5 * e, basis, G, b, len(t))[0]) / 2e-5
                        for e in np.eye(6)])
        np.testing.assert_allclose(grad, num, atol=1e-7)

    def test_solution_is_the_posterior_mean(self):
        X, t, basis, F, theta = self._problem()
        _, _, (keep, w, cf, r) = evidence(theta, basis, F.T @ F, F.T @ t, len(t))
        lam = np.concatenate([[10.0], np.exp(basis.log_prior_var(theta[:4], theta[4])[0]).ravel()])[keep]
        Fk = F[:, keep]; s2 = np.exp(2 * theta[5])
        np.testing.assert_allclose(w, np.linalg.solve(Fk.T @ Fk / s2 + np.diag(1 / lam), Fk.T @ t / s2), atol=1e-8)


class TestAdditiveLearner(unittest.TestCase):

    def test_arguments(self):
        with self.assertRaises(ValueError):
            AdditiveGlobalMean(refresh=1.0)
        self.assertIsInstance(make_global_mean('additive', max_features=200), AdditiveGlobalMean)
        gm = AdditiveGlobalMean()
        with self.assertRaises(ValueError):
            gm.observe(np.zeros(2), np.array([1.0, 2.0]), np.array([0.1, 0.1]))

    def test_learns_an_additive_target_from_the_sums(self):
        rng = np.random.RandomState(1)
        X = rng.rand(1000, 4); y = _additive(X); Xt = rng.rand(500, 4); yt = _additive(Xt)
        gm = AdditiveGlobalMean()
        for i in range(1000):
            gm.observe(X[i], y[i], 1e-3)
        self.assertLess(np.sqrt(np.mean((gm.current.predict(Xt)[:, 0] - yt) ** 2)), 1e-3 * yt.std())
        self.assertGreater(gm.additive_share(), 0.999)
        self.assertEqual(gm.n_refits, 5)                               # at 60, 120, 240, 480 and 960 points
        self.assertGreater(gm.n_solves, gm.n_refits)
        self.assertEqual(gm.current.version, gm.n_solves)
        # the sums hold every point
        gm._flush()
        F = gm.basis(X)
        np.testing.assert_allclose(gm.G, F.T @ F, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(gm.b, F.T @ y, rtol=1e-9, atol=1e-9)

    def test_share_is_small_without_additive_structure(self):
        rng = np.random.RandomState(2)
        X = rng.rand(500, 4); y = np.sin(8 * X[:, 0] * X[:, 1]) * np.cos(8 * X[:, 2] * X[:, 3])
        gm = AdditiveGlobalMean()
        for i in range(500):
            gm.observe(X[i], y[i], 1e-3)
        self.assertLess(gm.additive_share(), 0.3)

    def test_sums_are_rebuilt_when_the_stream_leaves_the_basis(self):
        rng = np.random.RandomState(3)
        X = np.vstack([0.3 * rng.rand(300, 3), rng.rand(700, 3)]); y = np.sin(6 * X[:, 0]) + X[:, 1] + X[:, 2] ** 2
        gm = AdditiveGlobalMean()
        for i in range(1000):
            gm.observe(X[i], y[i], 1e-3)
        self.assertGreaterEqual(gm.n_rebuilds, 2)
        Xt = rng.rand(300, 3); yt = np.sin(6 * Xt[:, 0]) + Xt[:, 1] + Xt[:, 2] ** 2
        self.assertLess(np.sqrt(np.mean((gm.current.predict(Xt)[:, 0] - yt) ** 2)), 1e-2 * yt.std())


class TestAdditiveInTree(unittest.TestCase):

    def test_tree_models_the_residual(self):
        rng = np.random.RandomState(4)
        X = rng.rand(600, 4); y = _additive(X) + 0.2 * _product(X)
        Xt = rng.rand(400, 4); yt = _additive(Xt) + 0.2 * _product(Xt)
        gpt = GPTree(GPR=Default_GPR(n_restarts_optimizer=0), Nbar=60, retrain_every_n_points=20, global_mean='additive')
        self.assertIsInstance(gpt.global_mean, AdditiveGlobalMean)
        np.random.seed(0); _stream(gpt, X, y)
        plain = _tree(None); _stream(plain, X, y)
        rmse = lambda p: float(np.sqrt(np.mean((p - yt) ** 2)))
        e_global = rmse(gpt.global_mean.current.predict(Xt)[:, 0])
        e_tree, e_plain = rmse(_predict(gpt, Xt)[0]), rmse(_predict(plain, Xt)[0])
        self.assertLess(e_tree, e_global)                              # the leaves add to the additive part
        self.assertLess(e_tree, 0.6 * e_plain)
        for leaf in gpt.root.leaves:
            self.assertIsInstance(leaf._fitted_global, AdditiveSnapshot)

    def test_sigma(self):
        rng = np.random.RandomState(5)
        X = rng.rand(400, 4); y = _additive(X) + _product(X)
        gpt = _tree(AdditiveGlobalMean()); _stream(gpt, X, y)
        Xt = rng.rand(200, 4)
        mu, sd = _predict(gpt, Xt)
        self.assertTrue(np.all(np.isfinite(sd)) and np.all(sd > 0))
        leaf = gpt.root.leaves[0]; x = leaf.my_X_data[:5] + 0.01
        with contextlib.redirect_stdout(io.StringIO()):
            total = leaf.predict(x)[1]
        self.assertTrue(np.all(total >= leaf._predict_independent(x, True)[1] - 1e-12))   # never below the leaf GP's own sigma
        snap = gpt.global_mean.current
        self.assertTrue(np.all(snap.outside_variance(X[:snap.n_fit]) == 0.0))      # the points seen when it was solved
        self.assertTrue(np.all(snap.outside_variance(1.5 + rng.rand(20, 4)) > 0.0))
        z = np.abs(mu - (_additive(Xt) + _product(Xt))) / sd
        self.assertGreater(np.mean(z <= 2), 0.6)

    def test_without_standard_scaling(self):
        rng = np.random.RandomState(7)
        X = rng.rand(300, 3); f = lambda Z: np.sin(4 * Z[:, 0]) + Z[:, 1] * Z[:, 2]
        gpt = _tree(AdditiveGlobalMean(), use_standard_scaling=False); _stream(gpt, X, f(X))
        Xt = rng.rand(100, 3)
        mu, sd = _predict(gpt, Xt)
        self.assertTrue(np.all(np.isfinite(mu)) and np.all(np.isfinite(sd)) and np.all(sd > 0))
        self.assertLess(np.sqrt(np.mean((mu - f(Xt)) ** 2)), 0.1)

    def test_pickle(self):
        rng = np.random.RandomState(8)
        X = rng.rand(200, 3); y = X[:, 0] + np.sin(4 * X[:, 1]) * X[:, 2]
        gpt = _tree(AdditiveGlobalMean()); _stream(gpt, X, y)
        Xt = rng.rand(20, 3)
        mu, sd = _predict(gpt, Xt)
        clone = pickle.loads(pickle.dumps(gpt))
        mu2, sd2 = _predict(clone, Xt)
        np.testing.assert_allclose(mu2, mu); np.testing.assert_allclose(sd2, sd)
        _stream(clone, X[:30] * 0.9, y[:30])                           # and it keeps learning


class TestLeafHooks(unittest.TestCase):

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
            Xq = rng.rand(7, 2)
            A, X_train = gpr.mean_weights(Xq)
            np.testing.assert_allclose(A.T @ y[:, 0], np.ravel(gpr.predict(Xq)), atol=1e-8)
            np.testing.assert_allclose(X_train, X)
            theta = gpr.get_kernel().theta.copy()
            self.assertTrue(gpr.refit_fixed(X, 3 * y))
            np.testing.assert_allclose(gpr.get_kernel().theta, theta)
            np.testing.assert_allclose(np.ravel(gpr.predict(Xq)), 3 * (A.T @ y[:, 0]), atol=1e-7)


if __name__ == '__main__':
    unittest.main()
