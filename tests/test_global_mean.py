"""Tests for the global model + residual tree (``GPTree(global_mean=...)``)."""

import io
import os
import sys
import pickle
import contextlib
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo.gptree import GPTree
from pygptreeo.gpnode import GPNode
from pygptreeo.default_gpr import Default_GPR
from pygptreeo.global_mean import (CoverageReservoir, AdditiveGPGlobalMean, GlobalMeanLearner,
                                   GlobalMeanSnapshot, make_global_mean)

warnings.filterwarnings("ignore")


def _stream(gpt, X, Y, S):
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(X.shape[0]):
            gpt.update_tree(X[i:i + 1], Y[i:i + 1], S[i:i + 1])


class FakeSnapshot:
    """A snapshot that predicts a known function (for unit tests of the node logic)."""

    def __init__(self, version, fn, n_outputs=1):
        self.version, self.fn, self.n_outputs = version, fn, n_outputs

    def predict(self, X):
        return np.asarray(self.fn(np.atleast_2d(X)), dtype=float).reshape(X.shape[0], self.n_outputs)


class FakeLearner(GlobalMeanLearner):
    def __init__(self):
        self.current = None
        self.n_seen = 0

    def observe(self, x, y, sigma):
        self.n_seen += 1
        return False

    def publish(self, version, fn, n_outputs=1):
        self.current = FakeSnapshot(version, fn, n_outputs)


class TestCoverageReservoir(unittest.TestCase):

    def test_fills_then_keeps_coverage(self):
        rng = np.random.RandomState(0)
        res = CoverageReservoir(size=20, n_features=2, n_outputs=1)
        # 20 points all in a small cluster fill the reservoir
        for _ in range(20):
            self.assertTrue(res.add(0.5 + 0.01 * rng.randn(2), [1.0], [0.1]))
        self.assertTrue(res.full)
        self.assertEqual(res.turnover, 20)
        d_min_before = res._D.min()
        # a point coincident with an existing one does not improve coverage: rejected
        self.assertFalse(res.add(res.X[0].copy(), [2.0], [0.1]))
        # a far-away point does: accepted, replacing one of the closest pair
        self.assertTrue(res.add(np.array([0.0, 0.0]), [3.0], [0.2]))
        self.assertEqual(res.turnover, 21)
        self.assertEqual(res.n, 20)
        self.assertGreater(res._D.min(), d_min_before)
        self.assertTrue(np.any(np.all(res.X == 0.0, axis=1)))
        # its y and sigma travelled with it
        k = int(np.where(np.all(res.X == 0.0, axis=1))[0][0])
        self.assertEqual(res.y[k, 0], 3.0)
        self.assertEqual(res.sigma[k, 0], 0.2)

    def test_multi_output_rows(self):
        res = CoverageReservoir(size=5, n_features=1, n_outputs=3)
        res.add([0.1], [1, 2, 3], [0.1, 0.2, 0.3])
        self.assertEqual(res.y.shape, (1, 3))
        self.assertEqual(res.sigma.shape, (1, 3))


class TestAdditiveGPGlobalMean(unittest.TestCase):

    def _feed(self, learner, n, d=2, p=1, seed=0):
        rng = np.random.RandomState(seed)
        X = rng.rand(n, d)
        Y = np.column_stack([np.sin(3 * X[:, 0]) + 0.5 * np.cos(5 * X[:, 1]) + j for j in range(p)])
        published = []
        for i in range(n):
            if learner.observe(X[i], Y[i], 1e-3):
                published.append(i + 1)
        return X, Y, published

    def test_first_fit_and_snapshot(self):
        learner = AdditiveGPGlobalMean(reservoir_size=60, min_points=40, n_restarts_optimizer=0)
        X, Y, published = self._feed(learner, 45)
        self.assertEqual(published, [40])
        snap = learner.current
        self.assertIsInstance(snap, GlobalMeanSnapshot)
        self.assertEqual(snap.version, 1)
        self.assertEqual(snap.n_fit, 40)
        pred = snap.predict(X[:7])
        self.assertEqual(pred.shape, (7, 1))
        # the fit is good on its own data
        self.assertLess(np.sqrt(np.mean((pred[:, 0] - Y[:7, 0]) ** 2)), 0.05)
        # default kernel built for the input dimension
        self.assertEqual(np.shape(learner.kernel.k2.k2.length_scale), (2,))
        # per-point noise variances reached the GP
        self.assertEqual(np.shape(snap.gp.alpha), (40,))

    def test_refit_requires_turnover(self):
        # With min_turnover=0 every due refit happens (first at min_points, then when the
        # count has doubled, then at the latest every reservoir_size points).
        learner_all = AdditiveGPGlobalMean(reservoir_size=30, min_points=30, min_turnover=0.0,
                                           n_restarts_optimizer=0)
        _, _, published = self._feed(learner_all, 130)
        self.assertEqual(published, [30, 60, 90, 120])
        self.assertEqual(learner_all.current.version, 4)
        self.assertEqual(learner_all.n_skipped, 0)
        # A high turnover requirement skips most due refits once the coverage reservoir
        # has settled (a uniform stream then rarely improves the minimum separation).
        learner_all = AdditiveGPGlobalMean(reservoir_size=30, min_points=30, min_turnover=0.0,
                                           n_restarts_optimizer=0)
        learner_strict = AdditiveGPGlobalMean(reservoir_size=30, min_points=30, min_turnover=1.0,
                                              n_restarts_optimizer=0)
        self._feed(learner_all, 400)
        self._feed(learner_strict, 400)
        self.assertGreater(learner_strict.n_skipped, 0)
        self.assertLess(learner_strict.current.version, learner_all.current.version)
        self.assertLessEqual(learner_strict.current.version, 3)

    def test_multi_output(self):
        learner = AdditiveGPGlobalMean(reservoir_size=50, min_points=30, n_restarts_optimizer=0)
        X, Y, _ = self._feed(learner, 35, p=3)
        self.assertEqual(learner.current.n_outputs, 3)
        pred = learner.current.predict(X[:4])
        self.assertEqual(pred.shape, (4, 3))
        np.testing.assert_allclose(pred[:, 1] - pred[:, 0], 1.0, atol=0.05)

    def test_make_global_mean(self):
        self.assertIsNone(make_global_mean(None))
        learner = make_global_mean('additive_gp', reservoir_size=77, min_points=10)
        self.assertIsInstance(learner, AdditiveGPGlobalMean)
        self.assertEqual(learner.reservoir_size, 77)
        self.assertIs(make_global_mean(learner), learner)
        with self.assertRaises(ValueError):
            make_global_mean('nope')
        with self.assertRaises(ValueError):
            make_global_mean(None, reservoir_size=5)
        with self.assertRaises(TypeError):
            make_global_mean(3.0)


class TestNodeResidual(unittest.TestCase):
    """Unit tests of the node logic with a fake snapshot predicting a known function."""

    def _node(self, learner, n=60, seed=1):
        rng = np.random.RandomState(seed)
        X = rng.rand(n, 1)
        y = 5.0 + 2.0 * X[:, 0] + np.sin(6 * X[:, 0])            # global part + residual
        node = GPNode(0, my_GPR=Default_GPR(), Nbar=1000, retrain_every_n_points=1000, global_mean=learner)
        node.init_data_set(1)
        for i in range(n):
            node.store_point(X[i:i + 1], float(y[i]), 1e-3, increment_buffer=False)
        return node, X, y

    def test_fits_residual_and_adds_back(self):
        learner = FakeLearner()
        learner.publish(1, lambda X: 5.0 + 2.0 * X[:, 0])
        node, X, y = self._node(learner)
        with contextlib.redirect_stdout(io.StringIO()):
            node.fit_my_GPR(force_training=True)
        self.assertEqual(node._fitted_global.version, 1)
        # raw targets stay stored
        np.testing.assert_allclose(node.my_y_data[:, 0], y[::-1])
        # the GP was trained on the (standardised) residual sin(6x), not on y
        resid = y[::-1] - (5.0 + 2.0 * X[::-1, 0])
        scaled = node.y_scalers[0].transform(resid[:, None])[:, 0]
        np.testing.assert_allclose(node.my_GPRs[0].sklearn_gpr.y_train_.ravel(), scaled, atol=1e-10)
        # and the prediction is global + residual, i.e. y
        with contextlib.redirect_stdout(io.StringIO()):
            mu, sd = node.predict(X[:10])
        self.assertEqual(mu.shape, (10, 1))
        self.assertLess(np.sqrt(np.mean((mu[:, 0] - y[:10]) ** 2)), 0.02)

    def test_no_global_model_is_unchanged(self):
        node, X, y = self._node(None)
        with contextlib.redirect_stdout(io.StringIO()):
            node.fit_my_GPR(force_training=True)
        self.assertIsNone(node._fitted_global)
        np.testing.assert_allclose(node.y_scalers[0].inverse_transform(
            node.my_GPRs[0].sklearn_gpr.y_train_.reshape(-1, 1))[:, 0], y[::-1], atol=1e-10)

    def test_refresh_rule_refits_on_newer_version(self):
        learner = FakeLearner()
        learner.publish(1, lambda X: 5.0 + 2.0 * X[:, 0])
        node, X, y = self._node(learner)
        fits = []
        orig = node.fit_my_GPR
        node.fit_my_GPR = lambda force_training=False: (fits.append(1), orig(force_training))[1]
        with contextlib.redirect_stdout(io.StringIO()):
            node.fit_my_GPR(force_training=True)
            node.predict(X[:3])                    # current version: no refit
        self.assertEqual(len(fits), 1)
        learner.publish(2, lambda X: 5.0 + 2.0 * X[:, 0] + 1.0)   # a newer (shifted) snapshot
        with contextlib.redirect_stdout(io.StringIO()):
            mu, _ = node.predict(X[:10])
        self.assertEqual(len(fits), 2)             # refit before predicting
        self.assertEqual(node._fitted_global.version, 2)
        self.assertLess(np.sqrt(np.mean((mu[:, 0] - y[:10]) ** 2)), 0.02)   # still predicts y
        with contextlib.redirect_stdout(io.StringIO()):
            node.predict(X[:3])
        self.assertEqual(len(fits), 2)             # and not again

    def test_children_inherit_snapshot(self):
        learner = FakeLearner()
        learner.publish(3, lambda X: 5.0 + 2.0 * X[:, 0])
        node, X, y = self._node(learner)
        with contextlib.redirect_stdout(io.StringIO()):
            node.fit_my_GPR(force_training=True)
            node.generate_children(Default_GPR(), 1)
        for child in node.children:
            self.assertIs(child._fitted_global, node._fitted_global)
            self.assertIs(child.global_mean, learner)


class TestTreeIntegration(unittest.TestCase):

    def _data(self, n=600, seed=3):
        rng = np.random.RandomState(seed)
        X = rng.rand(n, 2)
        y = np.sin(3 * X[:, 0]) + 0.5 * np.cos(5 * X[:, 1])       # exactly additive
        return X, y[:, None], np.full((n, 1), 1e-3)

    def test_default_is_no_global_model(self):
        gpt = GPTree(Nbar=50)
        self.assertIsNone(gpt.global_mean)
        X, Y, S = self._data(120)
        _stream(gpt, X, Y, S)
        self.assertTrue(all(leaf._fitted_global is None for leaf in gpt.root.leaves))

    def test_residual_tree_beats_plain_tree_on_additive_target(self):
        X, Y, S = self._data()
        Xt = np.random.RandomState(9).rand(500, 2)
        yt = np.sin(3 * Xt[:, 0]) + 0.5 * np.cos(5 * Xt[:, 1])
        errs = {}
        for name, gm in [('plain', None), ('global', 'additive_gp')]:
            np.random.seed(0)
            gpt = GPTree(Nbar=50, retrain_every_n_points=25, use_calibrated_sigma=False, global_mean=gm,
                         global_mean_kwargs=None if gm is None else dict(reservoir_size=100, min_points=100,
                                                                         n_restarts_optimizer=1, random_state=0))
            _stream(gpt, X, Y, S)
            with contextlib.redirect_stdout(io.StringIO()):
                mu, sd = gpt.predict(Xt)
            self.assertTrue(np.all(np.isfinite(mu)) and np.all(np.isfinite(sd)))
            errs[name] = float(np.sqrt(np.mean((mu[:, 0] - yt) ** 2)))
            if gm is not None:
                self.assertGreaterEqual(gpt.global_mean.current.version, 1)
                # after predicting, every leaf holds the current snapshot (refresh rule)
                self.assertTrue(all(leaf._fitted_global is gpt.global_mean.current for leaf in gpt.root.leaves))
        self.assertLess(errs['global'], errs['plain'])

    def test_batch_fit_and_pickle(self):
        X, Y, S = self._data(300)
        gpt = GPTree(Nbar=50, use_calibrated_sigma=False, global_mean='additive_gp',
                     global_mean_kwargs=dict(reservoir_size=80, min_points=60, n_restarts_optimizer=0))
        with contextlib.redirect_stdout(io.StringIO()):
            gpt.fit(X, Y, S)
        self.assertIsNotNone(gpt.global_mean.current)
        self.assertTrue(all(leaf._fitted_global is gpt.global_mean.current for leaf in gpt.root.leaves))
        with contextlib.redirect_stdout(io.StringIO()):
            mu, _ = gpt.predict(X[:20])
        self.assertLess(np.sqrt(np.mean((mu[:, 0] - Y[:20, 0]) ** 2)), 0.05)
        restored = pickle.loads(pickle.dumps(gpt))
        with contextlib.redirect_stdout(io.StringIO()):
            mu2, _ = restored.predict(X[:20])
        np.testing.assert_allclose(mu2, mu)

    def test_multi_output(self):
        rng = np.random.RandomState(5)
        X = rng.rand(300, 2)
        Y = np.column_stack([np.sin(3 * X[:, 0]), np.cos(4 * X[:, 1])])
        S = np.full(Y.shape, 1e-3)
        gpt = GPTree(Nbar=60, n_outputs=2, use_calibrated_sigma=False, global_mean='additive_gp',
                     global_mean_kwargs=dict(reservoir_size=80, min_points=60, n_restarts_optimizer=0))
        _stream(gpt, X, Y, S)
        self.assertEqual(gpt.global_mean.current.n_outputs, 2)
        with contextlib.redirect_stdout(io.StringIO()):
            mu, sd = gpt.predict(X[:10])
        self.assertEqual(mu.shape, (10, 2))
        self.assertLess(np.sqrt(np.mean((mu - Y[:10]) ** 2)), 0.05)

    def test_learner_instance_and_bad_spec(self):
        learner = AdditiveGPGlobalMean(reservoir_size=40, min_points=20)
        gpt = GPTree(global_mean=learner)
        self.assertIs(gpt.global_mean, learner)
        self.assertIs(gpt.root.global_mean, learner)
        with self.assertRaises(ValueError):
            GPTree(global_mean='something_else')
        with self.assertRaises(ValueError):
            GPTree(global_mean=learner, global_mean_kwargs=dict(reservoir_size=5))


if __name__ == '__main__':
    unittest.main()
