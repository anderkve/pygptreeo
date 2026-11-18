"""Tests for output_model='aligned': the warp, the combination of warped curves,
the weights of the GP mean, and the leaves of a tree."""

import io
import os
import sys
import contextlib
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo.gptree import GPTree
from pygptreeo.default_gpr import Default_GPR
from pygptreeo.output_alignment import AlignedCurves, OutputWarp, _RowSplines

warnings.filterwarnings("ignore")

GRID = np.linspace(0.0, 1.0, 64)


def moving_peak(X):
    """A peak whose position and width depend on the inputs, on a slope."""
    centre = 0.3 + 0.4 * X[:, :1]
    width = 0.04 + 0.03 * X[:, 1:2]
    return np.exp(-0.5 * ((GRID - centre) / width) ** 2) + 0.5 * GRID


def fixed_shapes(X):
    """Two shapes at fixed positions with amplitudes that depend on the inputs."""
    return np.sin(3 * X[:, :1]) * np.sin(2 * np.pi * GRID) + X[:, 1:2] ** 2 * np.cos(np.pi * GRID)


def _tree(model, **kwargs):
    return GPTree(GPR=Default_GPR(), Nbar=100, n_outputs=len(GRID), output_model=model,
                  retrain_every_n_points=25, **kwargs)


def _fit(model, target, n=150, seed=0, **kwargs):
    np.random.seed(seed)
    X = np.random.RandomState(seed).rand(n, 2)
    Y = target(X)
    gpt = _tree(model, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        gpt.fit(X, Y, np.full(Y.shape, 1e-4))
    return gpt


def _rmse(gpt, target, seed=1):
    X = np.random.RandomState(seed).rand(200, 2)
    mu, sd = gpt.predict(X, mode='loop')
    return float(np.sqrt(np.mean((mu - target(X)) ** 2)))


class TestOutputWarp(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(0)
        self.warp = OutputWarp(n_outputs=64, n_features=2)
        self.warp.C = 0.3 * rng.randn(2, self.warp.K)
        self.delta = rng.randn(20, 2)

    def test_identity_without_coefficients_and_at_the_centre(self):
        grid = np.arange(64.0)
        np.testing.assert_allclose(OutputWarp(64, 2).phi(self.delta), np.tile(grid, (20, 1)), atol=1e-12)
        np.testing.assert_allclose(self.warp.phi(np.zeros((1, 2)))[0], grid, atol=1e-12)

    def test_monotone_and_onto_the_grid(self):
        phi = self.warp.phi(self.delta)
        self.assertTrue(np.all(np.diff(phi, axis=1) > 0))
        np.testing.assert_allclose(phi[:, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(phi[:, -1], 63.0, atol=1e-10)

    def test_inverse(self):
        inverse = self.warp.inverse(self.delta)
        back = _RowSplines(self.warp.phi(self.delta))(inverse)
        np.testing.assert_allclose(back, np.tile(np.arange(64.0), (20, 1)), atol=1e-6)

    def test_jacobian(self):
        phi, jac = self.warp.phi(self.delta, jacobian=True)
        C = self.warp.C.copy()
        C[1, 3] += 1e-6
        np.testing.assert_allclose((self.warp.phi(self.delta, C) - phi) / 1e-6, jac[1, 3], atol=1e-4)

    def test_fit_aligns_a_moving_peak(self):
        rng = np.random.RandomState(1)
        X = rng.rand(80, 2)
        F = moving_peak(X)
        delta = (X - X.mean(axis=0)) / X.std(axis=0)
        warp = OutputWarp(64, 2)
        left = warp.fit(F, delta)
        self.assertLess(left, 0.1)
        # the peaks of the aligned curves lie closer together
        spread = lambda curves: np.std(np.argmax(curves - 0.5 * GRID, axis=1))
        self.assertLess(spread(warp.register(F, delta)), 0.3 * spread(F))


class TestAlignedCurves(unittest.TestCase):

    def test_combine_returns_a_curve_at_its_own_offset(self):
        rng = np.random.RandomState(2)
        X = rng.rand(30, 2)
        F = moving_peak(X)
        delta = (X - X.mean(axis=0)) / X.std(axis=0)
        warp = OutputWarp(64, 2)
        warp.C = 0.2 * rng.randn(2, warp.K)
        curves = AlignedCurves(warp, F, delta)
        curves.centre_weights = np.zeros(30)
        np.testing.assert_allclose(curves.combine(delta, np.eye(30)), F, atol=1e-5)

    def test_with_curve_leaves_the_original(self):
        rng = np.random.RandomState(3)
        X = rng.rand(10, 2)
        curves = AlignedCurves(OutputWarp(64, 2), moving_peak(X), X)
        more = curves.with_curve(moving_peak(X[:1])[0], X[0])
        self.assertEqual(curves.F.shape[0], 10)
        self.assertEqual(more.F.shape[0], 11)
        self.assertEqual(more.centre_weights[-1], 0.0)
        np.testing.assert_allclose(more.centre_weights.sum(), 1.0)


class TestMeanWeights(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(4)
        self.X = rng.rand(25, 2)
        self.Y = np.column_stack([np.sin(4 * self.X[:, 0]), self.X[:, 1] ** 2, self.X.sum(axis=1)])
        self.gpr = Default_GPR(alpha=1e-6)
        self.gpr.fit(self.X, self.Y)

    def test_weights_give_the_mean(self):
        Xt = np.random.RandomState(5).rand(7, 2)
        np.testing.assert_allclose(self.gpr.mean_weights(Xt) @ self.Y, self.gpr.predict(Xt), atol=1e-8)

    def test_leave_one_out(self):
        weights, variance = self.gpr.leave_one_out()
        np.testing.assert_allclose(np.diag(weights), 0.0)
        hyperparameters = self.gpr.get_hyperparameters()
        for i in (0, 7, 24):
            keep = np.arange(25) != i
            without = Default_GPR(alpha=1e-6)
            without.refit_fixed(self.X[keep], self.Y[keep], hyperparameters)
            mean, std = without.predict(self.X[i:i+1], return_std=True)
            np.testing.assert_allclose(weights[i] @ self.Y, mean[0], atol=1e-6)
            # the variance of the observed target: that of the function and the noise
            np.testing.assert_allclose(variance[i], std[0, 0] ** 2 + 1e-6, rtol=1e-4)

    def test_none_without_a_fit(self):
        self.assertIsNone(Default_GPR().mean_weights(self.X))
        self.assertIsNone(Default_GPR().leave_one_out())


class TestAlignedTree(unittest.TestCase):

    def test_lower_error_for_a_moving_peak(self):
        aligned = _fit('aligned', moving_peak)
        shared = _fit('shared', moving_peak)
        self.assertTrue(all(leaf.alignment is not None for leaf in aligned.root.leaves))
        self.assertLess(_rmse(aligned, moving_peak), 0.5 * _rmse(shared, moving_peak))

    def test_as_shared_where_an_alignment_does_not_help(self):
        aligned = _fit('aligned', fixed_shapes)
        shared = _fit('shared', fixed_shapes)
        self.assertTrue(all(leaf.alignment is None for leaf in aligned.root.leaves))
        X = np.random.RandomState(1).rand(50, 2)
        np.testing.assert_allclose(aligned.predict(X, mode='loop')[0], shared.predict(X, mode='loop')[0], atol=1e-10)

    def test_stream_with_appended_points_and_a_split(self):
        rng = np.random.RandomState(6)
        np.random.seed(6)
        X = rng.rand(140, 2)
        Y = moving_peak(X)
        gpt = _tree('aligned')
        with contextlib.redirect_stdout(io.StringIO()):
            gpt.batch_update_tree(X, Y, np.full(Y.shape, 1e-4))
        self.assertFalse(gpt.root.is_leaf)
        for leaf in gpt.root.leaves:
            self.assertIsNotNone(leaf.alignment)
            # one curve for each training point of the GP, appended ones included
            self.assertEqual(leaf.alignment.F.shape[0], leaf.my_GPRs[0].sklearn_gpr.X_train_.shape[0])
        Xt = rng.rand(100, 2)
        mu, sd = gpt.predict(Xt)
        self.assertTrue(np.all(np.isfinite(mu)) and np.all(sd > 0))
        self.assertLess(np.sqrt(np.mean((mu - moving_peak(Xt)) ** 2)), 0.02)

    def test_appended_curve_is_predicted(self):
        gpt = _fit('aligned', moving_peak, n=60, append_points_between_fits=True)
        leaf = gpt.root.leaves[0]
        n = leaf.alignment.F.shape[0]
        x = np.array([[0.31, 0.62]])
        y = moving_peak(x)
        before = leaf.predict(x)[0]
        self.assertTrue(leaf.append_point(x, y, 1e-4))
        self.assertEqual(leaf.alignment.F.shape[0], n + 1)
        after = leaf.predict(x)[0]
        self.assertLess(np.abs(after - y).max(), 0.2 * np.abs(before - y).max() + 1e-3)

    def test_sleeping_leaf_predicts_as_before(self):
        gpt = _fit('aligned', moving_peak, n=60)
        leaf = gpt.root.leaves[0]
        x = np.random.RandomState(7).rand(5, 2)
        before = leaf.predict(x)[0]
        size = leaf.nbytes_fitted()
        self.assertTrue(leaf.sleep())
        self.assertLess(leaf.nbytes_fitted(), size)
        np.testing.assert_allclose(leaf.predict(x)[0], before, atol=1e-8)
        self.assertFalse(leaf.asleep)

    def test_saved_tree_predicts_as_before(self):
        import joblib, tempfile
        gpt = _fit('aligned', moving_peak, n=60)
        x = np.random.RandomState(8).rand(5, 2)
        before = gpt.predict(x)
        with tempfile.TemporaryDirectory() as folder:
            gpt.save(os.path.join(folder, 'tree.joblib'))
            loaded = joblib.load(os.path.join(folder, 'tree.joblib'))
        for a, b in zip(before, loaded.predict(x)):
            np.testing.assert_allclose(a, b)

    def test_memory_limit(self):
        rng = np.random.RandomState(9)
        np.random.seed(9)
        X = rng.rand(400, 2)
        Y = moving_peak(X)
        gpt = _tree('aligned', max_memory_mb=1.5)
        with contextlib.redirect_stdout(io.StringIO()):
            gpt.batch_update_tree(X, Y, np.full(Y.shape, 1e-4))
        self.assertLessEqual(gpt.memory_mb, 1.5)
        mu, sd = gpt.predict(rng.rand(20, 2))
        self.assertTrue(np.all(np.isfinite(mu)) and np.all(sd > 0))

    def test_alignment_waits_after_rejections(self):
        gpt = _fit('aligned', fixed_shapes, n=60)
        leaf = gpt.root.leaves[0]
        self.assertEqual((leaf._alignment_rejections, leaf._alignment_wait), (1, 1))
        with contextlib.redirect_stdout(io.StringIO()):
            leaf.fit_my_GPR(force_training=True)     # waits
            self.assertEqual((leaf._alignment_rejections, leaf._alignment_wait), (1, 0))
            leaf.fit_my_GPR(force_training=True)     # tries again
        self.assertEqual((leaf._alignment_rejections, leaf._alignment_wait), (2, 3))

    def test_without_standard_scaling(self):
        aligned = _fit('aligned', moving_peak, use_standard_scaling=False)
        shared = _fit('shared', moving_peak, use_standard_scaling=False)
        self.assertLess(_rmse(aligned, moving_peak), _rmse(shared, moving_peak))

    def test_needs_a_grid(self):
        with self.assertRaises(ValueError):
            GPTree(n_outputs=4, output_model='aligned')


class TestAlignedTreeWithGlobalModel(unittest.TestCase):
    """A leaf with an alignment models the curves themselves, and a leaf without
    one the residual of the global model."""

    def _stream(self, model, target, with_global, n=220, seed=0):
        from pygptreeo.global_mean import AdditiveGPGlobalMean
        rng = np.random.RandomState(seed)
        np.random.seed(seed)
        X = rng.rand(n, 2)
        Y = target(X)
        learner = AdditiveGPGlobalMean(reservoir_size=60, min_points=50, n_restarts_optimizer=0, random_state=0) if with_global else None
        gpt = _tree(model, global_mean=learner)
        with contextlib.redirect_stdout(io.StringIO()):
            gpt.batch_update_tree(X, Y, np.full(Y.shape, 1e-4))
        return gpt

    def test_aligned_leaves_do_without_the_global_model(self):
        gpt = self._stream('aligned', moving_peak, with_global=True)
        alone = self._stream('aligned', moving_peak, with_global=False)
        self.assertIsNotNone(gpt.global_mean.current)
        for leaf in gpt.root.leaves:
            self.assertIsNotNone(leaf.alignment)
            self.assertIsNone(leaf._fitted_global)
        X = np.random.RandomState(1).rand(50, 2)
        np.testing.assert_allclose(gpt.predict(X, mode='loop')[0], alone.predict(X, mode='loop')[0], atol=1e-8)

    def test_other_leaves_model_the_residual(self):
        gpt = self._stream('aligned', fixed_shapes, with_global=True)
        shared = self._stream('shared', fixed_shapes, with_global=True)
        X = np.random.RandomState(1).rand(50, 2)
        np.testing.assert_allclose(gpt.predict(X, mode='loop')[0], shared.predict(X, mode='loop')[0], atol=1e-8)
        for leaf in gpt.root.leaves:
            self.assertIsNone(leaf.alignment)
            self.assertIs(leaf._fitted_global, gpt.global_mean.current)

    def test_aligned_leaf_is_not_refitted_for_a_new_snapshot(self):
        gpt = self._stream('aligned', moving_peak, with_global=True)
        leaf = gpt.root.leaves[0]
        gpr = leaf.my_GPRs[0]
        gpt.global_mean.fit()                      # a newer snapshot
        leaf.predict(np.array([[0.4, 0.5]]))
        self.assertIs(leaf.my_GPRs[0], gpr)
        self.assertIsNotNone(leaf.alignment)


if __name__ == '__main__':
    unittest.main()
