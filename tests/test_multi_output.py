"""Tests for multi-output handling: child-GP inheritance, per-point noise,
the 'shared' and 'pca' output models, and the output basis itself."""

import io
import os
import sys
import contextlib
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF

from pygptreeo.gptree import GPTree
from pygptreeo.gpnode import GPNode
from pygptreeo.default_gpr import Default_GPR
from pygptreeo.output_basis import OutputBasis, OutputBasisLearner

warnings.filterwarnings("ignore")


def _stream(gpt, X, Y, S):
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(X.shape[0]):
            gpt.update_tree(X[i:i+1], Y[i:i+1], S[i:i+1])


def _two_output_stream(n=150, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 1)
    Y = np.hstack([np.sin(6 * X), -np.sin(6 * X)])   # output 1 = -output 0
    S = np.full((n, 2), 1e-3)
    return X, Y, S


class TestChildGPInheritance(unittest.TestCase):
    """After a split, a child must predict each output with that output's GP."""

    def test_children_get_one_clone_per_output_gp(self):
        X, Y, S = _two_output_stream()
        gpt = GPTree(GPR=Default_GPR(), Nbar=100, n_outputs=2, retrain_every_n_points=100,
                     use_calibrated_sigma=False, splitting_strategy='standard')
        _stream(gpt, X, Y, S)
        self.assertFalse(gpt.root.is_leaf)
        for leaf in gpt.root.leaves:
            # children have not retrained yet: they still hold the parent's GPs
            self.assertEqual(len(leaf.my_GPRs), 2)
            self.assertIsNot(leaf.my_GPRs[0], leaf.my_GPRs[1])
        Xt = np.linspace(0.05, 0.95, 7).reshape(-1, 1)
        with contextlib.redirect_stdout(io.StringIO()):
            mu, sd = gpt.predict(Xt)
        self.assertEqual(mu.shape, (7, 2))
        rmse0 = np.sqrt(np.mean((mu[:, 0] - np.sin(6 * Xt[:, 0])) ** 2))
        rmse1 = np.sqrt(np.mean((mu[:, 1] + np.sin(6 * Xt[:, 0])) ** 2))
        self.assertLess(rmse0, 0.05)
        self.assertLess(rmse1, 0.05)


class TestNoiseReachesBackend(unittest.TestCase):
    """The per-point sigma passed to the tree must be used as the GP's noise."""

    def test_alpha_set_on_sklearn_gpr(self):
        node = GPNode(0, my_GPR=Default_GPR(), Nbar=50, retrain_every_n_points=1)
        node.init_data_set(1)
        rng = np.random.RandomState(0)
        for i in range(20):
            node.store_point(rng.rand(1, 1), float(rng.rand()), 0.1 * (i + 1))
        with contextlib.redirect_stdout(io.StringIO()):
            node.fit_my_GPR(force_training=True)
        alpha = node.my_GPRs[0].sklearn_gpr.alpha
        self.assertEqual(np.shape(alpha), (20,))
        # scaled variance: (sigma / y_scale)^2, in the node's storage order (newest first)
        expected = (node.my_sigma_data[:, 0] / node.y_scalers[0].scale_[0]) ** 2
        np.testing.assert_allclose(alpha, expected)


class TestOutputBasis(unittest.TestCase):

    def test_exact_recovery_of_low_rank_outputs(self):
        rng = np.random.RandomState(0)
        p, k, n = 12, 3, 200
        W_true = np.linalg.qr(rng.randn(p, k))[0]
        Z = rng.randn(n, k) * np.array([3.0, 2.0, 1.0])
        Y = 5.0 + Z @ W_true.T
        learner = OutputBasisLearner(n_outputs=p, n_components=0.999, min_points=20)
        for y in Y:
            learner.observe(y)
        basis = learner.current
        self.assertEqual(basis.n_components, k)
        self.assertGreater(basis.explained_variance_ratio.sum(), 0.999)
        rec = basis.reconstruct(basis.project(Y))
        np.testing.assert_allclose(rec, Y, atol=1e-8)
        np.testing.assert_allclose(basis.resid_var, 0.0, atol=1e-12)
        # orthonormal columns
        np.testing.assert_allclose(basis.W.T @ basis.W, np.eye(k), atol=1e-10)

    def test_variance_propagation_and_truncation(self):
        rng = np.random.RandomState(1)
        p, n = 6, 300
        Y = rng.randn(n, p) @ rng.randn(p, p)
        learner = OutputBasisLearner(n_outputs=p, n_components=2, min_points=10)
        for y in Y:
            learner.observe(y)
        b = learner.current
        self.assertEqual(b.n_components, 2)
        z_var = np.array([[0.5, 2.0]])
        mean, var = b.reconstruct(np.zeros((1, 2)), z_var)
        expected = b.scale ** 2 * ((b.W ** 2) @ z_var[0] + b.resid_var)
        np.testing.assert_allclose(var[0], expected)
        # noise propagation: isotropic per-point noise maps to the same variance on every score
        sig = np.full((1, p), 0.3)
        nz = b.project_noise(sig)
        np.testing.assert_allclose(nz, (0.3 / b.scale) ** 2, rtol=1e-10)

    def test_noise_rule_keeps_signal_components_only(self):
        rng = np.random.RandomState(4)
        p, k, n = 30, 4, 400
        W_true = np.linalg.qr(rng.randn(p, k))[0]
        Z = rng.randn(n, k) * np.array([4.0, 3.0, 2.0, 1.0])
        noise_std = 0.05
        Y = Z @ W_true.T + noise_std * rng.randn(n, p)
        learner = OutputBasisLearner(n_outputs=p, n_components='noise', min_points=20)
        for y in Y:
            learner.observe(y, sigma=noise_std)
        self.assertEqual(learner.current.n_components, k)
        # a fixed variance fraction that reaches into the noise keeps many more
        learner2 = OutputBasisLearner(n_outputs=p, n_components=0.9999, min_points=20)
        for y in Y:
            learner2.observe(y, sigma=noise_std)
        self.assertGreater(learner2.current.n_components, k)
        # and a cap is respected
        learner3 = OutputBasisLearner(n_outputs=p, n_components='noise', max_components=2, min_points=20)
        for y in Y:
            learner3.observe(y, sigma=noise_std)
        self.assertEqual(learner3.current.n_components, 2)

    def test_refit_schedule_and_versioning(self):
        rng = np.random.RandomState(2)
        learner = OutputBasisLearner(n_outputs=4, n_components=2, min_points=10, reservoir_size=50)
        versions = []
        for i in range(200):
            if learner.observe(rng.randn(4)):
                versions.append((i + 1, learner.current.version))
        # first fit at min_points, then when the count has doubled: 10, 20, 40, 80, 160
        self.assertEqual([n for n, _ in versions], [10, 20, 40, 80, 160])
        self.assertEqual([v for _, v in versions], [1, 2, 3, 4, 5])
        self.assertEqual(learner._reservoir.shape[0], 50)


class TestOutputModels(unittest.TestCase):

    def _data(self, n=120, p=8, seed=3):
        rng = np.random.RandomState(seed)
        X = rng.rand(n, 2)
        t = np.linspace(0, 1, p)
        # f(t; x) sampled on a t-grid: rank-3 structure plus a little noise
        Y = (np.sin(2 * np.pi * t)[None, :] * X[:, :1]
             + np.cos(2 * np.pi * t)[None, :] * X[:, 1:]
             + (t ** 2)[None, :] * (X[:, :1] * X[:, 1:]))
        Y = Y + 0.01 * rng.randn(*Y.shape)
        S = np.full(Y.shape, 0.01)
        return X, Y, S, t

    def test_shared_mode_fits_and_predicts(self):
        X, Y, S, _ = self._data()
        gpt = GPTree(GPR=Default_GPR(), Nbar=60, n_outputs=Y.shape[1], output_model='shared',
                     use_calibrated_sigma=False, splitting_strategy='standard')
        with contextlib.redirect_stdout(io.StringIO()):
            gpt.fit(X, Y, S)
        for leaf in gpt.root.leaves:
            self.assertEqual(len(leaf.my_GPRs), 1)
            self.assertIsNotNone(leaf.y_common_scaler)
        with contextlib.redirect_stdout(io.StringIO()):
            mu, sd = gpt.predict(X[:20])
        self.assertEqual(mu.shape, (20, Y.shape[1]))
        self.assertEqual(sd.shape, (20, Y.shape[1]))
        self.assertTrue(np.all(np.isfinite(mu)) and np.all(sd > 0))
        self.assertLess(np.sqrt(np.mean((mu - Y[:20]) ** 2)), 0.05)

    def test_shared_mode_requires_multitarget_backend(self):
        class NoMultitarget(Default_GPR().__class__):
            def supports_multitarget(self):
                return False
        gpr = Default_GPR()
        gpr.__class__ = NoMultitarget
        with self.assertRaises(ValueError):
            GPTree(GPR=gpr, n_outputs=3, output_model='shared')

    def test_pca_mode_end_to_end(self):
        X, Y, S, _ = self._data()
        p = Y.shape[1]
        gpt = GPTree(GPR=Default_GPR(), Nbar=60, n_outputs=p, output_model='pca',
                     output_basis_min_points=30,
                     use_calibrated_sigma=False, splitting_strategy='standard')
        with contextlib.redirect_stdout(io.StringIO()):
            gpt.fit(X, Y, S)
        basis = gpt.output_basis.current
        self.assertIsNotNone(basis)
        self.assertLess(basis.n_components, p)
        # the tree hands the per-point sigma to the basis learner
        self.assertTrue(np.all(np.isfinite(gpt.output_basis._reservoir_noise)))
        np.testing.assert_allclose(gpt.output_basis._reservoir_noise, 0.01 ** 2)
        for leaf in gpt.root.leaves:
            self.assertEqual(len(leaf.my_GPRs), basis.n_components)
            self.assertIs(leaf._fitted_basis, basis)
        with contextlib.redirect_stdout(io.StringIO()):
            mu, sd = gpt.predict(X[:20])
        self.assertEqual(mu.shape, (20, p))
        self.assertEqual(sd.shape, (20, p))
        self.assertTrue(np.all(np.isfinite(mu)) and np.all(sd > 0))
        self.assertLess(np.sqrt(np.mean((mu - Y[:20]) ** 2)), 0.05)

    def test_pca_mode_streaming_children_reconstruct_with_parent_basis(self):
        X, Y, S, _ = self._data(n=140)
        p = Y.shape[1]
        gpt = GPTree(GPR=Default_GPR(), Nbar=80, n_outputs=p, output_model='pca',
                     output_basis_min_points=20, retrain_every_n_points=40,
                     use_calibrated_sigma=False, splitting_strategy='standard')
        _stream(gpt, X, Y, S)
        self.assertFalse(gpt.root.is_leaf)
        # the basis was refit after the split (doubling schedule), so a child that has
        # not retrained yet must still carry the basis its inherited GPs were fit in
        versions = {leaf._fitted_basis.version for leaf in gpt.root.leaves}
        self.assertTrue(all(v >= 1 for v in versions))
        with contextlib.redirect_stdout(io.StringIO()):
            mu, sd = gpt.predict(X[:10])
        self.assertEqual(mu.shape, (10, p))
        self.assertTrue(np.all(np.isfinite(mu)))
        self.assertLess(np.sqrt(np.mean((mu - Y[:10]) ** 2)), 0.1)

    def test_pca_mode_before_basis_returns_prior_and_trains_later(self):
        X, Y, S, _ = self._data(n=40)
        p = Y.shape[1]
        gpt = GPTree(GPR=Default_GPR(), Nbar=100, n_outputs=p, output_model='pca',
                     output_basis_min_points=30, retrain_every_n_points=5,
                     use_calibrated_sigma=False)
        _stream(gpt, X[:20], Y[:20], S[:20])
        self.assertIsNone(gpt.output_basis.current)
        self.assertFalse(gpt.root._is_fitted())
        with contextlib.redirect_stdout(io.StringIO()):
            mu, sd = gpt.predict(X[:3])
        self.assertEqual(mu.shape, (3, p))
        _stream(gpt, X[20:], Y[20:], S[20:])
        self.assertIsNotNone(gpt.output_basis.current)
        self.assertTrue(gpt.root._is_fitted())

    def test_min_lengthscale_pools_over_outputs(self):
        # output 0 varies only along dim 0, output 1 only (and faster) along dim 1:
        # pooled length scales must pick dim 1 even though output 0 alone says dim 0.
        rng = np.random.RandomState(5)
        X = rng.rand(150, 2)
        Y = np.column_stack([np.sin(2 * np.pi * 0.7 * X[:, 0]),
                             np.sin(2 * np.pi * 2.5 * X[:, 1])])
        S = np.full(Y.shape, 1e-3)
        gpt = GPTree(GPR=Default_GPR(n_restarts_optimizer=2), Nbar=400, n_outputs=2,
                     split_dimension_criteria='min_lengthscale', use_calibrated_sigma=False)
        _stream(gpt, X, Y, S)
        with contextlib.redirect_stdout(io.StringIO()):
            gpt.root.fit_my_GPR(force_training=True)
            ls0 = gpt.root.my_GPRs[0].get_length_scales(2)
            gpt.root.compute_split_position_and_overlap(theta=1e-4)
        self.assertEqual(int(np.argmin(ls0)), 0)
        self.assertEqual(gpt.root.split_index, 1)

    def test_invalid_output_model(self):
        with self.assertRaises(ValueError):
            GPTree(n_outputs=2, output_model='nope')


if __name__ == '__main__':
    unittest.main()
