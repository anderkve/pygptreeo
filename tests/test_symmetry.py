"""Tests for pygptreeo.symmetry: Symmetry, SymmetrizedKernel and GPTree(symmetry=...)."""

import io
import contextlib
import os
import sys
import tempfile
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel

from pygptreeo import GPTree, Default_GPR, Symmetry, SymmetrizedKernel
from pygptreeo.symmetry import _pair_eval


def flip1(X):
    Y = X.copy(); Y[:, 1] = 1 - X[:, 1]; return Y


def flip02(X):
    Y = X.copy(); Y[:, 0] = 1 - X[:, 0]; Y[:, 2] = 1 - X[:, 2]; return Y


def shift0(X):
    Y = X.copy(); Y[:, 0] += 1.0; return Y


def f_sym(X):
    """Symmetric under x1 -> 1 - x1."""
    return np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1]) + X[:, 2]


def f_asym(X):
    return np.sin(2 * np.pi * X[:, 0]) * np.sin(2 * np.pi * X[:, 1]) + X[:, 2]


def f_perm(X):
    """Symmetric under x0 <-> x1."""
    return np.sin(2 * np.pi * (X[:, 0] + X[:, 1])) + X[:, 0] * X[:, 1] + 0.3 * X[:, 2]


class TestSymmetry(unittest.TestCase):

    def test_closure_and_orbits(self):
        sym = Symmetry(generators=[flip1, flip02], n_features=3)
        self.assertEqual(sym.n_elements, 4)
        self.assertIsNone(sym.orbits)
        s3 = Symmetry(permutations=[[1, 0, 2], [0, 2, 1]])
        self.assertEqual(s3.n_elements, 6)
        self.assertEqual(s3.orbits, [[0, 1, 2]])
        swap = Symmetry(permutations=[[1, 0, 2, 3]])
        self.assertEqual(swap.n_elements, 2)
        self.assertEqual(swap.orbits, [[0, 1], [2], [3]])
        with self.assertRaises(ValueError):
            Symmetry(generators=[shift0], n_features=2, max_order=50).n_elements

    def test_canonical(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(size=(50, 3))
        sym = Symmetry(generators=[flip1, flip02])     # dimension inferred from X
        C = sym.canonical(X)
        for g in sym.elements():
            np.testing.assert_allclose(sym.canonical(g(X)), C)
        np.testing.assert_allclose(sym.canonical(C), C)
        self.assertTrue((C[:, 0] <= 0.5 + 1e-12).all() and (C[:, 1] <= 0.5 + 1e-12).all())
        s3 = Symmetry(permutations=[[1, 0, 2], [0, 2, 1]])
        Cs = s3.canonical(X)
        self.assertTrue((np.diff(Cs, axis=1) >= -1e-12).all())

    def test_periodic(self):
        sym = Symmetry.periodic(dims=[1], periods=2.0, n_images=1)
        self.assertEqual(sym.n_elements, 3)
        X = np.array([[0.3, 2.5], [0.1, -0.5]])
        np.testing.assert_allclose(sym.canonical(X), [[0.3, 0.5], [0.1, 1.5]])
        imgs = sym.images(X)
        self.assertEqual(imgs.shape, (3, 2, 2))
        self.assertEqual(sorted(imgs[:, 0, 1].tolist()), [0.5, 2.5, 4.5])


class TestSymmetrizedKernel(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(1)
        self.X = rng.uniform(size=(7, 3))
        self.Y = rng.uniform(size=(5, 3))

    def test_pair_eval_matches_stacked(self):
        X, Y = self.X, self.Y
        for k0 in [ConstantKernel(2.0) * RBF(length_scale=[0.3, 0.5, 0.9]),
                   ConstantKernel(2.0) * Matern(nu=1.5, length_scale=[0.3, 0.5, 0.9]),
                   Matern(nu=2.5, length_scale=0.4), Matern(nu=0.5, length_scale=[0.3, 0.5, 0.9]) + WhiteKernel(0.1),
                   ConstantKernel(2.0, "fixed") * RBF(length_scale=[0.3, 0.5, 0.9])]:
            K, G = _pair_eval(k0, X, Y)
            Kz, Gz = k0(np.vstack([X, Y]), eval_gradient=True)
            np.testing.assert_allclose(K, Kz[:7, 7:])
            np.testing.assert_allclose(G, Gz[:7, 7:])

    def _check_gradient(self, k):
        X = self.X
        K, G = k(X, eval_gradient=True)
        t0 = k.theta.copy()
        for i in range(t0.size):
            e = np.eye(t0.size)[i] * 1e-6
            num = (k.clone_with_theta(t0 + e)(X) - k.clone_with_theta(t0 - e)(X)) / 2e-6
            np.testing.assert_allclose(G[:, :, i], num, atol=1e-6, rtol=1e-5)
        return K

    def test_gradient_invariance_diag(self):
        X, Y = self.X, self.Y
        sym = Symmetry(generators=[flip1, flip02], n_features=3)
        s3 = Symmetry(permutations=[[1, 0, 2], [0, 2, 1]])
        swap = Symmetry(permutations=[[1, 0, 2]])
        for k0, s in [(ConstantKernel(2.0) * RBF(length_scale=[0.3, 0.5, 0.9]), sym),
                      (ConstantKernel(2.0) * Matern(nu=1.5, length_scale=[0.5, 0.5, 0.5]), s3),
                      (ConstantKernel(2.0) * RBF(length_scale=[0.3, 0.3, 0.9]) + WhiteKernel(0.01), swap),
                      (Matern(nu=2.5, length_scale=0.4), sym)]:
            k = SymmetrizedKernel(k0, s, alpha=0.7)
            K = self._check_gradient(k)
            np.testing.assert_allclose(K, K.T)
            np.testing.assert_allclose(np.diag(K), k.diag(X))
            k1 = clone(k); k1.alpha = 1.0
            for g in s.elements():
                np.testing.assert_allclose(k1(X, Y), k1(g(X), Y))
                np.testing.assert_allclose(k1(X, Y), k1(X, g(Y)))
            k0c = clone(k); k0c.alpha = 1e-9
            np.testing.assert_allclose(k0c(X, Y), k0(X, Y), atol=1e-7)
            kc = k.clone_with_theta(k.theta + 0.1)
            np.testing.assert_allclose(kc.theta, k.theta + 0.1)

    def test_hyperparameters_and_tying(self):
        swap = Symmetry(permutations=[[1, 0, 2, 3]])
        k = SymmetrizedKernel(ConstantKernel(2.0) * RBF(length_scale=[0.3, 0.5, 0.9, 1.3]), swap, alpha=0.5)
        names = [(h.name, h.n_elements) for h in k.hyperparameters]
        self.assertEqual(names, [("kernel__k1__constant_value", 1), ("kernel__k2__length_scale", 3), ("alpha", 1)])
        self.assertEqual(k.theta.size, 5)
        self.assertEqual(k.bounds.shape, (5, 2))
        k(self.X[:, :4] if self.X.shape[1] >= 4 else np.hstack([self.X, self.X[:, :1]]))
        np.testing.assert_allclose(k.kernel.k2.length_scale, [0.3, 0.3, 0.9, 1.3])   # tied to the orbit's first entry
        # alpha bounds (0, 1): the lower bound is the floor, the upper bound 1
        np.testing.assert_allclose(np.exp(k.bounds[-1]), [1e-6, 1.0])
        kf = SymmetrizedKernel(RBF(length_scale=np.ones(4)), swap, alpha=0.5, alpha_bounds="fixed")
        self.assertEqual(kf.theta.size, 3)
        self.assertEqual(kf.n_dims, 3)
        untied = SymmetrizedKernel(RBF(length_scale=[0.3, 0.5, 0.9, 1.3]), swap, tie_length_scales=False)
        self.assertEqual(untied.theta.size, 5)

    def test_input_transform(self):
        X, Y = self.X, self.Y
        sym = Symmetry(generators=[flip1, flip02], n_features=3)
        k = SymmetrizedKernel(ConstantKernel(2.0) * RBF(length_scale=[0.3, 0.5, 0.9]), sym, alpha=0.6)
        shift, scale = X.mean(axis=0), X.std(axis=0)
        kt = clone(k)
        kt.set_params(input_transform=(shift, scale), kernel__k2__length_scale=np.array([0.3, 0.5, 0.9]) / scale)
        np.testing.assert_allclose(kt((X - shift) / scale, (Y - shift) / scale), k(X, Y))

    def test_gp_fit_learns_alpha(self):
        rng = np.random.default_rng(2)
        Xt = rng.uniform(size=(200, 3))
        sym = Symmetry(generators=[flip1], n_features=3)
        alphas = {}
        for f in (f_sym, f_asym):
            k = SymmetrizedKernel(ConstantKernel(1.0) * RBF(length_scale=0.5 * np.ones(3), length_scale_bounds=(1e-2, 1e2)),
                                  sym, alpha=0.9)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp = GaussianProcessRegressor(kernel=k, alpha=1e-8, normalize_y=True).fit(Xt, f(Xt))
            alphas[f.__name__] = gp.kernel_.alpha
        self.assertGreater(alphas["f_sym"], 0.8)
        self.assertLess(alphas["f_asym"], 0.5)


def _tree(symmetry, mode, alpha=0.9, alpha_bounds=(0.0, 1.0), Nbar=60, **kw):
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(3))
    if symmetry is not None and mode == 'mixed':
        kernel = SymmetrizedKernel(kernel, symmetry, alpha=alpha, alpha_bounds=alpha_bounds)
    return GPTree(GPR=Default_GPR(kernel=kernel), Nbar=Nbar, retrain_every_n_points=20,
                  symmetry=symmetry, symmetry_mode=mode, **kw)


class TestGPTreeSymmetry(unittest.TestCase):

    def _stream(self, gpt, X, y, sigma=1e-3):
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(len(X)):
                gpt.update_tree(X[i:i + 1], y[i:i + 1].reshape(1, 1), sigma)

    def test_mixed_routing_and_invariance(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(size=(400, 3)); y = f_sym(X)
        sym = Symmetry(generators=[flip1], n_features=3)
        gpt = _tree(sym, 'mixed', alpha=1.0, alpha_bounds="fixed")
        self._stream(gpt, X, y)
        self.assertGreater(len(gpt.root.leaves), 3)
        # every leaf's region is a box in canonical coordinates: a point and its image share the leaves
        Xe = rng.uniform(size=(20, 3))
        for x in Xe:
            _, _, names = gpt.predict(x[None, :], return_leaf_names=True)
            _, _, names_g = gpt.predict(flip1(x[None, :]), return_leaf_names=True)
            self.assertEqual(sorted(names), sorted(names_g))
        # with alpha = 1 the prediction is exactly symmetric
        mu, sd = gpt.predict(Xe); mu_g, sd_g = gpt.predict(flip1(Xe))
        np.testing.assert_allclose(mu, mu_g, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(sd, sd_g, rtol=1e-6, atol=1e-8)
        mu_l, _ = gpt.predict(Xe, mode='loop')
        np.testing.assert_allclose(mu_l, mu, rtol=1e-8, atol=1e-8)

    def test_mixed_generalizes_across_the_symmetry(self):
        """Trained on x1 < 0.5 only, the symmetric prior predicts the other half."""
        rng = np.random.default_rng(4)
        X = rng.uniform(size=(300, 3)); X[:, 1] *= 0.5; y = f_sym(X)
        Xe = rng.uniform(size=(200, 3)); Xe[:, 1] = 0.5 + 0.5 * Xe[:, 1]
        sym = Symmetry(generators=[flip1], n_features=3)
        errs = {}
        for name, gpt in [("plain", _tree(None, 'mixed')), ("mixed", _tree(sym, 'mixed')), ("fold", _tree(sym, 'fold'))]:
            self._stream(gpt, X, y)
            mu, _ = gpt.predict(Xe)
            errs[name] = np.sqrt(np.mean((mu.ravel() - f_sym(Xe)) ** 2))
        self.assertLess(errs["mixed"], 0.2 * errs["plain"])
        self.assertLess(errs["fold"], 0.2 * errs["plain"])

    def test_fitted_alpha_and_tied_scaler(self):
        rng = np.random.default_rng(5)
        X = rng.uniform(size=(300, 3))
        sym = Symmetry(permutations=[[1, 0, 2]])
        gpt = _tree(sym, 'mixed')
        self._stream(gpt, X, f_perm(X))
        alphas, scales = [], []
        for leaf in gpt.root.leaves:
            if leaf.my_GPR.is_trained():
                alphas.append(leaf.my_GPR.get_kernel().alpha)
                scales.append(leaf.X_scaler.scale_)
        self.assertTrue(alphas)
        self.assertGreater(np.median(alphas), 0.7)
        for s in scales:
            self.assertEqual(s[0], s[1])
        Xe = rng.uniform(size=(30, 3))
        mu, _ = gpt.predict(Xe); mu_g, _ = gpt.predict(Xe[:, [1, 0, 2]])
        self.assertLess(np.max(np.abs(mu - mu_g)), 0.05)
        gpt_a = _tree(sym, 'mixed')
        self._stream(gpt_a, X, f_asym(X[:, [1, 0, 2]]) if False else np.sin(2 * np.pi * X[:, 0]) * X[:, 1] ** 2)
        a_asym = [l.my_GPR.get_kernel().alpha for l in gpt_a.root.leaves if l.my_GPR.is_trained()]
        self.assertLess(np.median(a_asym), 0.5)

    def test_splits_cover_an_orbit(self):
        """With tied length scales the split must still alternate between exchangeable inputs."""
        rng = np.random.default_rng(7)
        X = rng.uniform(size=(2500, 3))
        y = np.sin(4 * np.pi * X[:, 0]) + np.sin(4 * np.pi * X[:, 1]) + 0.1 * X[:, 2]
        gpt = _tree(Symmetry(permutations=[[1, 0, 2]]), 'mixed', Nbar=100)
        self._stream(gpt, X, y)
        counts = {0: 0, 1: 0, 2: 0}

        def walk(node):
            if node.is_leaf:
                return
            counts[int(node.split_index)] += 1
            walk(node.left); walk(node.right)
        walk(gpt.root)
        self.assertGreater(counts[0], 0)
        self.assertGreater(counts[1], 0)
        self.assertLess(counts[2], counts[0] + counts[1])

    def test_warning_and_save(self):
        sym = Symmetry(generators=[flip1], n_features=3)
        kernel = SymmetrizedKernel(ConstantKernel(1.0) * RBF(length_scale=np.ones(3)), sym)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with contextlib.redirect_stdout(io.StringIO()):
                GPTree(GPR=Default_GPR(kernel=kernel), Nbar=50)
        self.assertTrue(any("SymmetrizedKernel" in str(x.message) for x in w))
        with self.assertRaises(ValueError):
            GPTree(symmetry=sym, symmetry_mode='nope')
        rng = np.random.default_rng(6)
        X = rng.uniform(size=(120, 3))
        gpt = _tree(sym, 'mixed')
        self._stream(gpt, X, f_sym(X))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tree.joblib")
            gpt.save(path)
            loaded = joblib.load(path)
        Xe = rng.uniform(size=(10, 3))
        np.testing.assert_allclose(loaded.predict(Xe)[0], gpt.predict(Xe)[0])


if __name__ == '__main__':
    unittest.main()
