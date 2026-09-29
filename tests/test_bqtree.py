import unittest

import numpy as np

from pygptreeo.bqtree import BayesianPolynomialLeaf, BQTree, PolynomialFeatures


class TestPolynomialFeatures(unittest.TestCase):
    def test_feature_count(self):
        for d in (1, 2, 5):
            self.assertEqual(PolynomialFeatures(d, 0).p, 1)
            self.assertEqual(PolynomialFeatures(d, 1).p, 1 + d)
            self.assertEqual(PolynomialFeatures(d, 2).p, 1 + d + d * (d + 1) // 2)

    def test_transform_values(self):
        f = PolynomialFeatures(2, 2)
        phi = f.transform(np.array([[2.0, 3.0]]))[0]
        np.testing.assert_allclose(phi, [1, 2, 3, 4, 6, 9])


class TestBayesianPolynomialLeaf(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(1)
        d = 3
        A = self.rng.randn(d, d); self.A = A + A.T; self.b = self.rng.randn(d)
        self.f = lambda X: np.einsum('ni,ij,nj->n', X, self.A, X) + X @ self.b + 0.5
        self.X = self.rng.rand(80, d)
        self.y = self.f(self.X)
        self.s2 = np.full(80, 1e-8)

    def test_recovers_exact_quadratic(self):
        leaf = BayesianPolynomialLeaf(3)
        leaf.set_points(self.X, self.y, self.s2)
        Xt = self.rng.rand(20, 3)
        mu, sd = leaf.predict(Xt)
        np.testing.assert_allclose(mu[:, 0], self.f(Xt), rtol=1e-3, atol=1e-3)
        self.assertLess(leaf.tau2, 1e-4)
        self.assertTrue(np.all(sd > 0))

    def test_recursive_matches_batch(self):
        leaf = BayesianPolynomialLeaf(3, rebuild_every=10 ** 6, select_prior_scale=False)
        for i in range(self.X.shape[0]):
            leaf.add_point(self.X[i], self.y[i], self.s2[i])
        # Batch solve in the same frame, same prior and misfit
        batch = BayesianPolynomialLeaf(3, select_prior_scale=False)
        batch.features.centre = leaf.features.centre.copy(); batch.features.scale = leaf.features.scale.copy()
        batch.tau2 = leaf.tau2; batch.y_mean = leaf.y_mean.copy(); batch.y_scale = leaf.y_scale
        Phi = batch.features.transform(self.X)
        Yz = (self.y[:, None] - batch.y_mean) / batch.y_scale
        r = (self.s2 + batch.tau2) / batch.y_scale ** 2
        m, S, _ = batch._solve(Phi, Yz, r, batch._prior_diag())
        np.testing.assert_allclose(leaf.m, m, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(leaf.S, S, rtol=1e-5, atol=1e-8)

    def test_misfit_is_estimated(self):
        # a cubic term the quadratic cannot fit -> positive misfit variance
        X = self.rng.rand(80, 1); y = 5.0 * X[:, 0] ** 3
        leaf = BayesianPolynomialLeaf(1)
        leaf.set_points(X, y, np.full(80, 1e-10))
        self.assertGreater(leaf.tau2, 1e-6)
        mu, sd = leaf.predict(np.array([[0.5]]))
        self.assertGreaterEqual(sd[0, 0], np.sqrt(leaf.tau2))

    def test_uncertainty_grows_away_from_data(self):
        leaf = BayesianPolynomialLeaf(1)
        X = self.rng.rand(30, 1) * 0.2; y = np.sin(X[:, 0])
        leaf.set_points(X, y, np.full(30, 1e-6))
        _, sd_in = leaf.predict(np.array([[0.1]]))
        _, sd_out = leaf.predict(np.array([[2.0]]))
        self.assertGreater(sd_out[0, 0], 5 * sd_in[0, 0])


class TestBQTree(unittest.TestCase):
    def test_stream_learns_and_splits(self):
        rng = np.random.RandomState(2)
        f = lambda X: np.sin(3 * X[:, 0]) * np.cos(2 * X[:, 1]) + X[:, 0] ** 2
        t = BQTree(Nbar=40)
        X = rng.rand(600, 2); y = f(X)
        for i in range(600):
            t.update_tree(X[i:i + 1], y[i], 1e-3)
        self.assertGreater(len(t.leaves), 4)
        Xt = rng.rand(200, 2)
        mu, sd = t.predict(Xt)
        self.assertEqual(mu.shape, (200, 1)); self.assertEqual(sd.shape, (200, 1))
        nrmse = np.sqrt(np.mean((mu[:, 0] - f(Xt)) ** 2)) / (y.max() - y.min())
        self.assertLess(nrmse, 0.02)
        self.assertTrue(np.all(sd > 0))

    def test_loop_and_recursive_predict_agree(self):
        rng = np.random.RandomState(3)
        t = BQTree(Nbar=30, theta=0.1)
        X = rng.rand(200, 2); y = X[:, 0] * X[:, 1]
        for i in range(200):
            t.update_tree(X[i:i + 1], y[i], 1e-3)
        Xt = rng.rand(5, 2)
        m1, s1 = t.predict(Xt, mode='recursive'); m2, s2 = t.predict(Xt, mode='loop')
        np.testing.assert_allclose(m1, m2, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(s1, s2, rtol=1e-8, atol=1e-10)

    def test_multi_output(self):
        rng = np.random.RandomState(4)
        t = BQTree(Nbar=50, n_outputs=2)
        X = rng.rand(300, 2); Y = np.column_stack([X[:, 0] ** 2, X[:, 0] * X[:, 1]])
        for i in range(300):
            t.update_tree(X[i:i + 1], Y[i:i + 1], np.array([[1e-3, 1e-3]]))
        mu, sd = t.predict(rng.rand(10, 2))
        self.assertEqual(mu.shape, (10, 2)); self.assertEqual(sd.shape, (10, 2))


if __name__ == '__main__':
    unittest.main()
