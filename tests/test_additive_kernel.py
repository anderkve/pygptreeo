"""Tests for the additive leaf kernel factory `make_additive_kernel`."""

import unittest
import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from pygptreeo import GPTree, make_additive_kernel
from pygptreeo.adapters import SklearnGPAdapter


def fit_gpr(kernel, X, y):
    gpr = SklearnGPAdapter(GaussianProcessRegressor(
        kernel=kernel, alpha=1e-6, n_restarts_optimizer=0))
    gpr.fit(X, y.reshape(-1, 1))
    return gpr


class TestMakeAdditiveKernel(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        rng = np.random.RandomState(0)
        self.nd = 4
        self.X = rng.uniform(0, 1, (50, self.nd))
        # An additive-with-pairwise target.
        self.y = (np.sum(np.sin(2 * np.pi * self.X), axis=1)
                  + self.X[:, 0] * self.X[:, 1])

    def test_rescue_adds_a_term(self):
        k_no = make_additive_kernel(self.nd, rescue=False)
        k_yes = make_additive_kernel(self.nd, rescue=True)
        # The rescued kernel is a Sum (two top-level terms); the bare one is not.
        self.assertNotIn("+", repr(k_no))
        self.assertIn("+", repr(k_yes))

    def test_fits_and_predicts(self):
        for rescue in (False, True):
            gpr = fit_gpr(make_additive_kernel(self.nd, interaction_depth=2, rescue=rescue),
                          self.X, self.y)
            self.assertTrue(gpr.is_trained())
            mu, sd = gpr.predict(self.X[:5], return_std=True)
            self.assertEqual(mu.shape[0], 5)
            self.assertTrue(np.all(np.isfinite(mu)))
            self.assertTrue(np.all(sd >= 0))

    def test_length_scales_exposed_for_split_criterion(self):
        # The 'min_lengthscale' split criterion needs per-dimension length scales.
        for rescue in (False, True):
            gpr = fit_gpr(make_additive_kernel(self.nd, rescue=rescue), self.X, self.y)
            ls = gpr.get_length_scales(self.nd)
            self.assertIsNotNone(ls)
            self.assertEqual(ls.shape, (self.nd,))
            self.assertTrue(np.all(ls > 0))

    def test_depth1_has_no_pairwise_terms(self):
        k = make_additive_kernel(3, interaction_depth=1, rescue=False)
        # ConstantKernel * AdditiveKernel -> the additive factor is k2.
        add = k.k2
        self.assertEqual(add.n_terms, 3)  # only main effects
        k2 = make_additive_kernel(3, interaction_depth=2, rescue=False).k2
        self.assertEqual(k2.n_terms, 3 + 3)  # 3 mains + C(3,2)=3 pairs

    def test_integration_with_gptree_stream(self):
        nd = 3
        rng = np.random.RandomState(1)
        X = rng.uniform(0, 1, (400, nd))
        y = np.sum(np.sin(2 * np.pi * X), axis=1)
        kernel = make_additive_kernel(nd, interaction_depth=2, rescue=True)
        gpr = SklearnGPAdapter(GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, n_restarts_optimizer=0))
        gpt = GPTree(GPR=gpr, Nbar=80, theta=1e-4, retrain_every_n_points=40,
                     use_standard_scaling=True, splitting_strategy="gradual",
                     max_n_pred_leaves=3, aggregation="moe",
                     split_dimension_criteria="min_lengthscale")
        for xi, yi in zip(X, y):
            gpt.update_tree(xi.reshape(1, -1), np.array([[yi]]), 1e-6)
        yp, ys = gpt.predict(X[:20])
        self.assertEqual(yp.shape, (20, 1))
        self.assertTrue(np.all(np.isfinite(yp)))
        self.assertGreater(len(gpt.root.leaves), 1)


if __name__ == "__main__":
    unittest.main()
