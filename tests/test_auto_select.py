"""Tests for AutoSelectGPR (automatic per-region baseline-vs-additive selection)."""
import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from pygptreeo import GPTree, AutoSelectGPR, make_auto_gpr
from pygptreeo.adapters import SklearnGPAdapter
from pygptreeo.kernels import make_order_additive_kernel


def _auto(d, margin=0.0, nr=2):
    base = SklearnGPAdapter(GaussianProcessRegressor(
        kernel=ConstantKernel(1.0, (1e-3, 1e8)) * Matern(
            nu=1.5, length_scale=[1.0] * d, length_scale_bounds=[(1e-5, 1e5)] * d),
        alpha=1e-6, n_restarts_optimizer=nr))
    add = SklearnGPAdapter(GaussianProcessRegressor(
        kernel=make_order_additive_kernel(d, 2, rescue=False), alpha=1e-6,
        n_restarts_optimizer=nr))
    return AutoSelectGPR(base, add, lml_margin=margin)


def _additive_data(d, n, seed):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, (n, d))
    y = (np.sin(3 * X[:, 0]) + 2 * X[:, 1] ** 2 + np.cos(4 * X[:, 2])
         + X[:, 3] + np.sin(5 * X[:, 4 % d]))
    return X, (y - y.mean()) / y.std()


def _high_order_data(d, n, seed):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, (n, d))
    y = np.prod(2 * X - 1, axis=1)   # mean-zero product: purely d-way interaction
    return X, (y - y.mean()) / y.std()


class TestSelection:
    def test_picks_additive_on_additive_data(self):
        g = _auto(5)
        g.fit(*_additive_data(5, 120, 0))
        assert g.verdict == "additive"
        assert g.last_lml[1] > g.last_lml[0]   # additive evidence higher

    def test_picks_baseline_on_pure_high_order(self):
        g = _auto(5)
        g.fit(*_high_order_data(5, 150, 1))
        assert g.verdict == "baseline"


class TestInheritance:
    def test_additive_commit_inherited_and_warm(self):
        g = _auto(5)
        g.fit(*_additive_data(5, 120, 0))
        assert g.verdict == "additive"
        child = g.clone()
        assert child._committed_additive          # additive verdict propagates
        assert child.is_trained()                 # warm start from parent

    def test_baseline_verdict_not_committed_but_warm(self):
        g = _auto(5)
        g.fit(*_high_order_data(5, 150, 1))
        assert g.verdict == "baseline"
        child = g.clone()
        assert not child._committed_additive       # children must re-test
        assert child.is_trained()                  # still warm

    def test_clone_before_fit_is_untrained(self):
        assert not _auto(5).clone().is_trained()

    def test_committed_child_skips_baseline_fit(self):
        """An additive-committed clone fits only the additive kernel."""
        g = _auto(5)
        g.fit(*_additive_data(5, 120, 0))
        child = g.clone()
        child.last_lml = None
        child.fit(*_additive_data(5, 120, 2))
        assert child.verdict == "additive"
        assert child.last_lml is None              # no comparison was run


class TestEndToEnd:
    def test_separable_target_selects_additive_in_tree(self):
        d = 5
        X, y = _additive_data(d, 1500, 0)
        gpt = GPTree(GPR=make_auto_gpr(d, lml_margin=0.1), Nbar=80, theta=1e-4,
                     retrain_every_n_points=80, use_standard_scaling=True,
                     splitting_strategy="gradual", split_dimension_criteria="max_variance")
        for xi, yi in zip(X, y):
            gpt.update_tree(xi.reshape(1, -1), np.array([[yi]]), 1e-6)
        verdicts = [l.my_GPRs[0].verdict for l in gpt.root.leaves]
        assert any(v == "additive" for v in verdicts)   # additive kernel adopted
        Xte = np.random.RandomState(9).uniform(0, 1, (50, d))
        yp, ys = gpt.predict(Xte)
        assert yp.shape == (50, 1) and np.all(np.isfinite(yp)) and np.all(ys >= 0)

    def test_default_tree_still_works_without_auto(self):
        """Sanity: a normal GPTree (no AutoSelectGPR) is unaffected."""
        from pygptreeo import Default_GPR
        gpt = GPTree(GPR=Default_GPR(alpha=1e-6), Nbar=50)
        X = np.random.RandomState(0).uniform(0, 1, (200, 3))
        for xi in X:
            gpt.update_tree(xi.reshape(1, -1), np.array([[float(xi.sum())]]), 1e-6)
        yp, _ = gpt.predict(X[:10])
        assert yp.shape == (10, 1)


def test_log_marginal_likelihood_adapter():
    g = SklearnGPAdapter(GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * Matern(nu=1.5), alpha=1e-6))
    assert g.log_marginal_likelihood() == float("-inf")     # untrained
    g.fit(np.random.RandomState(0).rand(20, 3), np.random.RandomState(1).rand(20))
    assert np.isfinite(g.log_marginal_likelihood())


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
