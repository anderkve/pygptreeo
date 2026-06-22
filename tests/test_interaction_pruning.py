"""Tests for region-local interaction pruning.

Covers the discovery signal (PairInteractionScreen), the kernel-pruning helpers
(prune_additive_kernel / build_pruned_gpr), and the end-to-end GPTree wiring.
"""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor

from pygptreeo import GPTree
from pygptreeo.adapters import SklearnGPAdapter
from pygptreeo.interaction_screen import PairInteractionScreen
from pygptreeo.kernels import (make_additive_kernel, prune_additive_kernel,
                               build_pruned_gpr, build_warm_pruned_gpr,
                               AdditiveKernel)


def _find_additive(kernel):
    if isinstance(kernel, AdditiveKernel):
        return kernel
    p = kernel.get_params(deep=False)
    if "k1" in p:
        return _find_additive(kernel.k1) or _find_additive(kernel.k2)
    return None


# --------------------------------------------------------------------------- #
# PairInteractionScreen
# --------------------------------------------------------------------------- #
class TestInteractionScreen:
    def test_separable_target_has_no_active_pairs(self):
        """A purely additive (separable) target -> no pair clears threshold."""
        rng = np.random.RandomState(0)
        X = rng.uniform(0, 1, (4000, 4))
        y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + np.cos(4 * X[:, 2]) + X[:, 3]
        scr = PairInteractionScreen(4, warmup_points=0)
        for xi, yi in zip(X, y):
            scr.update(xi, yi)
        assert scr.active_pairs() == []

    def test_coupled_pair_is_detected(self):
        """A strong x0*x1 coupling -> pair (0,1) is the one kept."""
        rng = np.random.RandomState(1)
        X = rng.uniform(0, 1, (4000, 4))
        y = 5.0 * X[:, 0] * X[:, 1] + X[:, 2] + X[:, 3]
        scr = PairInteractionScreen(4, warmup_points=0)
        for xi, yi in zip(X, y):
            scr.update(xi, yi)
        assert (0, 1) in scr.active_pairs()

    def test_not_ready_returns_none(self):
        scr = PairInteractionScreen(4, warmup_points=1000)
        for xi in np.random.rand(50, 4):
            scr.update(xi, 0.0)
        assert not scr.is_ready()
        assert scr.active_pairs() is None

    def test_copy_is_independent(self):
        """A child's inherited copy keeps the parent's counts but diverges after."""
        scr = PairInteractionScreen(3, warmup_points=0)
        for xi in np.random.rand(100, 3):
            scr.update(xi, float(xi[0]))
        child = scr.copy()
        assert child.n_seen == scr.n_seen
        child.update(np.zeros(3), 1.0)
        assert child.n_seen == scr.n_seen + 1  # parent unaffected


# --------------------------------------------------------------------------- #
# Kernel pruning helpers
# --------------------------------------------------------------------------- #
class TestKernelPruning:
    def test_prune_reduces_terms_and_keeps_mains(self):
        k = make_additive_kernel(5, interaction_depth=2, rescue=True)
        assert _find_additive(k).n_terms == 5 + 10
        pk = prune_additive_kernel(k, [(0, 1), (2, 3)])
        assert _find_additive(pk).n_terms == 5 + 2  # 5 mains + 2 pairs

    def test_prune_to_empty_is_main_effects_only(self):
        k = make_additive_kernel(4, interaction_depth=2)
        pk = prune_additive_kernel(k, [])
        ak = _find_additive(pk)
        assert ak.n_terms == 4
        assert all(len(t) == 1 for t in ak.interaction_terms)

    def test_pruned_terms_survive_sklearn_clone(self):
        """sklearn clones the kernel on fit; pruned term set must persist."""
        k = prune_additive_kernel(make_additive_kernel(5, interaction_depth=2), [(0, 1)])
        assert _find_additive(clone(k)).n_terms == 5 + 1

    def test_prune_preserves_rescue_term(self):
        k = make_additive_kernel(4, interaction_depth=2, rescue=True)
        pk = prune_additive_kernel(k, [])
        # Rescue Matern still present -> kernel is a Sum with two operands.
        assert "k1" in pk.get_params(deep=False)

    def test_original_kernel_untouched(self):
        k = make_additive_kernel(4, interaction_depth=2)
        prune_additive_kernel(k, [(0, 1)])
        assert _find_additive(k).n_terms == 4 + 6  # original unchanged

    def test_build_pruned_gpr_is_untrained_and_fits_to_pruned(self):
        tmpl = SklearnGPAdapter(GaussianProcessRegressor(
            kernel=make_additive_kernel(5, interaction_depth=2, rescue=True), alpha=1e-6))
        X = np.random.RandomState(0).uniform(0, 1, (30, 5))
        tmpl.fit(X, np.random.rand(30))          # template is now trained
        g = build_pruned_gpr(tmpl, [(0, 1)])
        assert not g.is_trained()                # child born untrained
        assert _find_additive(g.get_kernel()).n_terms == 5 + 1
        g.fit(X, np.random.rand(30))             # fitting keeps the pruned set
        assert _find_additive(g.get_kernel()).n_terms == 5 + 1

    def test_build_warm_pruned_gpr_predicts_warm_then_fits_pruned(self):
        """The tree's child builder: warm (trained) now, pruned kernel on next fit."""
        tmpl = SklearnGPAdapter(GaussianProcessRegressor(
            kernel=make_additive_kernel(5, interaction_depth=2, rescue=True), alpha=1e-6))
        X = np.random.RandomState(0).uniform(0, 1, (30, 5))
        parent = tmpl.clone()
        parent.fit(X, np.random.rand(30))            # parent trained (full kernel)

        child = build_warm_pruned_gpr(parent, tmpl, [(0, 1)])
        assert child.is_trained()                    # warm: can predict immediately
        child.predict(X, return_std=True)            # works without an extra fit
        # The kernel queued for the child's NEXT fit is pruned...
        assert _find_additive(child._gpr.kernel).n_terms == 5 + 1
        # ...and after it retrains, its fitted kernel is the pruned one.
        child.fit(X, np.random.rand(30))
        assert _find_additive(child.get_kernel()).n_terms == 5 + 1

    def test_reset_training(self):
        g = SklearnGPAdapter(GaussianProcessRegressor(
            kernel=make_additive_kernel(3, interaction_depth=2), alpha=1e-6))
        g.fit(np.random.rand(20, 3), np.random.rand(20))
        assert g.is_trained()
        g.reset_training()
        assert not g.is_trained()


# --------------------------------------------------------------------------- #
# End-to-end GPTree wiring
# --------------------------------------------------------------------------- #
class TestEndToEnd:
    def _make_tree(self, prune, d=4):
        gpr = SklearnGPAdapter(GaussianProcessRegressor(
            kernel=make_additive_kernel(d, interaction_depth=2, rescue=True),
            alpha=1e-6, n_restarts_optimizer=0))
        return GPTree(GPR=gpr, Nbar=60, theta=1e-4, retrain_every_n_points=40,
                      use_standard_scaling=True, splitting_strategy="gradual",
                      split_dimension_criteria="max_variance",
                      prune_interactions=prune, interaction_warmup=120)

    def test_separable_target_prunes_leaf_kernels(self):
        """On a separable target the tree should drive leaves toward main-effects."""
        d = 4
        rng = np.random.RandomState(0)
        X = rng.uniform(0, 1, (1500, d))
        y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + np.cos(4 * X[:, 2]) + X[:, 3]
        gpt = self._make_tree(prune=True, d=d)
        for xi, yi in zip(X, y):
            gpt.update_tree(xi.reshape(1, -1), np.array([[yi]]), 1e-6)
        terms = [_find_additive(l.my_GPRs[0].get_kernel()).n_terms
                 for l in gpt.root.leaves]
        full = d + d * (d - 1) // 2
        assert min(terms) < full           # at least some leaf was pruned
        # Pruned leaves use only main effects (d terms) on a separable target.
        assert any(t == d for t in terms)

    def test_pruning_does_not_break_predictions(self):
        """Pruned tree predicts finite values of correct shape, no worse-by-much."""
        d = 4
        rng = np.random.RandomState(2)
        X = rng.uniform(0, 1, (1200, d))
        y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + np.cos(4 * X[:, 2]) + X[:, 3]
        Xte = rng.uniform(0, 1, (300, d))
        yte = np.sin(3 * Xte[:, 0]) + Xte[:, 1] ** 2 + np.cos(4 * Xte[:, 2]) + Xte[:, 3]

        preds = {}
        for prune in (False, True):
            np.random.seed(7)
            gpt = self._make_tree(prune=prune, d=d)
            for xi, yi in zip(X, y):
                gpt.update_tree(xi.reshape(1, -1), np.array([[yi]]), 1e-6)
            yp, ys = gpt.predict(Xte)
            assert yp.shape == (300, 1) and np.all(np.isfinite(yp))
            preds[prune] = np.sqrt(np.mean((yp[:, 0] - yte) ** 2))
        # On a separable target, pruning must not materially hurt accuracy.
        assert preds[True] <= 1.5 * preds[False]

    def test_default_off_is_unchanged(self):
        """prune_interactions defaults off -> no screen is created."""
        gpt = self._make_tree(prune=False)
        gpt.update_tree(np.zeros((1, 4)), np.array([[0.0]]), 1e-6)
        assert gpt.root.interaction_screen is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
