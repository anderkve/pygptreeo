"""Tests for ``GPTree(max_memory_mb=...)``: at the limit the tree puts the leaves whose
GPs were unused for the longest time to sleep, and then removes the leaves that
have gone longest without a point."""

import os
import sys
import unittest
import warnings

import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo import GPTree, Default_GPR
from pygptreeo import gptree as gptree_module

warnings.filterwarnings("ignore")

NBAR = 60


def _tree(d=2, **kw):
    """A tree whose leaf GPs have fixed hyperparameters, so a fit is one factorisation."""
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=[1.0] * d, length_scale_bounds="fixed")
    kw.setdefault('Nbar', NBAR)
    return GPTree(GPR=Default_GPR(kernel=kernel), **kw)


def _target(X):
    return np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2


def _sweep(n, rng, length=10.0, width=0.3):
    """A cluster of 2D points that moves along the first input."""
    X = width * rng.randn(n, 2)
    X[:, 0] += np.linspace(0.0, length, n)
    return X


def _stream(gpt, X, predict=False):
    """Feed the points to the tree; with ``predict``, the prediction made for each one first."""
    y = _target(X)
    predictions = []
    for i in range(X.shape[0]):
        if predict:
            mu, sd = gpt.predict(X[i:i + 1])
            predictions.append((mu[0, 0], sd[0, 0]))
        gpt.update_tree(X[i:i + 1], y[i:i + 1, None], 1e-3)
    return np.array(predictions)


def _uniform(n, seed):
    return 4.0 * np.random.RandomState(seed).rand(n, 2)


def _scan(gpt):
    """Counted afresh: the size of the leaves, the size of what the GPs awake hold,
    and the number of leaves."""
    leaves = gpt.root.leaves
    overhead = gptree_module.NODE_OVERHEAD_BYTES + gptree_module.GPR_OVERHEAD_BYTES * len(leaves[0].my_GPRs)
    return (sum(leaf.nbytes_points() + leaf.nbytes_fitted() + overhead for leaf in leaves),
            sum(leaf.nbytes_fitted() for leaf in leaves if not leaf.asleep), len(leaves))


class TreeChecks(unittest.TestCase):

    def assertConsistent(self, gpt):
        """Links, names and split information agree with the node positions."""
        root = gpt.root
        self.assertIsNone(root.parent); self.assertIsNone(root.is_left); self.assertEqual(root.name, "0")
        inner = [node for node in root if not node.is_leaf]
        self.assertEqual(len(inner), len(root.leaves) - 1)
        for node in inner:
            self.assertEqual(node.children, [node.left, node.right])
            for side, child in enumerate(node.children):
                self.assertIs(child.parent, node)
                self.assertEqual(child.is_left, side == 0)
                self.assertEqual(child.name, node.name + str(side))
        for leaf in root.leaves:
            self.assertIsNone(leaf.children)

    def assertAccountsMatch(self, gpt):
        """The running accounts agree with a fresh count over the leaves."""
        size, awake, n_leaves = _scan(gpt)
        self.assertEqual((gpt._leaf_bytes, gpt._fitted_bytes, len(gpt._leaves)), (size, awake, n_leaves))
        self.assertEqual(set(map(id, gpt._leaves)), set(map(id, gpt.root.leaves)))
        self.assertEqual(set(map(id, gpt._fitted)),
                         {id(leaf) for leaf in gpt.root.leaves if leaf.nbytes_fitted() > 0 and not leaf.asleep})
        self.assertAlmostEqual(gpt.memory_mb, (gpt._fixed_bytes + size) / 1e6)


class TestMemoryLimit(TreeChecks):

    def test_memory_stays_within_the_limit(self):
        np.random.seed(0)
        X = _sweep(2500, np.random.RandomState(0))
        gpt = _tree(max_memory_mb=0.6)
        y = _target(X)
        for i in range(X.shape[0]):
            gpt.predict(X[i:i + 1])
            self.assertLessEqual(gpt.memory_mb, 0.6)
            gpt.update_tree(X[i:i + 1], y[i:i + 1, None], 1e-3)
            self.assertLessEqual(gpt.memory_mb, 0.6)
        self.assertAccountsMatch(gpt)
        self.assertConsistent(gpt)
        asleep = sum(leaf.asleep for leaf in gpt.root.leaves)
        self.assertGreater(asleep, 0)
        self.assertGreater(len(gpt.root.leaves) - asleep, 0)

        np.random.seed(0)
        unlimited = _tree()
        _stream(unlimited, X)
        self.assertGreater(unlimited.memory_mb, 3 * 0.6)
        self.assertGreater(len(unlimited.root.leaves), 3 * len(gpt.root.leaves))
        self.assertIsNone(unlimited.max_memory_mb)

    def test_sleeping_alone_changes_no_prediction(self):
        """A limit that the tree can meet by putting leaves to sleep costs nothing but the time to wake them."""
        X = _uniform(1500, 1)
        np.random.seed(1)
        unlimited = _tree(Nbar=80, retrain_every_n_points=1)
        reference = _stream(unlimited, X, predict=True)
        np.random.seed(1)
        gpt = _tree(Nbar=80, retrain_every_n_points=1, max_memory_mb=0.7 * unlimited.memory_mb)
        predictions = _stream(gpt, X, predict=True)
        np.testing.assert_allclose(predictions, reference, atol=1e-8)
        self.assertEqual(len(gpt.root.leaves), len(unlimited.root.leaves))
        self.assertGreater(sum(leaf.asleep for leaf in gpt.root.leaves), 5)
        self.assertLessEqual(gpt.memory_mb, 0.7 * unlimited.memory_mb)
        self.assertAccountsMatch(gpt)

    def test_leaves_are_removed_once_the_fitted_ones_take_half(self):
        np.random.seed(2)
        gpt = _tree(max_memory_mb=0.5)
        _stream(gpt, _uniform(2400, 2))
        self.assertLess(sum(leaf.n_points for leaf in gpt.root.leaves), 1200)      # leaves were removed
        one_gp = max(gpt._fitted.values())
        self.assertLessEqual(gpt._fitted_bytes, gptree_module.FITTED_SHARE * 0.5e6 + one_gp)
        self.assertGreater(gpt._fitted_bytes, gptree_module.FITTED_SHARE * 0.5e6 - 3 * one_gp)
        self.assertAccountsMatch(gpt)
        self.assertConsistent(gpt)

    def test_a_sweeping_stream_keeps_the_region_it_is_in(self):
        """With enough memory for the moving cluster, the tree is as accurate there as one without a limit."""
        rng = np.random.RandomState(3)
        X = _sweep(2400, rng)
        X_front = X[-100:] + 0.02 * rng.randn(100, 2)
        rmse = {}
        for limit in (0.5, None):
            np.random.seed(3)
            gpt = _tree(max_memory_mb=limit, retrain_every_n_points=5)
            _stream(gpt, X)
            mu, _ = gpt.predict(X_front)
            rmse[limit] = np.sqrt(np.mean((mu[:, 0] - _target(X_front)) ** 2))
            if limit is not None:
                stored = np.vstack([leaf.my_X_data for leaf in gpt.root.leaves])
                self.assertGreater(stored[:, 0].min(), 5.0)
        self.assertLess(rmse[0.5], 1.1 * rmse[None])

    def test_predictions_remove_no_leaf(self):
        np.random.seed(4)
        gpt = _tree(max_memory_mb=0.5)
        X = _uniform(2000, 4)
        _stream(gpt, X)
        n_leaves = len(gpt.root.leaves)
        self.assertGreater(sum(leaf.asleep for leaf in gpt.root.leaves), 2)
        X_test = _uniform(300, 5)
        mu, sd = gpt.predict(X_test)                             # reaches every leaf
        self.assertLessEqual(gpt.memory_mb, 0.5)
        mu_loop, sd_loop = gpt.predict(X_test, mode='loop')
        self.assertLessEqual(gpt.memory_mb, 0.5)
        np.testing.assert_allclose(mu, mu_loop, atol=1e-8)
        np.testing.assert_allclose(sd, sd_loop, atol=1e-8)
        self.assertEqual(len(gpt.root.leaves), n_leaves)
        self.assertAccountsMatch(gpt)

    def test_a_gp_used_for_predictions_stays_fitted(self):
        np.random.seed(5)
        gpt = _tree(max_memory_mb=0.5)
        _stream(gpt, _uniform(2000, 6))
        leaf = next(leaf for leaf in gpt.root.leaves if leaf.asleep)
        x = leaf.my_X_data[:1]
        gpt.predict(x)
        self.assertFalse(leaf.asleep)
        self.assertIs(next(reversed(gpt._fitted)), leaf)         # the most recently used
        self.assertIsNot(next(reversed(gpt._leaves)), leaf)      # but not the last a point reached

    def test_batch_fit_respects_the_limit(self):
        np.random.seed(6)
        X = _uniform(1500, 7)
        gpt = _tree(max_memory_mb=0.5)
        gpt.fit(X, _target(X), np.full(1500, 1e-3))
        self.assertLessEqual(gpt.memory_mb, 0.5)
        self.assertAccountsMatch(gpt)
        self.assertConsistent(gpt)
        mu, sd = gpt.predict(X[:50])
        self.assertTrue(np.all(np.isfinite(mu)) and np.all(np.isfinite(sd)))

    def test_a_saved_tree_carries_on(self):
        import tempfile
        np.random.seed(11)
        gpt = _tree(max_memory_mb=0.5)
        X = _uniform(1600, 12)
        _stream(gpt, X[:1200])
        with tempfile.TemporaryDirectory() as folder:
            gpt.save(os.path.join(folder, "tree.pkl"))
            loaded = GPTree.load(os.path.join(folder, "tree.pkl"))
        for tree in (gpt, loaded):
            self.assertAccountsMatch(tree)
            self.assertEqual(tree._fitted_bytes, 0)         # every leaf was put to sleep
        np.testing.assert_array_equal(loaded.predict(X[:30])[0], gpt.predict(X[:30])[0])
        _stream(loaded, X[1200:])
        self.assertLessEqual(loaded.memory_mb, 0.5)
        self.assertAccountsMatch(loaded)
        self.assertConsistent(loaded)

    def test_memory_estimate_without_a_limit(self):
        np.random.seed(7)
        gpt = _tree()
        self.assertEqual(gpt.memory_mb, 0.0)
        _stream(gpt, _uniform(400, 8))
        self.assertAlmostEqual(gpt.memory_mb, _scan(gpt)[0] / 1e6)

    def test_invalid_limits(self):
        for value in (0, -1.0):
            with self.assertRaises(ValueError):
                GPTree(max_memory_mb=value)
        gpt = _tree(max_memory_mb=0.02)                          # less than one leaf with its points
        with self.assertRaisesRegex(ValueError, "too small for a single leaf"):
            _stream(gpt, _uniform(200, 9))

    def test_the_global_model_counts(self):
        kwargs = dict(reservoir_size=300, min_points=40, n_restarts_optimizer=0)
        fixed_mb = 8 * 300 * (301 + 2 + 2 + 300 + 2 + 2) / 1e6   # the reservoir and the GP fitted on it
        gpt = GPTree(Nbar=NBAR, max_memory_mb=0.9 * fixed_mb, global_mean='additive_gp', global_mean_kwargs=kwargs)
        X = _uniform(300, 10)
        with self.assertRaisesRegex(ValueError, "global model"):
            gpt.update_tree(X[:1], _target(X[:1])[:, None], 1e-3)
        np.random.seed(8)
        gpt = GPTree(Nbar=NBAR, max_memory_mb=fixed_mb + 0.3, global_mean='additive_gp', global_mean_kwargs=kwargs)
        _stream(gpt, X)
        self.assertGreater(gpt.memory_mb, fixed_mb)
        self.assertLessEqual(gpt.memory_mb, fixed_mb + 0.3)
        self.assertAlmostEqual(gpt._fixed_bytes / 1e6, fixed_mb)

    def test_with_an_output_basis(self):
        rng = np.random.RandomState(11)
        X = _sweep(700, rng, length=4.0)
        Y = _target(X)[:, None] * np.linspace(1.0, 2.0, 6)[None, :] + 1e-3 * rng.randn(700, 6)
        fixed_mb = 8 * (200 * 7 + 6 * 8) / 1e6
        rmse = {}
        for limit in (fixed_mb + 0.35, None):
            np.random.seed(9)
            gpt = GPTree(Nbar=NBAR, max_memory_mb=limit, n_outputs=6, output_model='pca', output_basis_reservoir_size=200)
            for i in range(700):
                gpt.update_tree(X[i:i + 1], Y[i:i + 1], 1e-3)
                if limit is not None:
                    self.assertLessEqual(gpt.memory_mb, limit)
            rmse[limit] = np.sqrt(np.mean((gpt.predict(X[-40:])[0] - Y[-40:]) ** 2))
            if limit is not None:
                self.assertAlmostEqual(gpt._fixed_bytes / 1e6, fixed_mb)
                self.assertGreater(sum(leaf.asleep for leaf in gpt.root.leaves), 0)
                self.assertAccountsMatch(gpt)
        self.assertLess(rmse[fixed_mb + 0.35], 1.1 * rmse[None])


class TestSleepingLeaf(unittest.TestCase):
    """A leaf put to sleep and woken again predicts as before, whatever its output model."""

    def _check(self, n_outputs=1, **kw):
        np.random.seed(10)
        rng = np.random.RandomState(12)
        X = 4.0 * rng.rand(260, 2)
        Y = np.column_stack([_target(X) * (1.0 + 0.3 * k) + 0.1 * k * X[:, 0] for k in range(n_outputs)])
        gpt = GPTree(Nbar=NBAR, retrain_every_n_points=20, n_outputs=n_outputs, **kw)
        for i in range(X.shape[0]):
            gpt.update_tree(X[i:i + 1], Y[i:i + 1], 1e-3)
        X_test = 4.0 * rng.rand(50, 2)
        checked = 0
        for leaf in gpt.root.leaves:
            if not leaf.my_GPRs[0].is_trained():
                continue
            before = leaf.predict(X_test, use_calibrated_sigma=True)
            fitted = leaf.nbytes_fitted()
            self.assertTrue(leaf.sleep())
            self.assertTrue(leaf.asleep)
            self.assertLess(leaf.nbytes_fitted(), 0.3 * fitted)
            self.assertFalse(leaf.sleep())                       # nothing more to release
            after = leaf.predict(X_test, use_calibrated_sigma=True)
            self.assertFalse(leaf.asleep)
            self.assertEqual(leaf.nbytes_fitted(), fitted)
            if gpt.append_points_between_fits:
                # the factor of a GP with appended points is restored to numerical precision
                np.testing.assert_allclose(after[0], before[0], rtol=0, atol=1e-7 * np.ptp(Y))
                np.testing.assert_allclose(after[1], before[1], rtol=1e-4)
            else:
                np.testing.assert_array_equal(after[0], before[0])
                np.testing.assert_array_equal(after[1], before[1])
            checked += 1
        self.assertGreater(checked, 2)

    def test_one_output(self):
        self._check()

    def test_without_appended_points(self):
        self._check(append_points_between_fits=False)
        self._check(n_outputs=3, output_model='shared', append_points_between_fits=False)

    def test_independent_outputs(self):
        self._check(n_outputs=3)

    def test_shared_kernel(self):
        self._check(n_outputs=3, output_model='shared')

    def test_output_basis(self):
        self._check(n_outputs=5, output_model='pca', output_basis_min_points=30)

    def test_on_the_residual_of_a_global_model(self):
        self._check(global_mean='additive_gp', global_mean_kwargs=dict(reservoir_size=60, min_points=40, n_restarts_optimizer=0))

    def test_without_standard_scaling(self):
        self._check(use_standard_scaling=False)

    def test_a_leaf_without_a_fit_or_points_does_not_sleep(self):
        gpt = _tree()
        self.assertFalse(gpt.root.sleep())
        _stream(gpt, _uniform(5, 13))
        self.assertFalse(gpt.root.my_GPRs[0].is_trained())
        self.assertFalse(gpt.root.sleep())

    def test_sklearn_adapter_releases_and_restores_a_fit(self):
        rng = np.random.RandomState(14)
        X = rng.rand(40, 2); y = _target(X)[:, None]
        gpr = Default_GPR()
        self.assertEqual(gpr.nbytes(), 0)
        self.assertFalse(gpr.release())                          # not fitted
        self.assertFalse(gpr.restore())
        gpr.set_observation_noise(np.full(40, 1e-4))
        gpr.fit(X, y)
        before = gpr.predict(X[:5] + 0.01, return_std=True)
        self.assertEqual(gpr.nbytes(), 8 * 40 * (40 + 2 + 1 + 1))   # the factor, the inputs, the targets, the weights
        self.assertFalse(gpr.restore())                          # nothing released
        self.assertTrue(gpr.release())
        self.assertEqual(gpr.nbytes(), 8 * 40 * (2 + 1))
        self.assertTrue(gpr.is_trained())
        self.assertFalse(gpr.release())
        self.assertTrue(gpr.restore())
        after = gpr.predict(X[:5] + 0.01, return_std=True)
        np.testing.assert_array_equal(after[0], before[0]); np.testing.assert_array_equal(after[1], before[1])


class TestCollapseLeaf(TreeChecks):

    def _grown_tree(self):
        np.random.seed(0)
        gpt = _tree(Nbar=20)
        _stream(gpt, _uniform(400, 0))
        return gpt

    def _strictly_inside(self, node, X):
        """Rows of ``X`` that belong to ``node`` with probability one."""
        return X[node.marg_prob(X)[:, 0] == 1.0]

    def _check(self, gpt, leaf):
        parent = leaf.parent
        sibling = parent.right if leaf.is_left else parent.left
        grandparent, name, was_left = parent.parent, parent.name, parent.is_left
        X = _uniform(400, 8)
        X_sibling = self._strictly_inside(sibling, X)
        X_leaf = self._strictly_inside(leaf, X)
        self.assertGreater(min(X_sibling.shape[0], X_leaf.shape[0]), 0)
        before = gpt.predict(X_sibling)
        n_leaves = len(gpt.root.leaves)

        gpt._collapse_leaf(leaf)

        self.assertEqual(len(gpt.root.leaves), n_leaves - 1)
        self.assertNotIn(id(leaf), [id(node) for node in gpt.root])
        self.assertNotIn(id(parent), [id(node) for node in gpt.root])
        self.assertIs(sibling.parent, grandparent)
        self.assertEqual(sibling.name, name)
        self.assertEqual(sibling.is_left, was_left)
        if grandparent is None:
            self.assertIs(gpt.root, sibling)
        else:
            self.assertIs(grandparent.children[0 if was_left else 1], sibling)
            self.assertIs(grandparent.left if was_left else grandparent.right, sibling)
        # The sibling predicts as before in its own region, and now also covers the leaf's
        after = gpt.predict(X_sibling)
        np.testing.assert_allclose(after[0], before[0]); np.testing.assert_allclose(after[1], before[1])
        np.testing.assert_allclose(sibling.marg_prob(X_leaf), 1.0)
        self.assertConsistent(gpt)
        return sibling

    def test_sibling_leaf_becomes_the_parent(self):
        gpt = self._grown_tree()
        leaf = next(node for node in gpt.root.leaves if node.parent.left.is_leaf and node.parent.right.is_leaf)
        sibling = self._check(gpt, leaf)
        self.assertTrue(sibling.is_leaf)
        X_leaf = leaf.my_X_data
        mu, sd = gpt.predict(X_leaf)
        mu_sibling, sd_sibling = sibling.predict(X_leaf, use_calibrated_sigma=True)
        np.testing.assert_allclose(mu, mu_sibling); np.testing.assert_allclose(sd, sd_sibling)

    def test_sibling_subtree_takes_the_parents_place(self):
        gpt = self._grown_tree()
        leaf = next(node for node in gpt.root.leaves
                    if not (node.parent.left.is_leaf and node.parent.right.is_leaf) and node.parent.parent is not None)
        sibling = self._check(gpt, leaf)
        self.assertFalse(sibling.is_leaf)

    def test_sibling_of_a_child_of_the_root_becomes_the_root(self):
        np.random.seed(9)
        gpt = _tree(Nbar=20)
        X = _uniform(70, 9)
        X[:, 0] *= 0.2
        X[30:, 0] += 3.0                 # the root's left child stays a leaf
        _stream(gpt, X)
        leaf = gpt.root.left if gpt.root.left.is_leaf else gpt.root.right
        self.assertFalse((gpt.root.right if leaf.is_left else gpt.root.left).is_leaf)
        sibling = self._check(gpt, leaf)
        self.assertIs(gpt.root, sibling)


if __name__ == '__main__':
    unittest.main()
