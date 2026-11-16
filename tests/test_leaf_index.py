"""Tests for the balanced index over the leaves (``LeafIndex``) and its use by the tree."""

import os
import sys
import tempfile
import unittest
import warnings

import joblib
import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo import GPTree, Default_GPR
from pygptreeo.leaf_index import LeafIndex
from pygptreeo.neighbours import NeighbourSearch

warnings.filterwarnings("ignore")


def _fixed_gpr(d):
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=[1.0] * d, length_scale_bounds="fixed")
    return Default_GPR(kernel=kernel)


def _tree(d, **kw):
    kw.setdefault('Nbar', 20)
    return GPTree(GPR=_fixed_gpr(d), **kw)


def _stream(gpt, X):
    y = np.sin(3.0 * X[:, 0]) + 0.5 * X[:, -1] ** 2
    for i in range(X.shape[0]):
        gpt.update_tree(X[i:i + 1], y[i:i + 1, None], 1e-3)


def _drifting(n, d, rng, step=0.01):
    """A narrow cluster that moves steadily along the first input."""
    X = 0.5 + 0.02 * rng.randn(n, d)
    X[:, 0] = step * np.arange(n) + 0.02 * rng.randn(n)
    return X


def _by_the_cuts(gpt, x):
    """The leaf that the cuts give the point, or None within an overlap."""
    node = gpt.root
    while not node.is_leaf:
        p = node.prob_func(x)[0, 0]
        if 0.0 < p < 1.0:
            return None
        node = node.children[int(p)]
    return node


class _Leaf:
    """What the index needs of a leaf."""

    def __init__(self, X):
        self.my_X_data = X
        self.n_features = X.shape[1]


class TestIndex(unittest.TestCase):

    def _check(self, index, leaves, rng, n_queries=200):
        d = index.n_features
        self.assertEqual(len(index), len(leaves))
        self.assertTrue(all(leaf in index for leaf in leaves))
        # the boxes of a group hold those of its children
        groups = [index.root]
        while groups:
            group = groups.pop()
            self.assertLessEqual(len(group.children), index.fanout)
            if not group.bottom:
                for i, child in enumerate(group.children):
                    self.assertIs(child.parent, group)
                    cell_lo, cell_hi, data_lo, data_hi = child.union()
                    self.assertTrue(np.all(group.cell_lo[i] <= cell_lo) and np.all(group.cell_hi[i] >= cell_hi))
                    self.assertTrue(np.all(group.data_lo[i] <= data_lo) and np.all(group.data_hi[i] >= data_hi))
                    groups.append(child)
        cells = {leaf: index.cell(leaf) for leaf in leaves}
        for q in rng.uniform(-0.5, 1.5, (n_queries, d)):
            holding = [leaf for leaf, (lo, hi) in cells.items() if np.all(lo <= q) and np.all(q < hi)]
            self.assertEqual(len(holding), 1)
            self.assertIs(index.locate(q), holding[0])
            # the leaves within a distance are those whose points' box is
            w = rng.uniform(0.1, 1.0, d)
            radius = rng.uniform(0.0, 0.5)
            found, bounds = index.within(q, w, radius)
            expected = set()
            for leaf in leaves:
                gap = np.maximum(leaf.my_X_data.min(axis=0) - q, 0.0) + np.maximum(q - leaf.my_X_data.max(axis=0), 0.0)
                if ((gap / w) ** 2).sum() <= radius:
                    expected.add(leaf)
            self.assertEqual(set(found), expected)
            self.assertEqual(len(found), len(bounds))
            # the nearest leaves are those whose points' boxes are nearest
            distance = {}
            for leaf in leaves:
                gap = np.maximum(leaf.my_X_data.min(axis=0) - q, 0.0) + np.maximum(q - leaf.my_X_data.max(axis=0), 0.0)
                distance[leaf] = ((gap / w) ** 2).sum()
            m = int(rng.randint(1, 8))
            nearest = index.nearest(q, w, m)
            self.assertEqual(len(nearest), min(m, len(leaves)))
            np.testing.assert_allclose([distance[leaf] for leaf in nearest], sorted(distance.values())[:m])

    def test_cells_tile_the_space_through_splits_and_removals(self):
        rng = np.random.RandomState(0)
        for d, fanout in ((2, 4), (3, 6), (5, 32)):
            index = LeafIndex(d, fanout=fanout)
            first = _Leaf(rng.rand(40, d))
            index.add(first)
            leaves = [first]
            for step in range(300):
                if step % 7 == 6 and len(leaves) > 3:
                    # a leaf goes, and a neighbour across one of its faces takes its cell
                    leaf = leaves[rng.randint(len(leaves))]
                    lo, hi = index.cell(leaf)
                    merged = False
                    for other in leaves:
                        if other is leaf:
                            continue
                        other_lo, other_hi = index.cell(other)
                        for j in range(d):
                            rest = [k for k in range(d) if k != j]
                            if np.all(other_lo[rest] == lo[rest]) and np.all(other_hi[rest] == hi[rest]) \
                                    and (other_hi[j] == lo[j] or other_lo[j] == hi[j]):
                                index.remove(leaf)
                                index.set_cell(other, np.minimum(lo, other_lo), np.maximum(hi, other_hi))
                                leaves.remove(leaf)
                                merged = True
                                break
                        if merged:
                            break
                    continue
                leaf = leaves[rng.randint(len(leaves))]
                X = leaf.my_X_data
                if len(X) < 4:
                    continue
                j = rng.randint(d)
                position = np.median(X[:, j])
                left, right = _Leaf(X[X[:, j] < position]), _Leaf(X[X[:, j] >= position])
                if len(left.my_X_data) == 0 or len(right.my_X_data) == 0:
                    continue
                index.split(leaf, left, right, j, position)
                leaves.remove(leaf)
                leaves += [left, right]
                # a point arrives in one of them
                x = rng.uniform(-0.2, 1.2, d)
                target = index.locate(x)
                target.my_X_data = np.vstack((target.my_X_data, x))
                index.grow(target, x)
            self._check(index, leaves, rng)
            if fanout < 32:
                self.assertGreaterEqual(index.depth(), 2)

    def test_depth_stays_logarithmic_whatever_the_order(self):
        """Leaves split one after the other at one end, as on a drifting stream."""
        index = LeafIndex(1, fanout=8)
        leaf = _Leaf(np.array([[0.0], [1.0]]))
        index.add(leaf)
        for step in range(1, 2000):
            left, right = _Leaf(np.array([[step - 1.0], [step - 0.5]])), _Leaf(np.array([[step + 0.0], [step + 1.0]]))
            index.split(leaf, left, right, 0, step - 0.25)
            leaf = right
        self.assertEqual(len(index), 2000)
        self.assertLessEqual(index.depth(), 6)
        self.assertEqual(index.locate(np.array([1500.6])).my_X_data[0, 0], 1500.0)


class TestTree(unittest.TestCase):

    def _check_tree(self, gpt, d, rng, n_queries=300, overlap_seen=False):
        index = gpt.leaf_index
        leaves = gpt.root.leaves
        self.assertEqual(len(index), len(leaves))
        located = 0
        for q in rng.uniform(-0.3, 1.3, (n_queries, d)):
            expected = _by_the_cuts(gpt, q[None, :])
            found = index.locate(q)
            self.assertIs(found, expected)
            located += found is not None
            if found is not None:
                self.assertEqual(found.marg_prob(q[None, :])[0, 0], 1.0)
                self.assertIs(gpt._leaf_of(q[None, :]), found)
        self.assertGreater(located, 0)
        if not overlap_seen:
            self.assertGreater(located, 0.9 * n_queries)
        for leaf in leaves:
            if leaf.n_points:
                group = index._group[leaf]
                i = group.children.index(leaf)
                np.testing.assert_array_equal(group.data_lo[i], leaf.my_X_data.min(axis=0))
                np.testing.assert_array_equal(group.data_hi[i], leaf.my_X_data.max(axis=0))

    def test_the_index_gives_the_leaf_of_the_cuts(self):
        for d, n in ((2, 900), (4, 1500)):
            rng = np.random.RandomState(d)
            np.random.seed(d)
            gpt = _tree(d)
            _stream(gpt, rng.rand(n, d))
            self.assertGreater(len(gpt.root.leaves), 40)
            self._check_tree(gpt, d, rng)

    def test_with_overlapping_children(self):
        rng = np.random.RandomState(1)
        np.random.seed(1)
        gpt = _tree(2, theta=0.2)
        _stream(gpt, rng.rand(800, 2))
        self._check_tree(gpt, 2, rng, overlap_seen=True)
        # a query within an overlap is still a mixture of the leaves on both sides
        blended = 0
        for q in rng.rand(200, 2):
            leaves, weights = gpt._leaves_of(q[None, :])
            self.assertAlmostEqual(sum(weights), 1.0, places=12)
            for leaf, weight in zip(leaves, weights):
                self.assertAlmostEqual(leaf.marg_prob(q[None, :])[0, 0], weight, places=12)
            blended += len(leaves) > 1
        self.assertGreater(blended, 20)

    def test_after_leaves_were_removed(self):
        rng = np.random.RandomState(2)
        np.random.seed(2)
        gpt = _tree(2, Nbar=30, max_memory_mb=0.4)
        _stream(gpt, rng.rand(1500, 2))
        self.assertLess(sum(leaf.n_points for leaf in gpt.root.leaves), 1500)
        self._check_tree(gpt, 2, rng)

    def test_a_drifting_stream(self):
        """Every split adds a level to the tree of cuts; the index stays shallow."""
        rng = np.random.RandomState(3)
        np.random.seed(3)
        gpt = _tree(2, Nbar=8, retrain_every_n_points=8, predict_from='neighbours', n_neighbours=8)
        X = _drifting(9000, 2, rng)
        _stream(gpt, X)
        depth = max(len(leaf.name) for leaf in gpt.root.leaves) - 1
        self.assertGreater(depth, 1100)             # beyond the recursion limit
        self.assertLessEqual(gpt.leaf_index.depth(), 4)
        mean, std = gpt.predict(X[-200:])
        self.assertLess(np.sqrt(np.mean((mean[:, 0] - np.sin(3.0 * X[-200:, 0]) - 0.5 * X[-200:, 1] ** 2) ** 2)), 0.05)
        self.assertTrue(np.all(std > 0.0))
        self._check_tree(gpt, 2, rng, n_queries=100)
        # the same holds where the cuts are followed
        leaves, weights = GPTree._leaves_of(_without_index(gpt), X[-1:])
        self.assertEqual(len(leaves), 1)
        self.assertIs(leaves[0], gpt.leaf_index.locate(X[-1]))
        # a tree this deep is saved, and the copy predicts and learns
        with tempfile.TemporaryDirectory() as folder:
            gpt.save(os.path.join(folder, 'tree.joblib'))
            loaded = joblib.load(os.path.join(folder, 'tree.joblib'))
        np.testing.assert_array_equal(loaded.predict(X[-200:])[0], mean)
        self.assertEqual(len(loaded.root.leaves), len(gpt.root.leaves))
        self.assertTrue(all(leaf.parent.children[0 if leaf.is_left else 1] is leaf for leaf in loaded.root.leaves))
        _stream(loaded, X[-1:] + 0.01 * np.arange(1, 41)[:, None])
        self._check_tree(loaded, 2, rng, n_queries=100)

    def test_a_tree_without_an_index(self):
        """As one saved before the index existed: the cuts are followed."""
        rng = np.random.RandomState(5)
        np.random.seed(5)
        gpt = _tree(2)
        X = rng.rand(400, 2)
        _stream(gpt, X)
        expected = gpt.predict(X[:50])
        state = dict(gpt.__dict__)
        for name in ('leaf_index', 'neighbours', 'predict_from'):
            del state[name]
        old = GPTree.__new__(GPTree)
        old.__setstate__(state)
        self.assertIsNone(old.leaf_index)
        np.testing.assert_array_equal(old.predict(X[:50])[0], expected[0])
        self.assertEqual(old.n_leaves, len(gpt.root.leaves))
        _stream(old, rng.rand(60, 2))

    def test_the_search_of_a_tree_without_an_index(self):
        """Every leaf is looked at, and the points found are the same."""
        rng = np.random.RandomState(4)
        np.random.seed(4)
        gpt = _tree(3, Nbar=30)
        _stream(gpt, np.vstack((rng.rand(700, 3), _drifting(500, 3, rng, step=0.002))))
        search = NeighbourSearch(gpt, n_neighbours=40)
        plain = NeighbourSearch(_without_index(gpt), n_neighbours=40)
        for q in rng.uniform(-0.1, 1.1, (80, 3)):
            w = rng.uniform(0.05, 1.0, 3)
            np.testing.assert_allclose(np.sort((((search.find(q, w)[0] - q) / w) ** 2).sum(axis=1)),
                                       np.sort((((plain.find(q, w)[0] - q) / w) ** 2).sum(axis=1)), rtol=1e-12)


class _without_index:
    """A view of a tree that has no index of its leaves."""

    def __init__(self, gpt):
        self.root = gpt.root
        self.leaf_index = None


if __name__ == '__main__':
    unittest.main()
