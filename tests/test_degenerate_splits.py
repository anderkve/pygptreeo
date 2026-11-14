"""Tests for splits of leaves whose points coincide in one or all input dimensions."""

import contextlib
import io
import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo import GPTree, Default_GPR
from pygptreeo.gpnode import GPNode


def _stream(gpt, X, y):
    """Predict each point, then update the tree with it; the predictions."""
    preds = []
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        for i in range(len(X)):
            mu, sd = gpt.predict(X[i:i + 1])
            gpt.update_tree(X[i:i + 1], y[i:i + 1, None], np.full((1, 1), 1e-3))
            preds.append((mu[0, 0], sd[0, 0]))
    return np.array(preds)


def _inner_nodes(gpt):
    return [node for node in gpt.root if not node.is_leaf]


class TestConstantInputDimension(unittest.TestCase):
    """All points share the value of one input, e.g. inputs clipped onto a face of the box."""

    def setUp(self):
        rng = np.random.RandomState(0)
        self.X = rng.rand(260, 3)
        self.X[:, 1] = 1.0
        self.y = self.X[:, 0] + 0.5 * self.X[:, 2]     # smooth: long fitted length scales

    def _check(self, criterion):
        np.random.seed(3)
        gpt = GPTree(Nbar=50, split_dimension_criteria=criterion)
        preds = _stream(gpt, self.X, self.y)
        self.assertTrue(np.all(np.isfinite(preds)), criterion)
        inner = _inner_nodes(gpt)
        self.assertGreater(len(inner), 2, criterion)
        for node in inner:
            self.assertNotEqual(node.split_index, 1, criterion)
            self.assertGreater(node.overlap, 0.0, criterion)

    def test_min_lengthscale(self):
        self._check('min_lengthscale')

    def test_random(self):
        self._check('random')

    def test_max_spread(self):
        self._check('max_spread')

    def test_max_variance(self):
        self._check('max_variance')

    def test_max_uncertainty(self):
        self._check('max_uncertainty')


class TestZeroOverlap(unittest.TestCase):

    def test_prob_func_is_a_hard_split(self):
        node = GPNode(0, my_GPR=Default_GPR(), Nbar=10)
        node.split_index, node.split_position, node.overlap = 0, 0.5, 0.0
        x = np.array([[0.2, 9.0], [0.5, 9.0], [0.8, 9.0]])
        np.testing.assert_array_equal(node.prob_func(x), [[0.0], [0.5], [1.0]])

    def test_prob_func_with_overlap_is_unchanged(self):
        node = GPNode(0, my_GPR=Default_GPR(), Nbar=10)
        node.split_index, node.split_position, node.overlap = 0, 0.5, 0.2
        x = np.array([[0.2], [0.45], [0.5], [0.55], [0.8]])
        np.testing.assert_allclose(node.prob_func(x), [[0.0], [0.25], [0.5], [0.75], [1.0]])

    def test_theta_zero(self):
        """With an odd Nbar the median split position is one of the leaf's points."""
        rng = np.random.RandomState(1)
        X = rng.rand(160, 2)
        y = np.sin(4 * X[:, 0]) + X[:, 1]
        np.random.seed(1)
        gpt = GPTree(Nbar=51, theta=0.0)
        preds = _stream(gpt, X, y)
        self.assertTrue(np.all(np.isfinite(preds)))
        self.assertGreater(len(gpt.root.leaves), 1)

    def test_identical_points(self):
        """A leaf filled with copies of one point still splits without an error."""
        X = np.tile([[0.3, 0.7]], (70, 1))
        y = np.full(70, 2.0)
        np.random.seed(2)
        gpt = GPTree(Nbar=30)
        preds = _stream(gpt, X, y)
        self.assertTrue(np.all(np.isfinite(preds)))


if __name__ == '__main__':
    unittest.main()
