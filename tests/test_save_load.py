"""Tests for ``GPTree.save`` and ``GPTree.load``: the leaves are saved asleep,
and the file is replaced in one step."""

import gc
import os
import pickle
import sys
import tempfile
import unittest
import warnings

import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo import GPTree, Default_GPR

warnings.filterwarnings("ignore")


def _tree(**kw):
    """A tree whose leaf GPs have fixed hyperparameters, so a fit is one factorisation."""
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=[1.0, 1.0], length_scale_bounds="fixed")
    kw.setdefault('Nbar', 40)
    kw.setdefault('retrain_every_n_points', 10)
    return GPTree(GPR=Default_GPR(kernel=kernel), **kw)


def _target(X):
    return np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2


def _stream(gpt, X):
    y = _target(X)
    for i in range(X.shape[0]):
        gpt.update_tree(X[i:i + 1], np.tile(y[i:i + 1, None], (1, gpt.n_outputs)), 1e-3)


def _factors(gpt):
    """Number of GPs of the tree's leaves that hold a Cholesky factor."""
    return sum(hasattr(gpr._gpr, 'L_') for leaf in gpt.root.leaves for gpr in leaf.my_GPRs)


class TestSaveAndLoad(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.rng = np.random.RandomState(0)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = os.path.join(self.folder.name, 'tree.pkl')

    def test_the_leaves_are_saved_asleep(self):
        gpt = _tree()
        _stream(gpt, 4.0 * self.rng.rand(600, 2))
        n_leaves = len(gpt.root.leaves)
        self.assertEqual(_factors(gpt), n_leaves)
        gpt.save(self.path)
        loaded = GPTree.load(self.path)
        for tree in (gpt, loaded):
            self.assertEqual(len(tree.root.leaves), n_leaves)
            self.assertTrue(all(leaf.asleep for leaf in tree.root.leaves))
            self.assertEqual(_factors(tree), 0)
        # a prediction wakes the leaf that makes it, and no other
        loaded.predict(np.array([[1.0, 1.0]]))
        self.assertEqual(_factors(loaded), 1)

    def test_a_loaded_tree_predicts_as_the_saved_one(self):
        for options in (dict(), dict(n_outputs=3), dict(n_outputs=3, output_model='shared'), dict(predict_from='neighbours')):
            np.random.seed(1)
            gpt = _tree(**options)
            _stream(gpt, 4.0 * self.rng.rand(500, 2))
            X_test = 4.0 * self.rng.rand(100, 2)
            mean, sigma = gpt.predict(X_test)
            gpt.save(self.path)
            loaded = GPTree.load(self.path)
            mean_loaded, sigma_loaded = loaded.predict(X_test)
            # as before the leaves slept, to the precision of their factors
            np.testing.assert_allclose(mean_loaded, mean, rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(sigma_loaded, sigma, rtol=1e-5)
            # and exactly as the tree that was saved
            np.testing.assert_array_equal(mean_loaded, gpt.predict(X_test)[0])
            np.testing.assert_array_equal(sigma_loaded, gpt.predict(X_test)[1])

    def test_a_saved_and_a_loaded_tree_learn_on(self):
        """Both end as a tree that was never saved."""
        X, X_test = 4.0 * self.rng.rand(900, 2), 4.0 * self.rng.rand(100, 2)
        for options in (dict(), dict(predict_from='neighbours')):
            np.random.seed(3)
            reference = _tree(**options)
            _stream(reference, X[:500])
            random_state = np.random.get_state()
            _stream(reference, X[500:])
            np.random.seed(3)
            gpt = _tree(**options)
            _stream(gpt, X[:500])
            gpt.save(self.path)
            loaded = GPTree.load(self.path)
            n_leaves = len(loaded.root.leaves)
            for tree in (gpt, loaded):
                np.random.set_state(random_state)
                _stream(tree, X[500:])
                self.assertGreater(len(tree.root.leaves), n_leaves)
                self.assertEqual(len(tree.root.leaves), len(reference.root.leaves))
                for a, b in zip(tree.predict(X_test), reference.predict(X_test)):
                    np.testing.assert_allclose(a, b, rtol=1e-5)

    def test_a_tree_without_points(self):
        gpt = _tree()
        gpt.save(self.path)
        loaded = GPTree.load(self.path)
        _stream(loaded, 4.0 * self.rng.rand(100, 2))
        self.assertEqual(sum(leaf.n_points for leaf in loaded.root.leaves), 100)

    def test_the_count_of_points_is_saved(self):
        gpt = _tree()
        X = 4.0 * self.rng.rand(150, 2)
        _stream(gpt, X[:100])
        gpt.save(self.path)
        loaded = GPTree.load(self.path)
        self.assertEqual(loaded.n_points_seen, 100)
        _stream(loaded, X[100:])
        self.assertEqual(loaded.n_points_seen, 150)

    def test_the_file_is_replaced_in_one_step(self):
        gpt = _tree()
        X = 4.0 * self.rng.rand(300, 2)
        _stream(gpt, X[:200])
        gpt.save(self.path)
        with open(self.path, 'rb') as opened:
            # a reader that has the file open still gets the whole of the first tree
            _stream(gpt, X[200:])
            gpt.save(self.path)
            first = pickle.load(opened)
        self.assertEqual(sum(leaf.n_points for leaf in first.root.leaves), 200)
        self.assertEqual(sum(leaf.n_points for leaf in GPTree.load(self.path).root.leaves), 300)
        self.assertEqual(os.listdir(self.folder.name), ['tree.pkl'])

    def test_one_random_state_for_the_gps(self):
        """The GPs of the leaves share one random state, which is saved once."""
        np.random.seed(2)
        gpt = GPTree(Nbar=40, retrain_every_n_points=40)
        _stream(gpt, 4.0 * self.rng.rand(200, 2))
        self.assertGreater(len(gpt.root.leaves), 3)
        gpt.save(self.path)
        loaded = GPTree.load(self.path)
        self.assertEqual(len({id(gpr._gpr._rng) for leaf in loaded.root.leaves for gpr in leaf.my_GPRs}), 1)

    def test_the_scores_of_the_nodes(self):
        """Saved as one array per node, they are loaded as they were."""
        gpt = _tree(predict_from='neighbours')
        _stream(gpt, 4.0 * self.rng.rand(400, 2))
        self.assertTrue(any(len(leaf._scores) > 0 for leaf in gpt.root.leaves))
        gpt.save(self.path)
        loaded = GPTree.load(self.path)
        for saved, leaf in zip(gpt.root.leaves, loaded.root.leaves):
            self.assertEqual(leaf._scores.maxlen, saved._scores.maxlen)
            np.testing.assert_array_equal(np.array(leaf._scores), np.array(saved._scores))

    def test_the_garbage_collector_is_left_as_it_was(self):
        gpt = _tree()
        _stream(gpt, 4.0 * self.rng.rand(50, 2))
        self.addCleanup(gc.enable)
        for enabled in (True, False):
            gc.enable() if enabled else gc.disable()
            gpt.save(self.path)
            GPTree.load(self.path)
            self.assertEqual(gc.isenabled(), enabled)


if __name__ == '__main__':
    unittest.main()
