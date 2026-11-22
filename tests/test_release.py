"""Tests for ``GPTree.release``: a released tree is freed when it is dropped,
without the garbage collector."""

import gc
import os
import sys
import tempfile
import unittest
import warnings
import weakref

import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo import GPTree, Default_GPR

warnings.filterwarnings("ignore")


def _tree(**kw):
    """A tree of some 80 leaves whose leaf GPs have fixed hyperparameters, with
    a few predictions made."""
    np.random.seed(0)
    rng = np.random.RandomState(0)
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=[1.0, 1.0], length_scale_bounds="fixed")
    gpt = GPTree(GPR=Default_GPR(kernel=kernel), Nbar=20, retrain_every_n_points=10, **kw)
    X = 4.0 * rng.rand(1200, 2)
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
    for i in range(X.shape[0]):
        gpt.update_tree(X[i:i + 1], y[i:i + 1, None], 1e-3)
    gpt.predict(4.0 * rng.rand(20, 2))
    return gpt


def _watch(gpt):
    """Weak references to the tree and to each of its nodes."""
    nodes = [gpt.root]
    for node in nodes:
        if not node.is_leaf:
            nodes += node.children
    return [weakref.ref(gpt)] + [weakref.ref(node) for node in nodes]


def _alive(refs):
    return sum(ref() is not None for ref in refs)


class TestRelease(unittest.TestCase):

    def setUp(self):
        # without the collector, only objects that are in no loop are freed
        gc.collect()
        gc.disable()
        self.addCleanup(gc.enable)

    def test_a_released_tree_is_freed_when_dropped(self):
        for options in (dict(), dict(predict_from='neighbours'), dict(max_memory_mb=1.5),
                        dict(splitting_strategy='standard')):
            with self.subTest(**options):
                gpt = _tree(**options)
                self.assertFalse(gpt.leaf_index.root.bottom)        # the index has groups with parents
                refs = _watch(gpt)
                gpt.release()
                del gpt
                self.assertEqual(_alive(refs), 0)

    def test_a_tree_that_is_only_dropped_waits_for_the_collector(self):
        gpt = _tree()
        refs = _watch(gpt)
        del gpt
        self.assertGreater(_alive(refs), 0)
        gc.collect()
        self.assertEqual(_alive(refs), 0)

    def test_a_loaded_tree(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'tree.pkl')
            _tree().save(path)
            gpt = GPTree.load(path)
        gpt.predict(np.array([[1.0, 1.0]]))
        refs = _watch(gpt)
        gpt.release()
        del gpt
        self.assertEqual(_alive(refs), 0)

    def test_a_tree_without_an_index(self):
        gpt = _tree()
        gpt.leaf_index = None       # as for a tree with a cut that is not along an input
        gc.collect()                # frees the index that was dropped
        refs = _watch(gpt)
        gpt.release()
        del gpt
        self.assertEqual(_alive(refs), 0)

    def test_a_tree_without_points(self):
        gpt = GPTree(GPR=Default_GPR())
        refs = _watch(gpt)
        gpt.release()
        del gpt
        self.assertEqual(_alive(refs), 0)

    def test_the_loaded_tree_that_replaces_a_released_one_predicts(self):
        gpt = _tree()
        X_test = 4.0 * np.random.RandomState(1).rand(30, 2)
        mean, sigma = gpt.predict(X_test)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'tree.pkl')
            gpt.save(path)
            old, gpt = gpt, GPTree.load(path)
        old.release()
        del old
        mean_loaded, sigma_loaded = gpt.predict(X_test)
        np.testing.assert_allclose(mean_loaded, mean, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(sigma_loaded, sigma, rtol=1e-9, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
