"""Tests for the feature network global model with several outputs."""

import contextlib
import io
import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo.gptree import GPTree
from pygptreeo.default_gpr import Default_GPR
from pygptreeo.feature_net import TORCH_AVAILABLE, NetGlobalMean

warnings.filterwarnings("ignore")


def _stream(gpt, X, Y, S):
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(X.shape[0]):
            gpt.update_tree(X[i:i + 1], Y[i:i + 1], S[i:i + 1])


def _targets(X):
    """Three correlated smooth outputs of a 3D input on [0, 1]^3."""
    f = np.sin(3 * X[:, 0]) * np.cos(2 * X[:, 1]) + 0.5 * X[:, 2] ** 2
    return np.column_stack([f, 2.0 * f + 1.0, f ** 2 - X[:, 2]])


def _nrmse(P, Yt):
    return np.sqrt(np.mean((P - Yt) ** 2, axis=0)) / (Yt.max(axis=0) - Yt.min(axis=0))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class TestFeatureNetMultiOutput(unittest.TestCase):

    def _run(self, output_model, **tree_kw):
        rng = np.random.RandomState(21)
        N = 600
        X = rng.rand(N, 3); Y = _targets(X); S = np.full((N, 3), 1e-3)
        gm = NetGlobalMean(steps=200, polish_steps=0, min_points=60, random_state=0)
        gpt = GPTree(GPR=Default_GPR(), Nbar=60, retrain_every_n_points=20, n_outputs=3,
                     output_model=output_model, global_mean=gm, **tree_kw)
        _stream(gpt, X, Y, S)
        Xt = rng.rand(300, 3); Yt = _targets(Xt)
        P, Sd = gpt.predict(Xt)
        return gpt, gm, P, Sd, Yt

    def test_independent_outputs(self):
        gpt, gm, P, Sd, Yt = self._run('independent')
        self.assertEqual(gm.current.n_outputs, 3)
        self.assertEqual(gm.current.predict(Yt[:2, :3] * 0 + 0.5).shape, (2, 3))
        self.assertEqual(np.asarray(gm.error_scale).shape, (3,))
        self.assertEqual(P.shape, (300, 3)); self.assertEqual(Sd.shape, (300, 3))
        self.assertTrue(np.all(Sd > 0))
        self.assertTrue(np.all(_nrmse(P, Yt) < 0.05))

    def test_pca_outputs(self):
        gpt, gm, P, Sd, Yt = self._run('pca', output_basis_min_points=60)
        self.assertEqual(P.shape, (300, 3)); self.assertEqual(Sd.shape, (300, 3))
        self.assertTrue(np.all(Sd > 0))
        self.assertTrue(np.all(_nrmse(P, Yt) < 0.05))
        for leaf in gpt.root.leaves:
            self.assertEqual(leaf._fitted_global.version, gm.current.version)


if __name__ == '__main__':
    unittest.main()
