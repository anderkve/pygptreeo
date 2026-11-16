"""Tests for predictions conditioned on a query's nearest stored points
(``predict_from='neighbours'``)."""

import os
import sys
import tempfile
import unittest
import warnings

import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo import GPTree, Default_GPR, GPRegressorInterface
from pygptreeo.neighbours import NeighbourSearch

warnings.filterwarnings("ignore")


def _fixed_gpr(d, length_scale=1.0):
    """A GP with fixed hyperparameters, so a fit is one factorisation."""
    kernel = (ConstantKernel(1.0, constant_value_bounds="fixed")
              * RBF(length_scale=[length_scale] * d, length_scale_bounds="fixed"))
    return Default_GPR(kernel=kernel)


def _target(X):
    return np.sin(3.0 * X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.2 * X[:, -1]


def _stream(gpt, X, y=None, sigma=1e-3):
    y = _target(X) if y is None else y
    for i in range(X.shape[0]):
        gpt.update_tree(X[i:i + 1], np.atleast_2d(y[i]), sigma)


def _grown_tree(n=600, d=2, seed=0, **kw):
    np.random.seed(seed)
    kw.setdefault('Nbar', 30)
    gpt = GPTree(GPR=_fixed_gpr(d), **kw)
    _stream(gpt, np.random.RandomState(seed).rand(n, d))
    return gpt


def _stored_points(gpt):
    leaves = [leaf for leaf in gpt.root.leaves if leaf.n_points > 0]
    return (np.vstack([leaf.my_X_data for leaf in leaves]), np.vstack([leaf.my_y_data for leaf in leaves]),
            np.vstack([leaf.my_sigma_data for leaf in leaves]))


def _leaf_of(gpt, x):
    node = gpt.root
    while not node.is_leaf:
        node = node.children[int(node.prob_func(x)[0, 0] > 0.5)]
    return node


class TestCondition(unittest.TestCase):

    def test_matches_a_fit_to_the_points_with_the_kernel_kept(self):
        rng = np.random.RandomState(0)
        X, X_other, X_query = rng.rand(50, 3), rng.rand(40, 3), rng.rand(7, 3)
        noise = rng.uniform(1e-6, 1e-3, 40)
        for n_targets in (1, 2):
            def values(Z):
                return np.column_stack([np.sin(3 * Z[:, 0]) * (1 + k) + Z[:, 1] ** 2 for k in range(n_targets)])
            gpr = Default_GPR()
            self.assertIsNone(gpr.condition(X_other, values(X_other), noise, X_query))       # no fit
            gpr.set_observation_noise(np.full(50, 1e-6))
            gpr.fit(X, values(X))
            mean, std, consistency = gpr.condition(X_other, values(X_other), noise, X_query)
            reference = GaussianProcessRegressor(kernel=gpr.sklearn_gpr.kernel_, optimizer=None, alpha=noise)
            reference.fit(X_other, values(X_other))
            ref_mean, ref_std = reference.predict(X_query, return_std=True)
            np.testing.assert_allclose(mean, ref_mean.reshape(7, n_targets), rtol=1e-7, atol=1e-9)
            np.testing.assert_allclose(std, ref_std.reshape(7, -1)[:, 0], rtol=1e-6, atol=1e-9)
            self.assertEqual(consistency.shape, (n_targets,))
            # the fit is as it was
            before = gpr.predict(X_query, return_std=True)
            own = gpr.condition(X, values(X), np.full(50, 1e-6), X_query)
            np.testing.assert_allclose(own[0], np.reshape(before[0], (7, n_targets)), rtol=1e-7, atol=1e-9)

    def test_consistency_tells_points_the_kernel_does_not_describe(self):
        rng = np.random.RandomState(1)
        X = rng.rand(80, 2)
        gpr = Default_GPR()
        gpr.set_observation_noise(np.full(80, 1e-8))
        gpr.fit(X, np.sin(2 * X[:, 0]) + X[:, 1])
        X_other = rng.rand(60, 2)
        smooth = gpr.condition(X_other, np.sin(2 * X_other[:, 0]) + X_other[:, 1], np.full(60, 1e-8), X[:1])
        jump = gpr.condition(X_other, np.sin(2 * X_other[:, 0]) + X_other[:, 1] + 3.0 * (X_other[:, 0] > 0.5),
                             np.full(60, 1e-8), X[:1])
        self.assertLess(smooth[2][0], 10.0)
        self.assertGreater(jump[2][0], 100.0)

    def test_a_backend_without_it_is_refused(self):
        class Bare(GPRegressorInterface):
            def fit(self, X, y): return self
            def predict(self, X, return_std=False): return np.zeros(len(X)), np.ones(len(X))
            def is_trained(self): return False
            def set_observation_noise(self, alpha): pass
            def clone(self): return Bare()
            def get_kernel(self): return None
            def set_kernel(self, kernel): pass
        with self.assertRaises(ValueError):
            GPTree(GPR=Bare(), predict_from='neighbours')
        with self.assertRaises(ValueError):
            GPTree(predict_from='nearest')
        GPTree(GPR=Bare())      # predicting from the leaf needs no conditioning


class TestSearch(unittest.TestCase):

    def _check_exact(self, gpt, d, k=25, n_queries=60, seed=0):
        X, y, sigma = _stored_points(gpt)
        search = NeighbourSearch(gpt, n_neighbours=k)
        rng = np.random.RandomState(seed)
        for _ in range(n_queries):
            q = rng.uniform(-0.2, 1.2, d)
            w = rng.uniform(0.05, 1.0, d)
            Xn, yn, sn = search.find(q, w)
            found = np.sort((((Xn - q) / w) ** 2).sum(axis=1))
            d2 = (((X - q) / w) ** 2).sum(axis=1)
            np.testing.assert_allclose(found, np.sort(d2)[:k], rtol=1e-12)
            # the targets and noise levels are those of the points found
            for row, target, noise in zip(Xn, yn, sn):
                i = int(np.argmin(((X - row) ** 2).sum(axis=1)))
                np.testing.assert_array_equal(target, y[i])
                np.testing.assert_array_equal(noise, sigma[i])

    def test_finds_the_nearest_points(self):
        self._check_exact(_grown_tree(n=600, d=2), d=2)
        self._check_exact(_grown_tree(n=900, d=4, seed=1), d=4)

    def test_with_overlapping_children(self):
        self._check_exact(_grown_tree(n=600, d=2, theta=0.3), d=2)

    def test_with_standard_splitting(self):
        self._check_exact(_grown_tree(n=600, d=3, splitting_strategy='standard'), d=3)

    def test_after_leaves_were_removed(self):
        gpt = _grown_tree(n=900, d=2, max_memory_mb=0.4)
        self.assertLess(sum(leaf.n_points for leaf in gpt.root.leaves), 900)
        self._check_exact(gpt, d=2)

    def test_fewer_points_than_asked_for(self):
        gpt = _grown_tree(n=12, d=2)
        Xn, yn, sn = NeighbourSearch(gpt, n_neighbours=50).find(np.array([0.5, 0.5]), np.ones(2))
        self.assertEqual(Xn.shape, (12, 2))
        self.assertIsNone(NeighbourSearch(GPTree(), n_neighbours=5).find(np.zeros(2), np.ones(2)))     # an empty tree

    def test_a_limit_on_the_leaves_searched(self):
        """The points then come from the leaves whose points' boxes are nearest to the query."""
        gpt = _grown_tree(n=600, d=2)
        leaves = [leaf for leaf in gpt.root.leaves if leaf.n_points > 0]
        rng = np.random.RandomState(0)
        for q in rng.uniform(-0.1, 1.1, (40, 2)):
            for m in (1, 3):
                Xn, _, _ = NeighbourSearch(gpt, n_neighbours=10, max_leaves=m).find(q, np.ones(2))
                gaps = [np.maximum(leaf.my_X_data.min(axis=0) - q, 0.0) + np.maximum(q - leaf.my_X_data.max(axis=0), 0.0)
                        for leaf in leaves]
                nearest = np.argsort([(gap ** 2).sum() for gap in gaps], kind='stable')[:m]
                points = np.vstack([leaves[i].my_X_data for i in nearest])
                np.testing.assert_allclose(np.sort(((Xn - q) ** 2).sum(axis=1)),
                                           np.sort(((points - q) ** 2).sum(axis=1))[:10])


class TestPredictions(unittest.TestCase):

    def test_equal_a_gp_on_the_neighbours_with_the_leaf_kernel(self):
        np.random.seed(0)
        rng = np.random.RandomState(0)
        gpt = GPTree(Nbar=40, retrain_every_n_points=10, predict_from='neighbours', n_neighbours=30,
                     neighbour_consistency=np.inf, use_calibrated_sigma=False)
        _stream(gpt, rng.rand(400, 2))
        X, y, sigma = _stored_points(gpt)
        for q in rng.rand(15, 2):
            leaf = _leaf_of(gpt, q[None, :])
            d2 = (((X - q) / leaf.fit_std) ** 2).sum(axis=1)
            idx = np.argsort(d2)[:30]
            xs, ys = leaf.X_scaler, leaf.y_scalers[0]
            reference = GaussianProcessRegressor(kernel=leaf.my_GPRs[0].sklearn_gpr.kernel_, optimizer=None,
                                                 alpha=(sigma[idx, 0] / ys.scale_[0]) ** 2)
            reference.fit(xs.transform(X[idx]), ys.transform(y[idx]))
            ref_mean, ref_std = reference.predict(xs.transform(q[None, :]), return_std=True)
            mean, std = gpt.predict(q[None, :])
            np.testing.assert_allclose(mean[0, 0], ref_mean[0] * ys.scale_[0] + ys.mean_[0], rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(std[0, 0], ref_std[0] * ys.scale_[0], rtol=1e-4, atol=1e-8)

    def _both(self, n=260, d=2, **kw):
        """The predictions along a stream of a tree that predicts from its leaves
        and of one that predicts from neighbours no kernel is allowed to fit."""
        rng = np.random.RandomState(3)
        X = rng.rand(n, d)
        n_outputs = kw.get('n_outputs', 1)
        Y = np.column_stack([_target(X) * (1.0 + 0.3 * k) + 0.1 * k * X[:, 0] for k in range(n_outputs)])
        out = []
        for options in (dict(), dict(predict_from='neighbours', neighbour_consistency=0.0)):
            np.random.seed(3)
            gpt = GPTree(Nbar=50, retrain_every_n_points=10, **options, **kw)
            preds = []
            for i in range(n):
                preds.append(np.hstack(gpt.predict(X[i:i + 1])))
                gpt.update_tree(X[i:i + 1], Y[i:i + 1], 1e-3)
            out.append(np.vstack(preds))
        return out

    def test_the_leaf_predicts_where_the_neighbours_do_not_fit(self):
        for kw in (dict(), dict(n_outputs=2), dict(n_outputs=2, output_model='shared'),
                   dict(n_outputs=3, output_model='pca', output_basis_components=2),
                   dict(use_standard_scaling=False)):
            leaf, neighbours = self._both(**kw)
            np.testing.assert_allclose(neighbours, leaf, rtol=1e-9, atol=1e-12, err_msg=str(kw))

    def test_all_output_models(self):
        rng = np.random.RandomState(4)
        X = rng.rand(300, 2)
        Y = np.column_stack([_target(X) * (1.0 + 0.3 * k) + 0.1 * k * X[:, 0] for k in range(3)])
        X_test = rng.rand(40, 2)
        Y_test = np.column_stack([_target(X_test) * (1.0 + 0.3 * k) + 0.1 * k * X_test[:, 0] for k in range(3)])
        for kw in (dict(output_model='independent'), dict(output_model='shared'),
                   dict(output_model='pca', output_basis_components=2)):
            errors = {}
            for mode in ('leaf', 'neighbours'):
                np.random.seed(4)
                gpt = GPTree(Nbar=50, retrain_every_n_points=10, n_outputs=3, predict_from=mode, **kw)
                _stream(gpt, X, Y)
                mean, std = gpt.predict(X_test)
                self.assertEqual(mean.shape, (40, 3))
                self.assertEqual(std.shape, (40, 3))
                self.assertTrue(np.all(np.isfinite(mean)) and np.all(std >= 0.0))
                errors[mode] = np.sqrt(np.mean((mean - Y_test) ** 2))
            self.assertLess(errors['neighbours'], errors['leaf'], msg=str(kw))

    def test_lower_error_and_calibrated_sigma_on_a_stream(self):
        rng = np.random.RandomState(5)
        X = rng.rand(700, 3)
        y = np.sin(4 * X[:, 0]) * np.cos(3 * X[:, 1]) + X[:, 2] ** 2
        result = {}
        for mode in ('leaf', 'neighbours'):
            np.random.seed(5)
            gpt = GPTree(Nbar=50, retrain_every_n_points=25, predict_from=mode)
            errors, inside = [], []
            for i in range(len(X)):
                mean, std = gpt.predict(X[i:i + 1])
                gpt.update_tree(X[i:i + 1], y[i:i + 1, None], 1e-4)
                if i >= 350:
                    errors.append(mean[0, 0] - y[i])
                    inside.append(abs(mean[0, 0] - y[i]) <= std[0, 0])
            result[mode] = (np.sqrt(np.mean(np.square(errors))), np.mean(inside))
        self.assertLess(result['neighbours'][0], 0.8 * result['leaf'][0])
        # were a point among its own neighbours, the registered errors and sigma would collapse
        self.assertGreater(result['neighbours'][1], 0.5)
        self.assertLess(result['neighbours'][1], 0.9)

    def test_a_jump_in_the_target(self):
        """Leaves beside a jump predict from their own points where their neighbours lie across it."""
        rng = np.random.RandomState(6)
        X = rng.rand(900, 2)
        y = np.sin(2 * X[:, 0]) + X[:, 1] + 5.0 * (X[:, 0] > 0.5)
        trees = {}
        for mode in ('leaf', 'neighbours'):
            np.random.seed(6)
            trees[mode] = GPTree(Nbar=50, retrain_every_n_points=25, predict_from=mode, use_calibrated_sigma=False)
            _stream(trees[mode], X, y, sigma=1e-4)
        X_test = rng.rand(400, 2)
        X_test = X_test[np.abs(X_test[:, 0] - 0.5) > 0.02]
        truth = np.sin(2 * X_test[:, 0]) + X_test[:, 1] + 5.0 * (X_test[:, 0] > 0.5)
        errors = {mode: np.abs(trees[mode].predict(X_test)[0][:, 0] - truth) for mode in trees}
        self.assertLess(errors['neighbours'].max(), max(2.0 * errors['leaf'].max(), 0.5))
        # some queries had neighbours across the jump, and their prediction is the leaf's own
        gpt = trees['neighbours']
        own, conditioned = 0, 0
        for q in X_test:
            leaf = _leaf_of(gpt, q[None, :])
            Xn, yn, sn = gpt.neighbours.find(q, leaf.fit_std)
            ys = leaf.y_scalers[0]
            consistency = leaf.my_GPRs[0].condition(leaf.X_scaler.transform(Xn), ys.transform(yn)[:, 0],
                                                    (sn[:, 0] / ys.scale_[0]) ** 2, leaf.X_scaler.transform(q[None, :]))[2][0]
            if consistency > gpt.neighbours.consistency_limit:
                own += 1
                np.testing.assert_allclose(gpt.predict(q[None, :])[0], leaf._predict_own(q[None, :])[0], rtol=1e-9)
            else:
                conditioned += 1
        self.assertGreater(own, 5)
        self.assertGreater(conditioned, 100)

    def test_a_sleeping_leaf_stays_asleep(self):
        np.random.seed(7)
        rng = np.random.RandomState(7)
        gpt = GPTree(Nbar=40, retrain_every_n_points=10, predict_from='neighbours', neighbour_consistency=np.inf)
        _stream(gpt, rng.rand(300, 2))
        q = rng.rand(1, 2)
        leaf = _leaf_of(gpt, q)
        before = gpt.predict(q)
        self.assertTrue(leaf.sleep())
        after = gpt.predict(q)
        self.assertTrue(leaf.asleep)
        np.testing.assert_allclose(after[0], before[0], rtol=1e-12)
        np.testing.assert_allclose(after[1], before[1], rtol=1e-12)

    def test_with_a_memory_limit(self):
        np.random.seed(8)
        rng = np.random.RandomState(8)
        gpt = GPTree(Nbar=40, retrain_every_n_points=10, predict_from='neighbours', max_memory_mb=0.5)
        X = rng.rand(700, 2)
        _stream(gpt, X)
        self.assertLessEqual(gpt.memory_mb, 0.5)
        mean, std = gpt.predict(X[-50:])
        self.assertLess(np.sqrt(np.mean((mean[:, 0] - _target(X[-50:])) ** 2)), 0.05)

    def test_with_a_global_model(self):
        np.random.seed(9)
        rng = np.random.RandomState(9)
        X = rng.rand(260, 2)
        gpt = GPTree(Nbar=50, retrain_every_n_points=10, predict_from='neighbours', global_mean='additive_gp',
                     global_mean_kwargs=dict(reservoir_size=40, min_points=60, n_restarts_optimizer=0, random_state=0))
        _stream(gpt, X)
        self.assertIsNotNone(gpt.global_mean.current)
        X_test = rng.rand(60, 2)
        mean, std = gpt.predict(X_test)
        self.assertLess(np.sqrt(np.mean((mean[:, 0] - _target(X_test)) ** 2)), 0.02)
        self.assertTrue(np.all(std > 0.0))

    def test_after_a_batch_fit_and_a_save(self):
        rng = np.random.RandomState(10)
        X = rng.rand(300, 2)
        np.random.seed(10)
        gpt = GPTree(Nbar=50, predict_from='neighbours', n_neighbours=40, neighbour_leaves=6)
        gpt.fit(X, _target(X)[:, None], np.full(300, 1e-3))
        X_test = rng.rand(30, 2)
        mean, std = gpt.predict(X_test)
        self.assertLess(np.sqrt(np.mean((mean[:, 0] - _target(X_test)) ** 2)), 0.02)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'tree.joblib')
            gpt.save(path)
            loaded = joblib.load(path)
        self.assertIs(loaded.neighbours.tree, loaded)
        np.testing.assert_allclose(loaded.predict(X_test)[0], mean, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
