"""Tests for predictions conditioned on a query's nearest stored points
(``predict_from='neighbours'``)."""

import os
import sys
import tempfile
import unittest
import warnings
from unittest import mock

import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo import GPTree, Default_GPR, GPRegressorInterface
from pygptreeo import gpnode as gpnode_module
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
            self.assertEqual(std.shape, (7, n_targets))
            np.testing.assert_allclose(std, np.tile(ref_std.reshape(7, -1)[:, :1], (1, n_targets)), rtol=1e-6, atol=1e-9)
            self.assertEqual(consistency.shape, (n_targets,))
            # the fit is as it was
            before = gpr.predict(X_query, return_std=True)
            own = gpr.condition(X, values(X), np.full(50, 1e-6), X_query)
            np.testing.assert_allclose(own[0], np.reshape(before[0], (7, n_targets)), rtol=1e-7, atol=1e-9)

    def test_targets_normalised_by_the_gp(self):
        """With ``normalize_y`` each target has its own scale, in the mean and in sigma."""
        rng = np.random.RandomState(2)
        X, X_other, X_query = rng.rand(50, 2), rng.rand(40, 2), rng.rand(5, 2)

        def values(Z):
            return np.column_stack((np.sin(3 * Z[:, 0]) + Z[:, 1], 40.0 * np.cos(2 * Z[:, 1]) + 100.0))
        kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0])
        gpr = Default_GPR(kernel=kernel, normalize_y=True)
        gpr.set_observation_noise(np.full(50, 1e-6))
        gpr.fit(X, values(X))
        mean, std, _ = gpr.condition(X, values(X), np.full(50, 1e-6), X_query)
        ref_mean, ref_std = gpr.predict(X_query, return_std=True)
        np.testing.assert_allclose(mean, ref_mean, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(std, ref_std, rtol=1e-5, atol=1e-9)
        self.assertGreater(std[0, 1], 5.0 * std[0, 0])

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
            Xn, yn, sn, growth = search.find(q, w)
            self.assertIsNone(growth)
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
        for fade in (None, 0.5):
            Xn, yn, sn, growth = NeighbourSearch(gpt, n_neighbours=50, fade=fade).find(np.array([0.5, 0.5]), np.ones(2))
            self.assertEqual(Xn.shape, (12, 2))
            self.assertIsNone(growth)           # no stored point beyond the neighbours
        self.assertIsNone(NeighbourSearch(GPTree(), n_neighbours=5).find(np.zeros(2), np.ones(2)))     # an empty tree

    def test_neighbours_that_fade_out(self):
        """The same points, with a noise that grows from ``fade`` to the first point beyond them."""
        k = 25
        for d, max_leaves in ((2, None), (4, None), (2, 6)):
            gpt = _grown_tree(n=700, d=d, seed=d)
            X = _stored_points(gpt)[0]
            rng = np.random.RandomState(d)
            for q in rng.uniform(-0.1, 1.1, (40, d)):
                w = rng.uniform(0.05, 1.0, d)
                plain = NeighbourSearch(gpt, n_neighbours=k, max_leaves=max_leaves).find(q, w)[0]
                Xn, _, _, growth = NeighbourSearch(gpt, n_neighbours=k, max_leaves=max_leaves, fade=0.6).find(q, w)
                distance = np.sqrt((((Xn - q) / w) ** 2).sum(axis=1))
                np.testing.assert_allclose(np.sort(distance), np.sort(np.sqrt((((plain - q) / w) ** 2).sum(axis=1))), rtol=1e-12)
                self.assertEqual(growth.shape, (k,))
                if max_leaves is None:
                    beyond = np.sort(np.sqrt((((X - q) / w) ** 2).sum(axis=1)))[k]
                    t = np.clip((distance / beyond - 0.6) / 0.4, 0.0, 1.0)
                    np.testing.assert_allclose(growth, t ** 2 / (1.0 - t) ** 2, rtol=1e-9)
                    self.assertTrue(np.all(growth[distance <= 0.6 * beyond] == 0.0))        # the near ones keep their noise
                self.assertTrue(np.all(np.diff(growth[np.argsort(distance)]) >= 0.0))
                self.assertGreater(growth.max(), 0.0)
        with self.assertRaises(ValueError):
            NeighbourSearch(gpt, n_neighbours=k, fade=1.0)
        with self.assertRaises(ValueError):
            GPTree(predict_from='neighbours', neighbour_fade=-0.1)

    def test_a_limit_on_the_leaves_searched(self):
        """The points then come from the leaves whose points' boxes are nearest to the query."""
        gpt = _grown_tree(n=600, d=2)
        leaves = [leaf for leaf in gpt.root.leaves if leaf.n_points > 0]
        rng = np.random.RandomState(0)
        for q in rng.uniform(-0.1, 1.1, (40, 2)):
            for m in (1, 3):
                Xn = NeighbourSearch(gpt, n_neighbours=10, max_leaves=m).find(q, np.ones(2))[0]
                gaps = [np.maximum(leaf.my_X_data.min(axis=0) - q, 0.0) + np.maximum(q - leaf.my_X_data.max(axis=0), 0.0)
                        for leaf in leaves]
                nearest = np.argsort([(gap ** 2).sum() for gap in gaps], kind='stable')[:m]
                points = np.vstack([leaves[i].my_X_data for i in nearest])
                np.testing.assert_allclose(np.sort(((Xn - q) ** 2).sum(axis=1)),
                                           np.sort(((points - q) ** 2).sum(axis=1))[:10])


class TestPredictions(unittest.TestCase):

    def test_equal_a_gp_on_the_neighbours_with_the_leaf_kernel(self):
        """Without fading and with it, where the noise of the outer neighbours is raised."""
        k = 30
        for fade in (None, 0.5):
            np.random.seed(0)
            rng = np.random.RandomState(0)
            gpt = GPTree(Nbar=40, retrain_every_n_points=10, predict_from='neighbours', n_neighbours=k, neighbour_fade=fade,
                         neighbour_consistency=np.inf, neighbour_leaves=None, use_calibrated_sigma=False)
            _stream(gpt, rng.rand(400, 2))
            X, y, sigma = _stored_points(gpt)
            raised = 0
            for q in rng.rand(15, 2):
                leaf = _leaf_of(gpt, q[None, :])
                if leaf.prefers_own:
                    continue
                distance = np.sqrt((((X - q) / leaf.fit_std) ** 2).sum(axis=1))
                order = np.argsort(distance)
                idx = order[:k]
                noise = sigma[idx, 0] ** 2
                if fade is not None:
                    t = np.clip((distance[idx] / distance[order[k]] - fade) / (1.0 - fade), 0.0, 1.0)
                    self.assertIsNotNone(leaf._predicted_variance())
                    noise = noise + t ** 2 / (1.0 - t) ** 2 * (noise + leaf._predicted_variance()[0])
                    raised += int(np.sum(t > 0.0))
                xs, ys = leaf.X_scaler, leaf.y_scalers[0]
                reference = GaussianProcessRegressor(kernel=leaf.my_GPRs[0].sklearn_gpr.kernel_, optimizer=None,
                                                     alpha=noise / ys.scale_[0] ** 2)
                reference.fit(xs.transform(X[idx]), ys.transform(y[idx]))
                ref_mean, ref_std = reference.predict(xs.transform(q[None, :]), return_std=True)
                mean, std = gpt.predict(q[None, :])
                np.testing.assert_allclose(mean[0, 0], ref_mean[0] * ys.scale_[0] + ys.mean_[0], rtol=1e-6, atol=1e-8)
                np.testing.assert_allclose(std[0, 0], ref_std[0] * ys.scale_[0], rtol=1e-4, atol=1e-8)
            self.assertEqual(raised > 0, fade is not None)

    def test_fading_neighbours_give_a_prediction_without_steps(self):
        """Along a path within a leaf the mean steps where the set of neighbours changes, unless they fade out."""
        np.random.seed(2)
        rng = np.random.RandomState(2)
        gpt = GPTree(GPR=_fixed_gpr(2, length_scale=0.5), Nbar=60, predict_from='neighbours', n_neighbours=30,
                     neighbour_leaves=None, neighbour_consistency=np.inf)
        _stream(gpt, rng.rand(700, 2), sigma=1e-2)
        self.assertEqual(gpt.neighbours.fade, 0.9)
        compared = 0
        for leaf in gpt.root.leaves:
            lo, hi = gpt.leaf_index.cell(leaf)
            lo, hi = np.maximum(lo, 0.0), np.minimum(hi, 1.0)
            if leaf.prefers_own or leaf._predicted_variance() is None or np.any(hi - lo < 0.05):
                continue
            path = lo + np.linspace(0.35, 0.65, 3000)[:, None] * (hi - lo)
            self.assertTrue(all(gpt.leaf_index.locate(q) is leaf for q in path[::50]))
            largest = {}
            for fade in (None, 0.9):
                gpt.neighbours.fade = fade
                step = np.diff(gpt.predict(path)[0][:, 0])
                # what a step adds to the mean of the steps before and after it
                largest[fade] = np.abs(step[1:-1] - 0.5 * (step[:-2] + step[2:])).max()
            gpt.neighbours.fade = 0.9
            if largest[None] > 1e-6:
                self.assertLess(largest[0.9], 0.05 * largest[None])
                compared += 1
        self.assertGreater(compared, 3)

    def _both(self, n=260, d=2, **kw):
        """The predictions along a stream of a tree that predicts from its leaves
        and of one that predicts from neighbours no kernel is allowed to fit."""
        rng = np.random.RandomState(3)
        X = rng.rand(n, d)
        n_outputs = kw.get('n_outputs', 1)
        Y = np.column_stack([_target(X) * (1.0 + 0.3 * k) + 0.1 * k * X[:, 0] for k in range(n_outputs)])
        out = []
        for options in (dict(), dict(predict_from='neighbours', neighbour_consistency=0.0, neighbour_blend=None)):
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
            trees[mode] = GPTree(Nbar=50, retrain_every_n_points=25, predict_from=mode, use_calibrated_sigma=False,
                                 neighbour_fade=None, neighbour_blend=None)
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
            Xn, yn, sn, _ = gpt.neighbours.find(q, leaf.fit_std)
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

    def test_a_merged_point_is_predicted_before_the_merge(self):
        rng = np.random.RandomState(4)
        np.random.seed(4)
        gpt = GPTree(GPR=_fixed_gpr(2), Nbar=40, predict_from='neighbours',
                     enable_point_merging=True, merge_distance_threshold=0.02, min_points_before_merging=10)
        X = rng.rand(150, 2)
        _stream(gpt, X)
        merged = moved = 0
        for x in X[:80] + 0.001:
            x = x[None, :]
            leaf = _leaf_of(gpt, x)
            if leaf.merge_candidate(x) is None:
                continue
            before, n_merges = gpt.predict(x)[0], leaf.n_merges
            gpt.update_tree(x, np.array([[_target(x)[0] + 0.5]]), 1e-3)
            self.assertEqual(leaf.n_merges, n_merges + 1)
            merged += 1
            np.testing.assert_allclose(leaf.mu_preds_list[0][0], np.ravel(before)[0])
            moved += not np.allclose(gpt.predict(x)[0], before)
        self.assertGreater(merged, 5)
        self.assertGreater(moved, 0)        # the merged point is among the neighbours afterwards

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


class TestBlend(unittest.TestCase):
    """Predictions within the zones at the cuts (``neighbour_blend``)."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(11)
        rng = np.random.RandomState(11)
        cls.gpt = GPTree(Nbar=40, retrain_every_n_points=10, predict_from='neighbours', n_neighbours=30,
                         neighbour_leaves=None, neighbour_consistency=np.inf, use_calibrated_sigma=False)
        cls.X = rng.rand(700, 2)
        _stream(cls.gpt, cls.X, sigma=1e-3)
        cls.queries = rng.rand(400, 2)

    def _in_zones(self, gpt, queries, n_leaves=2):
        for q in queries:
            leaves, weights = gpt._leaves_of(q[None, :])
            if len(leaves) >= n_leaves and all(leaf._can_condition() and not leaf.prefers_own for leaf in leaves):
                yield q[None, :], leaves, np.array(weights)

    def test_the_covariance_of_the_backend(self):
        rng = np.random.RandomState(0)
        X, Z = rng.rand(40, 2), rng.rand(6, 2)
        Y = np.column_stack((np.sin(3 * X[:, 0]), 40.0 * X[:, 1] + 100.0))
        for normalize_y in (False, True):
            gpr = Default_GPR(normalize_y=normalize_y)
            self.assertIsNone(gpr.covariance(Z))
            gpr.set_observation_noise(np.full(40, 1e-6))
            gpr.fit(X, Y)
            K, scale, mean = gpr.covariance(Z)
            np.testing.assert_allclose(K, gpr.sklearn_gpr.kernel_(Z))
            np.testing.assert_allclose(scale, Y.std(axis=0) ** 2 if normalize_y else np.ones(2))
            np.testing.assert_allclose(mean, Y.mean(axis=0) if normalize_y else np.zeros(2), atol=1e-12)
        self.assertIsNone(GPRegressorInterface.covariance(gpr, Z))

    def test_equal_a_gp_with_the_sum_of_the_kernels(self):
        gpt = self.gpt
        X, y, sigma = _stored_points(gpt)
        k, fade, checked, three = gpt.neighbours.n_neighbours, gpt.neighbours.fade, 0, 0
        for x, leaves, weights in self._in_zones(gpt, self.queries):
            metric = sum(w * leaf.fit_std for w, leaf in zip(weights, leaves))
            distance = np.sqrt((((X - x[0]) / metric) ** 2).sum(axis=1))
            order = np.argsort(distance)
            idx = order[:k]
            Z = np.vstack((X[idx], x))
            K, mean0, unit = np.zeros((k + 1, k + 1)), 0.0, 0.0
            for w, leaf in zip(weights, leaves):
                xs, ys = leaf.X_scaler, leaf.y_scalers[0]
                K += w * ys.scale_[0] ** 2 * leaf.my_GPRs[0].sklearn_gpr.kernel_(xs.transform(Z))
                mean0 += w * ys.mean_[0]
                unit += w * leaf._predicted_variance()[0]
            t = np.clip((distance[idx] / distance[order[k]] - fade) / (1.0 - fade), 0.0, 1.0)
            noise = sigma[idx, 0] ** 2
            noise = noise + t ** 2 / (1.0 - t) ** 2 * (noise + unit)
            A = K[:k, :k] + np.diag(noise)
            ref_mean = mean0 + K[:k, k] @ np.linalg.solve(A, y[idx, 0] - mean0)
            ref_std = np.sqrt(K[k, k] - K[:k, k] @ np.linalg.solve(A, K[:k, k]))
            mean, std = gpt.predict(x)
            np.testing.assert_allclose(mean[0, 0], ref_mean, rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(std[0, 0], ref_std, rtol=1e-4, atol=1e-8)
            np.testing.assert_array_equal(gpt.predict(x, mode='loop')[0], mean)
            checked += 1
            three += len(leaves) > 2
        self.assertGreater(checked, 40)
        self.assertGreater(three, 0)

    def test_one_leaf_alone_is_its_own_prediction(self):
        gpt = self.gpt
        alone = 0
        for q in self.queries[:150]:
            x = q[None, :]
            leaves, weights = gpt._leaves_of(x)
            if len(leaves) > 1 or not leaves[0]._can_condition() or leaves[0].prefers_own:
                continue
            mixed = gpt.neighbours.predict_mixed(x, leaves, np.array(weights))
            mean, std = leaves[0].predict(x, return_std=True, use_calibrated_sigma=False)
            np.testing.assert_allclose(mixed[0], mean[0], rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(mixed[1], std[0], rtol=1e-6, atol=1e-10)
            alone += 1
        self.assertGreater(alone, 30)

    def test_no_step_at_a_cut(self):
        """Along a path through a cut the mean steps where the leaf changes, unless the kernels are mixed."""
        gpt = self.gpt
        index = gpt.leaf_index
        compared = 0
        for leaf in gpt.root.leaves:
            lo, hi, half = index.faces(leaf)
            data_lo, data_hi = index.data_box(leaf)
            for j in range(2):
                if not (np.isfinite(hi[j]) and half[j] > 0.0 and leaf._can_condition()):
                    continue
                # through the upper face along input j, at the middle of the leaf along the other input
                centre = 0.5 * (np.where(np.isfinite(lo), lo, data_lo) + np.where(np.isfinite(hi), hi, data_hi))
                path = np.tile(centre, (2000, 1))
                path[:, j] = np.linspace(hi[j] - 4 * half[j], hi[j] + 4 * half[j], 2000)
                largest = {}
                for blend in (None, 0.1):
                    gpt.neighbours.blend = blend
                    step = np.diff(gpt.predict(path)[0][:, 0])
                    largest[blend] = np.abs(step[1:-1] - 0.5 * (step[:-2] + step[2:])).max()
                if largest[None] > 1e-5:
                    self.assertLess(largest[0.1], 0.1 * largest[None])
                    compared += 1
        self.assertGreater(compared, 8)

    def test_a_leaf_that_predicts_from_its_own_points(self):
        """In its zones the predictions of the leaves are mixed."""
        gpt = self.gpt
        checked = 0
        for x, leaves, weights in self._in_zones(gpt, self.queries[:120]):
            leaves[0].prefers_own = True
            try:
                self.assertIsNone(gpt.neighbours.predict_mixed(x, leaves, weights))
                parts = [leaf.predict(x, return_std=True, use_calibrated_sigma=False) for leaf in leaves]
                mean = sum(w * mu[0, 0] for w, (mu, _) in zip(weights, parts))
                variance = sum(w * (sd[0, 0] ** 2 + (mu[0, 0] - mean) ** 2) for w, (mu, sd) in zip(weights, parts))
                predicted = gpt.predict(x)
                np.testing.assert_allclose(predicted[0][0, 0], mean, rtol=1e-9)
                np.testing.assert_allclose(predicted[1][0, 0], np.sqrt(variance), rtol=1e-9)
            finally:
                leaves[0].prefers_own = False
            checked += 1
        self.assertGreater(checked, 10)

    def _alone_equals_the_leaf(self, gpt, queries, minimum=20):
        """With one leaf and weight 1 the mixed prediction is the leaf's own from its neighbours."""
        checked = 0
        for q in queries:
            x = q[None, :]
            leaf = gpt.leaf_index.locate(q)
            if leaf is None or not leaf._can_condition() or leaf.prefers_own:
                continue
            mean, std = leaf.predict(x, return_std=True, use_calibrated_sigma=False)
            mixed = gpt.neighbours.predict_mixed(x, [leaf], np.array([1.0]))
            if mixed is None:
                continue
            np.testing.assert_allclose(mixed[0], mean[0], rtol=1e-7, atol=1e-9)
            np.testing.assert_allclose(mixed[1], std[0], rtol=1e-6, atol=1e-9)
            checked += 1
        self.assertGreater(checked, minimum)

    def test_targets_normalised_by_the_gp_without_scaling(self):
        np.random.seed(15)
        rng = np.random.RandomState(15)
        X = rng.rand(400, 2)
        for scaling in (True, False):
            gpt = GPTree(GPR=Default_GPR(normalize_y=True), Nbar=40, retrain_every_n_points=10, predict_from='neighbours',
                         n_neighbours=30, neighbour_consistency=np.inf, use_standard_scaling=scaling)
            _stream(gpt, X, 50.0 + 20.0 * _target(X), sigma=2.0)
            self._alone_equals_the_leaf(gpt, self.queries[:120])

    def test_with_a_global_model_in_the_zones(self):
        """The error of the global model is in sigma, and no leaf predicts with an outdated snapshot."""
        np.random.seed(16)
        rng = np.random.RandomState(16)
        gpt = GPTree(Nbar=40, retrain_every_n_points=10, predict_from='neighbours', n_neighbours=30,
                     neighbour_consistency=np.inf, global_mean='additive_gp',
                     global_mean_kwargs=dict(reservoir_size=40, min_points=60, n_restarts_optimizer=0, random_state=0))
        X = rng.rand(400, 2)
        _stream(gpt, X, _target(X) + 0.3 * np.sin(25.0 * X[:, 0] * X[:, 1]))        # not a sum of functions of one input
        self.assertIsNotNone(gpt.global_mean.current)
        self.assertGreater(float(np.max(gpt.global_mean.error_scale)), 0.0)
        self._alone_equals_the_leaf(gpt, self.queries[:120])
        gpt.global_mean.fit()
        current, refreshed = gpt.global_mean.current.version, 0
        for x, leaves, weights in self._in_zones(gpt, self.queries[:150]):
            if any(leaf._fitted_global is None or leaf._fitted_global.version < current for leaf in leaves):
                gpt.predict(x)
                self.assertTrue(all(leaf._fitted_global.version == current for leaf in leaves))
                refreshed += 1
        self.assertGreater(refreshed, 0)

    def test_a_memory_limit_with_leaves_that_predict_themselves(self):
        """Leaves woken for the prediction of a zone are counted, so that the limit holds."""
        np.random.seed(17)
        rng = np.random.RandomState(17)
        gpt = GPTree(Nbar=100, retrain_every_n_points=25, predict_from='neighbours', n_neighbours=40, max_memory_mb=1.2)
        X = rng.rand(1500, 2)
        y = np.sin(3.0 * X[:, 0]) + X[:, 1] + 3.0 * (X[:, 0] > 0.5)
        for i in range(len(X)):
            gpt.update_tree(X[i:i + 1], y[i:i + 1, None], 1e-3)
            if i % 50 == 49:
                awake = [leaf for leaf in gpt.root.leaves if leaf._is_fitted() and not leaf.asleep]
                self.assertTrue(all(leaf in gpt._fitted for leaf in awake))
                self.assertLessEqual(sum(gpt._leaf_size(leaf)[0] for leaf in gpt.root.leaves), 1.2e6)
        self.assertTrue(any(leaf.asleep for leaf in gpt.root.leaves))

    def test_no_more_leaves_than_allowed(self):
        np.random.seed(18)
        rng = np.random.RandomState(18)
        gpt = GPTree(Nbar=40, retrain_every_n_points=10, predict_from='neighbours', max_n_pred_leaves=1)
        _stream(gpt, rng.rand(400, 2))
        in_zone = 0
        for x in rng.rand(120, 2):
            x = x[None, :]
            several = len(gpt._leaves_of(x)[0]) > 1
            self.assertEqual(len(gpt._prediction_leaves(x)[0]), 1)
            before = gpt.predict(x)[0][0, 0]
            owner = gpt.leaf_index.locate(x[0])
            gpt.update_tree(x, np.array([[_target(x)[0]]]), 1e-3)
            if owner is not None and owner.is_leaf and gpt._prediction_leaves(x)[0][0] is owner:
                in_zone += several
        self.assertGreater(in_zone, 10)
        # a batch fit registers nothing before the leaves are fitted, and predicts nothing for it
        fresh = GPTree(Nbar=40, predict_from='neighbours')
        with mock.patch.object(fresh, '_predict_blended', wraps=fresh._predict_blended) as blended:
            X = rng.rand(300, 2)
            fresh.fit(X, _target(X)[:, None], np.full(300, 1e-3))
        self.assertEqual(blended.call_count, 0)

    def test_sleeping_leaves_stay_asleep(self):
        gpt = self.gpt
        checked = 0
        for x, leaves, weights in self._in_zones(gpt, self.queries[:80]):
            before = gpt.predict(x)
            asleep = [leaf for leaf in leaves if leaf.sleep()]
            try:
                after = gpt.predict(x)
                self.assertTrue(asleep and all(leaf.asleep for leaf in asleep))
                np.testing.assert_allclose(after[0], before[0], rtol=1e-12)
                np.testing.assert_allclose(after[1], before[1], rtol=1e-12)
            finally:
                for leaf in asleep:
                    leaf.wake()
            checked += 1
        self.assertGreater(checked, 5)

    def test_sigma_is_calibrated_on_the_prediction_of_the_zone(self):
        np.random.seed(12)
        rng = np.random.RandomState(12)
        gpt = GPTree(Nbar=40, retrain_every_n_points=10, predict_from='neighbours')
        _stream(gpt, rng.rand(500, 2))
        in_zone = 0
        for x in rng.rand(150, 2):
            x = x[None, :]
            leaves, _ = gpt._leaves_of(x)
            before = gpt.predict(x)[0][0, 0]
            owner = gpt.leaf_index.locate(x[0])
            if owner is None:
                continue
            gpt.update_tree(x, np.array([[_target(x)[0]]]), 1e-3)
            if owner.is_leaf:
                np.testing.assert_allclose(owner.mu_preds_list[0][0], before, rtol=1e-9)
                in_zone += len(leaves) > 1
        self.assertGreater(in_zone, 20)

    def test_other_output_models_mix_the_predictions(self):
        rng = np.random.RandomState(13)
        X = rng.rand(300, 2)
        Y = np.column_stack([_target(X) * (1.0 + 0.3 * k) + 0.1 * k * X[:, 0] for k in range(3)])
        for kw in (dict(output_model='shared'), dict(output_model='pca', output_basis_components=2), dict()):
            np.random.seed(13)
            gpt = GPTree(Nbar=50, retrain_every_n_points=10, n_outputs=3, predict_from='neighbours', **kw)
            _stream(gpt, X, Y)
            mixed = 0
            for x, leaves, weights in self._in_zones(gpt, self.queries[:200]):
                result = gpt.neighbours.predict_mixed(x, leaves, weights)
                self.assertEqual(result is None, bool(kw))
                mean, std = gpt.predict(x)
                self.assertTrue(np.all(np.isfinite(mean)) and np.all(std > 0.0))
                mixed += 1
            self.assertGreater(mixed, 5)

    def test_the_options(self):
        for blend in (0.0, 1.5, -0.1):
            with self.assertRaises(ValueError):
                GPTree(predict_from='neighbours', neighbour_blend=blend)
        np.random.seed(14)
        rng = np.random.RandomState(14)
        plain = GPTree(Nbar=40, retrain_every_n_points=10, predict_from='neighbours', neighbour_blend=None)
        leaf_tree = GPTree(Nbar=40, retrain_every_n_points=10)
        X = rng.rand(300, 2)
        _stream(plain, X)
        _stream(leaf_tree, X)
        for gpt in (plain, leaf_tree):
            self.assertFalse(gpt._blends())
            self.assertTrue(all(np.all(gpt.leaf_index.faces(leaf)[2] == 0.0) for leaf in gpt.root.leaves))
            self.assertTrue(all(len(gpt._leaves_of(q[None, :])[0]) == 1 for q in self.queries[:100]))
        self.assertTrue(self.gpt._blends())


class TestChoice(unittest.TestCase):
    """A leaf predicts from its own points while they predicted its last points clearly better."""

    def test_a_leaf_turns_to_its_own_gp_and_back(self):
        rng = np.random.RandomState(11)
        np.random.seed(11)
        # one leaf, whose neighbours are all its points: the two predictions are the same
        gpt = GPTree(Nbar=500, retrain_every_n_points=20, predict_from='neighbours')
        _stream(gpt, rng.rand(60, 2))
        leaf = gpt.root
        self.assertFalse(leaf.prefers_own)
        useless = lambda self, x: (np.full((len(x), 1), 50.0), np.ones((len(x), 1)))
        with mock.patch.object(type(leaf), '_predict_from_neighbours', useless):
            _stream(gpt, rng.rand(gpnode_module.MIN_SCORES, 2))
            self.assertTrue(leaf.prefers_own)
            q = rng.rand(5, 2)
            np.testing.assert_allclose(gpt.predict(q)[0], leaf._predict_own(q)[0], rtol=1e-9)
            self.assertLess(np.abs(gpt.predict(q)[0][:, 0] - _target(q)).max(), 0.5)
        # the neighbours predict as well as the leaf again: it returns to them
        _stream(gpt, rng.rand(gpnode_module.N_SCORES, 2))
        self.assertFalse(leaf.prefers_own)

    def test_children_start_with_the_record_of_their_parent(self):
        rng = np.random.RandomState(12)
        np.random.seed(12)
        gpt = GPTree(Nbar=40, retrain_every_n_points=10, predict_from='neighbours')
        _stream(gpt, rng.rand(39, 2))
        root = gpt.root
        self.assertTrue(root.is_leaf)
        root.prefers_own = True
        record = list(root._scores)
        self.assertGreater(len(record), 0)
        gpt.update_tree(rng.rand(1, 2), np.array([[0.3]]), 1e-3)
        self.assertFalse(root.is_leaf)
        for child in root.children:
            self.assertEqual(len(child._scores), len(root._scores))
            self.assertIsNot(child._scores, root._scores)

    def test_two_outputs(self):
        rng = np.random.RandomState(13)
        np.random.seed(13)
        X = rng.rand(120, 2)
        Y = np.column_stack((_target(X), 2.0 * _target(X) + X[:, 0]))
        gpt = GPTree(Nbar=60, retrain_every_n_points=10, n_outputs=2, predict_from='neighbours')
        _stream(gpt, X, Y)
        for leaf in gpt.root.leaves:
            self.assertEqual(np.array(leaf._scores).shape[1:], (2, 2))
            self.assertIn(leaf.prefers_own, (True, False))


if __name__ == '__main__':
    unittest.main()
