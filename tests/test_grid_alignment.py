"""Tests for the alignment of outputs on a grid of two dimensions
(``GPTree(output_model='aligned', output_shape=(p1, p2))``)."""

import os
import sys
import tempfile
import unittest
import warnings

import joblib
import numpy as np
from scipy.interpolate import RectBivariateSpline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo import GPTree, Default_GPR, GridWarp, AlignedImages, OutputWarp
from pygptreeo.output_alignment import OVERSAMPLING, _GridSplines, _grid, output_warp

warnings.filterwarnings("ignore")

SHAPE = (16, 16)


def _blob(U, shape=SHAPE):
    """Images of a blob whose centre moves along both axes with the first two
    inputs, and whose width grows with the third. Rows are flattened images."""
    u, v = np.meshgrid(*[(np.arange(p) + 0.5) / p for p in shape], indexing="ij")
    cu, cv, width = (c[:, None, None] for c in (0.35 + 0.3 * U[:, 0], 0.35 + 0.3 * U[:, 1], 0.10 + 0.05 * U[:, 2]))
    return np.exp(-0.5 * ((u - cu) ** 2 + (v - cv) ** 2) / width ** 2).reshape(U.shape[0], -1)


def _fixed_shapes(U, shape=SHAPE):
    """Images in which nothing moves: two shapes with amplitudes from the inputs."""
    u, v = np.meshgrid(*[(np.arange(p) + 0.5) / p for p in shape], indexing="ij")
    first, second = np.sin(np.pi * u) * np.sin(np.pi * v), np.sin(2.0 * np.pi * u) * np.sin(np.pi * v)
    return ((1.0 + U[:, :1, None]) * first + U[:, 1:2, None] * second).reshape(U.shape[0], -1)


def _offsets(U):
    return (U - U.mean(axis=0)) / U.std(axis=0)


class TestGridSplines(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(0)
        self.F = rng.randn(3, 12, 9)
        self.a, self.b = 11.0 * rng.rand(3, 40), 8.0 * rng.rand(3, 40)
        self.splines = _GridSplines(self.F)

    def test_interpolates_the_images(self):
        u, v = (np.tile(g, (3, 1)) for g in _grid((12, 9)))
        np.testing.assert_allclose(self.splines(u, v), self.F.reshape(3, -1), atol=1e-12)

    def test_is_the_tensor_product_spline(self):
        value, du, dv = self.splines(self.a, self.b, derivative=True)
        for i in range(3):
            reference = RectBivariateSpline(np.arange(12), np.arange(9), self.F[i], kx=3, ky=3, s=0)
            np.testing.assert_allclose(value[i], reference.ev(self.a[i], self.b[i]), atol=1e-10)
            np.testing.assert_allclose(du[i], reference.ev(self.a[i], self.b[i], dx=1), atol=1e-10)
            np.testing.assert_allclose(dv[i], reference.ev(self.a[i], self.b[i], dy=1), atol=1e-10)


class TestGridWarp(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(1)
        self.warp = GridWarp((14, 10), 3)
        self.warp.C = 0.4 * rng.randn(*self.warp.C.shape)
        self.delta = rng.randn(5, 3)

    def test_identity_without_a_warp_and_at_the_centre(self):
        u, v = (g.reshape(14, 10) for g in _grid((14, 10)))
        for a, b in (GridWarp((14, 10), 3).phi(self.delta), self.warp.phi(np.zeros((1, 3)))):
            np.testing.assert_allclose(a, np.broadcast_to(u, a.shape), atol=1e-12)
            np.testing.assert_allclose(b, np.broadcast_to(v, b.shape), atol=1e-12)

    def test_maps_the_grid_onto_itself(self):
        a, b = self.warp.phi(self.delta)
        np.testing.assert_allclose(a[:, 0, :], 0.0, atol=1e-12)
        np.testing.assert_allclose(a[:, -1, :], 13.0, atol=1e-12)
        np.testing.assert_allclose(b[:, :, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(b[:, :, -1], 9.0, atol=1e-12)
        self.assertTrue(np.all(np.diff(a, axis=1) > 0.0) and np.all(np.diff(b, axis=2) > 0.0))

    def test_the_warp_of_an_axis_varies_along_the_other(self):
        a, b = self.warp.phi(self.delta)
        self.assertGreater(np.abs(a[:, 7, 0] - a[:, 7, -1]).max(), 1e-3)
        self.assertGreater(np.abs(b[:, 0, 5] - b[:, -1, 5]).max(), 1e-3)

    def test_jacobian(self):
        for axis in (0, 1):
            _, d_knot = self.warp._axis(self.delta, self.warp.C, axis, jacobian=True)
            for i, l, k in ((0, 0, 0), (2, 1, 5), (1, 0, 7)):
                up, down = self.warp.C.copy(), self.warp.C.copy()
                up[axis, i, l, k] += 1e-6
                down[axis, i, l, k] -= 1e-6
                numeric = (self.warp._axis(self.delta, up, axis) - self.warp._axis(self.delta, down, axis)) / 2e-6
                factor = self.delta[:, i, None] * self.warp.cross[axis][None, :, l]          # (n, other)
                factor = factor[:, None, :] if axis == 0 else factor[:, :, None]
                np.testing.assert_allclose(d_knot[k] * factor, numeric, atol=1e-6)

    def test_inverse(self):
        a, b = self.warp.phi(self.delta)
        u, v = self.warp.inverse(self.delta)
        x, y = _grid((14, 10))
        np.testing.assert_allclose(_GridSplines(a)(u, v), np.broadcast_to(x, u.shape), atol=1e-6)
        np.testing.assert_allclose(_GridSplines(b)(u, v), np.broadcast_to(y, v.shape), atol=1e-6)

    def test_register_and_unwarp(self):
        x, y = _grid((14, 10))
        F = np.array([np.exp(-((x - 5.0 - i) ** 2 + (y - 4.0) ** 2) / 12.0) for i in range(5)])
        G = self.warp.register(F, self.delta)
        self.assertEqual(G.shape, (5, (OVERSAMPLING * 13 + 1) * (OVERSAMPLING * 9 + 1)))
        self.assertLess(np.abs(self.warp.unwarp(G, self.delta) - F).max(), 0.02)
        self.assertGreater(np.abs(G.reshape(5, OVERSAMPLING * 13 + 1, -1)[:, ::OVERSAMPLING, ::OVERSAMPLING].reshape(5, -1) - F).max(), 0.05)

    def test_start_from(self):
        ratio = np.array([2.0, 0.5, 1.0])
        other = GridWarp((14, 10), 3)
        other.start_from(self.warp, ratio)
        for a, b in zip(other.phi(self.delta / ratio), self.warp.phi(self.delta)):
            np.testing.assert_allclose(a, b, atol=1e-12)

    def test_output_warp(self):
        self.assertIsInstance(output_warp((40,), 3), OutputWarp)
        self.assertIsInstance(output_warp((8, 9), 3), GridWarp)


class TestGridWarpFit(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(2)
        self.U = 0.3 + 0.4 * rng.rand(60, 3)
        self.delta = _offsets(self.U)

    def test_aligns_a_blob_that_moves(self):
        F = _blob(self.U)
        warp = GridWarp(SHAPE, 3)
        left = warp.fit(F, self.delta)
        self.assertLess(left, 0.05)

        def centres(images, shape):
            """The centre of mass of each image along the two axes."""
            images = images.reshape((-1,) + shape)
            return np.column_stack([(images.sum(axis=2 - axis) @ np.arange(shape[axis])) / images.sum(axis=(1, 2)) for axis in (0, 1)])

        before = centres(F, SHAPE).std(axis=0)
        after = centres(warp.register(F, self.delta), warp.fine).std(axis=0) / OVERSAMPLING
        self.assertTrue(np.all(before > 0.4) and np.all(after < 0.3 * before), (before, after))

    def test_no_warp_where_nothing_moves(self):
        warp = GridWarp(SHAPE, 3)
        warp.fit(_fixed_shapes(self.U), self.delta)
        self.assertLess(np.abs(warp.C).max(), 1e-2)

    def test_warm_fit_stays(self):
        F = _blob(self.U)
        warp = GridWarp(SHAPE, 3)
        left = warp.fit(F, self.delta)
        C = warp.C.copy()
        self.assertLess(abs(warp.fit(F, self.delta) - left), 0.2 * left)
        self.assertLess(np.abs(warp.C - C).max(), 0.1 * np.abs(C).max())


class TestAlignedImages(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(3)
        self.U = 0.3 + 0.4 * rng.rand(40, 3)
        self.delta = _offsets(self.U)
        self.F = _blob(self.U)
        self.warp = GridWarp(SHAPE, 3)
        self.warp.fit(self.F, self.delta)
        self.images = self.warp.curves(self.F, self.warp.register(self.F, self.delta), self.delta)

    def test_combine_returns_an_image_from_its_own_weight(self):
        self.assertIsInstance(self.images, AlignedImages)
        weights = np.eye(40)[:5] - self.images.centre_weights        # the weights of the GP mean are for centred images
        np.testing.assert_allclose(self.images.combine(self.delta[:5], weights), self.F[:5], atol=5e-3)

    def test_with_curve(self):
        f, d = _blob(np.array([[0.5, 0.45, 0.55]])), np.array([[0.1, -0.2, 0.3]])
        more = self.images.with_curve(f[0], d[0])
        self.assertEqual(more.G.shape[0], 41)
        self.assertEqual(more.centre_weights[-1], 0.0)
        np.testing.assert_allclose(more.combine(d, np.eye(41)[-1:] - more.centre_weights), f, atol=5e-3)
        self.assertGreater(more.nbytes(), self.images.nbytes())


class TestAlignedImageTree(unittest.TestCase):

    def _tree(self, model, shape=SHAPE, **kw):
        return GPTree(GPR=Default_GPR(), Nbar=60, retrain_every_n_points=20, n_outputs=int(np.prod(shape)), output_model=model, output_shape=shape, **kw)

    def test_lower_error_than_shared_for_a_blob_that_moves(self):
        rng = np.random.RandomState(4)
        U, U_test = rng.rand(300, 3), rng.rand(100, 3)
        Y, truth = _blob(U), _blob(U_test)
        errors = {}
        for model in ('shared', 'aligned'):
            np.random.seed(0)
            gpt = self._tree(model)
            gpt.batch_update_tree(U, Y, np.full(Y.shape, 1e-4))
            mean, sigma = gpt.predict(U_test)
            self.assertEqual(sigma.shape, truth.shape)
            self.assertTrue(np.all(sigma > 0.0))
            errors[model] = np.sqrt(np.mean((mean - truth) ** 2))
            if model == 'aligned':
                self.assertTrue(any(isinstance(leaf.alignment, AlignedImages) for leaf in gpt.root.leaves))
                with tempfile.TemporaryDirectory() as folder:
                    gpt.save(os.path.join(folder, 'tree.joblib'))
                    loaded = joblib.load(os.path.join(folder, 'tree.joblib'))
                np.testing.assert_allclose(loaded.predict(U_test[:10])[0], mean[:10])
        self.assertLess(1.5 * errors['aligned'], errors['shared'])

    def test_stream_with_points_between_fits(self):
        rng = np.random.RandomState(5)
        U = 0.3 + 0.4 * rng.rand(70, 3)
        Y = _blob(U)
        gpt = self._tree('aligned')
        for i in range(70):
            gpt.update_tree(U[i:i + 1], Y[i:i + 1], np.full((1, Y.shape[1]), 1e-4))
        mean, _ = gpt.predict(U[-5:])
        self.assertLess(np.abs(mean - Y[-5:]).max(), 0.05)

    def test_same_predictions_as_shared_where_nothing_moves(self):
        rng = np.random.RandomState(6)
        U, U_test = rng.rand(120, 3), rng.rand(20, 3)
        Y = _fixed_shapes(U)
        predictions = []
        for model in ('shared', 'aligned'):
            np.random.seed(0)
            gpt = self._tree(model)
            gpt.batch_update_tree(U, Y, np.full(Y.shape, 1e-4))
            predictions.append(gpt.predict(U_test)[0])
        np.testing.assert_allclose(predictions[1], predictions[0], atol=1e-10)

    def test_output_shape_is_checked(self):
        for shape in ((16, 15), (4, 64), (4, 8, 8)):
            with self.assertRaises(ValueError):
                GPTree(GPR=Default_GPR(), n_outputs=256, output_model='aligned', output_shape=shape)
        gpt = GPTree(GPR=Default_GPR(), n_outputs=256, output_model='aligned', output_shape=(256,))
        self.assertEqual(gpt.root.output_shape, (256,))
        self.assertEqual(GPTree(GPR=Default_GPR(), n_outputs=256, output_model='aligned').root.output_shape, (256,))


if __name__ == '__main__':
    unittest.main()
