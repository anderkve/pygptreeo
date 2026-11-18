"""Alignment of curve-valued outputs for ``GPTree(output_model='aligned')``.

The outputs are a curve on a grid. A leaf learns a warp of the grid that depends
on the inputs and brings the features of its curves (a peak, an edge, the phase
of an oscillation) to common positions:

    phi(u; delta) = (p - 1) I(u) / I(p - 1),    I(u) = int_0^u exp(delta . w(s)) ds

``u`` is the grid index, ``delta`` the offset of the inputs from the mean of the
leaf's points in units of their standard deviation, and each ``w_i`` is piecewise
linear on a few knots. The warp is monotone, maps the grid onto itself and is
the identity at ``delta = 0``. The aligned curve is ``g(u) = f(phi(u; delta))``.

The leaf's GP is fitted to the aligned curves. A prediction combines the leaf's
curves with the weights of the GP mean, each curve warped from its own frame
into that of the query (:meth:`AlignedCurves.combine`).

Outputs on a grid of two dimensions (``GPTree(output_shape=(p1, p2))``) have a
warp of each axis that varies along the other one (:class:`GridWarp`):

    (u, v) -> (phi_1(u; v, delta), phi_2(v; u, delta))

The GP is fitted to the aligned images on a finer grid, and a prediction is
its mean read where each grid point of the query comes from
(:meth:`AlignedImages.combine`).
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import least_squares

N_KNOTS = 8             # knots of each w_i
BLURS = (1 / 16, 1 / 64)  # widths of the blurs that a fit from C = 0 also starts from, over the length of the grid
MAX_LOG_STRETCH = 2.0   # bound on |delta . w|: the grid is locally stretched by at most exp(2)
RIDGE = 1e-3            # weight of |w|^2 in the fit, which keeps a warp that the curves do not determine at zero
FIT_TOLERANCE = 1e-4    # relative change of the fit's cost or of the warp at which the fit stops
MIN_OUTPUTS = 16        # grid points below which there is no curve to align
N_CROSS = 2             # knots along the other axis of an image, between which the warp of an axis varies linearly
GRID_BLUR = 1 / 8       # width of the blur that a fit of a GridWarp from C = 0 also starts from, over the length of an axis
MAX_FIT_POINTS = 32     # points along an axis on which a GridWarp is fitted; a finer image is blurred and thinned out
OVERSAMPLING = 2        # points of an aligned image for each grid point along an axis
MIN_GRID = 8            # points along an axis below which an image is not aligned
GRID_RIDGE = 3e-2       # weight of |C|^2 in the fit of a GridWarp, which keeps it within the range of the offsets it is fitted on


class _RowSplines:
    """Cubic splines through the rows of ``F`` on the index grid, evaluated at one
    set of positions per row."""

    def __init__(self, F: np.ndarray):
        self.n, self.p = F.shape
        self.c = CubicSpline(np.arange(self.p), F, axis=1).c       # (4, p - 1, n)

    def __call__(self, pos: np.ndarray, derivative: bool = False):
        """Values at ``pos`` of shape (n, m), and the derivatives if asked for."""
        pos = np.clip(pos, 0.0, self.p - 1.0)
        idx = np.minimum(pos.astype(int), self.p - 2)
        t = pos - idx
        c = self.c[:, idx, np.arange(self.n)[:, None]]
        value = ((c[0] * t + c[1]) * t + c[2]) * t + c[3]
        if derivative:
            return value, (3.0 * c[0] * t + 2.0 * c[1]) * t + c[2]
        return value


def _cumulative(a: np.ndarray) -> np.ndarray:
    """Trapezoidal integral along the last axis, from the first grid point."""
    steps = 0.5 * (a[..., 1:] + a[..., :-1])
    return np.concatenate([np.zeros(a.shape[:-1] + (1,)), np.cumsum(steps, axis=-1)], axis=-1)


class OutputWarp:
    """The warp ``phi(u; delta)`` of a grid of ``n_outputs`` points for inputs of
    ``n_features`` dimensions. ``C`` holds the knot values of each ``w_i``."""

    def __init__(self, n_outputs: int, n_features: int, n_knots: int = N_KNOTS):
        self.p, self.d, self.K = int(n_outputs), int(n_features), int(n_knots)
        u = np.arange(self.p, dtype=float)
        knots = np.linspace(0.0, self.p - 1.0, self.K)
        self.B = np.column_stack([np.interp(u, knots, row) for row in np.eye(self.K)])   # (p, K) hat functions
        self.C = np.zeros((self.d, self.K))

    def phi(self, delta: np.ndarray, C=None, jacobian: bool = False):
        """The warp on the grid for each row of ``delta``: (n, p). With
        ``jacobian``, also its derivative with respect to ``C``: (d, K, n, p)."""
        C = self.C if C is None else C
        s = delta @ (C @ self.B.T)
        density = np.exp(np.clip(s, -MAX_LOG_STRETCH, MAX_LOG_STRETCH))
        integral = _cumulative(density)
        total = integral[:, -1:]
        phi = (self.p - 1) * integral / total
        if not jacobian:
            return phi
        free = np.abs(s) < MAX_LOG_STRETCH
        d_density = (density * free)[None, None] * delta.T[:, None, :, None] * self.B.T[None, :, None, :]
        d_integral = _cumulative(d_density)
        return phi, (self.p - 1) * (d_integral / total - integral * d_integral[..., -1:] / total ** 2)

    def inverse(self, delta: np.ndarray) -> np.ndarray:
        """``phi^-1`` at the grid points for each row of ``delta``: (n, p)."""
        u = np.arange(self.p, dtype=float)
        phi = self.phi(delta)
        splines = _RowSplines(phi)
        inv = np.vstack([np.interp(u, row, u) for row in phi])
        for _ in range(3):      # Newton steps on the splines through phi
            value, slope = splines(inv, derivative=True)
            inv = np.clip(inv - (value - u) / np.maximum(slope, 1e-3), 0.0, self.p - 1.0)
        return inv

    def register(self, F: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """The aligned curves ``g(u) = f(phi(u; delta))`` of the rows of ``F``."""
        return _RowSplines(F)(self.phi(delta))

    def start_from(self, other: 'OutputWarp', ratio: np.ndarray):
        """Take the warp of ``other``, whose offsets are ``ratio`` times these."""
        self.C = other.C * ratio[:, None]

    def curves(self, F: np.ndarray, aligned: np.ndarray, delta: np.ndarray) -> 'AlignedCurves':
        """What a leaf with the curves ``F`` at the offsets ``delta`` predicts with."""
        return AlignedCurves(self, F, delta)

    def fit(self, F: np.ndarray, delta: np.ndarray, max_nfev: int = 30) -> float:
        """Choose ``C`` so that the aligned curves are as linear in ``delta`` as
        possible, starting from the current ``C``. Returns the fraction that is
        left of the part of the curves that is not linear in ``delta``.

        From ``C = 0`` the fit is also run from a ``C`` fitted to the curves
        blurred along the grid, in which features that lie apart still
        overlap, and the better of the two results is kept."""
        cold = not np.any(self.C)
        left = self._fit(F, delta, max_nfev)
        if cold:
            direct, self.C = self.C, np.zeros_like(self.C)
            for blur in BLURS:
                width = blur * self.p
                step = max(1, int(width // 2))
                blurred = gaussian_filter1d(F, width, axis=1, mode="nearest")[:, ::step]
                coarse = OutputWarp(blurred.shape[1], self.d, self.K)
                coarse.C = self.C
                coarse._fit(blurred, delta, max_nfev)
                self.C = coarse.C
            from_blurred = self._fit(F, delta, max_nfev)
            if from_blurred < left:
                left = from_blurred
            else:
                self.C = direct
        return left

    def _fit(self, F, delta, max_nfev):
        n, p = F.shape
        A = np.hstack([np.ones((n, 1)), delta])
        A_pinv = np.linalg.pinv(A)

        def nonlinear(G):
            """``G`` (..., n, p) without its least-squares fit linear in delta."""
            return G - A @ (A_pinv @ G)

        norm = np.sqrt(np.sum(nonlinear(F) ** 2))
        if not norm > 0.0:
            return 1.0
        splines = _RowSplines(F)
        shape = self.C.shape
        n_par = self.C.size

        def residual(c):
            G = splines(self.phi(delta, c.reshape(shape)))
            return np.concatenate([nonlinear(G).ravel() / norm, RIDGE * c])

        def jacobian(c):
            phi, d_phi = self.phi(delta, c.reshape(shape), jacobian=True)
            slope = splines(phi, derivative=True)[1]
            J = nonlinear(slope * d_phi).reshape(n_par, n * p).T / norm
            return np.vstack([J, RIDGE * np.eye(n_par)])

        solution = least_squares(residual, self.C.ravel(), jac=jacobian, method="trf", max_nfev=max_nfev,
                                 ftol=FIT_TOLERANCE, xtol=FIT_TOLERANCE, gtol=FIT_TOLERANCE)
        self.C = solution.x.reshape(shape)
        return float(np.sum(solution.fun[:n * p] ** 2))


class AlignedCurves:
    """The curves of a leaf and their warp: what a prediction of the leaf combines.

    ``F`` (n, p) are the curves in the order of the GP's training points and
    ``delta`` (n, d) their offsets. ``centre_weights`` gives the mean of the
    aligned curves that the GP's targets were centred with, as weights of the
    curves.
    """

    def __init__(self, warp: OutputWarp, F: np.ndarray, delta: np.ndarray):
        self.warp, self.F, self.delta = warp, F, delta
        self.centre_weights = np.full(F.shape[0], 1.0 / F.shape[0])
        self._splines = None

    def with_curve(self, f: np.ndarray, delta: np.ndarray) -> 'AlignedCurves':
        """These curves and one more, which has no part in the mean."""
        more = AlignedCurves(self.warp, np.vstack((self.F, f.reshape(1, -1))), np.vstack((self.delta, delta.reshape(1, -1))))
        more.centre_weights = np.append(self.centre_weights, 0.0)
        return more

    def combine(self, delta: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """The curves at the offsets ``delta`` (m, d), given the weights (m, n) of
        the GP mean for the centred aligned curves: each curve is evaluated
        where the feature at a grid point of the query lies in its own frame,
        ``phi_n(phi^-1(x; delta))``, and the values are combined."""
        n, p = self.F.shape
        weights = weights + (1.0 - weights.sum(axis=1, keepdims=True)) * self.centre_weights
        if self._splines is None:
            self._splines = _RowSplines(self.F), _RowSplines(self.warp.phi(self.delta))
        curves, maps = self._splines
        inverse = self.warp.inverse(delta)
        out = np.empty((delta.shape[0], p))
        for i in range(delta.shape[0]):
            out[i] = weights[i] @ curves(maps(np.broadcast_to(inverse[i], (n, p))))
        return out

    def release(self):
        """Free the splines, which ``combine`` computes again."""
        self._splines = None

    def nbytes(self) -> int:
        """Bytes of the curves, their offsets and the splines through them."""
        splines = 0 if self._splines is None else sum(s.c.nbytes for s in self._splines)
        return self.F.nbytes + self.delta.nbytes + self.centre_weights.nbytes + splines


class _GridSplines:
    """Tensor-product cubic splines through the images ``F`` (n, p1, p2),
    evaluated at one set of positions per image."""

    def __init__(self, F: np.ndarray):
        self.n, self.p1, self.p2 = F.shape
        along_v = CubicSpline(np.arange(self.p2), F, axis=2).c            # (4, p2 - 1, n, p1)
        self.c = CubicSpline(np.arange(self.p1), along_v, axis=3).c       # (4, p1 - 1, 4, p2 - 1, n)

    def __call__(self, a: np.ndarray, b: np.ndarray, derivative: bool = False):
        """Values at the positions ``(a, b)``, each (n, m), and the derivatives
        along the two axes if asked for."""
        a, b = np.clip(a, 0.0, self.p1 - 1.0), np.clip(b, 0.0, self.p2 - 1.0)
        iu = np.minimum(a.astype(int), self.p1 - 2)
        iv = np.minimum(b.astype(int), self.p2 - 2)
        t, s = (a - iu)[..., None], b - iv
        c = self.c[:, iu, :, iv, np.arange(self.n)[:, None]]              # (n, m, 4 along u, 4 along v)
        along_u = ((c[:, :, 0] * t + c[:, :, 1]) * t + c[:, :, 2]) * t + c[:, :, 3]
        value = ((along_u[..., 0] * s + along_u[..., 1]) * s + along_u[..., 2]) * s + along_u[..., 3]
        if not derivative:
            return value
        slope_u = (3.0 * c[:, :, 0] * t + 2.0 * c[:, :, 1]) * t + c[:, :, 2]
        du = ((slope_u[..., 0] * s + slope_u[..., 1]) * s + slope_u[..., 2]) * s + slope_u[..., 3]
        dv = (3.0 * along_u[..., 0] * s + 2.0 * along_u[..., 1]) * s + along_u[..., 2]
        return value, du, dv


def _grid(shape, points=None):
    """Positions along the two axes of a grid of ``shape``, flattened row by
    row: its own points, or ``points`` equally spaced ones along each axis."""
    axes = [np.arange(p, dtype=float) if points is None else np.linspace(0.0, p - 1.0, m) for p, m in zip(shape, points or shape)]
    return [g.ravel() for g in np.meshgrid(*axes, indexing="ij")]


class GridWarp:
    """The warp of a grid of ``shape`` (p1, p2), flattened row by row, for inputs
    of ``n_features`` dimensions. Each axis has the warp of an OutputWarp, whose
    ``w`` varies linearly between ``n_cross`` knots along the other axis. ``C``
    (2, n_features, n_cross, n_knots) holds the knot values of the two axes."""

    def __init__(self, shape, n_features: int, n_knots: int = N_KNOTS, n_cross: int = N_CROSS):
        self.shape, self.d, self.K, self.Kc = tuple(int(p) for p in shape), int(n_features), int(n_knots), int(n_cross)
        self.B, self.cross = [], []
        for p, other in zip(self.shape, self.shape[::-1]):
            self.B.append(OutputWarp(p, 1, self.K).B)                               # (p, K) along the axis
            self.cross.append(OutputWarp(other, 1, self.Kc).B if self.Kc > 1 else np.ones((other, 1)))   # (other, Kc) across it
        self.C = np.zeros((2, self.d, self.Kc, self.K))
        self.fine = tuple(OVERSAMPLING * (p - 1) + 1 for p in self.shape)

    def _axis(self, delta, C, axis, jacobian=False):
        """The warp of ``axis`` on the grid, (n, p1, p2). With ``jacobian``, also
        its derivative with respect to the logarithm of its density at each
        knot (K, n, p1, p2); the derivative with respect to ``C[axis, i, l, k]``
        is that of knot ``k`` times ``delta[:, i]`` and column ``l`` of ``cross``."""
        p = self.shape[axis]
        knots = np.einsum("ni,ilk,ol->nok", delta, C[axis], self.cross[axis])       # (n, other, K)
        s = knots @ self.B[axis].T                                                  # one line along the axis for each (n, other)
        bounded = np.tanh(s / MAX_LOG_STRETCH)
        density = np.exp(MAX_LOG_STRETCH * bounded)
        integral = _cumulative(density)
        total = integral[..., -1:]
        phi = (p - 1) * integral / total
        to_grid = (lambda a: a) if axis == 1 else (lambda a: np.swapaxes(a, -1, -2))
        if not jacobian:
            return to_grid(phi)
        d_integral = _cumulative((density * (1.0 - bounded ** 2))[None] * self.B[axis].T[:, None, None, :])
        d_phi = (p - 1) * (d_integral / total - integral * d_integral[..., -1:] / total ** 2)
        return to_grid(phi), to_grid(d_phi)

    def phi(self, delta: np.ndarray, C=None):
        """Where the grid points go along the two axes: ``(a, b)``, each (n, p1, p2)."""
        C = self.C if C is None else C
        return self._axis(delta, C, 0), self._axis(delta, C, 1)

    def register(self, F: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """The aligned images of the rows of ``F`` (n, p1 p2), on the finer grid."""
        n = F.shape[0]
        a, b = self.phi(delta)
        u, v = (np.broadcast_to(g, (n, g.size)) for g in _grid(self.shape, self.fine))
        return _GridSplines(F.reshape((n,) + self.shape))(_GridSplines(a)(u, v), _GridSplines(b)(u, v))

    def inverse(self, delta: np.ndarray, steps: int = 6):
        """Where each grid point comes from: ``(u, v)``, each (n, p1 p2), with
        ``phi`` at ``(u, v)`` the grid point. Newton steps on the splines through phi."""
        n, (p1, p2) = delta.shape[0], self.shape
        a, b = self.phi(delta)
        x, y = (np.broadcast_to(g, (n, g.size)) for g in _grid(self.shape))
        along_u, along_v = _GridSplines(a), _GridSplines(b)
        u, v = 2.0 * x - a.reshape(n, -1), 2.0 * y - b.reshape(n, -1)
        for _ in range(steps):
            u, v = np.clip(u, 0.0, p1 - 1.0), np.clip(v, 0.0, p2 - 1.0)
            fa, au, av = along_u(u, v, derivative=True)
            fb, bu, bv = along_v(u, v, derivative=True)
            det = np.maximum(au * bv - av * bu, 1e-3)
            u, v = u - (bv * (fa - x) - av * (fb - y)) / det, v - (au * (fb - y) - bu * (fa - x)) / det
        return np.clip(u, 0.0, p1 - 1.0), np.clip(v, 0.0, p2 - 1.0)

    def unwarp(self, G: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """The images on the grid of the aligned images ``G`` (n, fine points)."""
        u, v = self.inverse(delta)
        return _GridSplines(G.reshape((G.shape[0],) + self.fine))(OVERSAMPLING * u, OVERSAMPLING * v)

    def start_from(self, other: 'GridWarp', ratio: np.ndarray):
        """Take the warp of ``other``, whose offsets are ``ratio`` times these."""
        self.C = other.C * ratio[None, :, None, None]

    def curves(self, F: np.ndarray, aligned: np.ndarray, delta: np.ndarray) -> 'AlignedImages':
        """What a leaf with the aligned images at the offsets ``delta`` predicts with."""
        return AlignedImages(self, aligned, delta)

    def fit(self, F: np.ndarray, delta: np.ndarray, max_iter: int = 30) -> float:
        """Choose ``C`` so that the aligned images are as linear in ``delta`` as
        possible, starting from the current ``C``, and from ``C = 0`` also from a
        fit to blurred images: as :meth:`OutputWarp.fit`."""
        images = F.reshape((F.shape[0],) + self.shape)
        step = [-(-p // MAX_FIT_POINTS) for p in self.shape]
        if max(step) > 1:
            images = gaussian_filter(images, [0.0] + [0.5 * (st - 1) for st in step], mode="nearest")[:, ::step[0], ::step[1]]
        cold = not np.any(self.C)
        left = self._fit(images, delta, max_iter)
        if cold:
            direct, self.C = self.C, np.zeros_like(self.C)
            widths = [GRID_BLUR * p for p in images.shape[1:]]
            step = [max(1, int(w // 2)) for w in widths]
            self._fit(gaussian_filter(images, [0.0] + widths, mode="nearest")[:, ::step[0], ::step[1]], delta, max_iter)
            from_blurred = self._fit(images, delta, max_iter)
            if from_blurred < left:
                left = from_blurred
            else:
                self.C = direct
        return left

    def _fit(self, images, delta, max_iter):
        """Damped Gauss-Newton steps from the current ``C`` on ``images``
        (n, q1, q2), a grid of any number of points over the same square."""
        n, shape = images.shape[0], images.shape[1:]
        q = shape[0] * shape[1]
        warp = self if shape == self.shape else GridWarp(shape, self.d, self.K, self.Kc)
        Q = np.linalg.qr(np.hstack([np.ones((n, 1)), delta]))[0]

        def nonlinear(G):
            return G - Q @ (Q.T @ G)

        norm = np.sqrt(np.sum(nonlinear(images.reshape(n, q)) ** 2))
        if not norm > 0.0:
            return 1.0
        splines = _GridSplines(images)
        factors = [np.einsum("ni,ol->ilno", delta, cross).reshape(self.d * self.Kc, 1, n, -1) for cross in warp.cross]

        def residual(c):
            a, b = warp.phi(delta, c.reshape(self.C.shape))
            return nonlinear(splines(a.reshape(n, q), b.reshape(n, q))) / norm

        def normal_equations(c, r):
            """``J^T J`` and ``J^T r`` of the residual, with the ridge."""
            C = c.reshape(self.C.shape)
            (a, da), (b, db) = warp._axis(delta, C, 0, jacobian=True), warp._axis(delta, C, 1, jacobian=True)
            _, gu, gv = splines(a.reshape(n, q), b.reshape(n, q), derivative=True)
            da = da * gu.reshape((n,) + shape) / norm                      # (K, n, q1, q2)
            db = db * gv.reshape((n,) + shape) / norm
            X = np.concatenate([(factors[0][..., None, :] * da[None]).reshape(-1, n, q),
                                (factors[1][..., :, None] * db[None]).reshape(-1, n, q)])     # the Jacobian before `nonlinear`
            linear = np.einsum("anq,nm->amq", X, Q).reshape(X.shape[0], -1)
            X = X.reshape(X.shape[0], -1)
            return X @ X.T - linear @ linear.T + GRID_RIDGE ** 2 * np.eye(c.size), X @ r.ravel() + GRID_RIDGE ** 2 * c

        cost_of = lambda c, r: float(np.sum(r ** 2)) + GRID_RIDGE ** 2 * float(c @ c)
        c = self.C.ravel().copy()
        r = residual(c)
        cost, damping = cost_of(c, r), 1e-3
        for _ in range(max_iter):
            H, g = normal_equations(c, r)
            while damping < 1e8:
                trial = c - np.linalg.solve(H + damping * np.diag(np.diag(H)), g)
                r_trial = residual(trial)
                if cost_of(trial, r_trial) < cost:
                    break
                damping *= 4.0
            else:
                break
            gain, moved = cost - cost_of(trial, r_trial), np.max(np.abs(trial - c))
            c, r, cost, damping = trial, r_trial, cost_of(trial, r_trial), damping / 3.0
            if gain < FIT_TOLERANCE * cost or moved < FIT_TOLERANCE * (FIT_TOLERANCE + np.max(np.abs(c))):
                break
        self.C = c.reshape(self.C.shape)
        return float(np.sum(r ** 2))


class AlignedImages:
    """The aligned images of a leaf and their warp: what a prediction of the leaf
    combines. ``G`` (n, fine points) are the aligned images in the order of the
    GP's training points, ``delta`` their offsets and ``centre_weights`` as for
    :class:`AlignedCurves`."""

    def __init__(self, warp: GridWarp, G: np.ndarray, delta: np.ndarray):
        self.warp, self.G, self.delta = warp, G, delta
        self.centre_weights = np.full(G.shape[0], 1.0 / G.shape[0])

    def with_curve(self, f: np.ndarray, delta: np.ndarray) -> 'AlignedImages':
        """These images and one more, which has no part in the mean."""
        delta = delta.reshape(1, -1)
        more = AlignedImages(self.warp, np.vstack((self.G, self.warp.register(f.reshape(1, -1), delta))), np.vstack((self.delta, delta)))
        more.centre_weights = np.append(self.centre_weights, 0.0)
        return more

    def combine(self, delta: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """The images at the offsets ``delta`` (m, d), given the weights (m, n) of
        the GP mean for the centred aligned images: their combination, read
        where each grid point of the query comes from."""
        weights = weights + (1.0 - weights.sum(axis=1, keepdims=True)) * self.centre_weights
        return self.warp.unwarp(weights @ self.G, delta)

    def release(self):
        """Nothing is kept that ``combine`` could compute again."""

    def nbytes(self) -> int:
        """Bytes of the aligned images and their offsets."""
        return self.G.nbytes + self.delta.nbytes + self.centre_weights.nbytes


def output_warp(shape, n_features: int):
    """The warp for outputs on a grid of ``shape``: one or two dimensions."""
    return OutputWarp(shape[0], n_features) if len(shape) == 1 else GridWarp(shape, n_features)
