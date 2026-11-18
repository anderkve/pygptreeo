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
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

N_KNOTS = 8             # knots of each w_i
BLURS = (1 / 16, 1 / 64)  # widths of the blurs that a fit from C = 0 also starts from, over the length of the grid
MAX_LOG_STRETCH = 2.0   # bound on |delta . w|: the grid is locally stretched by at most exp(2)
RIDGE = 1e-3            # weight of |w|^2 in the fit, which keeps a warp that the curves do not determine at zero
FIT_TOLERANCE = 1e-4    # relative change of the fit's cost or of the warp at which the fit stops
MIN_OUTPUTS = 16        # grid points below which there is no curve to align


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
