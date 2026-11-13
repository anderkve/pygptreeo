"""An additive global model computed from running sums (``GPTree(global_mean='additive')``).

``f(x) ~ c + sum_j g_j(x_j)``, each ``g_j`` a GP with an RBF kernel written in a fixed sine
basis. With fixed features the Bayesian linear model needs the stream only through
``G = Phi^T Phi``, ``b = Phi^T y`` and the column sums of ``Phi``, so

* a new point costs one update of the sums,
* the coefficients are one ``P x P`` solve, at any time,
* the hyperparameters (a length scale per input, an amplitude, and a noise level that
  also absorbs whatever is not additive) maximise a marginal likelihood that, with its
  gradient, is a function of the sums alone: its cost does not grow with the stream.

The GP leaves fit the residual, as for the other global models.
``examples/BENCHMARK_RESULTS_additive_sums.md`` has the measurements.
"""

import time
from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize

from pygptreeo.global_mean import GlobalMeanLearner, GlobalMeanSnapshot

CHUNK = 64                                   # points added to the sums at a time
LENGTH_SCALE_GRID = (0.5, 0.15, 0.04, 0.012)  # fractions of an input's range tried per coordinate


class SineFeatures:
    """A constant and ``m`` sines per input on ``[cen - L, cen + L]``.

    ``[lo, hi]`` is the range of the points the basis is built for; the interval is wider
    by ``margin`` times the half-range on each side. Inputs outside it are clipped.
    """

    def __init__(self, lo, hi, m, margin=0.5):
        self.lo, self.hi = np.array(lo, dtype=float), np.array(hi, dtype=float)
        self.cen = 0.5 * (self.lo + self.hi)
        self.L = (1.0 + margin) * np.maximum(0.5 * (self.hi - self.lo), 1e-12)
        self.m, self.d = int(m), len(self.lo)
        self.P = 1 + self.d * self.m
        self.freq = np.pi * np.arange(1, self.m + 1)[None, :] / (2 * self.L[:, None])      # (d, m)

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float)); n = len(X)
        u = np.clip(X - self.cen + self.L, 0.0, 2 * self.L)
        F = np.sin(u[:, :, None] * self.freq[None, :, :]) / np.sqrt(self.L)[None, :, None]
        return np.concatenate([np.ones((n, 1)), F.reshape(n, self.d * self.m)], axis=1)

    def outside(self, X):
        """True for the rows with a coordinate outside the interval."""
        return np.any(np.abs(np.atleast_2d(X) - self.cen) > self.L, axis=1)

    def log_prior_var(self, log_ls, log_amp):
        """Log prior variance of every sine coefficient (the spectral density of the RBF
        kernel at its frequency) and its derivative with respect to the log length scale
        of its input; both of shape (d, m)."""
        ls = np.exp(log_ls)[:, None]
        lv = 2 * log_amp - np.log(self.d) + 0.5 * np.log(2 * np.pi) + log_ls[:, None] - 0.5 * (ls * self.freq) ** 2
        return lv, 1.0 - (ls * self.freq) ** 2


def evidence(theta, basis, G, b, n):
    """Negative log marginal likelihood per point, its gradient, and the solution.

    Model: ``t ~ N(Phi w, s^2)``, ``w ~ N(0, Lambda(theta))``, for standardised targets
    (``sum(t^2) = n``), from ``G = Phi^T Phi`` and ``b = Phi^T t``.
    ``theta = [log length scales (d), log amplitude, log s]``.

    The solution is ``(keep, w, chol, r)``: the coefficients the prior has not switched
    off, their posterior mean, the Cholesky factor of ``B = I + r G r / s^2`` and
    ``r = sqrt(Lambda)``; the posterior covariance is ``r B^-1 r``. None if the
    factorisation fails.
    """
    d = basis.d
    lv, dlv = basis.log_prior_var(theta[:d], theta[d])
    lam = np.concatenate([[10.0], np.exp(lv).ravel()])
    keep = lam > 1e-14 * lam.max()
    lk = lam[keep]; r = np.sqrt(lk); s2 = np.exp(2 * theta[d + 1])
    Gk = G[np.ix_(keep, keep)]; bk = b[keep]
    B = (r[:, None] * Gk * r[None, :]) / s2
    B[np.diag_indices_from(B)] += 1.0
    try:
        cf = cho_factor(B, lower=True)
    except np.linalg.LinAlgError:
        return 1e10, np.zeros_like(theta), None
    w = r * cho_solve(cf, r * bk / s2)
    rss = n - 2 * w @ bk + w @ (Gk @ w)
    value = 0.5 * (n * np.log(2 * np.pi * s2) + 2 * np.sum(np.log(np.diag(cf[0]))) + (n - bk @ w) / s2) / n
    Linv = solve_triangular(cf[0], np.eye(len(lk)), lower=True, check_finite=False)
    diag_Binv = np.sum(Linv * Linv, axis=0)
    g_lam = np.zeros_like(lam); g_lam[keep] = 0.5 - 0.5 * (w ** 2 + lk * diag_Binv) / lk    # d value / d log lambda
    g = g_lam[1:].reshape(d, basis.m)
    n_eff = len(lk) - np.sum(diag_Binv)                                                     # effective number of coefficients
    grad = np.concatenate([np.sum(g * dlv, axis=1), [2 * np.sum(g)], [(n - n_eff) - rss / s2]]) / n
    return value, grad, (keep, w, cf, r)


class AdditiveSnapshot(GlobalMeanSnapshot):
    """One published solution. Besides ``predict`` it has ``leaf_variance``, which a leaf
    uses for its sigma in place of the learner's ``error_scale``."""

    def __init__(self, version, n_fit, basis, solution, mu, sd, theta, lo, hi):
        self.version, self.n_fit, self.basis = int(version), int(n_fit), basis
        self.lo, self.hi = lo, hi                                    # range of the points seen
        self.keep, self.w, self.chol, self.r = solution
        self.mu, self.sd = float(mu), float(sd)
        self.ls = np.exp(theta[:basis.d]); self.var1 = np.exp(2 * theta[basis.d]) / basis.d   # prior variance per input

    @property
    def n_outputs(self) -> int:
        return 1

    def features(self, X):
        return self.basis(X)[:, self.keep]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.mu + self.sd * (self.features(X) @ self.w)).reshape(-1, 1)

    def _quad(self, Q):
        """``q^T Cov(w) q`` for the rows of Q (standardised units)."""
        u = solve_triangular(self.chol[0], (Q * self.r).T, lower=True, check_finite=False)
        return np.sum(u * u, axis=0)

    def outside_variance(self, X):
        """Prior variance that returns where an input leaves the range the model has seen."""
        X = np.atleast_2d(X)
        gap = np.maximum(self.lo - X, 0.0) + np.maximum(X - self.hi, 0.0)
        return self.var1 * np.sum(1.0 - np.exp(-0.5 * (gap / self.ls) ** 2), axis=1)

    def leaf_variance(self, node, x, leaf_var):
        """Predictive variance of ``node`` at the rows of ``x``, given its GP's own
        variance: adds the variance of the coefficients, seen through
        ``q = phi(x) - Phi_train^T a(x)`` with ``a(x)`` the weights of the leaf GP's mean
        on its training residuals, and the variance that returns outside the seen range."""
        x = np.atleast_2d(np.asarray(x, dtype=float)); F = self.features(x)
        extra = self.outside_variance(x)
        gpr = node.my_GPRs[0]
        scaled = bool(node.use_standard_scaling) and node.X_scaler is not None
        mw = None
        if node.output_model == 'independent' and gpr.is_trained() and hasattr(gpr, 'mean_weights'):
            mw = gpr.mean_weights(node.X_scaler.transform(x) if scaled else x)
        if mw is None:
            return leaf_var + self.sd ** 2 * (extra + self._quad(F))[:, None]
        A, X_train = mw
        cache = getattr(node, '_additive_cache', None)
        if cache is None or cache[0] is not self or cache[1] is not X_train:
            cache = (self, X_train, self.features(node.X_scaler.inverse_transform(X_train) if scaled else X_train))
            node._additive_cache = cache
        if scaled:
            A = A + (1.0 - A.sum(axis=0)) / A.shape[0]                # the leaf centres its targets
        return leaf_var + self.sd ** 2 * (extra + self._quad(F - A.T @ cache[2]))[:, None]

    def __repr__(self) -> str:
        return f"AdditiveSnapshot(version={self.version}, n_fit={self.n_fit}, coefficients={int(np.sum(self.keep))})"


class AdditiveGlobalMean(GlobalMeanLearner):
    """Additive model from running sums as global model (single output).

    Parameters
    ----------
    max_features : int, default=600
        Budget of basis functions: every input gets ``max_features // d`` sines,
        between 16 and 128.
    min_points : int, default=60
        Observations before the first fit. The hyperparameters are fitted again
        whenever the observation count has doubled.
    refresh : float, default=1.25
        The coefficients are solved again when the observation count has grown by
        this factor.

    Use through ``GPTree(global_mean='additive')``.
    """

    cheap_leaf_refresh = True

    def __init__(self, max_features: int = 600, min_points: int = 60, refresh: float = 1.25):
        if refresh <= 1.0:
            raise ValueError("refresh must be larger than 1")
        self.max_features, self.min_points, self.refresh = int(max_features), int(min_points), float(refresh)
        self.current: Optional[AdditiveSnapshot] = None
        self.error_scale = None                                      # the snapshots provide leaf_variance instead
        self._X = []; self._y = []                                   # every point, for rebuilding the sums when the range grows
        self.basis = None; self.G = None; self.b = None; self.c = None; self.lo = None; self.hi = None
        self.pending = 0; self.outside = False
        self.theta = None; self.nll = None; self.mu = 0.0; self.sd = 1.0
        self.n_seen = 0; self.next_fit = self.min_points; self.next_solve = np.inf
        self.n_refits = 0; self.n_solves = 0; self.n_rebuilds = 0
        self.fit_seconds = 0.0; self.sum_seconds = 0.0; self.solve_seconds = 0.0

    # -- stream ------------------------------------------------------------------------
    def observe(self, x, y, sigma) -> bool:
        x = np.asarray(x, dtype=float).reshape(-1); y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != 1:
            raise ValueError("AdditiveGlobalMean supports a single output")
        self._X.append(x.copy()); self._y.append(float(y[0])); self.n_seen += 1
        self.lo = x.copy() if self.lo is None else np.minimum(self.lo, x)
        self.hi = x.copy() if self.hi is None else np.maximum(self.hi, x)
        if self.basis is not None:
            self.pending += 1
            if self.pending >= CHUNK:
                self._flush()
        if self.n_seen >= self.next_fit:
            self._flush(); self._fit(); self.next_fit = 2 * self.next_fit
            return self._solve()
        if self.n_seen >= self.next_solve:
            self._flush()
            return self._solve()
        return False

    def additive_share(self) -> Optional[float]:
        """Fraction of the variance of the targets that the fitted noise level leaves to
        the additive part."""
        return None if self.theta is None else float(1.0 - min(np.exp(2 * self.theta[-1]), 1.0))

    # -- sums --------------------------------------------------------------------------
    def _flush(self):
        """Add the points observed since the last flush to the sums."""
        if self.basis is None or self.pending == 0:
            return
        t0 = time.perf_counter()
        X = np.asarray(self._X[-self.pending:]); y = np.asarray(self._y[-self.pending:]); self.pending = 0
        if np.any(self.basis.outside(X)):
            self.outside = True
        F = self.basis(X)
        self.G += F.T @ F; self.b += F.T @ y; self.c += F.sum(axis=0)
        self.sum_seconds += time.perf_counter() - t0

    def _build(self):
        """A basis over the range seen so far and the sums of every stored point."""
        X = np.asarray(self._X); y = np.asarray(self._y)
        d = X.shape[1]; m = int(np.clip(self.max_features // d, 16, 128))
        self.basis = SineFeatures(X.min(axis=0), X.max(axis=0), m)
        P = self.basis.P; self.G = np.zeros((P, P)); self.b = np.zeros(P); self.c = np.zeros(P)
        for i in range(0, len(X), 512):
            F = self.basis(X[i:i + 512]); self.G += F.T @ F; self.b += F.T @ y[i:i + 512]; self.c += F.sum(axis=0)
        self.pending = 0; self.outside = False; self.n_rebuilds += 1

    # -- fitting -----------------------------------------------------------------------
    def _optimise(self, bt, n):
        """Maximise the evidence. It has a long and a short length-scale optimum per
        input, so after the gradient search every input is tried on a grid of length
        scales with the others fixed."""
        basis, G = self.basis, self.G
        d = basis.d; span = basis.hi - basis.lo + 1e-12
        fun = lambda p: evidence(p, basis, G, bt, n)[:2]
        bounds = [(np.log(2e-3 * r), np.log(1e2 * r)) for r in span] + [(-7.0, 3.0), (-9.0, 1.0)]
        lo = np.array([l for l, _ in bounds]); hi = np.array([h for _, h in bounds])

        def polish(p0):
            res = minimize(fun, np.clip(p0, lo, hi), jac=True, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=60, maxfun=80))
            return res.x, float(res.fun)

        if self.theta is not None and len(self.theta) == d + 2:
            starts = [self.theta]
        else:
            starts = [np.concatenate([np.log(f * span), [np.log(0.7)], [np.log(0.3)]]) for f in LENGTH_SCALE_GRID[:3]]
        x, val = min((polish(p0) for p0 in starts), key=lambda r: r[1])
        for _ in range(3):
            moved = False
            for j in range(d):
                for frac in LENGTH_SCALE_GRID:
                    p = x.copy(); p[j] = np.clip(np.log(frac * span[j]), lo[j], hi[j])
                    v = fun(p)[0]
                    if v < val - 1e-5:
                        x, val, moved = p, float(v), True
            if not moved:
                break
            x, val = polish(x)
        return x, val

    def _fit(self):
        """Hyperparameters by evidence; the basis is rebuilt first if the stream has left it."""
        t0 = time.perf_counter()
        if self.basis is None or self.outside:
            self._build()
        y = np.asarray(self._y)
        self.mu = float(y.mean()); self.sd = float(y.std())
        if not np.isfinite(self.sd) or self.sd <= 0.0:
            self.sd = 1.0
        self.theta, self.nll = self._optimise((self.b - self.mu * self.c) / self.sd, len(y))
        self.n_refits += 1; self.fit_seconds += time.perf_counter() - t0

    def _solve(self) -> bool:
        """Solve the coefficients from the sums and publish them."""
        t0 = time.perf_counter()
        self.next_solve = int(np.ceil(self.n_seen * self.refresh))
        _, _, solution = evidence(self.theta, self.basis, self.G, (self.b - self.mu * self.c) / self.sd, self.n_seen)
        if solution is None:
            return False
        self.n_solves += 1
        self.current = AdditiveSnapshot(self.n_solves, self.n_seen, self.basis, solution, self.mu, self.sd, self.theta,
                                        self.lo.copy(), self.hi.copy())
        self.solve_seconds += time.perf_counter() - t0
        return True

    def __repr__(self) -> str:
        return (f"AdditiveGlobalMean(max_features={self.max_features}, n_seen={self.n_seen}, n_refits={self.n_refits}, "
                f"n_solves={self.n_solves}, current={self.current!r})")
