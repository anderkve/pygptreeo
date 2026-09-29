"""BQTree: a GPTree-like partition with Bayesian local polynomial leaves.

The tree, the probabilistic routing, the mixture-of-experts prediction and the
per-leaf residual-quantile calibration follow :class:`pygptreeo.GPTree`. What
differs is the leaf model: instead of a Gaussian process refit by marginal
likelihood optimisation, each leaf is a Bayesian linear regression on a
low-order polynomial (default: full quadratic) in leaf-local standardised
coordinates.

* Every point enters the leaf posterior immediately through an exact rank-one
  (recursive least squares) update, weighted by its known noise variance plus
  the leaf's estimated misfit variance.
* Every ``rebuild_every`` points the leaf re-solves the posterior from its
  stored points: it re-centres its coordinates, re-estimates the misfit
  variance and picks the prior scale by marginal likelihood on a small grid.
* The predictive standard deviation is the posterior standard deviation of the
  polynomial plus the misfit variance, i.e. an uncertainty about the underlying
  function, as for GPTree.
* Beyond ``clip_margin`` standardised units outside the range of its fitted
  points a leaf holds its polynomial constant (per dimension) and lets the
  variance grow with the clipped distance, instead of extrapolating the
  quadratic.

Intended as a non-GP competitor for benchmarking; single- and multi-output
targets are supported (multi-output shares one posterior covariance).
"""

from typing import Optional, Union

import numpy as np
from scipy.linalg import cho_factor, cho_solve

DEFAULT_N_POINTS_PRED_PERF = 25   # window of prequential residuals for calibration
DEFAULT_SIGMA_SCALER = 10.0       # initial calibration scaler
TARGET_COVERAGE = 0.68            # coverage the calibration scaler aims at
PRIOR_SCALE_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)   # prior variance multipliers tried at a rebuild


def _feature_indices(d: int, degree: int):
    """Index pairs of the monomials: [] for 1, (i,) for z_i, (i, j) for z_i z_j, i <= j."""
    idx = [()]
    if degree >= 1:
        idx += [(i,) for i in range(d)]
    if degree >= 2:
        idx += [(i, j) for i in range(d) for j in range(i, d)]
    return idx


class PolynomialFeatures:
    """Monomials up to ``degree`` (0, 1 or 2) of standardised coordinates ``z = (x - centre) / scale``."""

    def __init__(self, d: int, degree: int = 2):
        if degree not in (0, 1, 2):
            raise ValueError("degree must be 0, 1 or 2")
        self.d = int(d)
        self.degree = int(degree)
        self.index = _feature_indices(self.d, self.degree)
        self.p = len(self.index)
        # order of every monomial (0, 1 or 2): the prior variance is set per order
        self.order = np.array([len(t) for t in self.index])
        if self.degree >= 2:
            self._ii = np.array([t[0] for t in self.index if len(t) == 2])
            self._jj = np.array([t[1] for t in self.index if len(t) == 2])
        self.centre = np.zeros(self.d)
        self.scale = np.ones(self.d)
        self.z_lo = np.full(self.d, -np.inf)   # range of the fitted points in z, for clipped extrapolation
        self.z_hi = np.full(self.d, np.inf)

    def set_frame(self, X: np.ndarray, clip_margin: Optional[float] = None):
        """Centre and scale from the given points (scale floored to avoid degeneracy).

        With ``clip_margin`` the polynomial is evaluated at most that many standardised
        units beyond the range of the fitted points (constant beyond, per dimension).
        """
        self.centre = X.mean(axis=0)
        s = X.std(axis=0)
        floor = 1e-3 * max(float(np.max(s)), 1e-12)
        self.scale = np.where(s > floor, s, max(floor, 1e-12))
        if clip_margin is None:
            self.z_lo = np.full(self.d, -np.inf); self.z_hi = np.full(self.d, np.inf)
        else:
            Z = (X - self.centre) / self.scale
            self.z_lo = Z.min(axis=0) - clip_margin; self.z_hi = Z.max(axis=0) + clip_margin

    def standardise(self, X: np.ndarray, clip: bool = False):
        """Standardised coordinates and, with ``clip``, the squared distance clipped away per point."""
        Z = (np.atleast_2d(X) - self.centre) / self.scale
        if not clip:
            return Z, np.zeros(Z.shape[0])
        Zc = np.clip(Z, self.z_lo, self.z_hi)
        return Zc, np.sum((Z - Zc) ** 2, axis=1)

    def transform(self, X: np.ndarray, clip: bool = False) -> np.ndarray:
        """Monomial features; clipping is used at prediction time only."""
        Z, _ = self.standardise(X, clip)
        n = Z.shape[0]
        cols = [np.ones((n, 1))]
        if self.degree >= 1:
            cols.append(Z)
        if self.degree >= 2:
            cols.append(Z[:, self._ii] * Z[:, self._jj])
        return np.hstack(cols)


class BayesianPolynomialLeaf:
    """Bayesian polynomial regression with exact recursive updates.

    Posterior over the coefficients ``w`` (shape ``(p, n_outputs)``) in
    standardised coordinates and standardised outputs, with prior
    ``N(0, s * diag(v_order))`` and per-point Gaussian noise of variance
    ``(sigma_i^2 + tau^2) / y_scale^2``, where ``tau^2`` is the leaf's misfit
    variance (in output units).
    """

    def __init__(self, d: int, n_outputs: int = 1, degree: int = 2,
                 prior_var_by_order=(1.0, 1.0, 1.0), rebuild_every: int = 10,
                 select_prior_scale: bool = True, min_points_for_frame: int = 3,
                 clip_margin: Optional[float] = 0.5):
        self.clip_margin = clip_margin
        self.features = PolynomialFeatures(d, degree)
        self.d, self.p, self.q = int(d), self.features.p, int(n_outputs)
        self.prior_var_by_order = np.asarray(prior_var_by_order, dtype=float)
        self.prior_scale = 1.0
        self.rebuild_every = int(rebuild_every)
        self.select_prior_scale = bool(select_prior_scale)
        self.min_points_for_frame = int(min_points_for_frame)

        self.X = np.empty((0, d))
        self.y = np.empty((0, n_outputs))
        self.s2 = np.empty((0,))              # noise variance per point (averaged over outputs)
        self.own = np.empty((0,), dtype=bool)  # own point (counts towards Nbar, used for splitting) or shared
        self.weight = np.empty((0,))          # fit weight (1 = full; noise variance is divided by it)
        self.n = 0
        self.n_since_rebuild = 0

        self.y_mean = np.zeros(n_outputs)
        self.y_scale = 1.0
        self.tau2 = 0.0                        # misfit variance, output units
        self.m = np.zeros((self.p, n_outputs))  # posterior mean (standardised units)
        self.S = self._prior_cov()              # posterior covariance
        self.n_rebuilds = 0

    # -- prior -------------------------------------------------------------------------
    def _prior_diag(self, scale: Optional[float] = None) -> np.ndarray:
        s = self.prior_scale if scale is None else scale
        return s * self.prior_var_by_order[self.features.order]

    def _prior_cov(self, scale: Optional[float] = None) -> np.ndarray:
        return np.diag(self._prior_diag(scale))

    # -- data ---------------------------------------------------------------------------
    @property
    def n_own(self) -> int:
        return int(np.count_nonzero(self.own))

    def add_point(self, x: np.ndarray, y: np.ndarray, s2: float, own: bool = True, weight: float = 1.0):
        """Store the point and fold it into the posterior (rank-one update)."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        y = np.asarray(y, dtype=float).reshape(1, -1)
        self.X = np.vstack((self.X, x)); self.y = np.vstack((self.y, y))
        self.s2 = np.append(self.s2, float(s2))
        self.own = np.append(self.own, bool(own)); self.weight = np.append(self.weight, float(weight))
        self.n += 1
        self.n_since_rebuild += 1
        if self.n_since_rebuild >= self.rebuild_every or self.n <= self.min_points_for_frame:
            self.rebuild()
        else:
            self._rank_one_update(x, y, float(s2), float(weight))

    def set_points(self, X: np.ndarray, y: np.ndarray, s2: np.ndarray, own=None, weight=None):
        self.X = np.asarray(X, dtype=float).reshape(-1, self.d)
        self.y = np.asarray(y, dtype=float).reshape(-1, self.q)
        self.s2 = np.asarray(s2, dtype=float).reshape(-1)
        self.n = self.X.shape[0]
        self.own = np.ones(self.n, dtype=bool) if own is None else np.asarray(own, dtype=bool).reshape(-1)
        self.weight = np.ones(self.n) if weight is None else np.asarray(weight, dtype=float).reshape(-1)
        self.rebuild()

    def _noise(self):
        """Effective noise variance per stored point, output units."""
        return (self.s2 + self.tau2) / self.weight

    # -- posterior ----------------------------------------------------------------------
    def _rank_one_update(self, x, y, s2, weight=1.0):
        phi = self.features.transform(x)[0]                       # (p,)
        r = (s2 + self.tau2) / weight / self.y_scale ** 2         # noise variance, standardised
        Sphi = self.S @ phi
        denom = r + phi @ Sphi
        gain = Sphi / denom                                       # (p,)
        resid = (y[0] - self.y_mean) / self.y_scale - phi @ self.m  # (q,)
        self.m = self.m + np.outer(gain, resid)
        self.S = self.S - np.outer(gain, Sphi)
        self.S = 0.5 * (self.S + self.S.T)

    def _solve(self, Phi, Yz, r, prior_diag):
        """Posterior mean/cov and log marginal likelihood for noise variances r (n,)."""
        w = 1.0 / r
        A = np.diag(1.0 / prior_diag) + (Phi * w[:, None]).T @ Phi
        b = (Phi * w[:, None]).T @ Yz                             # (p, q)
        c, low = cho_factor(A, lower=True)
        m = cho_solve((c, low), b)
        S = cho_solve((c, low), np.eye(self.p))
        logdetA = 2.0 * np.sum(np.log(np.diag(c)))
        # log evidence summed over outputs
        quad = np.sum(Yz * Yz * w[:, None]) - np.sum(b * m)
        logdet = logdetA + np.sum(np.log(prior_diag)) + np.sum(np.log(r))
        n = Phi.shape[0]
        lml = -0.5 * (quad + self.q * logdet + n * self.q * np.log(2 * np.pi))
        return m, S, lml

    def rebuild(self):
        """Re-solve the posterior from the stored points (re-centre, re-estimate misfit, pick prior scale)."""
        self.n_since_rebuild = 0
        self.n_rebuilds += 1
        if self.n == 0:
            self.m = np.zeros((self.p, self.q)); self.S = self._prior_cov()
            return
        own = self.own if self.n_own >= self.min_points_for_frame else np.ones(self.n, dtype=bool)
        if self.n >= self.min_points_for_frame:
            self.features.set_frame(self.X[own], self.clip_margin)
        self.y_mean = self.y[own].mean(axis=0)
        ys = float((self.y[own] - self.y_mean).std())
        self.y_scale = ys if np.isfinite(ys) and ys > 0 else 1.0
        Phi = self.features.transform(self.X)
        Yz = (self.y - self.y_mean) / self.y_scale

        # Two passes: fit, estimate the misfit from the in-sample residuals, refit.
        tau2 = self.tau2
        best = None
        for _ in range(2):
            r = (self.s2 + tau2) / self.weight / self.y_scale ** 2
            scales = PRIOR_SCALE_GRID if self.select_prior_scale else (self.prior_scale,)
            best = None
            for s in scales:
                m, S, lml = self._solve(Phi, Yz, r, self._prior_diag(s))
                if best is None or lml > best[2]:
                    best = (m, S, lml, s)
            m, S, _, s = best
            # effective degrees of freedom of the fit: p - tr(S S0^-1)
            dof = self.p - float(np.sum(np.diag(S) / self._prior_diag(s)))
            # misfit from the residuals on the own points, corrected for the fitted degrees of freedom
            n_own = int(own.sum())
            resid2 = np.sum((Yz[own] - Phi[own] @ m) ** 2) / self.q          # standardised
            denom = max(n_own - dof * n_own / max(self.n, 1), 1.0)
            tau2 = max(resid2 / denom * self.y_scale ** 2 - float(np.mean(self.s2[own])), 0.0)
        self.m, self.S, _, self.prior_scale = best
        self.tau2 = tau2

    # -- prediction ---------------------------------------------------------------------
    def predict(self, X: np.ndarray):
        """Mean (n, q) and standard deviation (n, q) of the underlying function."""
        Phi = self.features.transform(X, clip=True)
        mu = self.y_mean + self.y_scale * (Phi @ self.m)
        var_w = np.einsum('ij,jk,ik->i', Phi, self.S, Phi)
        # Beyond the clipped range the mean is held constant and the variance grows with
        # the squared distance clipped away (in units of the leaf's output variance).
        _, d2_out = self.features.standardise(X, clip=True)
        var = self.y_scale ** 2 * (np.maximum(var_w, 0.0) + d2_out) + self.tau2
        sd = np.sqrt(var)[:, None] * np.ones((1, self.q))
        return mu, sd

    def in_sample_rss(self, X, y, s2, weight=None, prior_diag=None) -> float:
        """Weighted residual sum of squares of a fresh ridge fit to (X, y): used to score candidate splits."""
        if X.shape[0] == 0:
            return 0.0
        Phi = self.features.transform(X)
        ym = y.mean(axis=0); ys = float((y - ym).std()); ys = ys if ys > 0 else 1.0
        Yz = (y - ym) / ys
        w = np.ones(X.shape[0]) if weight is None else weight
        r = (s2 + self.tau2) / w / ys ** 2
        pd = self._prior_diag() if prior_diag is None else prior_diag
        m, _, _ = self._solve(Phi, Yz, r, pd)
        return float(np.sum((Yz - Phi @ m) ** 2)) * ys ** 2


class BQNode:
    """A node of the BQTree: a leaf model plus DLGP-style split parameters."""

    def __init__(self, d: int, n_outputs: int, name: str, config: dict):
        self.d, self.q, self.name, self.config = d, n_outputs, name, config
        self.model = BayesianPolynomialLeaf(
            d, n_outputs, degree=config['degree'], prior_var_by_order=config['prior_var_by_order'],
            rebuild_every=config['rebuild_every'], select_prior_scale=config['select_prior_scale'],
            clip_margin=config['clip_margin'])
        self.parent = None
        self.children = None
        self.is_left = None
        self.lo = np.full(d, -np.inf)   # cell box from the ancestors' splits
        self.hi = np.full(d, np.inf)
        self.split_index = 0
        self.split_position = 0.0
        self.overlap = 1e-3
        # calibration
        self.residuals = np.array([])
        self.sigma_preds = np.array([])
        self.sigma_scaler = DEFAULT_SIGMA_SCALER
        self.n_points_pred_perf = DEFAULT_N_POINTS_PRED_PERF

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    @property
    def n_points(self) -> int:
        return self.model.n_own

    def expanded_box_contains(self, x: np.ndarray, margin: float) -> bool:
        """Is x inside the cell box widened by ``margin`` times the box width on every side?
        Sides that are unbounded are ignored; the width of a dimension with one unbounded
        side is taken from the leaf's own data spread."""
        x = x.reshape(-1)
        X = self.model.X[self.model.own] if self.model.n_own > 0 else self.model.X
        for j in range(self.d):
            lo, hi = self.lo[j], self.hi[j]
            if np.isfinite(lo) and np.isfinite(hi):
                w = hi - lo
            elif X.shape[0] > 1:
                w = float(X[:, j].max() - X[:, j].min())
            else:
                w = 0.0
            if np.isfinite(lo) and x[j] < lo - margin * w:
                return False
            if np.isfinite(hi) and x[j] > hi + margin * w:
                return False
        return True

    @property
    def leaves(self):
        if self.is_leaf:
            return [self]
        return self.children[0].leaves + self.children[1].leaves

    # -- routing (as GPNode.prob_func / marg_prob) ----------------------------------------
    def prob_func(self, x: np.ndarray) -> np.ndarray:
        prob = (x[:, self.split_index] - self.split_position) / self.overlap + 0.5
        return np.clip(prob, 0.0, 1.0).reshape(x.shape[0], 1)

    def marg_prob(self, x: np.ndarray) -> np.ndarray:
        p = np.ones((x.shape[0], 1))
        node = self
        while node.parent is not None:
            is_left = node.is_left
            node = node.parent
            p *= (1 - node.prob_func(x)) if is_left else node.prob_func(x)
        return p

    # -- calibration (as GPNode.register_pred_perf / update_sigma_scaler) ------------------
    def register_pred_perf(self, x: np.ndarray, y: np.ndarray):
        mu, sd = self.model.predict(x)
        keep = self.n_points_pred_perf - 1
        self.residuals = np.insert(self.residuals[:keep], 0, float(y[0, 0] - mu[0, 0]))
        self.sigma_preds = np.insert(self.sigma_preds[:keep], 0, float(sd[0, 0]))

    def update_sigma_scaler(self):
        if self.residuals.shape[0] < self.n_points_pred_perf:
            if self.sigma_preds.size > 0 and np.max(self.sigma_preds) > 0:
                self.sigma_scaler = float(np.max(np.abs(self.residuals) / (self.sigma_preds + 1e-10)))
            else:
                self.sigma_scaler = DEFAULT_SIGMA_SCALER
            return
        ratios = np.abs(self.residuals) / (self.sigma_preds + 1e-10)
        self.sigma_scaler = max(float(np.quantile(ratios, TARGET_COVERAGE)), 1e-9)

    def predict(self, x: np.ndarray, use_calibrated_sigma: bool = True):
        mu, sd = self.model.predict(x)
        if use_calibrated_sigma:
            sd = sd * self.sigma_scaler
        return mu, sd

    # -- splitting --------------------------------------------------------------------------
    def choose_split(self, theta: float):
        own = self.model.own
        X, y, s2 = self.model.X[own], self.model.y[own], self.model.s2[own]
        crit = self.config['split_criterion']
        spread = X.max(axis=0) - X.min(axis=0)
        if crit == 'max_spread':
            ref = self.config.get('root_scale')
            j = int(np.argmax(spread / ref)) if ref is not None else int(np.argmax(spread))
        elif crit == 'rss':
            best = None
            for jj in range(self.d):
                if spread[jj] <= 0:
                    continue
                pos = np.median(X[:, jj])
                left = X[:, jj] < pos
                if left.sum() < 2 or (~left).sum() < 2:
                    continue
                rss = (self.model.in_sample_rss(X[left], y[left], s2[left])
                       + self.model.in_sample_rss(X[~left], y[~left], s2[~left]))
                if best is None or rss < best[0]:
                    best = (rss, jj)
            j = best[1] if best is not None else int(np.argmax(spread))
        else:
            raise ValueError(f"unknown split_criterion '{crit}'")
        self.split_index = j
        self.split_position = float(np.median(X[:, j]))
        self.overlap = max(theta * float(spread[j]), 1e-12)

    def split(self, theta: float):
        self.choose_split(theta)
        cfg = self.config
        self.children = [BQNode(self.d, self.q, self.name + '0', cfg), BQNode(self.d, self.q, self.name + '1', cfg)]
        for k, child in enumerate(self.children):
            child.parent = self
            child.is_left = (k == 0)
            child.residuals = self.residuals.copy()
            child.sigma_preds = self.sigma_preds.copy()
            child.sigma_scaler = self.sigma_scaler
            child.model.tau2 = self.model.tau2
            child.model.prior_scale = self.model.prior_scale
        j = self.split_index
        self.children[0].lo, self.children[0].hi = self.lo.copy(), self.hi.copy()
        self.children[1].lo, self.children[1].hi = self.lo.copy(), self.hi.copy()
        self.children[0].hi[j] = self.split_position
        self.children[1].lo[j] = self.split_position
        M = self.model
        X, y, s2, own, wt = M.X, M.y, M.s2, M.own, M.weight
        # Own points are routed stochastically by the split function (as in GPTree)
        go_right = np.random.binomial(1, self.prob_func(X)[:, 0]).astype(bool)
        margin, share_w = cfg['fit_margin'], cfg['share_weight']
        for k, child in enumerate(self.children):
            side = go_right if k == 1 else ~go_right
            child_own = own & side
            if margin > 0:
                # give the child a frame first so the expanded box can use its data spread
                child.model.X = X[child_own]; child.model.own = np.ones(int(child_own.sum()), dtype=bool)
                shared = np.array([(not child_own[i]) and child.expanded_box_contains(X[i], margin) for i in range(X.shape[0])])
            else:
                shared = np.zeros(X.shape[0], dtype=bool)
            keep = child_own | shared
            w = np.where(child_own, 1.0, share_w * wt)
            child.model.set_points(X[keep], y[keep], s2[keep], own=child_own[keep], weight=w[keep])
        # An inner node keeps no data
        M.X = np.empty((0, self.d)); M.y = np.empty((0, self.q)); M.s2 = np.empty((0,))
        M.own = np.empty((0,), dtype=bool); M.weight = np.empty((0,))


class BQTree:
    """Tree of Bayesian local polynomial regressions with the GPTree interface.

    Parameters
    ----------
    Nbar : int
        Points a leaf holds before it splits.
    theta : float
        Overlap of sibling leaves as a fraction of the split dimension's spread.
    degree : int
        Polynomial degree of the leaf model (0, 1 or 2).
    split_criterion : 'rss' or 'max_spread'
        Split the dimension whose median split most reduces the leaves' residual sum
        of squares, or the widest dimension (in units of the root's data spread).
    rebuild_every : int
        Points between full re-solves of a leaf posterior (re-centring, misfit and
        prior-scale re-estimation); in between, points enter by rank-one updates.
    prior_var_by_order : tuple of 3 floats
        Prior variance of the constant, linear and quadratic coefficients (in
        standardised units) before the evidence-selected scale.
    select_prior_scale : bool
        Choose the prior scale by marginal likelihood on a small grid at each rebuild.
    use_calibrated_sigma : bool
        Multiply each leaf's sigma by its residual-quantile calibration scaler.
    clip_margin : float or None
        Evaluate a leaf's polynomial at most this many standardised units beyond the
        range of its fitted points (constant beyond, with a variance growing with the
        distance); None extrapolates the polynomial freely.
    fit_margin : float
        A point also enters the fit of every other leaf whose cell box, widened by
        this fraction of its width on each side, contains it (0 disables sharing).
        Shared points do not count towards ``Nbar`` and are not used for splitting.
    share_weight : float
        Fit weight of shared points (their noise variance is divided by it).
    n_outputs : int
        Number of output columns of ``y``.
    """

    def __init__(self, Nbar: int = 100, theta: float = 1e-4, degree: int = 2,
                 split_criterion: str = 'rss', rebuild_every: int = 10,
                 prior_var_by_order=(1.0, 1.0, 1.0), select_prior_scale: bool = True,
                 use_calibrated_sigma: bool = True, clip_margin: Optional[float] = 0.5,
                 fit_margin: float = 0.0, share_weight: float = 1.0,
                 n_outputs: int = 1, max_n_pred_leaves: Optional[int] = None):
        self.Nbar = int(Nbar)
        self.theta = float(theta)
        self.use_calibrated_sigma = bool(use_calibrated_sigma)
        self.n_outputs = int(n_outputs)
        self.max_n_pred_leaves = max_n_pred_leaves
        self.config = dict(degree=int(degree), split_criterion=split_criterion, rebuild_every=int(rebuild_every),
                           prior_var_by_order=tuple(prior_var_by_order), select_prior_scale=bool(select_prior_scale),
                           fit_margin=float(fit_margin), share_weight=float(share_weight),
                           clip_margin=None if clip_margin is None else float(clip_margin), root_scale=None)
        self.root = None
        self.n_features = 0
        self._root_X_stats = None

    # -- stream -------------------------------------------------------------------------------
    def update_tree(self, x: np.ndarray, y: Union[float, np.ndarray], sigma: Union[float, np.ndarray]):
        x = np.asarray(x, dtype=float).reshape(1, -1)
        y = np.asarray(y, dtype=float).reshape(1, -1)
        sigma = np.asarray(sigma, dtype=float).reshape(1, -1)
        s2 = float(np.mean(sigma ** 2))
        if self.root is None:
            self.n_features = x.shape[1]
            self.root = BQNode(self.n_features, self.n_outputs, "0", self.config)
        node = self.root
        while not node.is_leaf:
            node = node.children[int(np.random.binomial(1, node.prob_func(x)[0][0]))]
        if node.n_points > 0:
            node.register_pred_perf(x, y)
            if self.use_calibrated_sigma:
                node.update_sigma_scaler()
        node.model.add_point(x, y, s2)
        if self.config['fit_margin'] > 0:
            for leaf in self._leaves_near(x, self.config['fit_margin']):
                if leaf is not node:
                    leaf.model.add_point(x, y, s2, own=False, weight=self.config['share_weight'])
        if node is self.root:
            # data spread of the root, used by the 'max_spread' criterion to compare dimensions
            X = node.model.X
            if X.shape[0] >= 2:
                s = X.std(axis=0); self.config['root_scale'] = np.where(s > 0, s, 1.0)
        if node.n_points >= self.Nbar:
            node.split(self.theta)

    def fit(self, X_train, y_train, sigma_train, shuffle: bool = True):
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float).reshape(X_train.shape[0], -1)
        sigma_train = np.asarray(sigma_train, dtype=float).reshape(X_train.shape[0], -1)
        order = np.random.permutation(X_train.shape[0]) if shuffle else np.arange(X_train.shape[0])
        for i in order:
            self.update_tree(X_train[i:i + 1], y_train[i:i + 1], sigma_train[i:i + 1])

    def _leaves_near(self, x: np.ndarray, margin: float):
        """Leaves whose expanded cell box contains x."""
        out = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                if node.expanded_box_contains(x, margin):
                    out.append(node)
                continue
            j, s = node.split_index, node.split_position
            w = node.hi[j] - node.lo[j]
            if not np.isfinite(w):
                w = node.overlap / max(self.theta, 1e-12)   # spread of the data at the split
            xj = x[0, j]
            if xj <= s + margin * w:
                stack.append(node.children[0])
            if xj >= s - margin * w:
                stack.append(node.children[1])
        return out

    # -- prediction ---------------------------------------------------------------------------
    def _collect_leaves(self, x: np.ndarray):
        leaves, probs = [], []
        stack = [(self.root, 1.0)]
        while stack:
            node, prob = stack.pop()
            if prob <= 0:
                continue
            if node.is_leaf:
                leaves.append(node); probs.append(prob)
                continue
            pr = node.prob_func(x)[0, 0]
            stack.append((node.children[0], prob * (1 - pr)))
            stack.append((node.children[1], prob * pr))
        if self.max_n_pred_leaves and len(leaves) > self.max_n_pred_leaves:
            order = np.argsort(probs)[::-1][:self.max_n_pred_leaves]
            leaves = [leaves[i] for i in order]; probs = [probs[i] for i in order]
            tot = sum(probs); probs = [p / tot for p in probs]
        return leaves, probs

    def predict(self, X_test: np.ndarray, mode: str = 'recursive', **kwargs):
        """Mean and standard deviation, shape (n_test, n_outputs) each (mixture of experts)."""
        X_test = np.atleast_2d(np.asarray(X_test, dtype=float))
        n = X_test.shape[0]
        if self.root is None or self.root.n_points == 0 and self.root.is_leaf:
            return np.zeros((n, self.n_outputs)), np.full((n, self.n_outputs), np.inf)
        mean = np.zeros((n, self.n_outputs)); var = np.zeros((n, self.n_outputs))
        if mode == 'loop' or n > 8:
            for leaf in self.root.leaves:
                pt = leaf.marg_prob(X_test)
                if not np.any(pt > 0):
                    continue
                mu, sd = leaf.predict(X_test, self.use_calibrated_sigma)
                mean += pt * mu; var += pt * (sd ** 2 + mu ** 2)
        else:
            for i in range(n):
                x = X_test[i:i + 1]
                leaves, probs = self._collect_leaves(x)
                for leaf, pt in zip(leaves, probs):
                    mu, sd = leaf.predict(x, self.use_calibrated_sigma)
                    mean[i] += pt * mu[0]; var[i] += pt * (sd[0] ** 2 + mu[0] ** 2)
        var -= mean ** 2
        return mean, np.sqrt(np.maximum(var, 0.0))

    @property
    def leaves(self):
        return [] if self.root is None else self.root.leaves
