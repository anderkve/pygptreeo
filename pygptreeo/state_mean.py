"""A tree-wide linear state for an interaction-order GP prior (``GPTree(global_mean='state')``).

The global model is a Bayesian linear model whose features are

* a sine basis per input, which carries the additive (order-one) part of the prior,
* kernel functions centred on a coverage sample of the stream, which carry the smooth
  part of the pairwise and full-interaction terms,
* optionally the last hidden layer of a network trained on what the additive part leaves.

The prior (one length scale per input, one weight per interaction order, a noise level)
maximises the exact marginal likelihood on a coverage sample. The coefficients are solved
from every point stored in the tree, with the leaves as blocks: what the centres leave of
the prior is correlated only among the points one leaf owns. The GP leaves fit the residual
of this model, as for the other global models. ``examples/BENCHMARK_RESULTS_state_mean.md``
has the derivation and the measurements.
"""

import copy
import time
from math import comb
from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize

from pygptreeo.global_mean import CoverageReservoir, GlobalMeanLearner, GlobalMeanSnapshot


def fit_interaction_prior(X, t, x_range, max_order=1, starts=(0.5, 0.15), warm=None, evals=60):
    """Maximise the marginal likelihood of the interaction-order prior on ``(X, t)``.

    ``k = s1^2 e_1/d + s2^2 e_2/C(d,2) + sf^2 prod_j k_j + sig^2 I`` with ``k_j`` an RBF in
    input j and ``e_r`` the elementary symmetric polynomials of the ``k_j`` (``s2`` only for
    ``max_order=2``). ``t`` must be standardised. ``starts`` are initial length scales as
    fractions of ``x_range``; ``warm`` is a previous solution to start from as well.

    Returns ``dict(ls, s, sf, sig, nll, p)``: length scales, standard deviations of the
    orders, of the full term and of the noise, the negative log likelihood per point and
    the raw parameter vector.
    """
    n, d = X.shape; R = max_order
    if R not in (1, 2):
        raise ValueError("max_order must be 1 or 2")
    D2 = [(X[:, j, None] - X[None, :, j]) ** 2 for j in range(d)]
    eye = np.eye(n); c2 = max(comb(d, 2), 1)

    def fun(p):
        ls = np.exp(p[:d]); s = np.exp(p[d:d + R]); sf = np.exp(p[d + R]); sg = np.exp(p[d + R + 1])
        Kj = [np.exp(-0.5 * D2[j] / ls[j] ** 2) for j in range(d)]
        P = Kj[0].copy()
        for j in range(1, d):
            P *= Kj[j]
        S1 = sum(Kj)
        K = sf ** 2 * P + (sg ** 2 + 1e-8) * eye + (s[0] ** 2 / d) * S1
        if R == 2:
            E2 = 0.5 * (S1 * S1 - sum(k * k for k in Kj)); K += (s[1] ** 2 / c2) * E2
        try:
            cf = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(p)
        al = cho_solve(cf, t)
        val = (0.5 * t @ al + np.log(np.diag(cf[0])).sum()) / n
        Q = (cho_solve(cf, eye) - np.outer(al, al)) * (0.5 / n)       # d(value) / dK
        g = np.empty_like(p)
        for j in range(d):
            inner = sf ** 2 * P + (s[0] ** 2 / d) * Kj[j]
            if R == 2:
                inner = inner + (s[1] ** 2 / c2) * (S1 - Kj[j]) * Kj[j]
            g[j] = np.sum(Q * inner * D2[j]) / ls[j] ** 2
        g[d] = 2 * (s[0] ** 2 / d) * np.sum(Q * S1)
        if R == 2:
            g[d + 1] = 2 * (s[1] ** 2 / c2) * np.sum(Q * E2)
        g[d + R] = 2 * sf ** 2 * np.sum(Q * P)
        g[d + R + 1] = 2 * sg ** 2 * np.trace(Q)
        return val, g

    bounds = [(np.log(1e-2 * r), np.log(1e2 * r)) for r in x_range] + [(-9.0, 3.0)] * (R + 1) + [(-11.0, 1.0)]
    lo = [b[0] for b in bounds]; hi = [b[1] for b in bounds]
    inits = [np.concatenate([np.log(c * x_range), np.full(R, np.log(0.5)), [np.log(0.3)], [np.log(1e-2)]]) for c in starts]
    if warm is not None:
        inits.insert(0, warm)
    best = None
    for p0 in inits:
        res = minimize(fun, np.clip(p0, lo, hi), jac=True, method="L-BFGS-B", bounds=bounds, options=dict(maxfun=evals, maxiter=evals))
        if best is None or res.fun < best.fun:
            best = res
    p = best.x
    return dict(ls=np.exp(p[:d]), s=np.exp(p[d:d + R]), sf=float(np.exp(p[d + R])), sig=float(np.exp(p[d + R + 1])),
                nll=float(best.fun), p=p.copy())


class SineBasis:
    """Sine functions per input that represent an RBF GP on a padded interval.

    Input j with centre c, half-range S and length scale l gets the interval
    ``[c - L, c + L]``, ``L = b S`` with ``b = clip(3.2 l / S, 1.3, 8)``, and all sines of
    frequency up to ``cutoff / l`` (between 8 and ``max_basis`` of them). ``prior_var``
    gives the prior variance of each coefficient, the spectral density of the kernel at
    its frequency. Inputs outside the interval are clipped to it.
    """

    def __init__(self, lo, hi, ls, max_basis=128, cutoff=7.5):
        self.cen = 0.5 * (lo + hi); S = np.maximum(0.5 * (hi - lo), 1e-12)
        self.L = np.clip(3.2 * ls / S, 1.3, 8.0) * S
        self.m = np.clip(np.ceil(cutoff * 2 * self.L / (np.pi * ls)).astype(int), 8, max_basis)
        self.ls = np.array(ls, dtype=float); self.d = len(ls)
        self.P = 1 + int(self.m.sum())
        self.freq = [np.pi * np.arange(1, m + 1) / (2 * L) for m, L in zip(self.m, self.L)]

    def features(self, X):
        X = np.atleast_2d(X)
        cols = [np.ones((len(X), 1))]
        for j in range(self.d):
            u = np.clip(X[:, j:j + 1] - self.cen[j] + self.L[j], 0.0, 2 * self.L[j])
            cols.append(np.sin(u * self.freq[j]) / np.sqrt(self.L[j]))
        return np.concatenate(cols, axis=1)

    def prior_var(self, s1):
        out = [np.array([10.0])]                                      # constant term
        for j in range(self.d):
            l = self.ls[j]
            out.append((s1 ** 2 / self.d) * np.sqrt(2 * np.pi) * l * np.exp(-0.5 * (l * self.freq[j]) ** 2))
        return np.maximum(np.concatenate(out), 1e-30)


def _hidden(net, x_mu, x_sd, X):
    """Last hidden layer of a ``torch.nn.Sequential`` that ends in a linear layer."""
    import torch
    with torch.no_grad():
        return net[:-1](torch.tensor((np.atleast_2d(X) - x_mu) / x_sd, dtype=torch.float32)).double().numpy()


class StateSnapshot(GlobalMeanSnapshot):
    """One published state: basis, centres, coefficients ``w`` and the Cholesky factor
    ``chol`` of their precision. Besides ``predict`` it has ``leaf_variance``, which a
    leaf uses for its sigma in place of the learner's ``error_scale``.
    """

    def __init__(self, version, n_fit, prior, order, basis, centres, mu, sd, net=None):
        self.version, self.n_fit = int(version), int(n_fit)
        self.prior, self.order, self.basis, self.centres = prior, order, basis, centres
        self.mu, self.sd = float(mu), float(sd)
        self.net = net                                               # (frozen network, x_mu, x_sd, h_mu, h_sd) or None
        self.d = len(prior["ls"])
        self.rest_var = prior["sf"] ** 2 + (prior["s"][1] ** 2 if order >= 2 else 0.0)
        ev, V = np.linalg.eigh(self.rest(centres, centres))
        keep = ev > 1e-10 * ev[-1]
        self.W = V[:, keep] / np.sqrt(ev[keep])                      # whitening: unit prior variance per coefficient
        self.P1 = basis.P; self.P2 = self.P1 + int(keep.sum())
        self.w = None; self.chol = None

    @property
    def n_outputs(self) -> int:
        return 1

    def rest(self, A, B):
        """The prior above order one between the rows of A and of B (standardised units)."""
        pr = self.prior; ls = pr["ls"]; s2 = pr["s"][1] if self.order >= 2 else 0.0
        P = np.ones((len(A), len(B))); S1 = 0.0; Q = 0.0
        for j in range(self.d):
            kj = np.exp(-0.5 * ((A[:, j, None] - B[None, :, j]) / ls[j]) ** 2)
            P *= kj
            if s2 > 0:
                S1 = S1 + kj; Q = Q + kj * kj
        K = pr["sf"] ** 2 * P
        if s2 > 0:
            K += (s2 ** 2 / max(comb(self.d, 2), 1)) * 0.5 * (S1 * S1 - Q)
        return K

    def features(self, X):
        """Columns: sine basis ``[:P1]``, whitened kernel functions ``[P1:P2]``, network ``[P2:]``."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        parts = [self.basis.features(X), self.rest(X, self.centres) @ self.W]
        if self.net is not None:
            net, x_mu, x_sd, h_mu, h_sd = self.net
            parts.append((_hidden(net, x_mu, x_sd, X) - h_mu) / h_sd)
        return np.concatenate(parts, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.mu + self.sd * (self.features(X) @ self.w)).reshape(-1, 1)

    def _quad(self, Q):
        u = solve_triangular(self.chol, np.atleast_2d(Q).T, lower=True, check_finite=False)
        return np.sum(u * u, axis=0)

    def leaf_variance(self, node, x, leaf_var):
        """Predictive variance of ``node`` at the rows of ``x``, given its GP's own
        variance ``leaf_var``.

        Adds the variance of the coefficients, seen through ``q = phi(x) - Phi_train^T a(x)``
        with ``a(x)`` the weights of the leaf GP's mean on its training residuals. The local
        term is the larger of ``leaf_var`` and the prior variance of what the centres do not
        carry at ``x``, conditioned on the leaf's training points.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        F = self.features(x); psi = F[:, self.P1:self.P2]
        deficit = np.maximum(self.rest_var - np.sum(psi * psi, axis=1), 0.0)
        gpr = node.my_GPRs[0]
        scaled = bool(node.use_standard_scaling) and node.X_scaler is not None
        mw = None
        if node.output_model == 'independent' and gpr.is_trained() and hasattr(gpr, 'mean_weights'):
            mw = gpr.mean_weights(node.X_scaler.transform(x) if scaled else x)
        if mw is None:
            v_state, v_rest = self._quad(F), deficit
        else:
            A, X_train = mw
            cache = getattr(node, '_state_cache', None)
            if cache is None or cache[0] is not self or cache[1] is not X_train:
                Xt = node.X_scaler.inverse_transform(X_train) if scaled else X_train
                Ft = self.features(Xt); Psi = Ft[:, self.P1:self.P2]
                noise = gpr.get_observation_noise()
                s = float(node.y_scalers[0].scale_[0]) if scaled else 1.0
                white = self.prior["sig"] ** 2 if noise is None else np.maximum(np.asarray(noise, float) * (s / self.sd) ** 2, self.prior["sig"] ** 2)
                R = self.rest(Xt, Xt) - Psi @ Psi.T + np.diag(np.broadcast_to(white, (len(Xt),)) + 1e-8 * self.rest_var + 1e-12)
                cache = (self, X_train, Xt, Ft, cho_factor(R, lower=True))
                node._state_cache = cache
            _, _, Xt, Ft, cf = cache
            if scaled:
                A = A + (1.0 - A.sum(axis=0)) / A.shape[0]            # the leaf centres its targets
            v_state = self._quad(F - A.T @ Ft)
            kx = self.rest(Xt, x) - Ft[:, self.P1:self.P2] @ psi.T
            v_rest = np.maximum(deficit - np.sum(kx * cho_solve(cf, kx), axis=0), 0.0)
        return np.maximum(leaf_var, self.sd ** 2 * v_rest[:, None]) + self.sd ** 2 * v_state[:, None]

    def __repr__(self) -> str:
        return (f"StateSnapshot(version={self.version}, n_fit={self.n_fit}, basis={self.P1}, centres={self.P2 - self.P1}, "
                f"network={0 if self.net is None else len(self.net[3])})")


class StateGlobalMean(GlobalMeanLearner):
    """Interaction-order state as global model (single output).

    Parameters
    ----------
    max_order : {1, 2}, default=2
        2 adds a pairwise term (one weight for all pairs) to the additive and the
        full-interaction term of the prior.
    reservoir_size : int, default=1000
        Size of the coverage reservoir the prior is fitted on.
    n_centres : int, default=500
        Size of a second coverage reservoir whose points are the kernel centres.
    net : bool, default=False
        Add the last hidden layer of a feature network to the state. The network is
        refitted on the residual of the additive part, with held-out stopping, whenever
        the prior is due; the prior width of its coefficients is chosen by evidence on
        the held-out points. Needs PyTorch.
    net_kwargs : dict, optional
        Arguments of the :class:`~pygptreeo.feature_net.FeatureNetLearner` (default:
        ``holdout=0.2``).
    min_points : int, default=60
        Observations before the first prior fit. The prior is due again whenever the
        observation count has doubled.
    min_turnover : float, default=0.25
        A due prior fit is skipped while less than this fraction of a full reservoir
        has been replaced since the last fit. Once the reservoir is full, fits start
        from the previous hyperparameters only.
    refresh : float, default=1.25
        The state is solved again from the tree's leaves when the observation count has
        grown by this factor.
    net_min_points : int, default=200
        Observations before the first network fit.
    max_basis : int, default=128
        Largest number of sine functions per input.
    fit_evals : int, default=60
        Likelihood evaluations per optimiser start.
    random_state : int or None

    Use through ``GPTree(global_mean='state')``; a learner instance passed to ``GPTree``
    is attached to the tree by it.
    """

    cheap_leaf_refresh = True

    def __init__(self, max_order: int = 2, reservoir_size: int = 1000, n_centres: int = 500, net: bool = False,
                 net_kwargs: Optional[dict] = None, min_points: int = 60, min_turnover: float = 0.25,
                 refresh: float = 1.25, net_min_points: int = 200, max_basis: int = 128, fit_evals: int = 60,
                 random_state: Optional[int] = None):
        if max_order not in (1, 2):
            raise ValueError("max_order must be 1 or 2")
        if refresh <= 1.0:
            raise ValueError("refresh must be larger than 1")
        self.max_order, self.reservoir_size, self.n_centres = int(max_order), int(reservoir_size), int(n_centres)
        self.use_net, self.net_kwargs, self.net_min_points = bool(net), dict(net_kwargs or {}), int(net_min_points)
        self.min_points, self.min_turnover, self.refresh = int(min_points), float(min_turnover), float(refresh)
        self.max_basis, self.fit_evals, self.random_state = int(max_basis), int(fit_evals), random_state

        self.tree = None
        self.current: Optional[StateSnapshot] = None
        self.error_scale = None                                      # the snapshots provide leaf_variance instead
        self.prior = None; self.order = None; self.mu = 0.0; self.sd = 1.0
        self.fit_res = None; self.centre_res = None
        self.lo = None; self.hi = None
        self.net_learner = None; self._net = None; self._net_tau2 = None; self._net_changed = False
        self._X = []; self._y = []                                   # every point, only kept for the network
        self.n_seen = 0; self.next_fit = self.min_points; self.next_state = np.inf
        self.turnover_at_fit = 0; self.full_at_fit = False
        self.n_refits = 0; self.n_skipped = 0; self.n_states = 0
        self.fit_seconds = 0.0; self.state_seconds = 0.0; self.net_seconds = 0.0

    def attach(self, tree):
        """Give the learner the tree whose leaves hold the points."""
        self.tree = tree
        return self

    # -- stream ------------------------------------------------------------------------
    def observe(self, x, y, sigma) -> bool:
        x = np.asarray(x, dtype=float).reshape(1, -1); y = np.asarray(y, dtype=float).reshape(1, -1)
        if y.shape[1] != 1:
            raise ValueError("StateGlobalMean supports a single output")
        sigma = np.asarray(sigma, dtype=float).reshape(1, -1)[:, :1]
        if self.fit_res is None:
            d = x.shape[1]
            self.order = min(self.max_order, max(d - 1, 1))          # with two inputs the pairwise term is the full term
            self.fit_res = CoverageReservoir(self.reservoir_size, d, 1)
            self.centre_res = CoverageReservoir(self.n_centres, d, 1)
            self.lo = x[0].copy(); self.hi = x[0].copy()
        self.n_seen += 1
        self.fit_res.add(x, y, sigma); self.centre_res.add(x, y, sigma)
        self.lo = np.minimum(self.lo, x[0]); self.hi = np.maximum(self.hi, x[0])
        if self.use_net:
            self._X.append(x[0].copy()); self._y.append(float(y[0, 0]))
        if self.n_seen >= self.next_fit:
            self._fit_prior()
            if self.use_net and self.n_seen >= self.net_min_points:
                self._fit_net()
            self.next_fit = 2 * self.next_fit
            return self._solve_state()
        if self.n_seen >= self.next_state:
            return self._solve_state()
        return False

    def order_shares(self) -> Optional[dict]:
        """Share of the prior variance in the additive, pairwise and full-interaction term."""
        if self.prior is None:
            return None
        pr = self.prior
        w = np.array([pr["s"][0] ** 2, pr["s"][1] ** 2 if self.order >= 2 else 0.0, pr["sf"] ** 2])
        return dict(zip(("additive", "pairwise", "full"), (w / w.sum()).tolist()))

    # -- fitting -----------------------------------------------------------------------
    def _fit_prior(self):
        res = self.fit_res
        if self.prior is not None and self.full_at_fit and res.turnover - self.turnover_at_fit < self.min_turnover * res.size:
            self.n_skipped += 1
            return
        t0 = time.time()
        self.mu = float(res.y.mean()); self.sd = float(res.y.std())
        if not np.isfinite(self.sd) or self.sd <= 0.0:
            self.sd = 1.0
        x_range = np.maximum(self.hi - self.lo, 1e-12)
        warm = None if self.prior is None else self.prior["p"]
        starts = () if (warm is not None and self.full_at_fit) else (0.5, 0.15)
        self.prior = fit_interaction_prior(res.X, (res.y[:, 0] - self.mu) / self.sd, x_range, self.order,
                                           starts=starts, warm=warm, evals=self.fit_evals)
        self.turnover_at_fit = res.turnover; self.full_at_fit = res.full
        self.n_refits += 1; self.fit_seconds += time.time() - t0

    def _fit_net(self):
        """Refit the network on the residual of the additive part alone."""
        from pygptreeo.feature_net import FeatureNetLearner
        t0 = time.time(); pr = self.prior
        X = np.asarray(self._X); t = (np.asarray(self._y) - self.mu) / self.sd
        basis = SineBasis(self.lo, self.hi, pr["ls"], self.max_basis)
        F = basis.features(X); pv = basis.prior_var(pr["s"][0])
        white = pr["sig"] ** 2 + pr["sf"] ** 2 + (pr["s"][1] ** 2 if self.order >= 2 else 0.0) + 1e-12
        w_add = np.linalg.solve(F.T @ F / white + np.diag(1.0 / pv), F.T @ t / white)
        if self.net_learner is None:
            kwargs = dict(holdout=0.2, random_state=self.random_state); kwargs.update(self.net_kwargs)
            self.net_learner = FeatureNetLearner(**kwargs)
        ln = self.net_learner
        ln.fit_on(X, t - F @ w_add)
        net = copy.deepcopy(ln.net)                                  # the learner's network trains on at the next refit
        H = _hidden(net, ln.x_mu, ln.x_sd, X)
        self._net = (net, ln.x_mu.copy(), ln.x_sd.copy(), H.mean(axis=0), np.maximum(H.std(axis=0), 1e-12))
        self._net_changed = True
        self.net_seconds += time.time() - t0

    def _net_prior_var(self, snap, pv):
        """Prior variance of the network coefficients: evidence of the linear model on
        points the network was not trained on."""
        X = np.asarray(self._X); ho = np.flatnonzero(self.net_learner.held_out(X))
        if ho.size == 0:
            return 1e-3
        if ho.size > 1000:
            ho = np.random.RandomState(self.n_seen).choice(ho, 1000, replace=False)
        t = (np.asarray(self._y)[ho] - self.mu) / self.sd
        Z = snap.features(X[ho]); G = Z.T @ Z; b = Z.T @ t
        s2 = max(self.prior["sig"] ** 2, 1e-12); n_net = Z.shape[1] - snap.P2
        best = (np.inf, 1e-3)
        for log_tau2 in (-6.0, -5.0, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0):
            tau2 = 10.0 ** log_tau2
            A = G / s2 + np.diag(np.concatenate([1.0 / pv[:snap.P2], np.full(n_net, 1.0 / tau2)]))
            try:
                cf = cho_factor(A, lower=True)
            except np.linalg.LinAlgError:
                continue
            nll = n_net * np.log(tau2) + 2 * np.sum(np.log(np.diag(cf[0]))) - cho_solve(cf, b / s2) @ (b / s2)
            if nll < best[0]:
                best = (nll, tau2)
        return best[1]

    def _solve_state(self) -> bool:
        """Solve the coefficients from the points in the tree's leaves and publish them."""
        if self.tree is None:
            raise RuntimeError("StateGlobalMean is not attached to a tree; pass it to GPTree(global_mean=...)")
        self.next_state = int(np.ceil(self.n_seen * self.refresh))
        leaves = [lf for lf in self.tree.root.leaves if getattr(lf, 'my_X_data', None) is not None and lf.my_X_data.shape[0] > 0]
        if not leaves:
            return False
        t0 = time.time(); pr = self.prior
        basis = SineBasis(self.lo, self.hi, pr["ls"], self.max_basis)   # sized for the range explored so far
        snap = StateSnapshot(self.n_states + 1, self.n_seen, pr, self.order, basis, self.centre_res.X.copy(), self.mu, self.sd, self._net)
        P1, P2 = snap.P1, snap.P2
        n_net = 0 if self._net is None else len(self._net[3])
        pv = np.concatenate([basis.prior_var(pr["s"][0]), np.ones(P2 - P1), np.ones(n_net)])
        if n_net:
            if self._net_changed or self._net_tau2 is None:
                self._net_tau2 = self._net_prior_var(snap, pv); self._net_changed = False
            pv[P2:] = self._net_tau2
        P = len(pv); G = np.diag(1.0 / pv); rhs = np.zeros(P)
        for lf in leaves:
            Xc = lf.my_X_data; tc = (lf.my_y_data[:, 0] - self.mu) / self.sd
            Fc = snap.features(Xc); Psi = Fc[:, P1:P2]
            white = np.maximum((lf.my_sigma_data[:, 0] / self.sd) ** 2, pr["sig"] ** 2)
            Rc = snap.rest(Xc, Xc) - Psi @ Psi.T + np.diag(white + 1e-8 * snap.rest_var + 1e-12)
            try:
                cf = cho_factor(Rc, lower=True)
            except np.linalg.LinAlgError:
                cf = cho_factor(Rc + 1e-6 * np.mean(np.diag(Rc)) * np.eye(len(Xc)), lower=True)
            Z = cho_solve(cf, np.column_stack([Fc, tc]))
            G += Fc.T @ Z[:, :P]; rhs += Fc.T @ Z[:, P]
        snap.chol = np.linalg.cholesky(0.5 * (G + G.T))
        snap.w = cho_solve((snap.chol, True), rhs)
        self.n_states += 1
        self.current = snap
        self.state_seconds += time.time() - t0
        return True

    def __repr__(self) -> str:
        return (f"StateGlobalMean(max_order={self.max_order}, reservoir_size={self.reservoir_size}, n_centres={self.n_centres}, "
                f"net={self.use_net}, n_seen={self.n_seen}, n_refits={self.n_refits}, n_states={self.n_states}, current={self.current!r})")
