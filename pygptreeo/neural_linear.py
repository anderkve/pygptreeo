"""Neural-linear leaves: one shared feature network, Bayesian linear regression per leaf.

``GPTree(GPR=NeuralLinearGPR(FeatureNetLearner()))`` replaces the leaf GPs by
Bayesian linear regressions on the features of one tree-wide network. The
network learns the function's representation from everything the tree has seen;
each leaf re-weights those features on its own points, by default on the
residual of the network's own prediction, so a leaf with few points reverts to
the network rather than to a constant. The tree itself is unchanged.

* :class:`FeatureNetLearner`: an MLP (``depth`` layers of ``hidden`` units, SiLU)
  trained by least squares on a sample of the stream. The sample is every point
  (``reservoir_size=None``), a maximin coverage reservoir (``reservoir='coverage'``)
  or a uniform reservoir sample (``'uniform'``) of ``reservoir_size`` points.
  Every (re)fit runs a fixed number of Adam ``steps`` with a cosine schedule, so
  its cost does not depend on how long the stream has run; refits happen when the
  observation count has doubled since the last fit (or after ``refit_cap``
  points), warm-started from the previous weights. The last hidden layer is the
  feature map, the output layer the *head* ``h(x)``. The learner also tracks the
  head's prequential error with the observation noise subtracted
  (``error_scale``), the budget a residual leaf adds to its sigma.
* :class:`NetGlobalMean`: the same network as a tree-wide *global model*
  (``GPTree(global_mean='net')`` or ``GPTree(global_mean=NetGlobalMean(...))``):
  the leaves keep their GPs and model the residual of the network's head, with
  the global model's refresh rule and error budget. The hybrid for targets whose
  local structure a kernel describes better than the network's features.
* :class:`NeuralLinearGPR`: a :class:`GPRegressorInterface` backend. ``fit``
  regresses ``y - h(X)`` (``residual=True``) or ``y`` on ``[phi(X), 1]`` with a
  Gaussian prior on the weights and per-point noise ``alpha + s2``; the prior
  variance ``tau2`` and the extra noise ``s2`` are chosen by the evidence on a
  grid, by one eigendecomposition per ``s2``. The predicted sigma is the posterior
  sigma of the latent function plus, for residual leaves, the learner's
  ``error_scale`` in quadrature. A fit remembers the feature version it used and
  re-solves on the current features when asked to predict after a refit. A
  residual leaf also provides an uncertainty floor (``distance_floor``): its local
  leave-one-out error near its points, rising to the function's overall scale
  away from them, which the tree applies to the calibrated sigma.

The network expects raw inputs, so ``GPTree`` switches ``use_standard_scaling``
off for this backend (the learner standardises inputs and targets itself).
Requires PyTorch.
"""

from typing import Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

from pygptreeo.gp_interface import GPRegressorInterface
from pygptreeo.global_mean import CoverageReservoir, GlobalMeanLearner, GlobalMeanSnapshot


def _require_torch():
    if not TORCH_AVAILABLE:
        raise ImportError("pygptreeo.neural_linear needs PyTorch: pip install torch")


class UniformReservoir:
    """A uniform random sample of the stream of fixed ``size`` (reservoir sampling)."""

    def __init__(self, size: int, n_features: int, n_outputs: int, rng):
        self.size = int(size)
        self.X = np.empty((0, n_features)); self.y = np.empty((0, n_outputs)); self.sigma = np.empty((0, n_outputs))
        self.rng = rng
        self.turnover = 0
        self.n_offered = 0

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def full(self) -> bool:
        return self.n >= self.size

    def add(self, x, y, sigma) -> bool:
        x = np.asarray(x, float).reshape(1, -1); y = np.asarray(y, float).reshape(1, -1)
        sigma = np.asarray(sigma, float).reshape(1, -1)
        self.n_offered += 1
        if not self.full:
            self.X = np.vstack((self.X, x)); self.y = np.vstack((self.y, y)); self.sigma = np.vstack((self.sigma, sigma))
            self.turnover += 1
            return True
        j = self.rng.randint(0, self.n_offered)
        if j >= self.size:
            return False
        self.X[j] = x[0]; self.y[j] = y[0]; self.sigma[j] = sigma[0]
        self.turnover += 1
        return True


class _GrowingStore:
    """Every point of the stream (``reservoir_size=None``)."""

    def __init__(self, n_features: int, n_outputs: int):
        self.X = np.empty((0, n_features)); self.y = np.empty((0, n_outputs)); self.sigma = np.empty((0, n_outputs))
        self._bx, self._by, self._bs = [], [], []
        self.turnover = 0

    def _flush(self):
        if self._bx:
            self.X = np.vstack((self.X, np.array(self._bx))); self.y = np.vstack((self.y, np.array(self._by)))
            self.sigma = np.vstack((self.sigma, np.array(self._bs)))
            self._bx, self._by, self._bs = [], [], []

    @property
    def n(self) -> int:
        return self.X.shape[0] + len(self._bx)

    def add(self, x, y, sigma) -> bool:
        self._bx.append(np.asarray(x, float).ravel()); self._by.append(np.asarray(y, float).ravel())
        self._bs.append(np.asarray(sigma, float).ravel()); self.turnover += 1
        return True


class FeatureNetLearner:
    """Tree-wide feature network, trained on a sample of the stream at fixed cost per refit.

    Parameters
    ----------
    hidden, depth : int
        Width and number of hidden layers of the MLP (SiLU activations). The last
        hidden layer, of ``hidden`` units, is the feature map.
    steps : int, default=4000
        Adam steps per (re)fit, with the learning rate cosine-annealed to zero.
        Fixed, so a refit costs the same however many points the sample holds.
    batch_size : int, default=128
    lr : float, default=1e-3
    min_points : int, default=200
        Observations before the first fit. Until then leaves regress on
        ``[x, x^2]`` in place of the network's features.
    refit_cap : int or None
        A refit is due when the observation count has doubled since the last fit,
        or at the latest after this many observations (None: doubling only).
    min_turnover : float, default=0.0
        With a bounded reservoir, a due refit is carried out only if at least this
        fraction of the reservoir has been replaced since the last fit.
    reservoir_size : int or None, default=None
        None keeps every point (memory grows with the stream); an int bounds the
        training sample to that many points.
    reservoir : {'coverage', 'uniform'}
        For a bounded sample: a maximin coverage design of the explored region
        (``CoverageReservoir``), or a uniform random sample of the stream.
    warm_start : bool, default=True
        Continue each refit from the previous weights.
    steps_per_update : int or None, default=None
        None: a due refit runs all ``steps`` at once inside the ``observe`` call
        that triggered it (a latency spike of a few seconds). An int spreads the
        refit over the following observations, this many Adam steps per call, on a
        shadow copy of the network; the leaves keep the old features until the
        shadow has done ``steps`` and is published as the new version. The first
        fit always runs at once.
    error_window : int, default=200
        Memory, in observations, of the running estimate of the head's error.
    random_state : int or None
    """

    def __init__(self, hidden: int = 128, depth: int = 3, steps: int = 4000, batch_size: int = 128,
                 lr: float = 1e-3, min_points: int = 200, refit_cap: Optional[int] = None,
                 min_turnover: float = 0.0, reservoir_size: Optional[int] = None,
                 reservoir: str = 'coverage', warm_start: bool = True,
                 steps_per_update: Optional[int] = None, error_window: int = 200,
                 random_state: Optional[int] = None):
        _require_torch()
        if reservoir not in ('coverage', 'uniform'):
            raise ValueError("reservoir must be 'coverage' or 'uniform'")
        if min_points < 2:
            raise ValueError("min_points must be at least 2")
        if steps_per_update is not None and steps_per_update < 1:
            raise ValueError("steps_per_update must be at least 1")
        self.hidden, self.depth, self.steps, self.batch_size, self.lr = int(hidden), int(depth), int(steps), int(batch_size), float(lr)
        self.steps_per_update = int(steps_per_update) if steps_per_update is not None else None
        self._shadow = None                      # an in-progress amortised refit
        self.min_points = int(min_points)
        self.refit_cap = int(refit_cap) if refit_cap is not None else None
        self.min_turnover = float(min_turnover)
        self.reservoir_size = int(reservoir_size) if reservoir_size is not None else None
        self.reservoir_kind = reservoir
        self.warm_start = bool(warm_start)
        self.error_window = int(error_window)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.gen = torch.Generator().manual_seed(int(self.rng.randint(1 << 30)))

        self.sample = None
        self.n_features = None; self.n_outputs = None
        self.body = None; self.head_layer = None; self.net = None
        self.x_mu = None; self.x_sd = None; self.y_mu = None; self.y_sd = None
        self.version = 0
        self.n_seen = 0; self.n_seen_at_fit = 0; self.turnover_at_fit = 0
        self.n_refits = 0; self.n_skipped = 0; self.fit_seconds = 0.0; self.n_seen_at_first_fit = 0
        self.error_var = None; self.noise_var = None; self.error_scale = None

    # -- stream -------------------------------------------------------------------------
    @property
    def m(self) -> int:
        """Number of features."""
        return self.hidden

    @property
    def fitted(self) -> bool:
        return self.net is not None

    def observe(self, x, y, sigma) -> bool:
        """Register one observation; return True if the network was (re)fitted."""
        x = np.asarray(x, float).reshape(1, -1); y = np.asarray(y, float).reshape(1, -1)
        sigma = np.asarray(sigma, float).reshape(1, -1)
        if sigma.shape[1] == 1 and y.shape[1] > 1:
            sigma = np.tile(sigma, (1, y.shape[1]))
        if self.sample is None:
            self.n_features, self.n_outputs = x.shape[1], y.shape[1]
            if self.reservoir_size is None:
                self.sample = _GrowingStore(self.n_features, self.n_outputs)
            elif self.reservoir_kind == 'coverage':
                self.sample = CoverageReservoir(self.reservoir_size, self.n_features, self.n_outputs)
            else:
                self.sample = UniformReservoir(self.reservoir_size, self.n_features, self.n_outputs, self.rng)
        self.n_seen += 1
        if self.net is not None:
            # Prequential error of the head; its expectation is (h - f)^2 + noise^2.
            err2 = (self.head(x)[0] - y[0]) ** 2; nz2 = sigma[0] ** 2
            if self.error_var is None:
                self.error_var, self.noise_var = err2, nz2
            else:
                w = 1.0 / min(self.error_window, self.n_seen - self.n_seen_at_first_fit + 1)
                self.error_var = (1.0 - w) * self.error_var + w * err2
                self.noise_var = (1.0 - w) * self.noise_var + w * nz2
            self.error_scale = np.sqrt(np.maximum(self.error_var - self.noise_var, 0.0))
        self.sample.add(x, y, sigma)
        if self._shadow is not None:
            return self._advance_shadow()
        return self._maybe_fit()

    def _maybe_fit(self) -> bool:
        if self.n_seen < self.min_points:
            return False
        if self.net is None:
            self.fit(); return True
        due = self.n_seen >= 2 * self.n_seen_at_fit
        if self.refit_cap is not None and self.n_seen - self.n_seen_at_fit >= self.refit_cap:
            due = True
        if not due:
            return False
        if self.reservoir_size is not None and self.min_turnover > 0.0 and \
                self.sample.turnover - self.turnover_at_fit < self.min_turnover * self.reservoir_size:
            self.n_skipped += 1; self.n_seen_at_fit = self.n_seen
            return False
        if self.steps_per_update is not None:
            self._start_shadow(); return False
        self.fit(); return True

    # -- fitting ------------------------------------------------------------------------
    def _training_set(self):
        if isinstance(self.sample, _GrowingStore):
            self.sample._flush()
        X = self.sample.X; Y = self.sample.y
        if X.shape[0] < 2:
            raise RuntimeError("Need at least 2 observations to fit the feature network")
        x_mu = X.mean(0); x_sd = X.std(0); x_sd = np.where(x_sd > 0, x_sd, 1.0)
        y_mu = Y.mean(0); y_sd = float((Y - y_mu).std()) or 1.0
        Xt = torch.tensor((X - x_mu) / x_sd, dtype=torch.float32)
        Yt = torch.tensor((Y - y_mu) / y_sd, dtype=torch.float32)
        return Xt, Yt, x_mu, x_sd, y_mu, y_sd

    def _new_net(self):
        torch.manual_seed(int(self.rng.randint(1 << 30)))
        layers, w = [], self.n_features
        for _ in range(self.depth):
            layers += [nn.Linear(w, self.hidden), nn.SiLU()]; w = self.hidden
        body = nn.Sequential(*layers); head = nn.Linear(w, self.n_outputs)
        return nn.Sequential(body, head)

    def _train(self, net, Xt, Yt, opt, sched, state, n_steps):
        """Run n_steps minibatch steps on net; `state` carries the epoch permutation
        and position across calls."""
        n = Xt.shape[0]; batch = int(min(self.batch_size, max(2, n // 2)))
        net.train(); done = 0
        while done < n_steps:
            if state['perm'] is None or state['pos'] >= n:
                state['perm'] = torch.randperm(n, generator=self.gen); state['pos'] = 0
            idx = state['perm'][state['pos']:state['pos'] + batch]; state['pos'] += batch
            loss = ((net(Xt[idx]) - Yt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); done += 1
        net.eval()

    def _publish(self, net, scal, t_spent):
        self.net = net; self.body = net[0]; self.head_layer = net[1]
        self.x_mu, self.x_sd, self.y_mu, self.y_sd = scal
        if self.n_refits == 0:
            self.n_seen_at_first_fit = self.n_seen
        self.version += 1; self.n_refits += 1
        self.n_seen_at_fit = self.n_seen; self.turnover_at_fit = self.sample.turnover
        self.fit_seconds += t_spent

    def fit(self):
        """(Re)fit the network on the current sample in one go; callable directly to
        force a refit (it also discards an in-progress amortised refit)."""
        import time
        t0 = time.time()
        self._shadow = None
        Xt, Yt, x_mu, x_sd, y_mu, y_sd = self._training_set()
        net = self.net if (self.net is not None and self.warm_start) else self._new_net()
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.steps, eta_min=0.0)
        self._train(net, Xt, Yt, opt, sched, {'perm': None, 'pos': 0}, self.steps)
        self._publish(net, (x_mu, x_sd, y_mu, y_sd), time.time() - t0)

    def _start_shadow(self):
        """Begin an amortised refit on a copy of the network (or a fresh one)."""
        import copy
        Xt, Yt, x_mu, x_sd, y_mu, y_sd = self._training_set()
        net = copy.deepcopy(self.net) if self.warm_start else self._new_net()
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.steps, eta_min=0.0)
        self._shadow = dict(net=net, Xt=Xt, Yt=Yt, opt=opt, sched=sched, scal=(x_mu, x_sd, y_mu, y_sd),
                            state={'perm': None, 'pos': 0}, done=0, t=0.0)
        # The schedule restarts from this refit's start, so a second refit cannot
        # become due while this one is in progress.
        self.n_seen_at_fit = self.n_seen

    def _advance_shadow(self) -> bool:
        import time
        t0 = time.time(); s = self._shadow
        k = min(self.steps_per_update, self.steps - s['done'])
        self._train(s['net'], s['Xt'], s['Yt'], s['opt'], s['sched'], s['state'], k)
        s['done'] += k; s['t'] += time.time() - t0
        if s['done'] >= self.steps:
            self._shadow = None
            self._publish(s['net'], s['scal'], s['t'])
            return True
        return False

    @property
    def refit_in_progress(self) -> bool:
        return self._shadow is not None

    # -- queries ------------------------------------------------------------------------
    def _xt(self, X):
        return torch.tensor((np.atleast_2d(np.asarray(X, float)) - self.x_mu) / self.x_sd, dtype=torch.float32)

    def features(self, X) -> np.ndarray:
        """phi(X): (n, m)."""
        with torch.no_grad():
            return self.body(self._xt(X)).double().numpy()

    def head(self, X) -> np.ndarray:
        """h(X), the network's own prediction in original units: (n, n_outputs)."""
        with torch.no_grad():
            out = self.net(self._xt(X)).double().numpy()
        return out * self.y_sd + self.y_mu

    def gradient_of_linear(self, X, w) -> np.ndarray:
        """d/dx of ``w . phi(x)`` (``w`` in feature units) at each row of X: (n, d), in
        original input units."""
        Xt = self._xt(X).requires_grad_(True)
        f = self.body(Xt) @ torch.tensor(np.asarray(w, float), dtype=torch.float32)
        g, = torch.autograd.grad(f.sum(), Xt)
        return g.double().numpy() / self.x_sd

    def __repr__(self) -> str:
        return (f"FeatureNetLearner(hidden={self.hidden}, depth={self.depth}, steps={self.steps}, "
                f"reservoir_size={self.reservoir_size}, n_seen={self.n_seen}, n_refits={self.n_refits}, "
                f"version={self.version})")


class _NetSnapshot(GlobalMeanSnapshot):
    """An immutable view of one published version of a feature network's head."""

    def __init__(self, learner: FeatureNetLearner, version: int):
        self.learner = learner; self.version = int(version); self.n_fit = learner.n_seen_at_fit
        self.n_out = learner.n_outputs

    @property
    def n_outputs(self) -> int:
        return self.n_out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.learner.head(X)

    def __repr__(self) -> str:
        return f"_NetSnapshot(version={self.version}, n_fit={self.n_fit})"


class NetGlobalMean(GlobalMeanLearner):
    """A feature network as the tree's global model: GP leaves on the residual of its head.

    Wraps a :class:`FeatureNetLearner` (constructed from the keyword arguments) as
    a :class:`GlobalMeanLearner`. Each published network version is a new
    snapshot, so leaves refit against it on first use, as with the additive GP;
    ``error_scale`` is the learner's prequential error budget. The network's own
    weights keep changing only inside a refit, which is published atomically, so
    a snapshot's predictions are fixed until the next version.
    """

    def __init__(self, **learner_kwargs):
        _require_torch()
        self.learner = FeatureNetLearner(**learner_kwargs)
        self.current = None
        self.error_scale = None

    def observe(self, x, y, sigma) -> bool:
        self.learner.observe(x, y, sigma)
        self.error_scale = self.learner.error_scale
        if self.learner.fitted and (self.current is None or self.current.version != self.learner.version):
            self.current = _NetSnapshot(self.learner, self.learner.version)
            return True
        return False

    def __repr__(self) -> str:
        return f"NetGlobalMean({self.learner!r})"


LOO_K = 5            # fit points nearest the query whose leave-one-out residuals set the local floor
RAMP_START = 2.0     # spacings beyond the leaf's points where the floor starts rising to the function's scale
RAMP_LENGTH = 2.0    # spacings over which it rises


class NeuralLinearGPR(GPRegressorInterface):
    """Bayesian linear regression on a :class:`FeatureNetLearner`'s features, as a leaf model.

    Parameters
    ----------
    learner : FeatureNetLearner
        The shared feature network (one instance per tree; the tree feeds it the
        stream through :meth:`observe_stream`).
    residual : bool, default=True
        Regress ``y - h(x)`` (the residual of the network's head) rather than ``y``.
    distance_floor : bool, default=True
        For residual leaves: an uncertainty floor returned by :meth:`predict_floor`,
        which the tree applies to the calibrated sigma as ``max(sigma, floor)``. On
        the leaf's data the floor is the leaf's *local* leave-one-out error: the rms
        of the closed-form leave-one-out residuals of the regression at the
        ``LOO_K`` fit points nearest to ``x`` (observation noise subtracted), a
        stream-independent estimate of the error at new points there. Beyond
        ``RAMP_START`` nearest-neighbour spacings from the leaf's points it rises to
        the function's overall scale ``y_sd`` over ``RAMP_LENGTH`` spacings, as a
        kernel's variance rises to its prior amplitude:
        ``floor^2 = loo^2 + (y_sd^2 - loo^2) (1 - exp(-((r - RAMP_START) / RAMP_LENGTH)^2))``.
        Distances are in coordinates scaled by the leaf's per-dimension spread.
        With ``distance_floor=False`` the learner's stream-wide error budget is
        added to the model sigma instead (before calibration).
    log_s2 : array-like
        Grid of log10 extra-noise variances, relative to the leaf's target variance.
    log_tau2 : array-like
        Grid of log10 prior variances of the standardised feature weights.
    """

    def __init__(self, learner: FeatureNetLearner, residual: bool = True, distance_floor: bool = True,
                 log_s2=np.linspace(-8, 1, 13), log_tau2=np.linspace(-4, 4, 25)):
        _require_torch()
        self.learner = learner; self.residual = bool(residual); self.distance_floor = bool(distance_floor)
        self.x_scale = None; self.h = None       # per-dimension spread and nearest-neighbour spacing of the fit points
        self.loo = None                          # leave-one-out residual of each fit point, in y units
        self.log_s2 = np.asarray(log_s2, float); self.log_tau2 = np.asarray(log_tau2, float)
        self.alpha = 1e-10
        self.mu = None; self.Sigma = None; self.version = -1
        self.X_fit = None; self.y_fit = None; self.alpha_fit = None
        self.tau2 = None; self.s2 = None; self.y_shift = 0.0; self.y_scale = 1.0
        self.f_mu = None; self.f_sd = None
        self.n_solves = 0

    # -- stream hook used by GPTree -----------------------------------------------------
    def observe_stream(self, x, y, sigma) -> None:
        self.learner.observe(x, y, sigma)

    def requires_raw_inputs(self) -> bool:
        return True

    # -- features -----------------------------------------------------------------------
    def _phi(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        if self.learner.fitted:
            F = self.learner.features(X)
        else:
            F = np.hstack([X, X ** 2])
        return np.hstack([F, np.ones((F.shape[0], 1))])

    def _head(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        if self.residual and self.learner.fitted:
            return self.learner.head(X)[:, 0]
        return np.zeros(X.shape[0])

    def _std(self, Phi):
        out = Phi.copy(); out[:, :-1] = (Phi[:, :-1] - self.f_mu) / self.f_sd
        return out

    # -- interface ----------------------------------------------------------------------
    def set_observation_noise(self, alpha: Union[float, np.ndarray]) -> None:
        self.alpha = np.asarray(alpha, float).ravel() if isinstance(alpha, np.ndarray) else float(alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NeuralLinearGPR':
        X = np.atleast_2d(np.asarray(X, float)); y = np.asarray(y, float).ravel()
        if y.shape[0] != X.shape[0]:
            raise ValueError("NeuralLinearGPR fits one output; y must have one value per row of X")
        self.X_fit = X; self.y_fit = y
        self.alpha_fit = np.broadcast_to(np.asarray(self.alpha, float).ravel(), (len(y),)).astype(float)
        self._solve()
        return self

    def _solve(self):
        X, alpha = self.X_fit, self.alpha_fit
        y = self.y_fit - self._head(X)
        self.y_shift = float(y.mean()); sd = float(y.std()); self.y_scale = sd if sd > 0 else 1.0
        yc = (y - self.y_shift) / self.y_scale; a = alpha / self.y_scale ** 2
        Phi = self._phi(X)
        self.f_mu = Phi[:, :-1].mean(0); self.f_sd = Phi[:, :-1].std(0) + 1e-8
        Phi = self._std(Phi); n, m = Phi.shape
        inv_t2 = 10.0 ** (-self.log_tau2)                       # (T,)
        best = (-np.inf, None)
        for log_s2 in self.log_s2:
            lam = a + 10.0 ** log_s2; Li = 1.0 / lam
            G = (Phi.T * Li) @ Phi; b = Phi.T @ (Li * yc)
            ev_vals, V = np.linalg.eigh(G); ev_vals = np.maximum(ev_vals, 0.0)
            c = V.T @ b
            den = ev_vals[None, :] + inv_t2[:, None]            # (T, m)
            quad = (c[None, :] ** 2 / den).sum(1)
            logdet = np.log(den).sum(1)
            evidence = -0.5 * ((yc ** 2 * Li).sum() - quad + logdet + np.log(lam).sum()
                               + m * np.log(10.0 ** self.log_tau2) + n * np.log(2 * np.pi))
            j = int(np.argmax(evidence))
            if evidence[j] > best[0]:
                best = (evidence[j], (10.0 ** self.log_tau2[j], 10.0 ** log_s2, V, c, den[j]))
        _, (t2, s2, V, c, d) = best
        self.tau2, self.s2 = t2, s2
        self.mu = V @ (c / d)
        self.Sigma = (V / d) @ V.T
        if self.distance_floor:
            # Leave-one-out residuals in closed form: r_i / (1 - h_ii), h_ii the leverage.
            lam = a + s2; lev = (1.0 / lam) * np.einsum('ij,jk,ik->i', Phi, self.Sigma, Phi)
            self.loo = (yc - Phi @ self.mu) / np.maximum(1.0 - lev, 1e-6) * self.y_scale
            sd_x = X.std(0); self.x_scale = np.where(sd_x > 0, sd_x, 1.0)
            if n >= 2:
                from scipy.spatial.distance import cdist
                D = cdist(X / self.x_scale, X / self.x_scale); np.fill_diagonal(D, np.inf)
                self.h = float(np.median(D.min(axis=1)))
                if not np.isfinite(self.h) or self.h <= 0:
                    self.h = None
            else:
                self.h = None
        self.version = self.learner.version; self.n_solves += 1

    def predict(self, X: np.ndarray, return_std: bool = False):
        X = np.atleast_2d(np.asarray(X, float))
        if self.mu is None:
            n = X.shape[0]
            if self.learner.fitted:
                mean = self.learner.head(X)[:, 0]; std = np.full(n, self.learner.y_sd)
            else:
                mean = np.zeros(n); std = np.ones(n)
            return (mean, std) if return_std else mean
        if self.version != self.learner.version:
            self._solve()                                        # the features changed since this fit
        Phi = self._std(self._phi(X))
        mean = Phi @ self.mu * self.y_scale + self.y_shift + self._head(X)
        if not return_std:
            return mean
        var = np.einsum('ij,jk,ik->i', Phi, self.Sigma, Phi) * self.y_scale ** 2
        if self.residual and not self.distance_floor and self.learner.error_scale is not None:
            var = var + float(self.learner.error_scale[0]) ** 2
        return mean, np.sqrt(np.maximum(var, 0.0))

    def predict_floor(self, X: np.ndarray) -> np.ndarray:
        """The uncertainty floor at X (see ``distance_floor``), shape (n,); zeros when
        the floor is off or the leaf has no fit. The tree applies it as
        ``max(calibrated sigma, floor)``."""
        X = np.atleast_2d(np.asarray(X, float))
        if not (self.distance_floor and self.residual) or self.mu is None or self.X_fit is None:
            return np.zeros(X.shape[0])
        if self.version != self.learner.version:
            self._solve()
        y_sd = float(self.learner.y_sd) if self.learner.fitted else float(self.y_scale)
        n_fit = self.X_fit.shape[0]
        D = np.sqrt((((X[:, None, :] - self.X_fit[None, :, :]) / self.x_scale) ** 2).sum(-1))   # (n, n_fit)
        k = min(LOO_K, n_fit)
        nearest = np.argpartition(D, k - 1, axis=1)[:, :k] if k < n_fit else np.tile(np.arange(n_fit), (X.shape[0], 1))
        noise = float(np.mean(self.alpha_fit))
        loo2 = np.maximum((self.loo[nearest] ** 2).mean(axis=1) - noise, 0.0)
        floor2 = np.minimum(loo2, y_sd ** 2)
        if self.h is not None:
            r = D.min(axis=1) / self.h
            w = 1.0 - np.exp(-(np.maximum(r - RAMP_START, 0.0) / RAMP_LENGTH) ** 2)
            floor2 = floor2 + (y_sd ** 2 - floor2) * w
        return np.sqrt(floor2)

    def is_trained(self) -> bool:
        return self.mu is not None

    def get_kernel_covariance(self, X: np.ndarray) -> np.ndarray:
        Phi = self._phi(X)
        if self.f_mu is not None:
            Phi = self._std(Phi)
        return (self.tau2 if self.tau2 is not None else 1.0) * Phi @ Phi.T

    def clone(self) -> 'NeuralLinearGPR':
        c = NeuralLinearGPR(self.learner, self.residual, self.distance_floor, self.log_s2, self.log_tau2)
        for k in ('alpha', 'mu', 'Sigma', 'version', 'X_fit', 'y_fit', 'alpha_fit', 'tau2', 's2',
                  'y_shift', 'y_scale', 'f_mu', 'f_sd', 'x_scale', 'h', 'loo'):
            v = getattr(self, k); setattr(c, k, v.copy() if isinstance(v, np.ndarray) else v)
        return c

    def get_kernel(self):
        return (self.tau2, self.s2)

    def set_kernel(self, kernel) -> None:
        self.tau2, self.s2 = kernel

    def get_length_scales(self, n_features: int) -> Optional[np.ndarray]:
        """Inverse rms gradient of the fitted leaf mean per input dimension (the
        analogue of an ARD length scale), from the network's Jacobian."""
        if self.mu is None or not self.learner.fitted or self.X_fit is None:
            return None
        # The leaf mean is y_scale * sum_j mu_j (phi_j - f_mu_j) / f_sd_j + shift (+ head),
        # and the head is linear in phi too, so one weight vector gives the whole gradient.
        w = self.y_scale * self.mu[:-1] / self.f_sd
        if self.residual:
            w = w + self.learner.y_sd * self.learner.head_layer.weight.detach().double().numpy()[0]
        g = self.learner.gradient_of_linear(self.X_fit, w)
        return 1.0 / (np.sqrt((g ** 2).mean(0)) + 1e-12)

    def __repr__(self) -> str:
        return f"NeuralLinearGPR(residual={self.residual}, trained={self.is_trained()}, learner={self.learner!r})"
