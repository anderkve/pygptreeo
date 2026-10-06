"""Neural-linear leaves: one shared feature network, Bayesian linear regression per leaf.

``GPTree(GPR=NeuralLinearGPR(FeatureNetLearner()))`` replaces the leaf GPs by
Bayesian linear regressions on the features of one tree-wide network. The
network learns the function's representation from everything the tree has seen;
each leaf re-weights those features on its own points, by default on the
residual of the network's own prediction, so a leaf with few points reverts to
the network rather than to a constant. The tree itself is unchanged.

* :class:`FeatureNetLearner`: an MLP (``depth`` layers of ``hidden`` units, SiLU)
  trained by weighted least squares in two phases: full-batch L-BFGS on a
  maximin coverage reservoir of the stream (``reservoir_size`` points), then a
  fixed budget of minibatch Adam steps over every point seen (``polish_steps``),
  for the fine structure the reservoir cannot hold. Each point's squared error
  is weighted by the inverse of its observation-noise variance plus the
  network's own current error variance. Both phases run a fixed number of
  steps, so a refit costs the same however long the stream has run; refits
  happen when the observation count has doubled since the last fit,
  warm-started from the previous weights. The last hidden layer is the feature
  map, the output layer the *head* ``h(x)``. The learner also tracks the head's
  prequential error with the observation noise subtracted (``error_scale``),
  the budget a residual leaf adds to its sigma.
* :class:`NetGlobalMean`: the same network as a tree-wide *global model*
  (``GPTree(global_mean='net')`` or ``GPTree(global_mean=NetGlobalMean(...))``):
  the leaves keep their GPs and model the residual of the network's head, with
  the global model's refresh rule and error budget. The hybrid for targets whose
  local structure a kernel describes better than the network's features.
* :class:`NeuralLinearGPR`: a :class:`GPRegressorInterface` backend. ``fit``
  regresses ``y - h(X)``, the residual of the network's head, on ``[phi(X), 1]``
  with a Gaussian prior on the weights and per-point noise ``alpha + s2``; the
  prior variance ``tau2`` and the extra noise ``s2`` are chosen by the evidence
  on a grid, by one eigendecomposition per ``s2``. The predicted sigma is the
  posterior sigma of the latent function. A fit remembers the feature version
  it used and re-solves on the current features when asked to predict after a
  refit. The leaf also provides an uncertainty floor: its local leave-one-out
  error near its points, rising to the function's overall scale away from them,
  which the tree applies to the calibrated sigma.

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


class _GrowingStore:
    """Every point of the stream."""

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


BATCH_SIZE = 128              # Adam minibatch size
ERROR_WINDOW = 200            # observations the running estimate of the head's error remembers
POLISH_STEPS_PER_UPDATE = 8   # Adam steps per L-BFGS iteration in an amortised refit's polish phase (about equal cost)


class FeatureNetLearner:
    """Tree-wide feature network, trained on the stream at a fixed cost per refit.

    Parameters
    ----------
    hidden, depth : int
        Width and number of hidden layers of the MLP (SiLU activations). The last
        hidden layer, of ``hidden`` units, is the feature map.
    steps : int, default=300
        L-BFGS iterations per refit (full batch, strong-Wolfe line search) on the
        reservoir. An iteration costs one or a few passes over the reservoir. 0
        skips this phase.
    polish_steps : int, default=3000
        Adam minibatch steps per refit over every point seen after the L-BFGS
        phase, at ``lr`` cosine-annealed to zero: the fine structure the
        reservoir cannot hold, at a cost fixed by the step count. The store of
        every point costs ``d + 2`` floats per point, a few percent of the
        tree's own footprint. 0 skips this phase.
    lr : float, default=1e-3
        Adam's learning rate.
    min_points : int, default=200
        Observations before the first fit. Until then leaves regress on
        ``[x, x^2]`` in place of the network's features.
    reservoir_size : int or None, default=5000
        Points of the maximin coverage reservoir (``CoverageReservoir``) the
        L-BFGS phase trains on. None trains it on every point, at a cost per
        iteration that grows with the stream.
    steps_per_update : int or None, default=None
        None: a due refit runs both phases at once inside the ``observe`` call
        that triggered it (a latency spike of seconds). An int spreads the refit
        over the following observations, this many L-BFGS iterations (and
        ``POLISH_STEPS_PER_UPDATE`` times as many Adam steps) per call, on a
        shadow copy of the network; the leaves keep the old network until the
        shadow is published as the new version. The first fit always runs at once.
    random_state : int or None
    """

    def __init__(self, hidden: int = 128, depth: int = 3, steps: int = 300, polish_steps: int = 3000,
                 lr: float = 1e-3, min_points: int = 200, reservoir_size: Optional[int] = 5000,
                 steps_per_update: Optional[int] = None, random_state: Optional[int] = None):
        _require_torch()
        if steps < 0 or polish_steps < 0:
            raise ValueError("steps and polish_steps must be non-negative")
        if steps == 0 and polish_steps == 0:
            raise ValueError("steps and polish_steps cannot both be 0")
        if min_points < 2:
            raise ValueError("min_points must be at least 2")
        if steps_per_update is not None and steps_per_update < 1:
            raise ValueError("steps_per_update must be at least 1")
        self.hidden, self.depth, self.steps, self.polish_steps = int(hidden), int(depth), int(steps), int(polish_steps)
        self.lr = float(lr)
        self.min_points = int(min_points)
        self.reservoir_size = int(reservoir_size) if reservoir_size is not None else None
        self.steps_per_update = int(steps_per_update) if steps_per_update is not None else None
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.gen = torch.Generator().manual_seed(int(self.rng.randint(1 << 30)))

        self.sample = None                       # the reservoir (or every point when unbounded)
        self.store = None                        # every point, for the polish phase of a bounded reservoir
        self._shadow = None                      # an in-progress amortised refit
        self.n_features = None; self.n_outputs = None
        self.body = None; self.head_layer = None; self.net = None
        self.x_mu = None; self.x_sd = None; self.y_mu = None; self.y_sd = None
        self.version = 0
        self.n_seen = 0; self.n_seen_at_fit = 0
        self.n_refits = 0; self.fit_seconds = 0.0; self.n_seen_at_first_fit = 0
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
            else:
                self.sample = CoverageReservoir(self.reservoir_size, self.n_features, self.n_outputs)
                if self.polish_steps > 0:
                    self.store = _GrowingStore(self.n_features, self.n_outputs)
        self.n_seen += 1
        if self.net is not None:
            # Prequential error of the head; its expectation is (h - f)^2 + noise^2.
            err2 = (self.head(x)[0] - y[0]) ** 2; nz2 = sigma[0] ** 2
            if self.error_var is None:
                self.error_var, self.noise_var = err2, nz2
            else:
                w = 1.0 / min(ERROR_WINDOW, self.n_seen - self.n_seen_at_first_fit + 1)
                self.error_var = (1.0 - w) * self.error_var + w * err2
                self.noise_var = (1.0 - w) * self.noise_var + w * nz2
            self.error_scale = np.sqrt(np.maximum(self.error_var - self.noise_var, 0.0))
        self.sample.add(x, y, sigma)
        if self.store is not None:
            self.store.add(x, y, sigma)
        if self._shadow is not None:
            return self._advance_shadow()
        return self._maybe_fit()

    def _maybe_fit(self) -> bool:
        if self.n_seen < self.min_points:
            return False
        if self.net is None:
            self.fit(); return True
        if self.n_seen < 2 * self.n_seen_at_fit:
            return False
        if self.steps_per_update is not None:
            self._start_shadow(); return False
        self.fit(); return True

    # -- fitting ------------------------------------------------------------------------
    def _training_set(self, source, scal=None):
        """``source`` (the reservoir or the store) standardised as tensors, the
        per-point loss weights, and the scaling ``(x_mu, x_sd, y_mu, y_sd)`` a fit
        publishes with the net; ``scal`` reuses a scaling instead of computing one."""
        if isinstance(source, _GrowingStore):
            source._flush()
        X = source.X; Y = source.y; S = source.sigma
        if X.shape[0] < 2:
            raise RuntimeError("Need at least 2 observations to fit the feature network")
        if scal is None:
            x_mu = X.mean(0); x_sd = X.std(0); x_sd = np.where(x_sd > 0, x_sd, 1.0)
            y_mu = Y.mean(0); y_sd = float((Y - y_mu).std()) or 1.0
        else:
            x_mu, x_sd, y_mu, y_sd = scal
        Xt = torch.tensor((X - x_mu) / x_sd, dtype=torch.float32)
        Yt = torch.tensor((Y - y_mu) / y_sd, dtype=torch.float32)
        # Weights 1 / (sigma^2 + s^2), s the head's current error (zero before the
        # first fit; a floor of 1% of the targets' spread keeps them finite): the
        # likelihood for the noise the stream reports, tending to uniform where the
        # network's own error dominates it.
        err = self.error_scale
        var = (S / y_sd) ** 2 + (0.0 if err is None else (err / y_sd) ** 2)
        w = 1.0 / np.maximum(var, 1e-4)
        Wt = torch.tensor(w / w.mean(axis=0), dtype=torch.float32)
        return Xt, Yt, Wt, (x_mu, x_sd, y_mu, y_sd)

    def _new_net(self):
        torch.manual_seed(int(self.rng.randint(1 << 30)))
        layers, w = [], self.n_features
        for _ in range(self.depth):
            layers += [nn.Linear(w, self.hidden), nn.SiLU()]; w = self.hidden
        body = nn.Sequential(*layers); head = nn.Linear(w, self.n_outputs)
        return nn.Sequential(body, head)

    def _lbfgs_phase(self, net):
        """The L-BFGS phase on the reservoir (None when off) and the scaling this
        refit publishes."""
        Xt, Yt, Wt, scal = self._training_set(self.sample)
        if self.steps == 0:
            return None, scal
        # One iteration per step() call, so a refit can be amortised; max_eval must
        # be set explicitly (its default of max_iter * 5 // 4 = 1 cuts the line search).
        opt = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=1, max_eval=25, history_size=20,
                                line_search_fn='strong_wolfe', tolerance_grad=0.0, tolerance_change=0.0)
        return dict(Xt=Xt, Yt=Yt, Wt=Wt, opt=opt, sched=None, state=None), scal

    def _polish_phase(self, net, scal):
        """The Adam phase over every point, with this refit's scaling; None when off."""
        source = self.store if self.store is not None else self.sample
        if self.polish_steps == 0 or source.n < 2:
            return None
        Xt, Yt, Wt, _ = self._training_set(source, scal)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.polish_steps, eta_min=0.0)
        return dict(Xt=Xt, Yt=Yt, Wt=Wt, opt=opt, sched=sched, state={'perm': None, 'pos': 0})

    @staticmethod
    def _loss(net, X, Y, W):
        return (((net(X) - Y) ** 2) * W).mean()

    def _train(self, net, ph, n_steps):
        """Run n_steps of a phase on net: full-batch L-BFGS iterations (the optimiser
        carries its own history) or minibatch Adam steps (``ph['state']`` carries
        the epoch permutation and position across calls)."""
        net.train(); done = 0
        Xt, Yt, Wt, opt, sched, state = ph['Xt'], ph['Yt'], ph['Wt'], ph['opt'], ph['sched'], ph['state']
        if sched is None:
            def closure():
                opt.zero_grad(); loss = self._loss(net, Xt, Yt, Wt); loss.backward(); return loss
            while done < n_steps:
                opt.step(closure); done += 1
            net.eval(); return
        n = Xt.shape[0]; batch = int(min(BATCH_SIZE, max(2, n // 2)))
        while done < n_steps:
            if state['perm'] is None or state['pos'] >= n:
                state['perm'] = torch.randperm(n, generator=self.gen); state['pos'] = 0
            idx = state['perm'][state['pos']:state['pos'] + batch]; state['pos'] += batch
            loss = self._loss(net, Xt[idx], Yt[idx], Wt[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); done += 1
        net.eval()

    def _publish(self, net, scal, t_spent):
        self.net = net; self.body = net[0]; self.head_layer = net[1]
        self.x_mu, self.x_sd, self.y_mu, self.y_sd = scal
        if self.n_refits == 0:
            self.n_seen_at_first_fit = self.n_seen
        self.version += 1; self.n_refits += 1
        self.n_seen_at_fit = self.n_seen
        self.fit_seconds += t_spent

    def fit(self):
        """(Re)fit the network in one go; callable directly to force a refit (it
        also discards an in-progress amortised refit)."""
        import time
        t0 = time.time()
        self._shadow = None
        net = self.net if self.net is not None else self._new_net()
        ph, scal = self._lbfgs_phase(net)
        if ph is not None:
            self._train(net, ph, self.steps)
        ph = self._polish_phase(net, scal)
        if ph is not None:
            self._train(net, ph, self.polish_steps)
        self._publish(net, scal, time.time() - t0)

    def _start_shadow(self):
        """Begin an amortised refit on a copy of the network."""
        import copy
        net = copy.deepcopy(self.net)
        ph, scal = self._lbfgs_phase(net)
        self._shadow = dict(net=net, scal=scal, lbfgs=ph, done=0, polish=None, polish_started=False, polish_done=0, t=0.0)
        # The schedule restarts from this refit's start, so a second refit cannot
        # become due while this one is in progress.
        self.n_seen_at_fit = self.n_seen

    def _advance_shadow(self) -> bool:
        import time
        t0 = time.time(); s = self._shadow
        if s['done'] < self.steps:
            k = min(self.steps_per_update, self.steps - s['done'])
            self._train(s['net'], s['lbfgs'], k)
            s['done'] += k
        if s['done'] >= self.steps and not s['polish_started']:
            s['polish'] = self._polish_phase(s['net'], s['scal'])   # on the store as it stands now
            s['polish_started'] = True
        if s['polish'] is not None and s['polish_done'] < self.polish_steps:
            k = min(POLISH_STEPS_PER_UPDATE * self.steps_per_update, self.polish_steps - s['polish_done'])
            self._train(s['net'], s['polish'], k)
            s['polish_done'] += k
        s['t'] += time.time() - t0
        if s['done'] >= self.steps and (s['polish'] is None or s['polish_done'] >= self.polish_steps):
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
                f"polish_steps={self.polish_steps}, reservoir_size={self.reservoir_size}, n_seen={self.n_seen}, "
                f"n_refits={self.n_refits}, version={self.version})")


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


LOG_S2 = np.linspace(-8, 1, 13)    # grid of log10 extra-noise variances, relative to the leaf's target variance
LOG_TAU2 = np.linspace(-4, 4, 25)  # grid of log10 prior variances of the standardised feature weights
LOO_K = 5            # fit points nearest the query whose leave-one-out residuals set the local floor
RAMP_START = 2.0     # spacings beyond the leaf's points where the floor starts rising to the function's scale
RAMP_LENGTH = 2.0    # spacings over which it rises


class NeuralLinearGPR(GPRegressorInterface):
    """Bayesian linear regression on a :class:`FeatureNetLearner`'s features, as a leaf model.

    The leaf regresses ``y - h(x)``, the residual of the network's head, on the
    features; the prior variance and the extra noise are chosen by the evidence
    on the grids ``LOG_TAU2`` and ``LOG_S2``. :meth:`predict_floor` returns an
    uncertainty floor the tree applies to the calibrated sigma as
    ``max(sigma, floor)``: on the leaf's data it is the leaf's *local*
    leave-one-out error, the rms of the closed-form leave-one-out residuals at
    the ``LOO_K`` fit points nearest to ``x`` (observation noise subtracted), a
    stream-independent estimate of the error at new points there; beyond
    ``RAMP_START`` nearest-neighbour spacings from the leaf's points it rises to
    the function's overall scale ``y_sd`` over ``RAMP_LENGTH`` spacings, as a
    kernel's variance rises to its prior amplitude:
    ``floor^2 = loo^2 + (y_sd^2 - loo^2) (1 - exp(-((r - RAMP_START) / RAMP_LENGTH)^2))``.
    Distances are in coordinates scaled by the leaf's per-dimension spread.

    Parameters
    ----------
    learner : FeatureNetLearner
        The shared feature network (one instance per tree; the tree feeds it the
        stream through :meth:`observe_stream`).
    """

    def __init__(self, learner: FeatureNetLearner):
        _require_torch()
        self.learner = learner
        self.x_scale = None; self.h = None       # per-dimension spread and nearest-neighbour spacing of the fit points
        self.loo = None                          # leave-one-out residual of each fit point, in y units
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
        if self.learner.fitted:
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
        inv_t2 = 10.0 ** (-LOG_TAU2)                            # (T,)
        best = (-np.inf, None)
        for log_s2 in LOG_S2:
            lam = a + 10.0 ** log_s2; Li = 1.0 / lam
            G = (Phi.T * Li) @ Phi; b = Phi.T @ (Li * yc)
            ev_vals, V = np.linalg.eigh(G); ev_vals = np.maximum(ev_vals, 0.0)
            c = V.T @ b
            den = ev_vals[None, :] + inv_t2[:, None]            # (T, m)
            quad = (c[None, :] ** 2 / den).sum(1)
            logdet = np.log(den).sum(1)
            evidence = -0.5 * ((yc ** 2 * Li).sum() - quad + logdet + np.log(lam).sum()
                               + m * np.log(10.0 ** LOG_TAU2) + n * np.log(2 * np.pi))
            j = int(np.argmax(evidence))
            if evidence[j] > best[0]:
                best = (evidence[j], (10.0 ** LOG_TAU2[j], 10.0 ** log_s2, V, c, den[j]))
        _, (t2, s2, V, c, d) = best
        self.tau2, self.s2 = t2, s2
        self.mu = V @ (c / d)
        self.Sigma = (V / d) @ V.T
        # Leave-one-out residuals in closed form: r_i / (1 - h_ii), h_ii the leverage.
        lam = a + s2; lev = (1.0 / lam) * np.einsum('ij,jk,ik->i', Phi, self.Sigma, Phi)
        self.loo = (yc - Phi @ self.mu) / np.maximum(1.0 - lev, 1e-6) * self.y_scale
        sd_x = X.std(0); self.x_scale = np.where(sd_x > 0, sd_x, 1.0)
        self.h = None
        if n >= 2:
            from scipy.spatial.distance import cdist
            D = cdist(X / self.x_scale, X / self.x_scale); np.fill_diagonal(D, np.inf)
            h = float(np.median(D.min(axis=1)))
            if np.isfinite(h) and h > 0:
                self.h = h
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
        return mean, np.sqrt(np.maximum(var, 0.0))

    def predict_floor(self, X: np.ndarray) -> np.ndarray:
        """The uncertainty floor at X (see the class docstring), shape (n,); zeros
        when the leaf has no fit. The tree applies it as ``max(calibrated sigma, floor)``."""
        X = np.atleast_2d(np.asarray(X, float))
        if self.mu is None or self.X_fit is None:
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
        c = NeuralLinearGPR(self.learner)
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
        w = w + self.learner.y_sd * self.learner.head_layer.weight.detach().double().numpy()[0]
        g = self.learner.gradient_of_linear(self.X_fit, w)
        return 1.0 / (np.sqrt((g ** 2).mean(0)) + 1e-12)

    def __repr__(self) -> str:
        return f"NeuralLinearGPR(trained={self.is_trained()}, learner={self.learner!r})"
