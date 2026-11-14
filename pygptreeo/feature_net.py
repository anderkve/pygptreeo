"""A tree-wide feature network as GPTree's global model (``GPTree(global_mean='net')``).

``FeatureNetLearner`` trains an MLP on the stream at a fixed cost per refit;
``NetGlobalMean`` publishes each refit as a global-model snapshot whose residual
the GP leaves model. Requires PyTorch.
"""

import copy
import zlib
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

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


class _HeldOutBest:
    """The weights with the lowest loss on the held-out points of a refit."""

    def __init__(self, Xh, Yh, Wh):
        self.Xh, self.Yh, self.Wh = Xh, Yh, Wh
        self.best = np.inf; self.state = None

    def check(self, net):
        with torch.no_grad():
            loss = float((((net(self.Xh) - self.Yh) ** 2) * self.Wh).mean())
        if loss < self.best:
            self.best, self.state = loss, copy.deepcopy(net.state_dict())

    def restore(self, net):
        self.check(net)
        if self.state is not None:
            net.load_state_dict(self.state)


BATCH_SIZE = 128              # Adam minibatch size
HOLDOUT_CHECK_EVERY = 25      # L-BFGS iterations between two checks of the held-out loss (ten times as many Adam steps)
ERROR_WINDOW = 200            # observations the running estimate of the head's error remembers
POLISH_STEPS_PER_UPDATE = 8   # Adam steps per L-BFGS iteration in an amortised refit's polish phase (about equal cost)


class FeatureNetLearner:
    """Tree-wide feature network, trained on the stream at a fixed cost per refit.

    An MLP of ``depth`` hidden layers of ``hidden`` SiLU units with a linear
    output ``h(x)``. Each refit runs two phases of fixed length, so its cost does not depend on
    the stream's length: full-batch L-BFGS on a maximin coverage reservoir of
    the stream, then Adam minibatch steps over every point seen. The loss weights
    each point by ``1 / (sigma^2 + s^2)``, ``s`` the head's prequential error
    with the observation noise subtracted (``error_scale``). A refit is due when
    the observation count has doubled since the last one. A fraction ``holdout``
    of the points is never trained on, and a refit publishes the weights that had
    the lowest loss on them.

    Parameters
    ----------
    hidden : int, default=128
        Units per hidden layer.
    depth : int, default=5
        Hidden layers.
    steps : int, default=300
        L-BFGS iterations per refit on the reservoir; 0 skips the phase.
    polish_steps : int, default=3000
        Adam steps per refit over every point seen, with the learning rate
        cosine-annealed to zero; 0 skips the phase. The store of every point
        costs ``d + 2`` floats per point.
    lr : float, default=1e-3
        Adam's learning rate.
    min_points : int, default=200
        Observations before the first fit.
    reservoir_size : int or None, default=5000
        Points of the coverage reservoir (``CoverageReservoir``); None trains the
        L-BFGS phase on every point, at a cost per iteration that grows with the stream.
    steps_per_update : int or None, default=None
        None: a due refit runs at once inside the ``observe`` call. An int spreads
        it over the following observations, this many L-BFGS iterations (and
        ``POLISH_STEPS_PER_UPDATE`` times as many Adam steps) per call, on a copy
        of the network published when complete. The first fit always runs at once.
    holdout : float, default=0.2
        Fraction of the points that is never trained on; a refit publishes the
        weights with the lowest loss on them. A refit continues from the current
        weights, so without held-out points a fit to few or noisy points is carried
        along. 0 trains on every point and publishes the last weights.
    random_state : int or None
    """

    def __init__(self, hidden: int = 128, depth: int = 5, steps: int = 300, polish_steps: int = 3000,
                 lr: float = 1e-3, min_points: int = 200, reservoir_size: Optional[int] = 5000,
                 steps_per_update: Optional[int] = None, holdout: float = 0.2,
                 random_state: Optional[int] = None):
        _require_torch()
        if not (0.0 <= holdout < 1.0):
            raise ValueError("holdout must be in [0, 1)")
        self.holdout = float(holdout)
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
        self.net = None
        self.x_mu = None; self.x_sd = None; self.y_mu = None; self.y_sd = None
        self.version = 0
        self.n_seen = 0; self.n_seen_at_fit = 0
        self.n_refits = 0; self.fit_seconds = 0.0; self.n_seen_at_first_fit = 0
        self.error_var = None; self.noise_var = None; self.error_scale = None

    # -- stream -------------------------------------------------------------------------
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
    def held_out(self, X: np.ndarray) -> np.ndarray:
        """Boolean mask of the rows of ``X`` that are never trained on. A point's
        membership depends only on its coordinates, so it is the same wherever the
        point is stored."""
        X = np.ascontiguousarray(np.atleast_2d(np.asarray(X, float)))
        if self.holdout <= 0.0:
            return np.zeros(X.shape[0], dtype=bool)
        salt = 0 if self.random_state is None else int(self.random_state)
        h = np.array([zlib.crc32(row.tobytes(), salt) for row in X], dtype=np.uint64)
        return (h % 10000) < self.holdout * 10000

    def _training_set(self, source, scal=None):
        """``source`` standardised as tensors, the per-point loss weights, the
        scaling ``(x_mu, x_sd, y_mu, y_sd)`` and the mask of held-out rows;
        ``scal`` reuses a scaling."""
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
        # 1 / (sigma^2 + s^2), floored at 1% of the targets' spread, mean one
        err = self.error_scale
        var = (S / y_sd) ** 2 + (0.0 if err is None else (err / y_sd) ** 2)
        w = 1.0 / np.maximum(var, 1e-4)
        Wt = torch.tensor(w / w.mean(axis=0), dtype=torch.float32)
        return Xt, Yt, Wt, (x_mu, x_sd, y_mu, y_sd), self.held_out(X)

    def _split(self, Xt, Yt, Wt, hold, best):
        """Training rows, and the record of the best weights (created from the
        held-out rows if there is none yet)."""
        if not hold.any() or hold.all():
            return Xt, Yt, Wt, best
        tr = torch.tensor(np.flatnonzero(~hold)); ho = torch.tensor(np.flatnonzero(hold))
        if best is None:
            best = _HeldOutBest(Xt[ho], Yt[ho], Wt[ho])
        return Xt[tr], Yt[tr], Wt[tr], best

    def _new_net(self):
        torch.manual_seed(int(self.rng.randint(1 << 30)))
        layers, w = [], self.n_features
        for _ in range(self.depth):
            layers += [nn.Linear(w, self.hidden), nn.SiLU()]; w = self.hidden
        return nn.Sequential(*layers, nn.Linear(w, self.n_outputs))

    def _lbfgs_phase(self, net):
        """The L-BFGS phase on the reservoir (None when off) and the scaling this
        refit publishes."""
        Xt, Yt, Wt, scal, hold = self._training_set(self.sample)
        if self.steps == 0:
            return None, scal
        Xt, Yt, Wt, best = self._split(Xt, Yt, Wt, hold, None)
        # One iteration per step() call; max_eval's default of 1 would cut the line search.
        opt = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=1, max_eval=25, history_size=20,
                                line_search_fn='strong_wolfe', tolerance_grad=0.0, tolerance_change=0.0)
        return dict(Xt=Xt, Yt=Yt, Wt=Wt, opt=opt, sched=None, state=None, best=best, it=0), scal

    def _polish_phase(self, net, scal, best=None):
        """The Adam phase over every point, with this refit's scaling; None when off."""
        source = self.store if self.store is not None else self.sample
        if self.polish_steps == 0 or source.n < 2:
            return None
        Xt, Yt, Wt, _, hold = self._training_set(source, scal)
        Xt, Yt, Wt, best = self._split(Xt, Yt, Wt, hold, best)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.polish_steps, eta_min=0.0)
        return dict(Xt=Xt, Yt=Yt, Wt=Wt, opt=opt, sched=sched, state={'perm': None, 'pos': 0}, best=best, it=0)

    @staticmethod
    def _loss(net, X, Y, W):
        return (((net(X) - Y) ** 2) * W).mean()

    def _train(self, net, ph, n_steps):
        """Run n_steps of a phase: full-batch L-BFGS iterations, or minibatch Adam
        steps (``ph['state']`` carries the epoch permutation across calls). With
        held-out points their loss is checked at regular intervals."""
        net.train(); done = 0
        Xt, Yt, Wt, opt, sched, state, best = ph['Xt'], ph['Yt'], ph['Wt'], ph['opt'], ph['sched'], ph['state'], ph['best']
        every = HOLDOUT_CHECK_EVERY if sched is None else 10 * HOLDOUT_CHECK_EVERY

        def check():
            ph['it'] += 1
            if best is not None and ph['it'] % every == 0:
                net.eval(); best.check(net); net.train()

        if sched is None:
            def closure():
                opt.zero_grad(); loss = self._loss(net, Xt, Yt, Wt); loss.backward(); return loss
            while done < n_steps:
                opt.step(closure); done += 1
                check()
            net.eval(); return
        n = Xt.shape[0]; batch = int(min(BATCH_SIZE, max(2, n // 2)))
        while done < n_steps:
            if state['perm'] is None or state['pos'] >= n:
                state['perm'] = torch.randperm(n, generator=self.gen); state['pos'] = 0
            idx = state['perm'][state['pos']:state['pos'] + batch]; state['pos'] += batch
            loss = self._loss(net, Xt[idx], Yt[idx], Wt[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); done += 1
            check()
        net.eval()

    def _publish(self, net, scal, t_spent):
        self.net = net
        self.x_mu, self.x_sd, self.y_mu, self.y_sd = scal
        if self.n_refits == 0:
            self.n_seen_at_first_fit = self.n_seen
        self.version += 1; self.n_refits += 1
        self.n_seen_at_fit = self.n_seen
        self.fit_seconds += t_spent

    def fit(self):
        """(Re)fit the network in one go, discarding any amortised refit in progress."""
        import time
        t0 = time.time()
        self._shadow = None
        net = self.net if self.net is not None else self._new_net()
        ph, scal = self._lbfgs_phase(net)
        best = None
        if ph is not None:
            self._train(net, ph, self.steps); best = ph['best']
        ph = self._polish_phase(net, scal, best)
        if ph is not None:
            self._train(net, ph, self.polish_steps); best = ph['best']
        if best is not None:
            best.restore(net)
        self._publish(net, scal, time.time() - t0)

    def _start_shadow(self):
        """Begin an amortised refit on a copy of the network."""
        net = copy.deepcopy(self.net)
        ph, scal = self._lbfgs_phase(net)
        self._shadow = dict(net=net, scal=scal, lbfgs=ph, done=0, polish=None, polish_started=False, polish_done=0, t=0.0)
        self.n_seen_at_fit = self.n_seen         # no second refit is due while this one runs

    def _advance_shadow(self) -> bool:
        import time
        t0 = time.time(); s = self._shadow
        if s['done'] < self.steps:
            k = min(self.steps_per_update, self.steps - s['done'])
            self._train(s['net'], s['lbfgs'], k)
            s['done'] += k
        if s['done'] >= self.steps and not s['polish_started']:
            best = None if s['lbfgs'] is None else s['lbfgs']['best']
            s['polish'] = self._polish_phase(s['net'], s['scal'], best)   # on the store as it stands now
            s['polish_started'] = True
        if s['polish'] is not None and s['polish_done'] < self.polish_steps:
            k = min(POLISH_STEPS_PER_UPDATE * self.steps_per_update, self.polish_steps - s['polish_done'])
            self._train(s['net'], s['polish'], k)
            s['polish_done'] += k
        s['t'] += time.time() - t0
        if s['done'] >= self.steps and (s['polish'] is None or s['polish_done'] >= self.polish_steps):
            last = s['polish'] if s['polish'] is not None else s['lbfgs']
            if last is not None and last['best'] is not None:
                last['best'].restore(s['net'])
            self._shadow = None
            self._publish(s['net'], s['scal'], s['t'])
            return True
        return False

    @property
    def refit_in_progress(self) -> bool:
        return self._shadow is not None

    # -- prediction ---------------------------------------------------------------------
    def _xt(self, X):
        return torch.tensor((np.atleast_2d(np.asarray(X, float)) - self.x_mu) / self.x_sd, dtype=torch.float32)

    def head(self, X) -> np.ndarray:
        """h(X), the network's own prediction in original units: (n, n_outputs)."""
        with torch.no_grad():
            out = self.net(self._xt(X)).double().numpy()
        return out * self.y_sd + self.y_mu

    def __repr__(self) -> str:
        return (f"FeatureNetLearner(hidden={self.hidden}, depth={self.depth}, steps={self.steps}, "
                f"polish_steps={self.polish_steps}, reservoir_size={self.reservoir_size}, holdout={self.holdout}, n_seen={self.n_seen}, "
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

    Wraps a :class:`FeatureNetLearner` (built from the keyword arguments) as a
    :class:`GlobalMeanLearner`. Each published network version is a new snapshot.
    A leaf takes it up on first use by refitting its GP on the new residual with
    the kernel hyperparameters kept (``cheap_leaf_refresh``); ``error_scale`` is
    the learner's.
    """

    cheap_leaf_refresh = True

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
