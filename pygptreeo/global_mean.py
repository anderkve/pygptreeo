"""Global model + residual tree: a tree-wide mean model that the leaves correct.

With ``GPTree(global_mean=...)`` every leaf GP models the *residual* of one
tree-wide global model instead of the raw target. The global model is fitted on
a coverage sample of everything the tree has seen and captures smooth,
large-scale structure; the leaves capture the rest and, at their edges, revert
to the global model instead of to a leaf constant. ``global_mean=None`` (the
default) disables all of this.

* :class:`CoverageReservoir`: a fixed-size maximin design of the stream inputs
  (a newcomer replaces one point of the closest pair if that increases the
  minimum pairwise separation), so the sample covers the explored region
  regardless of how the stream moves through it.
* :class:`AdditiveGPGlobalMean`: a GP with an ``AdditiveMaternKernel(order=2)``
  shared by all outputs, fitted on the reservoir. First fit after ``min_points``
  observations; afterwards refit when the observation count has doubled (or
  ``refit_cap`` points have passed) and at least ``min_turnover`` of the
  reservoir has been replaced since the last fit. Every fit warm-starts from
  the previous hyperparameters and adds ``n_restarts_optimizer`` random restarts.
* :class:`GlobalMeanSnapshot`: an immutable fitted model with a version number.
  A leaf subtracts the current snapshot at fit time, adds the same snapshot back
  at predict time, and refits before predicting if a newer version exists.
"""

from typing import Optional, Union

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler


class GlobalMeanSnapshot:
    """An immutable fitted global model.

    ``predict(X)`` returns the mean in original units with shape
    ``(n_samples, n_outputs)``. Standardisation: ``X`` through ``x_scaler``,
    outputs centred per column (``y_mean``) and divided by one common ``y_scale``.
    """

    def __init__(self, version, x_scaler, y_mean, y_scale, gp, n_fit):
        self.version = int(version)
        self.x_scaler = x_scaler
        self.y_mean = np.asarray(y_mean, dtype=float).reshape(-1)
        self.y_scale = float(y_scale)
        self.gp = gp
        self.n_fit = int(n_fit)

    @property
    def n_outputs(self) -> int:
        return self.y_mean.shape[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        m = np.asarray(self.gp.predict(self.x_scaler.transform(X)), dtype=float)
        m = m.reshape(X.shape[0], -1)
        if m.shape[1] != self.n_outputs:          # a single-target backend for one output
            m = np.tile(m[:, :1], (1, self.n_outputs))
        return m * self.y_scale + self.y_mean

    def __repr__(self) -> str:
        return f"GlobalMeanSnapshot(version={self.version}, n_fit={self.n_fit}, n_outputs={self.n_outputs})"


class CoverageReservoir:
    """Fixed-size maximin design of the stream (a coverage sample, not a time sample).

    Points are compared in the coordinates they are given in (GPTree feeds raw
    inputs; the learner standardises them before fitting), so ``size`` points
    end up roughly evenly spread over the region the stream has explored.
    Rows keep their outputs and noise so the global model can be fitted with
    per-point noise.

    With ``value_weight > 0`` the design is maximin in the joint space of the
    inputs and the outputs, both scaled by their standard deviations at the
    moment the reservoir fills and the outputs multiplied by ``value_weight``:
    two points close in ``x`` but far in ``y`` count as far apart, so regions
    where the function varies fast keep more points than flat ones. With
    ``value_weight = 0`` (the default) the metric is the raw inputs alone.
    """

    def __init__(self, size: int, n_features: int, n_outputs: int, value_weight: float = 0.0):
        if size < 2:
            raise ValueError("reservoir size must be at least 2")
        self.size = int(size)
        self.value_weight = float(value_weight)
        self.X = np.empty((0, n_features))
        self.y = np.empty((0, n_outputs))
        self.sigma = np.empty((0, n_outputs))
        self._D = None                    # pairwise distances once full (inf on the diagonal)
        self._row_min = None              # per-row minimum of _D, kept so an offer costs O(size)
        self.turnover = 0                 # number of insertions/replacements so far
        self._scale = None                # (x_scale, y_scale) of the joint metric, fixed at fill

    def _coords(self, X, y):
        """The points in the metric's coordinates: raw inputs, or scaled inputs and weighted outputs."""
        if self.value_weight <= 0.0:
            return X
        xs, ys = self._scale
        return np.hstack((X / xs, self.value_weight * y / ys))

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def full(self) -> bool:
        return self.n >= self.size

    def add(self, x: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> bool:
        """Offer one point; return True if it entered the reservoir."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        y = np.asarray(y, dtype=float).reshape(1, -1)
        sigma = np.asarray(sigma, dtype=float).reshape(1, -1)
        if not self.full:
            self.X = np.vstack((self.X, x)); self.y = np.vstack((self.y, y)); self.sigma = np.vstack((self.sigma, sigma))
            self.turnover += 1
            if self.full:
                from scipy.spatial.distance import cdist
                if self.value_weight > 0.0:
                    xs = self.X.std(axis=0); ys = self.y.std(axis=0)
                    self._scale = (np.where(xs > 0, xs, 1.0), np.where(ys > 0, ys, 1.0))
                Z = self._coords(self.X, self.y)
                self._D = cdist(Z, Z)
                np.fill_diagonal(self._D, np.inf)
                self._row_min = self._D.min(axis=1)
            return True
        d_new = np.sqrt(((self._coords(self.X, self.y) - self._coords(x, y)) ** 2).sum(axis=1))
        i = int(np.argmin(self._row_min)); j = int(np.argmin(self._D[i]))
        if d_new.min() <= self._D[i, j]:
            return False                  # would not improve the minimum separation
        victim = i if self._row_min[i] <= self._row_min[j] else j
        old_col = self._D[:, victim].copy()
        self.X[victim] = x[0]; self.y[victim] = y[0]; self.sigma[victim] = sigma[0]
        self._D[victim, :] = d_new; self._D[:, victim] = d_new; self._D[victim, victim] = np.inf
        # Row minima: the victim's row is new; another row changes only through its
        # entry in the victim's column, and needs a rescan only if that entry was its minimum.
        self._row_min = np.minimum(self._row_min, d_new)
        stale = np.where(old_col <= self._row_min)[0]
        if stale.size:
            self._row_min[stale] = self._D[stale].min(axis=1)
        self._row_min[victim] = d_new[np.arange(self.size) != victim].min()
        self.turnover += 1
        return True


class GlobalMeanLearner:
    """Base class for tree-wide global models.

    Subclasses implement :meth:`observe`, which sees every (x, y, sigma) the tree
    receives and decides when to (re)fit, and expose the latest fitted model as
    :attr:`current` (a :class:`GlobalMeanSnapshot`, or ``None`` before the first
    fit). Snapshots must be immutable and carry increasing version numbers.

    :attr:`error_scale` (shape ``(n_outputs,)``, or ``None``) estimates the
    current snapshot's own error: its RMS prequential error with the
    observation-noise variance subtracted, i.e. how far the snapshot's mean is
    from the underlying function. A leaf adds this variance to its residual GP's
    predictive variance, since the residual GP cannot know how wrong the global
    model is at a new point. The reported sigma thus remains an uncertainty
    about the underlying function, not about noisy observations.
    """

    current: Optional[GlobalMeanSnapshot] = None
    error_scale: Optional[np.ndarray] = None

    def observe(self, x: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> bool:
        """Register one observation; return True if a new snapshot was published."""
        raise NotImplementedError


class AdditiveGPGlobalMean(GlobalMeanLearner):
    """GP global model on a coverage reservoir, refit on reservoir turnover.

    Parameters
    ----------
    kernel : sklearn kernel, optional
        Kernel of the global GP. Default: ``AdditiveMaternKernel(d, order=2)``,
        built at the first observation when the input dimension is known.
    reservoir_size : int, default=500
        Size of the coverage reservoir the model is fitted on.
    min_points : int, default=200
        Observations before the first fit. Until then leaves model the raw target.
    min_turnover : float, default=0.25
        A due refit is carried out only if at least this fraction of the reservoir
        has been replaced since the last fit, so the model stays fixed while the
        stream revisits known territory and is refit when it explores new territory.
    refit_cap : int or None, default=None
        A refit is due when the number of observations has doubled since the last
        fit, or at the latest after this many observations (default: the reservoir
        size).
    n_restarts_optimizer : int, default=2
        Random restarts of the hyperparameter optimiser on every fit, in addition
        to the warm start from the previous fit.
    alpha_floor : float, default=1e-8
        Minimum per-point noise variance (in the standardised output space) added
        to the kernel diagonal.
    random_state : int or None
        Seed for the optimiser restarts.
    """

    def __init__(self, kernel=None, reservoir_size: int = 500, min_points: int = 200,
                 min_turnover: float = 0.25, refit_cap: Optional[int] = None,
                 n_restarts_optimizer: int = 2, alpha_floor: float = 1e-8,
                 random_state: Optional[int] = None):
        if min_points < 2:
            raise ValueError("min_points must be at least 2")
        if not (0.0 <= min_turnover <= 1.0):
            raise ValueError("min_turnover must be in [0, 1]")
        self.kernel = kernel
        self.reservoir_size = int(reservoir_size)
        self.min_points = int(min_points)
        self.min_turnover = float(min_turnover)
        self.refit_cap = int(refit_cap) if refit_cap is not None else int(reservoir_size)
        self.n_restarts_optimizer = int(n_restarts_optimizer)
        self.alpha_floor = float(alpha_floor)
        self.random_state = random_state

        self.reservoir: Optional[CoverageReservoir] = None
        self.current: Optional[GlobalMeanSnapshot] = None
        self.error_var: Optional[np.ndarray] = None    # running mean of the snapshot's squared prequential error
        self.noise_var: Optional[np.ndarray] = None    # running mean of the observation-noise variance
        self.error_scale: Optional[np.ndarray] = None  # sqrt(max(error_var - noise_var, 0))
        self.error_window = 200                        # memory of the running means, in observations
        self.n_seen = 0
        self.n_seen_at_fit = 0
        self.turnover_at_fit = 0
        self.n_refits = 0
        self.n_skipped = 0
        self.n_seen_at_first_fit = 0

    # -- stream ------------------------------------------------------------------------
    def observe(self, x, y, sigma) -> bool:
        x = np.asarray(x, dtype=float).reshape(1, -1)
        y = np.asarray(y, dtype=float).reshape(1, -1)
        sigma = np.asarray(sigma, dtype=float).reshape(1, -1)
        if sigma.shape[1] == 1 and y.shape[1] > 1:
            sigma = np.tile(sigma, (1, y.shape[1]))
        if self.reservoir is None:
            self.reservoir = CoverageReservoir(self.reservoir_size, x.shape[1], y.shape[1])
            if self.kernel is None:
                from pygptreeo.kernels import AdditiveMaternKernel
                self.kernel = AdditiveMaternKernel(d=x.shape[1], order=min(2, x.shape[1]))
        self.n_seen += 1
        if self.current is not None:
            # Prequential error of the current snapshot at this point. Its expectation
            # is (mean - f)^2 + noise^2, so the noise variance is tracked alongside
            # and subtracted in error_scale.
            err2 = (self.current.predict(x)[0] - y[0]) ** 2
            nz2 = sigma[0] ** 2
            if self.error_var is None:
                self.error_var, self.noise_var = err2, nz2
            else:
                w = 1.0 / min(self.error_window, self.n_seen - self.n_seen_at_first_fit + 1)
                self.error_var = (1.0 - w) * self.error_var + w * err2
                self.noise_var = (1.0 - w) * self.noise_var + w * nz2
            self.error_scale = np.sqrt(np.maximum(self.error_var - self.noise_var, 0.0))
        self.reservoir.add(x, y, sigma)
        return self._maybe_fit()

    def _maybe_fit(self) -> bool:
        if self.n_seen < self.min_points:
            return False
        if self.current is None:
            self.fit()
            return True
        due = (self.n_seen >= 2 * self.n_seen_at_fit) or (self.n_seen - self.n_seen_at_fit >= self.refit_cap)
        if not due:
            return False
        if self.reservoir.turnover - self.turnover_at_fit < self.min_turnover * self.reservoir.size:
            # Too little new coverage: postpone and restart the schedule.
            self.n_skipped += 1
            self.n_seen_at_fit = self.n_seen
            return False
        self.fit()
        return True

    # -- fitting -----------------------------------------------------------------------
    def fit(self) -> GlobalMeanSnapshot:
        """Fit a new snapshot on the current reservoir (callable directly to force a refit)."""
        res = self.reservoir
        if res is None or res.n < 2:
            raise RuntimeError("Need at least 2 observations to fit the global model")
        x_scaler = StandardScaler().fit(res.X)
        y_mean = res.y.mean(axis=0)
        y_scale = float((res.y - y_mean).std())
        if not np.isfinite(y_scale) or y_scale <= 0.0:
            y_scale = 1.0
        Y = (res.y - y_mean) / y_scale
        # The GP takes one noise variance per point, so per-output noise variances
        # (standardised) are averaged over the outputs.
        alpha = np.maximum(np.mean((res.sigma / y_scale) ** 2, axis=1), self.alpha_floor)
        kernel = self.current.gp.kernel_ if self.current is not None else self.kernel   # warm start
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha,
                                      n_restarts_optimizer=self.n_restarts_optimizer,
                                      random_state=self.random_state)
        gp.fit(x_scaler.transform(res.X), Y if Y.shape[1] > 1 else Y[:, 0])
        version = 1 if self.current is None else self.current.version + 1
        if self.current is None:
            self.n_seen_at_first_fit = self.n_seen
        self.current = GlobalMeanSnapshot(version, x_scaler, y_mean, y_scale, gp, res.n)
        self.n_seen_at_fit = self.n_seen
        self.turnover_at_fit = res.turnover
        self.n_refits += 1
        return self.current

    def __repr__(self) -> str:
        return (f"AdditiveGPGlobalMean(reservoir_size={self.reservoir_size}, min_points={self.min_points}, "
                f"min_turnover={self.min_turnover}, n_restarts_optimizer={self.n_restarts_optimizer}, "
                f"n_seen={self.n_seen}, n_refits={self.n_refits}, current={self.current!r})")


def make_global_mean(spec: Union[None, str, GlobalMeanLearner], **kwargs) -> Optional[GlobalMeanLearner]:
    """Resolve the ``GPTree(global_mean=...)`` argument to a learner (or None)."""
    if spec is None:
        if kwargs:
            raise ValueError("global_mean_kwargs given but global_mean is None")
        return None
    if isinstance(spec, GlobalMeanLearner):
        if kwargs:
            raise ValueError("global_mean_kwargs cannot be combined with a learner instance")
        return spec
    if isinstance(spec, str):
        if spec == 'additive_gp':
            return AdditiveGPGlobalMean(**kwargs)
        if spec == 'net':
            from pygptreeo.neural_linear import NetGlobalMean
            return NetGlobalMean(**kwargs)
        raise ValueError(f"Unknown global_mean '{spec}'. Use None, 'additive_gp', 'net' or a GlobalMeanLearner instance.")
    raise TypeError("global_mean must be None, a string or a GlobalMeanLearner instance")
