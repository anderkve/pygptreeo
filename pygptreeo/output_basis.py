"""Global linear output basis (PCA) for multi-output GPTree leaves.

With ``GPTree(output_model='pca')`` the tree learns one global linear basis for
the output space from a reservoir sample of the stream, and every leaf models
the *scores* of its data in that basis with ``k`` independent GPs instead of
modelling all ``n_outputs`` columns separately. Predictions are mapped back to
output space through the basis, including the propagated GP variances and the
(fixed) truncation residual of the components that were dropped.

The basis is stored in an :class:`OutputBasis` (immutable once built, carries a
version number) and learned/refit by an :class:`OutputBasisLearner`, which is
shared by reference between the tree and all of its nodes. Leaves record the
basis version they were fitted with, so a refit of the basis never invalidates
an already trained leaf: each leaf simply picks up the new basis at its next
retrain, because leaves keep storing the raw outputs and project them at fit time.
"""

from typing import Optional, Union

import numpy as np


class OutputBasis:
    """A centred, scaled, truncated orthonormal basis for the output space.

    Outputs ``y`` (shape ``(n_outputs,)``) map to scores ``z`` (shape ``(k,)``) via

        z = ((y - mu) / scale) @ W

    and back via ``y = mu + scale * (z @ W.T)``. ``W`` has orthonormal columns, so
    with independent score GPs the reconstructed variance of output ``i`` is
    ``scale**2 * (sum_j W[i, j]**2 var_j + resid_var[i])``, where ``resid_var`` is
    the per-output variance (in scaled units) of the components not kept.

    Attributes
    ----------
    mu : np.ndarray, shape (n_outputs,)
        Per-output centring.
    scale : float
        One common scale for all outputs. A *common* scale (rather than one per
        output) keeps per-point observation noise identical across outputs in the
        scaled space, which is what makes noise propagate cleanly to the scores.
    W : np.ndarray, shape (n_outputs, k)
        Orthonormal basis vectors (principal directions) as columns.
    resid_var : np.ndarray, shape (n_outputs,)
        Truncation residual variance per output, in scaled units.
    explained_variance_ratio : np.ndarray, shape (k,)
        Fraction of total (scaled) variance carried by each kept component.
    version : int
        Increments every time the learner refits the basis.
    n_fit : int
        Number of reservoir rows the basis was fitted on.
    """

    def __init__(self, mu, scale, W, resid_var, explained_variance_ratio, version, n_fit):
        self.mu = np.asarray(mu, dtype=float)
        self.scale = float(scale)
        self.W = np.asarray(W, dtype=float)
        self.resid_var = np.asarray(resid_var, dtype=float)
        self.explained_variance_ratio = np.asarray(explained_variance_ratio, dtype=float)
        self.version = int(version)
        self.n_fit = int(n_fit)

    @property
    def n_outputs(self) -> int:
        return self.W.shape[0]

    @property
    def n_components(self) -> int:
        return self.W.shape[1]

    def project(self, Y: np.ndarray) -> np.ndarray:
        """Map outputs ``Y`` of shape (n, n_outputs) to scores of shape (n, k)."""
        return ((np.atleast_2d(Y) - self.mu) / self.scale) @ self.W

    def project_noise(self, sigma: np.ndarray) -> np.ndarray:
        """Map per-point, per-output noise std ``sigma`` (n, n_outputs) to per-score
        noise *variance* (n, k): ``var_z[:, j] = sum_i W[i, j]**2 (sigma[:, i]/scale)**2``."""
        var_scaled = (np.atleast_2d(sigma) / self.scale) ** 2
        return var_scaled @ (self.W ** 2)

    def reconstruct(self, z_mean: np.ndarray, z_var: Optional[np.ndarray] = None):
        """Map score means (n, k) [and variances (n, k)] back to output space.

        Returns ``y_mean`` of shape (n, n_outputs) and, if ``z_var`` is given,
        ``y_var`` of shape (n, n_outputs) that includes the truncation residual.
        """
        z_mean = np.atleast_2d(z_mean)
        y_mean = self.mu + self.scale * (z_mean @ self.W.T)
        if z_var is None:
            return y_mean
        z_var = np.atleast_2d(z_var)
        y_var = (self.scale ** 2) * (z_var @ (self.W.T ** 2) + self.resid_var)
        return y_mean, y_var

    def __repr__(self) -> str:
        return (f"OutputBasis(n_outputs={self.n_outputs}, n_components={self.n_components}, "
                f"version={self.version}, n_fit={self.n_fit}, "
                f"explained={self.explained_variance_ratio.sum():.5f})")


class OutputBasisLearner:
    """Learns and refits an :class:`OutputBasis` from a reservoir sample of the stream.

    Parameters
    ----------
    n_outputs : int
        Output dimensionality.
    n_components : {'noise', int, float}, default='noise'
        How many components to keep.

        * ``'noise'``: keep every component whose variance lies above the noise
          bulk, i.e. above ``v_n * (1 + sqrt(n_outputs / n))**2`` where ``v_n`` is
          the mean per-point observation-noise variance (in the basis' scaled
          units) and ``n`` the number of reservoir rows. This is the
          Marchenko-Pastur edge of the eigenvalues that pure noise would produce,
          so it keeps exactly the components that carry signal: fewer for noisy
          data, more for clean data. Requires ``sigma`` to be passed to
          :meth:`observe`; without it, all components above ``1e-12`` of the
          total variance are kept.
        * ``int``: a fixed number of components (capped by what the data supports).
        * ``float`` in (0, 1): the smallest number of components whose cumulative
          explained variance reaches this fraction. The dropped components set a
          reconstruction-error floor of roughly ``sqrt(1 - fraction) * scale``.
    max_components : int or None, default=None
        Upper cap on the number of components (``None``: no cap beyond what the
        data supports, ``min(n_outputs, n - 1)``).
    min_points : int, default=50
        Number of observed outputs required before the first basis is fitted.
        Leaves cannot train in 'pca' mode before this.
    refit_every : int or None, default=None
        Refit the basis every this many observed points. ``None`` uses a doubling
        schedule (refit when the number of points seen has doubled since the last
        fit), which is cheap and converges quickly.
    reservoir_size : int, default=2000
        Maximum number of output rows kept (uniform reservoir sampling).
    random_state : int or None
        Seed for the reservoir sampling.
    """

    def __init__(self, n_outputs: int, n_components: Union[str, int, float] = 'noise',
                 max_components: Optional[int] = None,
                 min_points: int = 50, refit_every: Optional[int] = None,
                 reservoir_size: int = 2000, random_state: Optional[int] = None):
        if isinstance(n_components, str):
            if n_components != 'noise':
                raise ValueError(f"n_components must be 'noise', a positive int or a float in (0, 1), got {n_components!r}")
        elif isinstance(n_components, float):
            if not (0.0 < n_components < 1.0):
                raise ValueError("A float n_components must be in (0, 1) (explained-variance fraction)")
        elif int(n_components) != n_components or n_components < 1:
            raise ValueError(f"n_components must be 'noise', a positive int or a float in (0, 1), got {n_components!r}")
        if max_components is not None and max_components < 1:
            raise ValueError("max_components must be at least 1")
        if min_points < 2:
            raise ValueError("min_points must be at least 2")
        self.n_outputs = int(n_outputs)
        self.n_components = n_components
        self.max_components = max_components
        self.min_points = int(min_points)
        self.refit_every = refit_every
        self.reservoir_size = int(reservoir_size)
        self._rng = np.random.RandomState(random_state)

        self._reservoir = np.empty((0, self.n_outputs))
        self._reservoir_noise = np.empty((0,))   # mean noise variance of each reservoir row (nan if unknown)
        self.n_seen = 0
        self.n_seen_at_last_fit = 0
        self.current: Optional[OutputBasis] = None

    def observe(self, y: np.ndarray, sigma=None) -> bool:
        """Register one output row (and its observation-noise std ``sigma``, scalar
        or per output); fit/refit the basis when due. Returns True if refit."""
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.shape[0] != self.n_outputs:
            raise ValueError(f"expected {self.n_outputs} outputs, got {y.shape[0]}")
        noise = np.nan if sigma is None else float(np.mean(np.asarray(sigma, dtype=float) ** 2))
        self.n_seen += 1
        if self._reservoir.shape[0] < self.reservoir_size:
            self._reservoir = np.vstack((self._reservoir, y[None, :]))
            self._reservoir_noise = np.append(self._reservoir_noise, noise)
        else:
            j = self._rng.randint(0, self.n_seen)
            if j < self.reservoir_size:
                self._reservoir[j] = y
                self._reservoir_noise[j] = noise
        return self._maybe_fit()

    def _maybe_fit(self) -> bool:
        if self.n_seen < self.min_points:
            return False
        if self.current is None:
            due = True
        elif self.refit_every is None:
            due = self.n_seen >= 2 * self.n_seen_at_last_fit
        else:
            due = (self.n_seen - self.n_seen_at_last_fit) >= self.refit_every
        if not due:
            return False
        self.fit()
        return True

    def fit(self) -> OutputBasis:
        """Fit the basis on the current reservoir (also usable to force a refit)."""
        Y = self._reservoir
        n = Y.shape[0]
        if n < 2:
            raise RuntimeError("Need at least 2 observed outputs to fit an output basis")
        mu = Y.mean(axis=0)
        Yc = Y - mu
        scale = float(Yc.std())
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        Ys = Yc / scale
        # Thin SVD of the centred, scaled reservoir: Ys = U S Vt, principal directions are rows of Vt.
        _, S, Vt = np.linalg.svd(Ys, full_matrices=False)
        var = S ** 2 / max(n - 1, 1)
        total = var.sum()
        ratio = var / total if total > 0 else np.zeros_like(var)
        max_k = min(self.n_outputs, n - 1, len(S))
        if self.max_components is not None:
            max_k = min(max_k, int(self.max_components))
        if isinstance(self.n_components, str):  # 'noise'
            known = self._reservoir_noise[np.isfinite(self._reservoir_noise)]
            if known.size > 0:
                v_n = float(np.mean(known)) / scale ** 2
                edge = v_n * (1.0 + np.sqrt(self.n_outputs / n)) ** 2
            else:
                edge = 1e-12 * total
            k = int(np.sum(var > edge))
        elif isinstance(self.n_components, float):
            cum = np.cumsum(ratio)
            k = int(np.searchsorted(cum, self.n_components) + 1)
        else:
            k = int(self.n_components)
        k = max(1, min(k, max_k))
        W = Vt[:k].T                                   # (n_outputs, k), orthonormal columns
        resid = Ys - (Ys @ W) @ W.T
        resid_var = (resid ** 2).mean(axis=0)          # per output, scaled units
        version = 1 if self.current is None else self.current.version + 1
        self.current = OutputBasis(mu, scale, W, resid_var, ratio[:k], version, n)
        self.n_seen_at_last_fit = self.n_seen
        return self.current

    def __repr__(self) -> str:
        return (f"OutputBasisLearner(n_outputs={self.n_outputs}, n_components={self.n_components!r}, "
                f"n_seen={self.n_seen}, current={self.current!r})")
