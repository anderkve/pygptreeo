"""PairInteractionScreen: region-local discovery of important coordinate pairs.

The additive leaf kernel (:func:`pygptreeo.kernels.make_additive_kernel`) carries
a pairwise term for every pair of inputs, but most targets only couple a few
pairs -- and an additively separable target couples none. Carrying the dead
pairs costs compute (a depth-2 kernel over ``d`` inputs has ``C(d,2)`` pairwise
terms, each evaluated on every kernel call and gradient) without buying any
accuracy. This object discovers, from the data itself, which pairs actually
interact, so the rest can be pruned when a leaf is created.

Discovery signal
----------------
A model-free 2-way ANOVA interaction measure. Bin ``x_i`` and ``x_j`` into a
``B x B`` grid and look at the cell-mean table. The part of each cell not
explained by its row mean + column mean (``cell - row - col + grand``) is the
pure pairwise interaction -- by construction blind to main effects. Its
count-weighted variance, divided by the total variance of ``y``, is the pair's
interaction score. This is cheap, decoupled from the GP, and -- crucially --
robust enough at large sample sizes to rank pairs reliably.

Region-local accumulation (the tree-structured part)
----------------------------------------------------
A screen kept on a single leaf's ~Nbar points is far too noisy to trust. Instead
each node carries a screen and, when it splits, every child *inherits a copy of
the parent's accumulated statistics* (:meth:`copy`) and then keeps updating it
with only the points routed into the child's region. So statistics both
**accumulate** (a deep leaf's screen is seeded by its whole ancestral chain) and
**localise** (over time a leaf's own region-local points come to dominate the
inherited prior). Different regions can therefore end up keeping different pairs.
Binning uses running per-dimension min/max inherited along with the counts, so
the inherited histogram and the new local points share one consistent grid.

Inputs are assumed bounded (true for the standard benchmarks); running min/max
converge quickly for bounded streams.
"""

from copy import deepcopy
from itertools import combinations

import numpy as np


class PairInteractionScreen:
    """Region-local detector of important pairwise coordinate interactions.

    Attributes:
        n_features (int): Number of input dimensions.
        n_bins (int): Bins per dimension for the 2-way tables.
        warmup_points (int): Accumulated points before the scores are trusted.
        n_seen (int): Number of points accumulated into this screen (including
            those inherited from ancestors).
    """

    def __init__(self, n_features, n_bins=6, warmup_points=150,
                 abs_floor=0.012, rel_factor=2.5, low_pct=40.0):
        self.n_features = int(n_features)
        self.n_bins = int(n_bins)
        self.warmup_points = int(warmup_points)
        # Pruning aggressiveness. A pair is kept only if its interaction score
        # clears BOTH an absolute floor and a relative margin above the
        # low-percentile (empirical noise floor) of all pair scores. Larger
        # values prune more aggressively; the rescue Matern term in the kernel
        # is the safety net that makes aggressive pruning low-risk.
        self.abs_floor = float(abs_floor)
        self.rel_factor = float(rel_factor)
        self.low_pct = float(low_pct)

        self.pairs = list(combinations(range(self.n_features), 2))
        B = self.n_bins
        # Per-pair binned sufficient statistics: count and sum(y) over a B x B grid.
        self._count = {p: np.zeros((B, B)) for p in self.pairs}
        self._sumy = {p: np.zeros((B, B)) for p in self.pairs}

        # Running per-dimension min/max (binning grid) and Welford stats for y.
        self._xmin = np.full(self.n_features, np.inf)
        self._xmax = np.full(self.n_features, -np.inf)
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0  # sum of squared deviations of y -> SS_tot
        self.n_seen = 0

    def copy(self):
        """Return a deep copy for a child node to inherit and keep updating."""
        return deepcopy(self)

    def update(self, x, y):
        """Accumulate one point into this region's interaction statistics.

        Args:
            x (np.ndarray): Input point, shape (n_features,) or (1, n_features).
            y (float): Target value (first output for multi-output).
        """
        x = np.asarray(x, dtype=float).ravel()
        y = float(np.ravel(y)[0])

        # Welford update for SS_tot (total variance of y).
        self._n += 1
        delta = y - self._mean
        self._mean += delta / self._n
        self._M2 += delta * (y - self._mean)

        # Update running input ranges, then bin into [min, max].
        self._xmin = np.minimum(self._xmin, x)
        self._xmax = np.maximum(self._xmax, x)
        self.n_seen += 1

        if not self.pairs:
            return

        span = self._xmax - self._xmin
        span = np.where(span > 0, span, 1.0)
        b = np.clip(((x - self._xmin) / span * self.n_bins).astype(int),
                    0, self.n_bins - 1)
        for (i, j) in self.pairs:
            self._count[(i, j)][b[i], b[j]] += 1.0
            self._sumy[(i, j)][b[i], b[j]] += y

    def _pair_score(self, pair):
        """Interaction variance share for one pair from its accumulated table."""
        cnt = self._count[pair]
        sm = self._sumy[pair]
        total = cnt.sum()
        if total < 1 or self._M2 <= 0:
            return 0.0
        grand = sm.sum() / total
        row_cnt = cnt.sum(1); col_cnt = cnt.sum(0)
        row_sum = sm.sum(1); col_sum = sm.sum(0)
        SS_int = 0.0
        for a in range(self.n_bins):
            if row_cnt[a] == 0:
                continue
            r = row_sum[a] / row_cnt[a]
            for bb in range(self.n_bins):
                c = cnt[a, bb]
                if c == 0 or col_cnt[bb] == 0:
                    continue
                m = sm[a, bb] / c
                col = col_sum[bb] / col_cnt[bb]
                SS_int += c * (m - r - col + grand) ** 2
        return SS_int / self._M2

    def scores(self):
        """Return ``{pair: interaction variance share}`` for all pairs."""
        return {p: self._pair_score(p) for p in self.pairs}

    def is_ready(self):
        """True once enough points have accumulated to trust the scores."""
        return self.n_seen >= self.warmup_points and bool(self.pairs)

    def active_pairs(self):
        """Pairs judged to carry real interaction signal (to be kept).

        Sparse / separable regions leave no pair above threshold, so the kernel
        can collapse to main effects only. Returns ``None`` if the screen is not
        ready yet (caller should then fall back to keeping all pairs).
        """
        if not self.is_ready():
            return None
        sc = self.scores()
        if not sc:
            return []
        vals = np.array(list(sc.values()))
        floor = max(np.percentile(vals, self.low_pct), 1e-12)
        threshold = max(self.abs_floor, self.rel_factor * floor)
        return [p for p, s in sc.items() if s >= threshold]
