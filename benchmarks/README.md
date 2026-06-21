# Benchmarks

## Incremental rank-1 Cholesky updates (`IncrementalGP`)

### What it does
In the streaming setting a GPTree leaf only re-incorporates new points when its
GP is fully re-fitted (every `retrain_every_n_points` points, an O(n³)
operation). Between refits the posterior ignores the most recent points.

The `IncrementalGP` backend can instead incorporate **every** new point
immediately via an **exact rank-1 Cholesky update** (O(n²)), while still
re-optimizing the kernel hyperparameters only periodically:

* `fit()` — full fit: optimize hyperparameters + build the Cholesky factor
  (delegated to scikit-learn). This is the periodic, expensive step.
* `add_observation()` — extend the Cholesky factor with one new point, holding
  the hyperparameters fixed.

`GPTree(..., incremental_updates=True)` (default) calls `add_observation` for
each streamed point between full refits. It is a harmless no-op for backends
that don't support it (e.g. the default scikit-learn one), so default behaviour
is unchanged.

On a split, a child node inherits a deep copy of the parent's fitted GP (warm
start, exactly like the scikit-learn backend), so it predicts immediately; the
child is re-fit on its own local data before rank-1 updates begin (tracked by an
internal `_gp_fitted_on_own_data` flag). Incremental-capable backends also
re-optimize hyperparameters when a leaf's data has grown substantially since its
last fit (geometric "doubling" schedule), keeping young leaves fresh.

### How to reproduce
```bash
OMP_NUM_THREADS=1 python benchmarks/benchmark_incremental.py            # optimized hyperparameters
OMP_NUM_THREADS=1 python benchmarks/benchmark_incremental.py --fixed    # fixed hyperparameters
OMP_NUM_THREADS=1 python benchmarks/benchmark_incremental.py --no-gold  # skip the slow R=1 reference
```
All regimes use the **same** `IncrementalGP` backend, differing only in *when*
points enter the GP:

| regime | `retrain_every_n_points` | `incremental_updates` |
|--------|--------------------------|-----------------------|
| `full (R=1)` | 1 | False | full refit at every point (most current, slow) |
| `lazy (R=R)` | R | False | recent points ignored until next refit (cheap) |
| `incremental` | R | True | rank-1 updates between refits (posterior current) |

### Findings

**1. Correctness — the rank-1 posterior is exact.**
With hyperparameters fixed, incremental updates reproduce a from-scratch refit to
machine precision (unit test: ~1e-7; single-leaf tree: 0.0 mean difference). The
incremental posterior is the *same* object as the full-refit posterior, just
computed cheaply.

**2. Cost — keeping the posterior current is O(n²)/point instead of O(n³)/point.**
Maintaining an always-current posterior (fixed hyperparameters, single leaf):

| N (points) | rank-1 incremental | refit every point | speedup |
|-----------:|-------------------:|------------------:|--------:|
| 300 | 0.11 s | 0.99 s | 8.7× |
| 600 | 0.14 s | 6.05 s | 42.8× |

The speedup grows with N (total cost O(n³) vs O(n⁴)). This is the core benefit:
if you want the posterior to reflect every streamed point, rank-1 updates make it
affordable; the alternative (refitting every point) becomes prohibitive as leaves
grow.

**3. In the full tree with optimized hyperparameters — a modest accuracy effect.**
Representative run (`Nbar=200`, `R=30`, `noise=1e-2`, 3 seeds, N=500):

| config | final RMSE | final NLPD | train time |
|--------|-----------:|-----------:|-----------:|
| lazy (R=30)        | 0.0173 | −2.94 | 1.5 s |
| incremental (R=30) | 0.0173 | −2.95 | 1.7 s |
| full (R=1)         | 0.0119 | −3.05 | 25.3 s |

Honest caveats:
* **vs full-refit-every-point:** incremental is ~15× faster but somewhat less
  accurate — not because of the points (those are exact) but because it
  re-optimizes *hyperparameters* less often. That gap is controllable via
  `retrain_every_n_points` and the growth-doubling schedule.
* **vs lazy refitting at the same cadence:** incremental is slightly better
  (~0–7% RMSE, averaged over checkpoints; noisy) by incorporating recent points,
  for a little extra per-point cost. The gain is modest here because warm
  inheritance and growth-doubling re-optimization already keep the lazy baseline
  reasonably fresh on this small, cheap-to-fit problem.

### When it's worth using
`IncrementalGP` pays off when **(a)** you need predictions that reflect every
point as it arrives (rather than tolerating staleness until the next refit), and
**(b)** full refits are expensive — large `Nbar` (big local GPs), costly kernels,
or many hyperparameters. In those regimes the O(n²) vs O(n³) per-point gap (and
its growth with N) is the deciding factor. For small, cheap-to-fit leaves the
accuracy difference from a tuned lazy schedule is marginal. It is a self-contained
optional backend and does not change the default behaviour.

Plot: `incremental.png` (from the optimized-hyperparameter run).
