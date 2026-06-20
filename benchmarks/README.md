# Benchmarks

## Incremental rank-1 Cholesky updates (`IncrementalGP`)

### What it does
In the streaming setting a GPTree leaf only re-incorporates new points when its
GP is fully re-fitted, which happens every `retrain_every_n_points` points and
costs O(n³). Between refits the posterior ignores the most recent points.

The `IncrementalGP` backend incorporates **every** new point immediately via an
**exact rank-1 Cholesky update** (O(n²)) while still re-optimizing the kernel
hyperparameters only periodically. Concretely:

* `fit()` does a full fit — optimizes hyperparameters and builds the Cholesky
  factor (delegating the optimization to scikit-learn). This is the periodic
  "re-optimization".
* `add_observation()` extends the Cholesky factor with one new point, holding the
  hyperparameters fixed. This is exact: the resulting posterior is identical (to
  machine precision) to a from-scratch refit with those hyperparameters.

`GPTree` calls `add_observation` for each streamed point between full refits
(see `GPTree(..., incremental_updates=True)`, on by default — it is a harmless
no-op for backends that don't support it, such as the default scikit-learn one).
For incremental-capable backends the leaf also does an early **bootstrap** fit
and re-optimizes on substantial data growth, so freshly-split leaves are usable
quickly rather than predicting the prior until their retrain buffer fills.

### How to reproduce
```bash
OMP_NUM_THREADS=1 python benchmarks/benchmark_incremental.py
# faster (skip the expensive full-refit-every-point reference):
OMP_NUM_THREADS=1 python benchmarks/benchmark_incremental.py --no-gold
```
All three regimes use the **same** `IncrementalGP` backend (identical noise
handling and hyperparameter optimization), differing only in *when* points enter
the GP:

| regime | `retrain_every_n_points` | `incremental_updates` |
|--------|--------------------------|-----------------------|
| `full (R=1)` | 1 | False | full refit at every point (accuracy gold standard, slow) |
| `lazy (R=R)` | R | False | recent points ignored until next refit (cheap) |
| `incremental` | R | True | rank-1 updates between refits (posterior always current) |

### Findings
Representative run (`Nbar=200`, `R=30`, `noise=1e-2`, 3 seeds, `N=500`):

| config | final RMSE | final NLPD | train time |
|--------|-----------:|-----------:|-----------:|
| lazy (R=30)        | 0.01431 | −2.964 | 1.85 s |
| incremental (R=30) | 0.01212 | −3.028 | 2.23 s |
| full (R=1)         | 0.01191 | −3.033 | 25.63 s |

* **Incremental matches the gold-standard accuracy** (RMSE 0.01212 vs 0.01191;
  NLPD −3.028 vs −3.033) while training **~11.5× faster** than refitting at every
  point.
* **Incremental beats the cheap lazy baseline** by ~15% RMSE at final N (and
  ~7.5% averaged over checkpoints) for ~20% more compute. The lazy curve
  sawtooths — good right after a refit, degrading until the next — while the
  incremental curve stays consistently low.
* Correctness: a unit test (`tests/test_incremental_gp.py`) verifies the rank-1
  updates reproduce a from-scratch full fit to ~1e-7.

### Notes / limitations
* The incremental path is skipped while a node holds *shared* points (gradual
  splitting); those are reconciled at the next full refit.
* Point *merging* modifies an existing stored point in place; with incremental
  updates the change is only reflected in the GP at the next full refit.
* Because per-point cost is O(n²) rather than an O(n³) refit, incremental updates
  also make larger `Nbar` (bigger local GPs) more affordable.

Plot: `incremental.png`.
