# Benchmark results: length-scale split (#1) & resolution split (#2)

Two new, composable options for `GPTree`/`GPNode`:

- **#1 — `split_dimension_criteria='min_lengthscale'`**: choose the split
  dimension from the GP's smallest fitted ARD length scale (split where the
  function varies fastest), instead of the raw data spread.
- **#2 — `split_on_resolution=True` (with `resolution_budget`)**: let a leaf
  split *before* reaching `Nbar` when its region spans more than
  `resolution_budget` length scales in some dimension. The early split is
  directed at that under-resolved dimension.

All runs: streaming, 20 000 points, batches of 2 000, `Nbar=200`,
`retrain_every_n_points=50`, `theta=1e-4`, anisotropic Matérn(3/2) ARD kernel,
`use_standard_scaling=True`, `use_calibrated_sigma=True`, `aggregation='moe'`,
`resolution_budget=3.0`. Reproduce with:

```bash
OMP_NUM_THREADS=1 python examples/benchmark_lengthscale_split.py aniso_chirp 20000
OMP_NUM_THREADS=1 python examples/benchmark_lengthscale_split.py eggholder   20000
```

Figures: `benchmark_lengthscale_split_aniso_chirp_20000.png`,
`benchmark_lengthscale_split_eggholder_20000.png` (raw batch metrics in the
matching `.csv` files).

## Target A — `aniso_chirp` (anisotropic + heterogeneous)

Rough, frequency-chirped along `x0` (smooth near `x0=0`, ~17 cycles/unit near
`x0=1`); smooth/linear along `x1`, but `x1` has the **largest input spread** so
it deliberately misleads the spread-based baseline; mild low-frequency `x2`.
This is the regime both ideas target.

Final-batch metrics (last 2 000 of 20 000 points):

| config              | leaves | NRMSE   | within 4% | coverage | total time |
|---------------------|-------:|--------:|----------:|---------:|-----------:|
| baseline            | 137    | 0.0010  | 0.967     | 0.675    | 350 s      |
| #1 min_lengthscale  | 141    | **0.0001** | 1.000  | 0.659    | 349 s      |
| #2 resolution       | 153    | 0.0002  | 0.995     | 0.640    | 314 s      |
| #1 + #2             | 150    | **0.0001** | 1.000  | 0.662    | **281 s**  |

- **#1 gives ~10× lower NRMSE** than baseline, at the same leaf count and run
  time. The baseline wastes most of its splits on the smooth, wide `x1`
  dimension; `min_lengthscale` instead splits the rough `x0` dimension where
  resolution is actually needed.
- **#2 gives ~5× lower NRMSE** *and* runs faster, by refining the rough
  high-`x0` region early instead of waiting for `Nbar`.
- **#1 + #2 is the sweet spot**: best accuracy (10× better) **and** the fastest
  run (~20% faster than baseline).
- Empirical coverage stays ~0.66–0.68 throughout: the calibration is unaffected.

## Target B — `eggholder` (roughly isotropic, uniformly rough) — honest reference

Eggholder violates the structural assumptions of both ideas (no privileged rough
direction; rough at all scales everywhere).

Final-batch metrics:

| config              | leaves | NRMSE   | within 4% | coverage | total time |
|---------------------|-------:|--------:|----------:|---------:|-----------:|
| baseline            | 139    | 0.0431  | 0.626     | 0.677    | 153 s      |
| #1 min_lengthscale  | 142    | 0.0446  | 0.630     | 0.664    | 151 s      |
| #2 resolution       | 588    | 0.1186  | 0.140     | 0.664    | 29 s       |
| #1 + #2             | 588    | 0.1186  | 0.140     | 0.664    | 29 s       |

- **#1 ≈ baseline**: with no anisotropy to exploit, the length-scale criterion
  picks essentially the same dimensions as `max_spread`. It is harmless.
- **#2 hurts here, and no budget rescues it**: eggholder is rough at *every*
  scale, so `spread_d / length_scale_d` is large in every leaf no matter how
  small. The resolution criterion therefore fires constantly and shrinks leaves
  toward the floor (`min_points_for_resolution_split`), producing 4× more,
  data-starved leaves (~34 points each) — worse accuracy, though ~5× faster.
  A budget sweep confirms this is not a tuning artifact: raising
  `resolution_budget` from 3 → 25 only drops the leaf count from 242 → 143 and
  leaves NRMSE stuck at ~0.14 (baseline ~0.07), because there is no scale at
  which eggholder looks locally smooth. On functions that are rough at all
  scales, `split_on_resolution` should simply be left off (its default). The one
  knob that *does* bound the downside is `min_points_for_resolution_split`, which
  caps how small (and how data-starved) a leaf can get.

## Takeaways

- **#1 (`min_lengthscale`) is a safe default for ARD kernels**: a large win on
  anisotropic problems and a no-op on isotropic ones, at no extra cost (it reuses
  the GP's already-fitted length scales — cheaper than the grid-based
  `max_uncertainty` criterion).
- **#2 (`split_on_resolution`) is a conditional tool, off by default**: it pays
  off only when roughness is *heterogeneous* — i.e. there is scale separation, so
  some regions can grow into large, data-rich leaves while others need refining.
  On functions that are rough at all scales (no scale separation, e.g.
  eggholder) it has nothing to exploit: it just shrinks every leaf and trades a
  lot of accuracy for speed, and no `resolution_budget` recovers the baseline.
  Enable it only when you expect that structure; `min_points_for_resolution_split`
  bounds the worst case.
- The two compose cleanly: a resolution-triggered split is directed at the
  under-resolved dimension, which is exactly the dimension `min_lengthscale`
  would choose — so **#1 + #2** gives the best accuracy *and* the fastest run on
  the structured target.
