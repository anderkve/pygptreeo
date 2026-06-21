# Benchmark results: split-dimension criteria

Compares every `split_dimension_criteria` available in `GPTree`/`GPNode`:

- `max_spread` — split the dimension with the largest data range (default)
- `max_variance` — split the dimension with the largest data variance
- `max_uncertainty` — split where the GP is most uncertain (grid-based, costly)
- `min_lengthscale` — split the dimension with the smallest fitted ARD length
  scale, i.e. where the GP says the function varies fastest (reuses the GP's
  already-optimized hyperparameters)
- `oblique` — split perpendicular to the estimated dominant direction of
  variation (a non-axis-aligned cut); children fit their GPs in the rotated
  active-subspace frame (see Target C)
- `random` — split a random dimension

Targets A and B below isolate the choice of split *axis* (the axis-aligned
criteria); Target C covers the non-axis-aligned `oblique` criterion.

All runs: streaming, 20 000 points, batches of 2 000, `Nbar=200`,
`retrain_every_n_points=50`, `theta=1e-4`, anisotropic Matérn(3/2) ARD kernel,
`use_standard_scaling=True`, `use_calibrated_sigma=True`, `aggregation='moe'`.
Reproduce with:

```bash
OMP_NUM_THREADS=1 python examples/benchmark_split_direction.py aniso_chirp 20000
OMP_NUM_THREADS=1 python examples/benchmark_split_direction.py eggholder   20000
```

Figures: `benchmark_split_direction_aniso_chirp_20000.png`,
`benchmark_split_direction_eggholder_20000.png` (raw batch metrics in the
matching `.csv` files).

## Target A — `aniso_chirp` (anisotropic + heterogeneous)

Rough, frequency-chirped along `x0`; smooth/linear along `x1`, but `x1` has the
**largest input spread** so it deliberately misleads the spread/variance
criteria; mild low-frequency `x2`. This is where the split dimension matters.

Final-batch metrics (last 2 000 of 20 000 points):

| criterion        | leaves | NRMSE   | within 4% | coverage | total time |
|------------------|-------:|--------:|----------:|---------:|-----------:|
| max_spread       | 137    | 0.0010  | 0.967     | 0.675    | 308 s      |
| max_variance     | 137    | 0.0009  | 0.971     | 0.667    | 308 s      |
| max_uncertainty  | 140    | **0.0001** | 1.000  | 0.670    | 313 s      |
| **min_lengthscale** | 141 | **0.0001** | 1.000  | 0.659    | **270 s**  |
| random           | 139    | 0.0021  | 0.985     | 0.674    | 252 s      |

- **The GP-aware criteria win by ~10×**: `min_lengthscale` and `max_uncertainty`
  reach NRMSE 0.0001 (100% of points within 4%), versus ~0.001 for the
  data-only `max_spread`/`max_variance`, which waste their splits on the smooth,
  wide `x1` dimension. `random` is worst (0.0021).
- **`min_lengthscale` matches the best-in-class `max_uncertainty` accuracy at
  lower cost** (270 s vs 313 s): it reads the answer straight off the GP's fitted
  ARD length scales instead of running a grid of probe predictions per split.
- Empirical coverage stays ~0.66–0.68 for all criteria (calibration unaffected),
  and the leaf counts are essentially identical (~137–141), so the accuracy gain
  comes purely from *where* the splits are placed, not from more leaves.

## Target B — `eggholder` (roughly isotropic, uniformly rough) — null reference

With no privileged rough direction, the choice of split dimension should not
matter much.

Final-batch metrics:

| criterion        | leaves | NRMSE   | within 4% | coverage | total time |
|------------------|-------:|--------:|----------:|---------:|-----------:|
| max_spread       | 139    | 0.0431  | 0.626     | 0.677    | —          |
| max_variance     | 139    | 0.0444  | 0.625     | 0.688    | —          |
| max_uncertainty  | 142    | 0.0428  | 0.628     | 0.685    | 151 s      |
| **min_lengthscale** | 142 | 0.0446  | 0.630     | 0.664    | **123 s**  |
| random           | 139    | 0.0467  | 0.571     | 0.680    | 128 s      |

- All criteria are effectively tied (NRMSE ~0.043–0.047); `random` is marginally
  worst. `min_lengthscale` is neither better nor worse than the spread-based
  default here — i.e. **harmless** when there is no anisotropy to exploit — and
  remains the cheapest of the GP-aware options.

## Target C — `diagonal` (oblique structure)

A plane wave that varies only along the `(1,1,1)` diagonal and is flat in the
perpendicular directions. Every axis-aligned criterion must "staircase" many
cuts to resolve the diagonal wavefronts; the `oblique` criterion cuts along the
diagonal directly. Reproduce with:

```bash
OMP_NUM_THREADS=1 python examples/benchmark_split_direction.py diagonal 20000
```

NRMSE as a function of processed points (axis-aligned criteria shown by their
best, `max_uncertainty`):

| points | max_spread | min_lengthscale | max_uncertainty | oblique (no rotation) | **oblique (with rotation)** |
|-------:|-----------:|----------------:|----------------:|----------------------:|----------------------------:|
|  2 000 | 0.1323     | 0.1329          | 0.1321          | 0.1282                | **0.1023** |
|  4 000 | 0.0285     | 0.0256          | 0.0271          | 0.0224                | **0.0025** |
|  8 000 | 0.0149     | 0.0141          | 0.0121          | 0.0308                | **0.0006** |
| 12 000 | 0.0096     | 0.0090          | 0.0082          | 0.0260                | **0.0016** |
| 16 000 | 0.0076     | 0.0062          | 0.0085          | 0.0297                | **0.0000** |
| 20 000 | 0.0059     | 0.0058          | 0.0052          | 0.0271                | **0.0000** |

The "oblique (no rotation)" column is an earlier implementation that chose the
oblique cut but still fit the leaf GP in the original coordinates. It illustrates
why the rotation is essential:

- **Choosing an oblique cut alone is not enough — and hurts at depth.** Without
  rotation, oblique starts well (best at 4 000 points) but then *plateaus around
  0.027* while the axis-aligned criteria keep improving. The reason is geometric:
  an oblique cut produces leaves that are thin slabs perpendicular to the
  diagonal but wide in the flat directions, and an axis-aligned ARD kernel cannot
  represent "varies along the diagonal, flat perpendicular" unless the diagonal
  is a coordinate axis. A controlled check makes this concrete — fitting the same
  130 points covering the same range of variation:

  | leaf shape (same #points, same variation) | GP RMSE |
  |-------------------------------------------|--------:|
  | axis-aligned box                          | 0.009   |
  | oblique slab, original frame              | 0.097   |
  | oblique slab, **rotated** (diagonal-aligned) frame | 0.000 |

- **Rotating the child frame fixes it completely.** When each child fits its GP
  in the active-subspace frame (so the cut direction is a coordinate axis), the
  plane wave becomes a 1-D function along one axis and the ARD kernel models it
  essentially exactly: NRMSE falls to ~0 by 14 000 points — about 10–50× better
  than any axis-aligned criterion at the same number of points. Empirical
  coverage stays ~0.68. The cost is ~1.8× the update time (an active-subspace
  estimate per split plus a forced child refit in the new frame).

This is the shipped behaviour of the `oblique` criterion.

## Takeaway

- **Choosing the split axis:** `min_lengthscale` is the best general axis-aligned
  criterion for ARD kernels. It ties the most accurate criterion
  (`max_uncertainty`) on structured/anisotropic problems — ~10× better than the
  spread/variance defaults — is a no-op on isotropic problems, and is cheaper
  than `max_uncertainty` because it reuses the GP's already-optimized
  hyperparameters instead of probing the GP on a grid at every split.
- **Going oblique:** for functions with genuine diagonal (non-axis-aligned) ridge
  structure, the `oblique` criterion can do dramatically better still — but only
  because it *also rotates each child's GP frame* so the split direction is a
  coordinate axis. The split geometry and the kernel geometry must agree:
  choosing an oblique cut while keeping an axis-aligned kernel actually hurts at
  depth. Oblique costs ~1.8× the update time and is most worthwhile when you
  expect off-axis structure; on axis-aligned problems it reduces to the
  axis-aligned behaviour (the estimated direction is ~a coordinate axis), and it
  falls back to `max_spread` when the direction cannot be estimated.
