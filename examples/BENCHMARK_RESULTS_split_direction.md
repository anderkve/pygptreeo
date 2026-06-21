# Benchmark results: split-dimension criteria

Compares every `split_dimension_criteria` available in `GPTree`/`GPNode`:

- `max_spread` — split the dimension with the largest data range (default)
- `max_variance` — split the dimension with the largest data variance
- `max_uncertainty` — split where the GP is most uncertain (grid-based, costly)
- `min_lengthscale` — split the dimension with the smallest fitted ARD length
  scale, i.e. where the GP says the function varies fastest (reuses the GP's
  already-optimized hyperparameters)
- `random` — split a random dimension

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

## Takeaway

`min_lengthscale` is the best general choice of split-dimension criterion for
ARD kernels: it ties the most accurate criterion (`max_uncertainty`) on
structured/anisotropic problems — ~10× better than the spread/variance defaults
— is a no-op on isotropic problems, and is cheaper than `max_uncertainty`
because it reuses hyperparameters the GP has already optimized rather than
probing the GP on a grid at every split. It requires an anisotropic (ARD) kernel
and a trained GP, and falls back to `max_spread` otherwise.
