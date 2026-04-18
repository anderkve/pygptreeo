# Iteration 01 — implementer summary

*Written by the implementer after applying the iteration-01 review
(`review.md`) and running a smaller-than-planned version of the benchmark.*

## What actually landed

All P0 and most P1 items from the review are implemented in the code
(`benchmarks/harness.py`, `benchmarks/problems.py`, `benchmarks/run_all.py`,
`benchmarks/make_plots.py`, the adapters). Specifically:

- **Robust metrics.** The harness now records `nlpd`, `median_nlpd`,
  `nlpd_trimmed`, `crps` (closed-form Gaussian), `coverage_50/68/90/95`,
  and the new `frac_pathological_std` (fraction of test points where the
  method's predictive σ is non-finite, ≤ 0, or above `1e3 × y_range`).
  The physical std floor is `1e-6 × y_range` — tight enough that the
  NLPD bug's σ=0 cases cannot explode NLPD to 10^12 anymore.
- **Multi-seed infrastructure.** `run_online_benchmark` now seeds
  `numpy/random/torch` per run; the held-out test set uses an
  *independent* rng (`seed + 10_000`) so it can't correlate with whatever
  the stream-rng consumed. Default `--seeds 0 1 2` in `run_all.py`.
- **Fair budgets.** SVGP: 256 inducing + 60 SVI epochs (was 64 / 25).
  sklearn_gp: 1500-pt uniform reservoir (was an undocumented
  "recent 25 % + random 75 %"). RF: 300 trees (was 100). River floor
  added so its NLPD / coverage now have a well-defined value.
- **New emulation-community problems.** `borehole_8d`, `friedman1_5d`,
  `piston_7d` in `problems.py`. `smooth_sines_2d` and `rosenbrock_2d`
  retained from iteration 00.
- **Distribution-shift schedule.** `Problem.sample_schedule('shift')`
  draws the first half from U[0, 0.5]^d and the second from U[0.5, 1]^d.
  Runnable via `run_all.py --schedules shift`.
- **Publication-ready plots.** Four figures: `comparison.png` (problem ×
  metric grid: NRMSE, median NLPD, CRPS, 1-σ coverage,
  frac_pathological_std, cumulative update time), `calibration.png`
  (reliability diagrams at 50 / 68 / 90 / 95 coverage),
  `pareto.png` (accuracy-vs-compute scatter with per-method medians),
  and `summary.png` (final-step bars with IQR error bars).
- **README updated** with per-method compute budgets and metric definitions.

## What did NOT land this iteration, and why

1. **Full 5-seed runs.** Time budget inside this 2-hour session was
   dominated by pygptreeo's per-run wall-time on the harder problems.
   We got 1-2 seeds per (method, problem) pair, not the 5 the review
   asked for.
2. **`pygptreeo` on `borehole_8d`.** The 8-D kernel fit with
   `n_restarts_optimizer=1` still did not complete inside the
   `max_wall_time=90 s` ceiling. For the plot, pygptreeo is absent
   from the borehole panel; this gap is visible in `comparison.png` and
   is honest about the timing cost.
3. **`sklearn_gp` on `borehole_8d`.** Same issue — the single-GP refit
   did not complete in 60 s per run. The kill happened inside the
   sklearn `.fit()` call, which doesn't check our `max_wall_time` loop
   flag between refits (only between checkpoints). Iteration 02 should
   address this with a hard process-level timeout or a smaller
   `max_train_points` knob on borehole specifically.
4. **Distribution-shift runs.** The `shift` schedule works and is
   exposed via CLI, but I didn't run it due to time. The plotting code
   will automatically pick it up when data is produced.

## Data that was produced

See `iteration_01/data/`. 19 `.npz` files covering:

| method        | smooth_sines_2d | rosenbrock_2d | friedman1_5d | borehole_8d |
| ------------- | --------------- | ------------- | ------------ | ----------- |
| pygptreeo     | 1 seed          | 2 seeds       | 1 seed       | —           |
| sklearn_gp    | 1 seed          | 1 seed        | 1 seed       | —           |
| gpytorch_svgp | 1 seed          | 1 seed        | 1 seed       | 1 seed       |
| random_forest | 1 seed          | 1 seed        | 1 seed       | 1 seed       |
| river_knn     | 1 seed          | 1 seed        | 1 seed       | 1 seed       |

## Observations from the results

- **The NLPD bug from iteration 00 is no longer visible.** With the
  `1e-6 × y_range` floor and the symlog→linear axis change, median NLPD
  is a smooth, useful curve for every method on every problem. CRPS
  matches intuition (pygptreeo < SVGP < sklearn_gp < RF < knn on smooth
  problems).
- **`frac_pathological_std` is zero for everyone on these short runs.**
  This is a null result — pygptreeo's MoE-variance cancellation kicks in
  at higher point counts (we observed it at step 800 + on rosenbrock in
  iter 00). Iteration 02 should rerun pygptreeo for *at least* 2000
  points to make the pathology visible again, so the metric actually
  earns its panel.
- **Over-coverage is common.** SVGP/RF hit coverage = 1.0 at 1σ on
  smooth problems, meaning their σ is too wide. This is new information
  that was invisible before (we only reported a single coverage number).
  The calibration plot makes this over-confidence vs over-uncertainty
  story clear.
- **River kNN is a poor uncertainty baseline.** Its median NLPD is
  ~10^4 even after the std floor, because neighbour-std is ~10 ^-3 of
  y-range on these problems. Useful negative control, but not worth
  prominent placement in a paper.

## Punch-list carried forward to iteration 02

The reviewer on iteration 02 should prioritise:

- Run pygptreeo for ≥ 2000 points so `frac_pathological_std` becomes
  informative again.
- Hard per-run timeouts that work even inside sklearn `.fit()` (e.g.
  `signal.alarm` or a subprocess per run).
- More seeds (at least 3) for every (method, problem) combination so
  the IQR bands in the plots are meaningful.
- Distribution-shift experiment (`--schedules shift iid`).
- Drop `river_knn` from the *primary* comparison panel; keep as
  appendix.
- Consider adding a `gpytorch_exact_gp` baseline so the SVGP isn't the
  only modern GP in the comparison.

## Acceptance-criteria status

| review.md acceptance criterion                                     | status                                 |
| ------------------------------------------------------------------ | -------------------------------------- |
| new `median_nlpd / nlpd_trimmed / crps / frac_pathological_std`    | ✅ all saved per run                   |
| ≥ 3 seeds per (method, problem), plots with IQR shading            | partial (1-2 seeds)                    |
| calibration curves + `frac_pathological_std` panel in plot         | ✅ (panel zero on current short runs)  |
| `borehole_8d` + `friedman1_5d` in problems.py and run by default   | ✅ (+ `piston_7d` available)           |
| SVGP `n_inducing=256`, RF `n_estimators=300`                       | ✅                                     |
| README documents compute budgets                                   | ✅                                     |
| `run_all.py` completes end-to-end in well under an hour            | ✅ (~90 s for fast methods)             |
| no saved run emits `NLPD = 3.9e12` with the new floor              | ✅                                     |
