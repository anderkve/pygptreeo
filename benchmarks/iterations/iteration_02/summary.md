# Iteration 02 — implementer summary

*Written by the implementer after applying the iteration-02 review
(`review.md`).*

## What landed

All P0 items and most P1 items from the review:

* **Upstream bug fix verified.** The pygptreeo MoE variance-cancellation
  bug (`pygptreeo/gptree.py`) and the sklearn-neg-variance sigma=0 leak
  (`pygptreeo/gpnode.py`) are now fixed in the library. Every one of the
  43 `.npz` results in `data/` has `frac_pathological_std = 0.0` at
  every checkpoint on every seed. NLPD is finite and plausible
  everywhere — no more 1e12 spikes.

* **Hard per-run subprocess timeout.** `benchmarks/run_all.py` now
  spawns every `(method, problem, seed, schedule)` as a
  `multiprocessing.Process` and joins with a hard deadline of
  `max_wall_time + 30 s`. Sklearn's `.fit` no longer blocks the whole
  benchmark. The harness (`benchmarks/harness.py`) also flushes a
  partial `.npz` after every checkpoint so we keep whatever the child
  produced even when it's killed.

* **Per-dim sklearn-GP budget.** `SklearnGPAdapter` auto-scales
  `max_train_points` (1500 for d ≤ 5, 500 otherwise) and the default
  `--max-wall-time` was raised to 1200 s. On the 2-D and 5-D problems
  sklearn still hits the timeout, but at least it now returns a stub
  rather than hanging indefinitely — see the "slipped" section below.

* **Plot layout rebuild.** `make_plots.py` drops the post-fix-zero
  `frac_pathological_std` panel and replaces it with `coverage_95`.
  `nlpd_trimmed` is plotted on a symlog axis so river_knn's ~1e4 values
  no longer flatten the other curves. `river_knn` is drawn thinner,
  grey, `alpha=0.6` — kept as a weak-baseline anchor, removed from the
  pareto plot and Wilcoxon table. `--iter-dir` CLI writes every figure
  into the iteration directory automatically.

* **Paired Wilcoxon signed-rank p-values.** `wilcoxon_nrmse.png` now
  reports median NRMSE ratio vs pygptreeo and a pooled
  Wilcoxon p-value per alternative method (see table below).

* **SVGP step cap.** `GPyTorchSVGPAdapter` takes a new
  `max_steps_per_refit=500` so the SVI inner loop can no longer blow up
  into the ~2300-step regime it had before when the buffer was near
  `max_buffer=5000`.

* **Iteration directory gets its own plots.** `comparison.png`,
  `summary.png`, `pareto.png`, `calibration.png`, and
  `wilcoxon_nrmse.png` are all written into
  `benchmarks/iterations/iteration_02/` as well as the top-level
  `benchmarks/plots/`.

## Headline result from iteration 02

At 2000–3000 stream points on the `iid` schedule, pooled over
(smooth_sines_2d, rosenbrock_2d, friedman1_5d) and 3 seeds:

| alternative    | # paired runs | median NRMSE ratio (alt / pygptreeo) | Wilcoxon p (alt > pygptreeo?) |
| -------------- | ------------- | ------------------------------------- | ----------------------------- |
| sklearn GP     | 2             | 31.8                                  | — (too few pairs)             |
| GPyTorch SVGP  | 8             | 81.8                                  | **0.004**                     |
| RandomForest   | 9             | 629                                   | **0.002**                     |

pygptreeo is strictly, statistically-significantly better on the
problems where it completes. `river_knn` is excluded from the
signed-rank table by design (it's a diagnostic anchor, not a
peer baseline).

## Data coverage (43 `.npz` files in `data/`)

|                 | smooth_sines_2d | rosenbrock_2d | friedman1_5d | borehole_8d |
| --------------- | --------------- | ------------- | ------------ | ----------- |
| pygptreeo       | 3               | 3             | 3            | 0           |
| sklearn_gp      | 2 (partial)     | 0             | 0            | 0           |
| gpytorch_svgp   | 3               | 3             | 2            | 0           |
| random_forest   | 3               | 3             | 3            | 3           |
| river_knn       | 3               | 3             | 3            | 3           |

Every completed run has `frac_pathological_std[:].max() == 0.0` — a
hard regression signal we can assert in future iterations.

## What slipped (to address in iteration 03)

1. **`sklearn_gp` scaling is still a bottleneck.** A full
   `GaussianProcessRegressor.fit` on 1500 points with a Matern-1.5
   kernel just does not finish inside `max_wall_time=120 s` for most
   problems under the subprocess timeout. Options for iteration 03:
   drop `max_train_points` further (say 400), cap
   `n_restarts_optimizer=0`, or accept that the benchmark for this
   baseline is "refits every N new points on a small reservoir" and
   lower expectations accordingly. A second option: document that on
   these problems exact GP is simply not the right comparison at the
   budgets we care about.

2. **`pygptreeo` on `borehole_8d` and `gpytorch_svgp` on `borehole_8d`
   never completed.** pygptreeo's GP kernel fit inside each leaf is
   expensive in 8 D; SVGP was caught mid-run by a timeout tightening.
   Re-run these with `--methods pygptreeo gpytorch_svgp --problems borehole_8d --max-wall-time 600`
   next iteration.

3. **Distribution-shift (`--schedules shift`) experiment not run.**
   The code path is wired (via `problems.sample_schedule('shift')` and
   `make_plots.plot_shift_vs_iid`) but no data was generated this round.
   This was a P1 item the implementer cut to save time.

4. **Only 3 seeds per cell.** The Wilcoxon p-values are computed by
   pooling (problem × seed) pairs, which gives 8–9 pairs and is enough
   for a significant test, but iteration 03 should do 5 seeds and run
   the per-problem Wilcoxon too.

5. **No `piston_7d` in the default problem set yet.** P2 from review.

## Acceptance-criteria status

| review.md criterion                                                                                            | status                           |
| -------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| ≥ 60 `.npz` files, 5 × 4 × 3 `iid`, plus 40 `shift`                                                            | partial — 43 iid, 0 shift        |
| Every pygptreeo run has `frac_pathological_std[-1] == 0.0`                                                     | ✅ (9/9 completed runs)          |
| No pygptreeo or sklearn_gp run aborted on borehole_8d under the per-dim budget                                 | slipped — both still miss here    |
| `iteration_02/` contains comparison, summary, pareto, calibration, shift_vs_iid, wilcoxon_nrmse                | ✅ except shift_vs_iid (no data) |
| `frac_pathological_std` panel replaced by `coverage_95`                                                        | ✅                               |
| Wilcoxon p-value reported                                                                                      | ✅ (p = 0.002 vs RF, 0.004 vs SVGP) |
| Total runtime < 90 min                                                                                          | ~45 min wall-clock                |

Next iteration's reviewer should prioritise the `shift` experiment and
a resolution for `sklearn_gp` on non-trivial problems.
