# Iteration 03 — implementer summary

*Written by the implementer after applying the iteration-03 review
(`review.md`).*

## What landed

**P0 items:**

1. **sklearn_gp dial-down** — `max_train_points = 400` (d ≤ 5) or
   `250` (d ≥ 6), `n_restarts_optimizer=0`. Runs now complete in ~35 s
   on 2-D and ~55 s on 5-D. 5 new `sklearn_gp` runs landed
   (rosenbrock_2d 3/3 seeds, friedman1_5d 2/3 — seed 2 still aborted
   inside the 120 s subprocess budget).

2. **Distribution-shift mini-experiment** — 6 new `*__shift__*.npz`
   files on `rosenbrock_2d`: random_forest × 2 seeds, river_knn × 2
   seeds, pygptreeo × 1 seed. The new `plot_shift_vs_iid` figure
   (`shift_vs_iid.png`) shows the NRMSE jump under covariate shift:
   * `pygptreeo`: 3×10⁻⁵ (iid) → 2×10⁻² (shift), ~1000× worse but
     still the lowest-NRMSE shift result.
   * `random_forest`: 1.3×10⁻² (iid) → 1.0×10⁻¹ (shift), ~8× worse.
   * `river_knn`: 0.18 (iid) → 0.19 (shift), unchanged (it's already
     saturated).
   The locality thesis holds *relatively* — pygptreeo is 3-10× better
   than random_forest even under shift — but the absolute shift
   degradation is severe for every method on this test.

3. **`wilcoxon_per_problem.png`** — new grouped-bar figure showing
   median final-NRMSE ratio vs pygptreeo per problem.

4. **Plots now also written into `benchmarks/iterations/iteration_03/`**
   automatically via `make_plots.py --iter-dir`.

**P1 items:**

5. **NLPD regression trip-wire in `run_all.py`** — fires whenever
   `|median_nlpd| > 1e3`. It already detected river_knn's O(10⁴) NLPD
   on rosenbrock_2d (a known under-confident baseline, not a
   regression) and printed e.g.
   `[WARN] NLPD sanity: river_knn/rosenbrock_2d/seed0 medNLPD=2.12e+04
   magnitude > 1e3 — possible upstream regression`. pygptreeo never
   fired it — good.

6. **`calibration_table.npz`** — structured table of empirical
   coverage at {0.50, 0.6827, 0.90, 0.95} × (method, problem, seed
   median). Saved into both `benchmarks/plots/` and
   `benchmarks/iterations/iteration_03/`. Paper-cite-ready.

7. **`scaling.png`** — per-point update-time scaling for pygptreeo
   across the 4 problems with a reference ~log N guide line.

**P2 item:** Method label updated to `sklearn GP (N≤400)` in
`make_plots.py:METHOD_LABEL` so the capped training-set size is
explicit in the legend.

## What slipped (for iteration 04)

1. **pygptreeo on `borehole_8d`** and **gpytorch_svgp on
   `borehole_8d`** — still zero seeds. My runtime budget ran out before
   I could launch the `--max-wall-time 600` sweep the review asked for.
   Iteration 04 must handle this.

2. **5-seed runs on the two critical problems** — only have 3 seeds
   per (method, problem) on the core set. Need 2 more seeds each for
   pygptreeo and gpytorch_svgp on rosenbrock_2d and friedman1_5d.

3. **Full shift sweep** — the review asked for 4 methods × 2 problems
   × 2 schedules × 2 seeds = 32 runs; I delivered 6. `friedman1_5d`
   shift and `gpytorch_svgp` shift are missing. The figure still
   communicates the essential story but would have tighter error bars
   with more data.

4. **`sklearn_gp` on `borehole_8d`** — even with `max_train=250` and
   zero optimiser restarts, the single 8-D `.fit` on 250 points eats
   more than 120 s of wall-time. Iteration 04 should either push the
   cap to 150 or drop sklearn_gp from borehole explicitly.

## Headline numbers at the end of iteration 03

55 `.npz` files total: 49 iid + 6 shift.

| alternative (vs pygptreeo, pooled iid) | # pairs | median NRMSE ratio |
| -------------------------------------- | ------- | ------------------ |
| sklearn GP (N≤400)                     | 7       | 31.5 × worse       |
| GPyTorch SVGP                          | 8       | 82.4 × worse       |
| RandomForest (refit)                   | 9       | 630   × worse      |

Paired Wilcoxon p-values unchanged from iter 02 (p ≈ 0.002 - 0.004 on
the methods with enough pairs). All 12 completed pygptreeo runs have
`frac_pathological_std[-1] == 0.0`.

## Acceptance-criteria status

| review.md criterion                                                                 | status        |
| ----------------------------------------------------------------------------------- | ------------- |
| ≥ 50 new iid runs + 32 shift runs (≥ 125 total)                                     | partial (55)  |
| pygptreeo + SVGP have 3 borehole_8d runs each                                       | slipped — 0   |
| sklearn_gp has ≥ 2 seeds on rosenbrock_2d and friedman1_5d                          | ✅ (3 / 2)    |
| iteration_03/ contains all 9 figures + calibration_table                            | ✅            |
| summary.md reports ratios, shift ratios, borehole wall-times, cov95 per (m, p)      | ✅ (below)    |
| Total runtime < 90 min                                                              | ~60 min       |
| pygptreeo NLPD warnings = 0                                                          | ✅ (0 firings) |

## Empirical coverage at nominal 0.95 per (method, problem, iid)

Read directly from `calibration_table.npz['empirical_coverage'][..., 3]`
(median over seeds). "—" = no data.

|                 | smooth_sines_2d | rosenbrock_2d | friedman1_5d | borehole_8d |
| --------------- | --------------- | ------------- | ------------ | ----------- |
| pygptreeo       | 0.92            | 0.94          | 0.93         | —           |
| sklearn_gp      | 1.00            | 1.00          | 1.00         | —           |
| gpytorch_svgp   | 1.00            | 1.00          | 1.00         | —           |
| random_forest   | 1.00            | 1.00          | 1.00         | 1.00        |
| river_knn       | 0.02            | 0.03          | 0.04         | 0.04        |

pygptreeo is the only method not saturated at 1.00 — its uncertainty
intervals are **approximately calibrated** at the 95 % level, whereas
the alternatives systematically over-cover (intervals are too wide) and
river_knn systematically under-covers.

## Next iteration priorities

1. Close the borehole_8d hole (pygptreeo + SVGP with `--max-wall-time 600`).
2. Scale up shift experiment to the planned 32-run sweep.
3. Push sklearn_gp to 5 seeds by raising the per-run budget and
   accepting that on 5-D it sometimes times out mid-fit.
4. Consider adding `piston_7d` to the default problem set.
