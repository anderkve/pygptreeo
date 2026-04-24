# Iteration 17 review — Seed variance, coverage drift, banana-5d, MMD precision, paired-error scatter

## Goal

Close the five referee-2 items that do **not** require a new
dependency: (i) multi-seed error bars on the static-stream and
delayed-acceptance tables, (ii) a coverage-drift reliability
supplement, (iii) a plausible explanation for the banana-5d MMD²
jump, (iv) higher-precision MMD² reporting so the ranking claim is
verifiable, (v) a paired-sample trusted-error scatter that
distinguishes reservoir-cap from genuine σ-gating. The streaming-GP
comparator (the one referee-2 item that needs new code) is pushed
to iter 18 and the realistic-likelihood run is left as future work.

## Plan

1. **Multi-seed iter-12 static-stream extension.** Run seeds 1 and 2
   on the five main methods (`pygptreeo_A`, `pygptreeo_D`,
   `sklearn_gp_A`, `gpytorch_svgp_A`, `random_forest_A`) × two
   problems (`rosenbrock_2d`, `borehole_8d`) × two schedules
   (`de`, `mcmc`) at `n_stream = 4000`, `--de-popsize 300`,
   `--max-wall-time 900`. 40 new `.npz` in
   `iteration_17/data/`. The iter-12 seed-0 data stays in place; the
   summary table aggregates the three seeds.

2. **Delayed-acceptance std already in iter 15.** Recompute the
   std across 3 seeds from the existing iter-15 `.npz` and report
   with explicit "n_seeds = 3" caption.

3. **Coverage-drift supplement.** Post-process every pygptreeo*
   `.npz` in `iteration_{09..17}/data/` and record
   `coverage_1sigma[-1]`. Define the coverage-drift invariant as
   `coverage_1sigma[-1] ∈ [0.60, 0.76]` at the final checkpoint
   (the nominal value is 0.6827; a ±0.08 band is large enough to
   accommodate honest under/over-coverage without flagging noise).
   Emit `iteration_17/plots/coverage_drift.md` listing every
   (method, problem, schedule, seed) cell that is outside the band,
   plus an aggregate "X / Y pygptreeo* runs within band" headline.

4. **Banana-5d diagnostic.** Run pygptreeo_D on `banana_5d` for
   seeds 1 and 2 at `τ_σ ∈ {3e-3, 1e-2}` and report mean ± std. If
   the outlier persists across seeds, write a paragraph in the
   summary explaining it in terms of the 3 nuisance-dimensions'
   coverage drift; if it was a seed-0 unlucky draw, the band
   shrinks and the outlier disappears. Either outcome is
   publishable — the point is to *know* rather than to assert.

5. **MMD² precision.** Recompute the MMD² for every iter-14 and
   iter-15 `.npz` with **four significant figures**, save as
   `mmd_rbf_joint_hires`. Produce `iteration_17/plots/mmd_pairwise.md`
   — a table of signed pairwise differences `mmd²(method_A) −
   mmd²(method_B)` with bootstrap 1σ from the 2000-sample MMD
   estimator, so the ranking claim is directly verifiable.

6. **Paired trusted-error scatter.** For every `(τ_σ, problem, seed)`
   cell where both `pygptreeo (A)` and `sklearn_gp (A)` trusted the
   same proposals (identified via the stream-step index already
   saved), scatter `|μ_pygp − f|` on the x-axis against
   `|μ_sklearn − f|` on the y-axis. Points on the diagonal mean the
   two methods make the same mistake; points below the diagonal
   mean pygptreeo is more accurate, above means sklearn is. A cloud
   clearly below the diagonal would finish the §1.2 referee
   argument; a cloud on the diagonal would force a retraction of
   the "mechanism difference matters" claim. Save as
   `iteration_17/plots/paired_trusted_err.png`.

   **Caveat**: iter 13's trust harness does not currently record
   the stream-step identity of trusted picks, only the per-batch
   statistics. The paired scatter therefore uses a simpler proxy:
   re-run a small `n_stream = 3000` cell on each method with the
   same stream seed and record the per-step `(μ, σ, |μ − f|)`
   triples. One problem × one τ_σ × one seed is enough to show the
   direction.

## Out-of-scope

- Streaming-GP comparator (NNGP / laGP / Vecchia): iter 18.
- Realistic global-fit posterior (BSM / GAMBIT): future work.
- Neural-net surrogate; adaptive MCMC; acquisition-function
  baselines: future work.
- Chapter draft_3: after iter 18 lands.

## Acceptance criteria

- `iteration_17/data/` has the 40 new static-stream `.npz` + 4 new
  banana-5d `.npz` + the paired-scatter cell files.
- `iteration_17/plots/coverage_drift.md` lists the out-of-band
  cells (target: 0 cells out of > 150 `pygptreeo*` runs).
- `iteration_17/plots/mmd_pairwise.md` gives signed differences
  with bootstrap 1σ.
- `iteration_17/plots/paired_trusted_err.png` exists.
- `iteration_17/summary.md` contains the multi-seed NRMSE table
  (mean ± std, n_seeds=3), the multi-seed DA table (mean ± std,
  n_seeds=3), the coverage-drift headline, the banana-5d
  explanation, and a one-paragraph reading of the paired-scatter.
- Reliability invariant extended: 100 % of pygptreeo* runs with
  `frac_pathological_std[-1] == 0` **and**
  `coverage_1sigma[-1] ∈ [0.60, 0.76]`.
- Sweep wall-time < 90 min.
