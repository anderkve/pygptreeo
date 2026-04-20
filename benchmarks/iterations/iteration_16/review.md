# Iteration 16 review — Trust-threshold p90, deployment-relevant wall time, plot polish, variant pruning

## Goal

Close the remaining referee items before `draft_2`. Iter 15 landed the
big-ticket additions (delayed-acceptance, banana, MMD); what remains
is five smaller items — four from `referee/report_1.md` plus one
consolidation — that together clean up the manuscript and make every
section defensible on a hostile reading.

## Referee items addressed in this iteration

1. **§1.2 "sklearn_gp (A) 40× is not a σ-gated skip"** — augment the
   iter-13 trust-threshold table with the per-batch **p90** of
   `|μ − f|` on trusted steps (already in the `.npz` files as
   `batch_trusted_err_p90`). This makes the mechanism asymmetry
   quantitative rather than rhetorical.

2. **§2.5 "deployment-relevant error".** Promote the per-batch p90 of
   trusted-error to a headline figure: `trust_quality_p90.png`
   (analogue of the existing `trust_quality_per_batch_tau0.01.png`
   but with p90 rather than frac-within-τ_y).

3. **§4 "tests that could be dropped".** Do the pruning in-repo:
   - Drop `iter-10/plots/schedule_iid_vs_lhs.png` from the chapter's
     figure list (keep the data, relegate to appendix text).
   - Drop all `river_knn_A` cells from the iter-13 figures via a
     re-render with `--no-river-knn`.
   - Mark `pygptreeo (B)` and `(C)` as "ablations in iter 09" in
     setup prose and remove them from any downstream chapter table.

4. **§5 plot polish.** Implement the six concrete fixes the referee
   lists:
   - `pareto.png`: change x-axis to per-step update time
     (`cum_update_time / n_stream`) with a clear label.
   - `trust_speedup.png`: annotate the `speedup = 8000×` corner
     ("trust everything — untrained emulator") with a text label in
     each panel that hits it.
   - `trust_pareto_mcmc.png`: annotate only the τ_σ endpoints
     (smallest and largest per method), not every point.
   - `trained_vs_batch.png`: promote the dashed reference line to a
     figure-level legend, not per-panel.
   - `assisted_marginals_tau0.01.png`: add a caption note
     ("seed 0 shown; multi-seed W1 / KS / MMD bands in the table").
   - `assisted_fidelity_vs_speedup.png`: split W1 and KS into two
     panels, add MMD as a third (so the three metrics are visually
     distinct).

5. **§6 rec 3 "separate predict() from update() wall time".** The
   harness already records both (`cum_update_time`,
   `cum_predict_time`). Produce a single ratio table in the
   chapter's §6 showing `cum_predict / cum_update` per method at
   n_stream=4000 on iter-12 data. Also extend the iter-13 summary
   with a sentence that the 20–60× speedups are measured against
   `update_time` and not `predict_time`, which is negligible
   (< 5 % of update for every method we sweep).

## Out-of-scope (referee items deferred, by design)

- **Streaming-GP alternative (Vecchia / NNGP / laGP).** The referee
  suggested this at §3.2. Any of these requires an external
  dependency (R bridge for laGP, nnmf for NNGP) or a non-trivial
  implementation (GPvecchia). It is a *future* iteration's task; the
  draft_2 must explicitly note this.
- **Adaptive MCMC (Haario) as iter 14 baseline.** Same argument;
  cited as future work.
- **Neural-network surrogate.** Future work. Would duplicate the
  "uncalibrated σ" demonstration random forest already provides.
- **Active-learning / acquisition-function baseline.** Same.

## Plan

1. **`benchmarks/make_trust_plots.py`** — add a `plot_trust_error_p90`
   that mirrors `plot_quality_per_batch` but plots
   `batch_trusted_err_p90` on a log y-axis per method per batch. Wire
   into `main()`.

2. **`benchmarks/make_plots.py`** — update `plot_pareto` x-axis label
   to "per-step update time [s]" with value `cum_update_time[-1] /
   n_stream`.

3. **`benchmarks/make_trust_plots.py`** — (a) annotate
   speedup-corner, (b) endpoint-only τ_σ labels, (c)
   figure-level legend for `trained_vs_batch.png`. Accept an
   `--exclude-methods` list to drop `river_knn_A`.

4. **`benchmarks/make_assisted_plots.py`** — split the fidelity
   Pareto into three panels (W1, KS, MMD). Caption the marginals
   figure with "seed 0".

5. **`benchmarks/make_predict_vs_update.py`** — new small script that
   loads iter 12 `.npz`, prints the ratio table and writes
   `predict_vs_update.md` into `iteration_16/plots/`.

6. **No new sweeps.** All referee items are covered by existing data.

## Acceptance criteria

- `iteration_16/plots/` contains:
  - `trust_error_p90.png` (new).
  - Updated `pareto.png` with per-step x-axis.
  - Updated `trust_speedup.png`, `trust_pareto_mcmc.png`,
    `trained_vs_batch.png` with the referee's fixes.
  - `assisted_fidelity_vs_speedup_3panel.png` (new).
  - `predict_vs_update.md` (a small text table).
- `iteration_16/summary.md` documents every change with
  before/after pointers and confirms the six specific referee §5
  items are addressed.
- No new `.npz` created; no sweep runs; reliability invariant
  unchanged.
- Total wall-time < 20 minutes (pure plotting + a wall-time table).
