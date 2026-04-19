# Iteration 12 review — Fast-adapting pygptreeo + wider DE coverage

## Goal

Three structural improvements responding to user feedback after iter 11:

1. **Adaptation speed.** A fast-changing input stream should reward
   smaller leaves and more frequent retraining. Add a new variant
   `pygptreeo_D` with `Nbar=100`, `retrain_every=100`, otherwise
   identical to `pygptreeo_A`. Run it under DE and MCMC alongside `_A`
   so we can quote the fast-adaptation premium on a streaming
   exploration workload.

2. **Wider DE coverage.** In a global-fit / GAMBIT setting we need to
   *map out* the profile likelihood across the 2σ confidence region,
   not just locate the best-fit point. Increase the DE `popsize` from
   100 to **300** so the per-generation trial set is roughly 3× wider
   and the early-generation visits sample more of the cube before DE
   begins tightening.

3. **Plot housekeeping.**
   - Pareto plot legend currently has duplicate labels and missing
     markers; deduplicate and add an explicit legend panel.
   - Per-iteration plot output: each iteration writes its plots to
     `iterations/iteration_X/plots/`, not the global `plots/`. The
     global `plots/` stays as the "latest snapshot".

## Plan

1. **`benchmarks/run_all.py`.**
   - Add `_make_pygptreeo_D(d)` with `Nbar=100, retrain_step=100,
     theta=1e-4, sigma_rel=1e-3, kernel_spec="matern+rq"`.
   - Register `"pygptreeo_D"` in the `METHODS` dict.
   - Plumb a `--de-popsize` CLI flag (default 100 to preserve iter 11
     numbers) that the schedule uses.

2. **`benchmarks/problems.py`.**
   - `_sample_differential_evolution` accepts a `popsize` kwarg
     (default 100). Threaded through `sample_schedule(..., de_popsize=...)`.
   - Run `run_online_benchmark` reads the CLI flag and forwards it to
     `sample_schedule`.

3. **`benchmarks/make_plots.py`.**
   - `plot_pareto`: deduplicate the `(label, handle)` list with a
     dict-by-label and append a 5th panel that is just the legend.
   - `main()` writes plots to `<iter-dir>/plots/` when `--iter-dir`
     is given, in addition to the global `plots/`.

4. **`make_plots.py` plot updates.**
   - Update the schedule-comparison drawer so `pygptreeo_A` and
     `pygptreeo_D` appear side-by-side as separate bars.
   - Add `pygptreeo_D` to `METHOD_ORDER`, `METHOD_LABEL`,
     `METHOD_COLOR`, `METHOD_LS`.

5. **Longer adaptive-sampling sweep.**
   - **Two problems only**: `rosenbrock_2d` and `borehole_8d` — these
     bracket the curvature × dimensionality space, and from iter 11
     they showed the most dramatic spread of behaviours.
   - **n_stream = 4000** (was 2000) on these problems.
   - **Six methods**: `pygptreeo_A`, `pygptreeo_D`, `sklearn_gp_A`,
     `gpytorch_svgp_A`, `random_forest_A`, `river_knn_A`.
   - **Two schedules**: `de` (popsize 300), `mcmc`.
   - **One seed** (seed 0) — n=4000 is enough to read trends from a
     single trace.
   - 6 × 2 × 2 = **24 runs**, hard wall-time 600 s each.

## Out-of-scope

- No new test functions.
- No change to MCMC proposal sigma.
- No iid / lhs re-runs.
- No change to method hyperparameters other than the `_D` variant.
- Trust-threshold infrastructure stays in **iter 13**.
- Posterior-comparison MCMC plots stay in **iter 14**.

## Acceptance criteria

- `_make_pygptreeo_D` registered and runs successfully on both problems.
- `_sample_differential_evolution(..., popsize=300)` works.
- `iteration_12/plots/` contains all the standard plots (comparison,
  summary, schedule_iid_vs_de.png, schedule_iid_vs_mcmc.png,
  schedule_de_vs_mcmc.png, pareto.png with deduped legend, etc).
- `iteration_12/data/<method>__<problem>__{de,mcmc}__seed0.npz` exists
  for the 24-run grid.
- `summary.md` reports a per-(method, problem) table of final NRMSE
  for each of `iid` (from iter 11 baseline), `de(popsize=300)`, `mcmc`,
  highlighting the `_A` vs `_D` improvement under streaming.
- Reliability still 100 % (every `pygptreeo_*` run has
  `frac_pathological_std[-1] == 0`).

## Out-of-scope but planned (queued)

- Iter 13: trust-threshold harness — track when the emulator's σ falls
  below a per-batch threshold; when it does, "use the emulator instead
  of the true function" and skip the next true evaluation. Plots:
  fraction-below-threshold by 1000-point batch, total true-function
  speedup vs threshold, n_trained vs n_seen.
- Iter 14: emulator-assisted MCMC. Two parallel chains: one always
  uses true f, one uses the emulator when σ < threshold. Compare
  posterior marginals and joint scatter; also plot KS distance
  between marginals as a fidelity vs speedup trade-off.
