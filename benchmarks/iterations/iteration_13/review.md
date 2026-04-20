# Iteration 13 review — Trust-threshold deployment sweep

## Goal

The continual emulator earns its keep when, mid-run, it can replace the
expensive truth at inputs where its own σ says "I know this region".
Iter 11/12 measured emulator quality when it sees *every* point; that
overstates the cost in deployment, where each trusted prediction is a
full evaluation skipped and a training point not added. This iteration
quantifies that headline number — total speedup at a chosen trust
threshold — and the quality of the predictions that get trusted, on
streams long enough for an MCMC chain to revisit the same region many
times.

## Plan

1. **Sweep grid (driver: `benchmarks/run_trust_all.py`).**
   - Methods: `pygptreeo_A`, `pygptreeo_D`, `sklearn_gp_A`,
     `gpytorch_svgp_A`, `random_forest_A` (5 methods; **drop
     `river_knn_A`** — its per-prediction σ collapses to 0 once it has a
     few neighbours, so trust is meaningless and it would always look
     "fast and wrong"; we already have its iter-11/12 numbers if a
     reviewer asks).
   - Problems: `rosenbrock_2d`, `borehole_8d` (2 problems — keep the
     2-D and 8-D contrast, drop friedman1_5d and smooth_sines_2d to
     trade test-function breadth for stream length, per the user's
     "longer runs, fewer functions" steer).
   - Schedules: `mcmc` and `iid`. Including `iid` as a control
     doubles the run count but is the only way to show that the
     trust-threshold story is qualitatively distinct on a
     return-visiting chain vs a uniform stream — and it falsifies the
     null hypothesis "speedups are just a function of the threshold,
     not the schedule".
   - τ_σ grid: `{1e-3, 3e-3, 1e-2, 3e-2, 1e-1}` (5 points, **wider**
     than the prototype's 4-point grid; we add `3e-3` and `3e-1` is
     dropped in favour of `1e-1` so the speedup curve shows both the
     "trust nothing" floor and the "trust everything" ceiling on a
     log axis).
   - `n_stream = 8000`, `batch_size = 1000`, `checkpoint_every = 500`,
     1 seed.
   - Total: 5 × 2 × 2 × 5 × 1 = **100 runs**. Budget check:
     `pygptreeo_A` on `borehole_8d` at n=8000 with the iter-12
     timings (~2× the n=4000 wall) is ≈ 25 min worst case, but
     trusted steps skip the (expensive) update so wall time *falls*
     with τ_σ. Cap with `--max-wall-time 1500` to bound the tail.
     Realistic total ≲ 120 min.

2. **Driver invocation.** `bash` block in the `regenerate_paper.sh`
   (or a new `regenerate_iter13.sh`) calling `run_trust_all.py` with
   the grid above, `--out-dir benchmarks/iterations/iteration_13/data`.
   Use `--no-subprocess` only if subprocess startup is the bottleneck;
   default to subprocess isolation for safety.

3. **Plots (`benchmarks/make_trust_plots.py`).** The three existing
   functions cover what we need; add one cut and one panel:
   - `plot_speedup_vs_threshold`: render **two rows** (one per
     schedule), one column per problem. Single shared legend.
   - `plot_quality_per_batch`: render the τ_y inner index used in
     the title (currently hard-coded to `tau_y_col=1` → `1e-3`); make
     `--tau-y-pick` a CLI arg and produce panels at `1e-3` and `1e-2`.
   - `plot_trust_pareto`: keep as-is, but render once per schedule
     (suffix `_mcmc.png` and `_iid.png`), so the dedup-legend panel
     stays uncluttered.
   - Add `plot_trained_vs_batch`: cumulative `n_trained` curve as a
     function of stream step, one panel per (problem, schedule), one
     line per (method, τ_σ). Makes the "MCMC revisits → trained
     count plateaus" argument visible at a glance — the central
     mechanistic point of the iteration.

4. **Output destinations.** All `.npz` into
   `iterations/iteration_13/data/`; all `.png` into both
   `iterations/iteration_13/plots/` and the global `plots/` (mirror
   the iter-12 convention).

5. **`summary.md` requirements.**
   - Per-(method, τ_σ) table for each (problem, schedule) reporting
     **(final NRMSE, speedup = n_stream / n_trained, mean
     trusted-prediction |μ − f|, frac_within_tau_y at τ_y = 1e-3)**.
     Two problems × two schedules = 4 tables, 5 methods × 5 τ_σ rows
     each.
   - One paragraph per schedule reading the headline result: which
     method gets the best speedup at any τ_σ that holds NRMSE within
     2× of its iter-12 baseline, and how much cheaper that is than
     the no-trust deployment.
   - Reliability one-liner over all `pygptreeo_*` runs (target 100 %).
   - Wall-time table.

## Out-of-scope

- Assisted-MCMC (closed-loop chain on the emulator): defer to iter 14.
  The harness exists in `mcmc_assisted.py` but we are not running it.
- Reference-vs-emulator posterior comparison: iter 14.
- New methods or new problems beyond the two named.
- Multi-seed reruns; one seed per cell is enough at this run count.
- Re-running iter 11/12 sweeps; iter 13 is purely additive.

## Acceptance criteria

- 100 `.npz` files in `benchmarks/iterations/iteration_13/data/`,
  named `<method>__<problem>__<schedule>__tau<tau>__seed0.npz`.
- `iterations/iteration_13/plots/` contains `trust_speedup.png` (now
  2 rows × 2 cols), `trust_quality_per_batch_tau1e-3.png`,
  `trust_quality_per_batch_tau1e-2.png`, `trust_pareto_mcmc.png`,
  `trust_pareto_iid.png`, `trained_vs_batch.png`.
- `iterations/iteration_13/summary.md` includes the four
  per-(method, τ_σ) tables, the headline paragraphs per schedule, the
  reliability line, and a wall-time table.
- Reliability: 100 % of `pygptreeo_*` runs end with
  `frac_pathological_std[-1] == 0`.
- Sweep wall-time ≲ 2 hours.
