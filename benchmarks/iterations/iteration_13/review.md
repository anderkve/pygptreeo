# Iteration 13 review — Trust-threshold deployment harness

## Goal

The whole reason for *continual* regression is to **avoid running the
expensive simulator** for inputs the emulator already knows. This
iteration adds the missing piece of the benchmark: a harness that
**deploys** the emulator with a trust threshold, *skipping* the true
function call whenever predictive uncertainty is below that threshold.
The result is a directly-quantifiable speedup along with batch-resolved
quality control.

## Setup

Each `(method, problem, schedule, threshold)` run:

- Streams `n_stream` candidate inputs `x_t` (in our case mostly MCMC,
  since that is the canonical revisit-heavy workload — DE explores too
  uniformly to benefit from an emulator).
- At each step, query the emulator → `(μ_t, σ_t)`.
- **Decide**: if `σ_t ≤ τ`, *trust* the emulator: predict `μ_t` and
  do NOT call the true function (no training point added). Otherwise:
  call the true function `f(x_t)`, predict `f(x_t)` (the truth is
  reported), and update the emulator on `(x_t, f(x_t))`.
- For diagnostics only (never for training), evaluate `f(x_t)` at the
  *trusted* steps too, so we can record the *would-have-been* error.
  In production you would not pay this cost.

## Metrics tracked per-step (cumulated and binned per 1000-step batch)

- `n_trained` — true-function calls actually issued
- `n_trusted` — true-function calls *avoided*
- `speedup = n_total / n_trained`
- `trusted_err` — `|μ_t − f(x_t)|` on trusted steps
- `frac_trusted_within_tau_y` — fraction of trusted steps where
  `|μ_t − f(x_t)| ≤ τ_y` for a configurable absolute τ_y on the
  target scale. We report it for several τ_y simultaneously.

In addition we keep the standard held-out test set `(X_test, y_test)`
evaluated every `checkpoint_every` steps to track NRMSE / coverage etc.
That gives the reader an independent "is the emulator any good"
yardstick beside the speedup story.

## Plan

1. **`benchmarks/trust_harness.py`** — new module exposing
   `run_trust_threshold_benchmark(...)`. Re-uses the existing metric
   helpers in `benchmarks/harness.py` for held-out evaluation.

2. **`benchmarks/run_trust_all.py`** — driver. Per-run output:
   `data_trust/<method>__<problem>__<schedule>__tau{τ}__seed0.npz`.
   Schema:
       - n_total[t], n_trained[t], n_trusted[t]
       - per-batch: trusted_err_med, trusted_err_p90, frac_within_τ_y_grid
       - checkpoints: nrmse, coverage, etc (same as harness.py)

3. **`benchmarks/make_trust_plots.py`** — three plots per problem:
   - `trust_speedup.png` — speedup curve as a function of τ_σ (one
     curve per method).
   - `trust_quality_per_batch.png` — for the chosen "production"
     threshold, the per-1000-step bar of the fraction of trusted
     predictions that landed within τ_y, by method.
   - `trust_pareto.png` — accuracy (NRMSE on held-out test) vs
     speedup (n_trained / n_total) — the deployment Pareto.

4. **Sweep**: 2 problems (rosenbrock_2d, borehole_8d) × 6 methods
   (`pygptreeo_A`, `pygptreeo_D`, `sklearn_gp_A`, `gpytorch_svgp_A`,
   `random_forest_A`, `river_knn_A`) × MCMC schedule × 4 thresholds
   `τ_σ ∈ {1e-3, 5e-3, 1e-2, 5e-2}` (σ relative to the emulator's
   observed running y-range). `n_stream = 8000`. 48 runs.

## Out-of-scope

- Iter 14 separately handles **emulator-assisted MCMC** where the
  emulator's σ-gated decisions actually affect the chain trajectory.
  Iter 13's chain is fixed up front (the inputs are pre-sampled).
- No new methods, no new problems.

## Acceptance criteria

- `run_trust_threshold_benchmark` exists and produces valid `.npz`.
- 48 `.npz` files in `benchmarks/iterations/iteration_13/data/`.
- Three plots in `iteration_13/plots/`.
- `summary.md` reports per-method speedup at the median trust
  threshold and the corresponding NRMSE on held-out test data.
- Reliability: every `pygptreeo_*` run has
  `frac_pathological_std[-1] == 0`.
