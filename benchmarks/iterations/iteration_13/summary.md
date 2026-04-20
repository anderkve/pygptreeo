# Iteration 13 — implementer summary

*Trust-threshold deployment sweep — 100 runs at n_stream = 8000.*

> **`Reliability: 40 / 40 pygptreeo* runs have frac_pathological_std[-1] == 0 (100.0 %)`**

## What landed

1. **Trust-threshold harness and driver** already committed in the
   scaffolding pass (`benchmarks/trust_harness.py`,
   `benchmarks/run_trust_all.py`). This iteration exercises them at
   full scale: 5 methods × 2 problems × 2 schedules × 5 τ_σ = 100
   runs, each n_stream = 8000 (2× the iter 12 length).

2. **Plot-module revisions per the reviewer's plan**
   (`benchmarks/make_trust_plots.py`):
   - `trust_speedup.png` is now a **two-row** panel (mcmc / iid rows),
     shared legend.
   - `trust_pareto_{mcmc,iid}.png` — per-schedule splits so the
     dedup-legend panel stays uncluttered.
   - New `trained_vs_batch.png` — cumulative true-function call count
     per stream step, making the MCMC-revisits-plateau visible.
   - `--tau-y-picks` CLI produces two batch-quality panels (τ_y = 1e-3
     and 1e-2) instead of a single hard-coded one.

3. **No changes to pygptreeo or problem definitions.**

## Headline speedups (NRMSE stays within 2 × of iter-12 baseline)

| method | problem | schedule | τ_σ | NRMSE | speedup |
|---|---|---|---|---|---|
| pygptreeo (A) | rosenbrock_2d | iid  | 3e-2 | 7.2 × 10⁻⁵ | **36.5×** |
| pygptreeo (A) | rosenbrock_2d | mcmc | 3e-3 | 2.7 × 10⁻³ | **22.7×** |
| pygptreeo (D) | rosenbrock_2d | iid  | 3e-2 | 1.8 × 10⁻³ | **59.3×** |
| pygptreeo (D) | rosenbrock_2d | mcmc | 3e-3 | 3.5 × 10⁻³ | **32.0×** |
| pygptreeo (A) | borehole_8d   | iid  | 3e-2 | 1.8 × 10⁻³ | **33.9×** |
| pygptreeo (A) | borehole_8d   | mcmc | 3e-3 | 3.3 × 10⁻² | **20.8×** |
| pygptreeo (D) | borehole_8d   | iid  | 3e-2 | 3.7 × 10⁻³ | **58.8×** |
| pygptreeo (D) | borehole_8d   | mcmc | 3e-3 | 2.5 × 10⁻² | **9.7×**  |
| sklearn GP (A) | rosenbrock_2d | iid | 1e-2 | 1.1 × 10⁻³ | 40.0× |
| sklearn GP (A) | borehole_8d   | iid | 1e-2 | 1.9 × 10⁻³ | 40.0× |
| SVGP (A)      | borehole_8d   | iid | 1e-1 | 8.0 × 10⁻³ | 13.3× |
| RandomForest (A)| borehole_8d | iid | 1e-1 | 4.2 × 10⁻² | 11.9× |

## Reading

- **Pygptreeo dominates the deployment-Pareto.** On both problems and
  both schedules it reaches 20–60× speedup while keeping NRMSE within
  a factor of 2 of the no-trust iter-12 baseline. The D variant
  (Nbar=100) gets the highest speedups — 58–59× on iid — because its
  smaller leaves report a tighter σ sooner in the stream, so more
  points get trusted.
- **sklearn GP's 40× speedup is an artefact of its rolling 400-point
  window**, not of the trust-gate. The method has a fixed
  `n_trained` ceiling regardless of τ_σ. Useful to document, but this
  speedup buys you the *worst* NRMSE among GP methods on rosenbrock
  (1.1 × 10⁻³ vs pygptreeo's 7.2 × 10⁻⁵).
- **SVGP's σ rarely falls below τ_σ**; its speedup saturates at ~10–20×
  only at high thresholds where NRMSE also degrades. The inducing-
  point approximation keeps predictive variance conservative.
- **Random forest has uncalibrated σ**: the per-batch mean
  `|μ − f|` on trusted points regularly exceeds 1.0 (versus all
  GP-family methods at ≤ 0.1). Its speedup only rises when the
  threshold is so loose that *any* prediction is trusted — at which
  point accuracy collapses. This is exactly why the reviewer dropped
  river_knn from the sweep.
- **The "trust everything" corner case (τ_σ = 0.1) is a dead giveaway**:
  pygptreeo's speedup jumps to 8000× but NRMSE jumps to the constant-
  prediction floor (2.3 × 10⁻¹), i.e. the model trained on 1 point.
  The panel plots `trust_speedup.png` show this divergence clearly.
- **MCMC revisits work**. In `trained_vs_batch.png` the pygptreeo
  curves plateau after ~500–1000 steps even though the stream runs to
  8000 — the chain keeps proposing points near already-sampled modes,
  where the emulator's σ is already below τ_σ. This is the central
  mechanistic argument for continual emulation in a global-fit
  deployment.

## Artefacts

- `iteration_13/plots/` — `trust_speedup.png`,
  `trust_pareto_{mcmc,iid}.png`, `trust_quality_per_batch_{0.001,
  0.01}.png`, `trained_vs_batch.png`.
- `iteration_13/data/` — 100 `.npz` files.

## Acceptance criteria check

- 100 `.npz` present. ✔
- All five required plots in `iteration_13/plots/`. ✔
- Per-(method, τ_σ) tables by (problem, schedule) reported above. ✔
- Reliability 40 / 40. ✔
- Sweep wall-time ≈ 65 min (well under 2 h budget). ✔

Total iter-13 runtime: ~65 min sweep + ~10 min plots/summary.
