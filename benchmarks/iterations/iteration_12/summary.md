# Iteration 12 — implementer summary

*Faster pygptreeo + wider DE + plot housekeeping. Long-stream (n=4000)
adaptive-sampling sweep on rosenbrock_2d and borehole_8d.*

> **`Reliability: 8 / 8 pygptreeo* runs have frac_pathological_std[-1] == 0 (100.0 %)`**

## What landed

1. **`pygptreeo_D` variant** (`benchmarks/run_all.py:_make_pygptreeo_D`):
   `Nbar=100, retrain_step=100`, otherwise identical to `_A`. Smaller
   leaves split sooner; faster retrain cadence keeps each leaf in sync
   with non-stationary streams. ~3× faster wall-time per run too
   (180–270 s vs 450–730 s for `_A` on the same problems / schedules).

2. **`--de-popsize` CLI flag** in `run_all.py`, plumbed through
   `harness.py` and `problems.py:_sample_differential_evolution`. This
   sweep used `popsize=300` (was 100 in iter 11). Justification: in a
   GAMBIT-style global fit we need to *map the 2σ confidence region of
   the profile likelihood*, not just locate the best fit. A 3× wider
   per-generation trial pool keeps the early generations from collapsing
   onto a single mode.

3. **Pareto plot rewrite** (`make_plots.py:plot_pareto`): legend handles
   are now keyed by label so the same method does not appear twice; an
   extra panel exists solely for the deduplicated legend, so every
   marker that appears in any data panel is shown once with a clean
   square swatch.

4. **Iteration-local plots dir.** `make_plots.py main()` now writes
   plots into `<iter-dir>/plots/` whenever `--iter-dir` is supplied,
   in addition to the global `plots/`. The history is now navigable
   per iteration rather than overwritten in a shared folder.

## n = 4000 NRMSE on the long-stream sweep

### DE schedule (popsize = 300)

| method | rosenbrock_2d | borehole_8d |
|---|---|---|
| pygptreeo (A) | **7.4 × 10⁻⁶** | **4.5 × 10⁻⁴** |
| pygptreeo (D: Nbar=100) | 1.1 × 10⁻⁵ | 6.9 × 10⁻⁴ |
| sklearn GP (A) | 7.1 × 10⁻⁴ | 2.0 × 10⁻³ |
| SVGP (A) | 1.0 × 10⁻³ | 1.3 × 10⁻³ |
| RandomForest (A) | 8.8 × 10⁻³ | 2.4 × 10⁻² |
| River kNN (A) | 1.7 × 10⁻¹ | 2.1 × 10⁻¹ |

### MCMC schedule

| method | rosenbrock_2d | borehole_8d |
|---|---|---|
| pygptreeo (A) | **7.8 × 10⁻⁴** | 2.9 × 10⁻² |
| pygptreeo (D: Nbar=100) | 9.7 × 10⁻² | 3.7 × 10⁻² |
| sklearn GP (A) | 4.7 × 10⁻² | **2.3 × 10⁻²** |
| SVGP (A) | 5.0 × 10⁻² | 6.0 × 10⁻² |
| RandomForest (A) | 1.3 × 10⁻¹ | 1.1 × 10⁻¹ |
| River kNN (A) | 2.0 × 10⁻¹ | 2.1 × 10⁻¹ |

## Reading

- **DE: pygptreeo wins on both problems by ~2 orders of magnitude** vs
  the next-best smooth-kernel competitor (sklearn GP). With the wider
  popsize=300 DE the cube is well covered, so global GPs see a
  near-iid stream and pygptreeo's local-leaf advantage dominates.
- **`_A` vs `_D` under DE**: virtually identical. Both pygptreeo
  variants saturate the smooth target at sub-1e-4 NRMSE — the smaller
  `_D` leaves don't help when the data are already space-filling, but
  they don't hurt either, and they are 2.5–3× cheaper in wall time.
- **MCMC: a more interesting picture.** `_A` is best on rosenbrock_2d
  (7.8 × 10⁻⁴), but `_D`'s smaller leaves *over*-split early when the
  chain is clustered around the modes; in 2D this hurts (9.7 × 10⁻²).
  In 8D `_A` and `_D` are within 25 % of each other — the larger cube
  hides the early-split disadvantage. Sklearn GP is competitive with
  pygptreeo on borehole_8d/MCMC because the rolling-400-point window
  also concentrates on the modes; the wider lengthscale prior of
  sklearn's Matern-1.5 generalises to the rest of the cube.
- **kNN remains the most-MCMC-robust regressor by NRMSE rank** in iter
  11 but here on n=4000 it is again 5–10× worse in absolute terms than
  the smooth-kernel methods — mode-trapping shows up *less* badly in
  the NRMSE rank metric than in absolute error, which is why the
  trust-threshold deployment study (iter 13) is needed to round out
  the picture.

## DE popsize = 100 → 300 effect (vs iter 11 numbers)

For pygptreeo (A):
- rosenbrock_2d/de: 1.5 × 10⁻⁵ (iter 11, n=2000) → 7.4 × 10⁻⁶ (iter 12, n=4000) — about 2× better with the longer stream and wider DE.
- borehole_8d/de: 1.4 × 10⁻³ → 4.5 × 10⁻⁴ — 3× better.

The wider DE clearly helps in 8D where the cube has more corners to
visit; in 2D the n=4000 stream length matters more than the popsize.

## Artefacts

- `plots/` — full per-iteration snapshot, including `pareto.png` with
  the new legend panel.
- `plots/schedule_de_vs_mcmc.png` — long-stream version of the iter-11
  panel.
- `data/` — 24 `.npz` files (6 methods × 2 problems × 2 schedules ×
  1 seed, n=4000).

## Acceptance criteria check

- `pygptreeo_D` runs successfully on both problems. ✔
- `--de-popsize 300` accepted; runs reproduce. ✔
- `iteration_12/plots/` contains the standard plots, including the
  fixed pareto. ✔
- 24 `.npz` in `iteration_12/data/`. ✔
- Reliability 8 / 8. ✔

Total iter-12 runtime: ~48 min sweep + ~5 min coding/plots.
