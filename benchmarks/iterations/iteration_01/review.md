# Iteration 01 review

*Written by the critical reviewer (Plan agent), for the implementer.*

## Summary of baseline weaknesses

- **NLPD is dominated by a handful of pathological points** (std=0 or std~1e37 from pygptreeo MoE/calibrated-sigma). The harness clips std to 1e-8 in `benchmarks/harness.py:83`, which silently converts "I have no idea" into "massive per-point penalty" and makes the reported mean NLPD essentially noise. The symlog plot hides rather than reveals this.
- **Per-method budgets are asymmetric and implicitly favour pygptreeo.** pygptreeo uses up to Nbar=200 points per leaf with many leaves (final update time ~100 s), while SVGP is starved at 64 inducing points and sklearn_gp is capped at 800 training points — yet the summary presents NRMSE side-by-side as if budgets were matched.
- **A single seed.** Seed 1 on `pygptreeo/rosenbrock_2d` gives final NLPD=2.07e+11 while seed 0 gives 22.0 — the method's own single-run NLPD variance exceeds every effect in the plot. No means/CIs reported.
- **"Random + recent" reservoir bias.** `sklearn_gp_adapter.py:58-62` and `rf_adapter.py:42-46` weight training on "random 75% + recent 25%" which is nothing any practitioner does and is not documented in the README.
- **Problem suite is ad-hoc.** Three optimisation test functions, all i.i.d. U[0,1]^d, n_stream=2000, no dimensionality variation and no distribution shift. This is not a paper-grade emulation benchmark.

## Prioritised punch-list for the implementer

1. **[P0] Robust NLPD handling (fix the reporting, not pygptreeo).**
   Replace the `np.clip(std, 1e-8, None)` hack in `benchmarks/harness.py:83` with: (a) a physically motivated std floor equal to `1e-3 * (y_test.max() - y_test.min())`; (b) additionally record `median_nlpd` and `nlpd_trimmed` (mean after trimming 5/95th percentiles of per-point NLPD); (c) record a new metric `frac_pathological_std` = fraction of test points with `std<=floor` or `std>=1e3*y_range` or non-finite. Compute **CRPS** for Gaussian predictives (closed form) and save it.

2. **[P0] Multi-seed + proper statistical reporting.**
   In `run_all.py` change default `--seeds` to `[0, 1, 2, 3, 4]`. In `harness.py` seed numpy/torch/random at run start. Switch plots to median + IQR shading; add `mean ± 1 std` in a secondary colour. Summary bars: median with IQR error bars.

3. **[P0] Match per-method budgets honestly, and document them.**
   SVGP: `n_inducing=256, n_epochs=60, retrain_every=200, max_buffer=5000, lr=5e-3`.
   sklearn_gp: `retrain_every=200, max_train_points=1500`.
   random_forest: `n_estimators=300, retrain_every=200`.
   Add a budget table in `benchmarks/README.md`.

4. **[P1] Replace "recent 25% + random 75%" reservoir with uniform reservoir sampling.**
   In sklearn_gp and RF adapters: `rng.choice(n, size=max_train_points, replace=False)` only. Reseed with `default_rng(self.random_state + self._n_refits)` to avoid identical subsets across refits.

5. **[P1] Expand the problem suite to emulation-community benchmarks.**
   Add `borehole_8d`, `friedman1_5d`, `piston_7d`. Keep `smooth_sines_2d` as an easy-GP sanity baseline and `rosenbrock_2d` for the curved-valley classic. Drop `step_3d` from the default set (flag as pathological).

6. **[P1] Longer streams + a distribution-shift variant.**
   Bump `--n-stream` default to 5000, `--checkpoint-every` to 500. Add a `sample_shifted` method on `Problem` that draws first half from U[0,0.5]^d then U[0.5,1]^d. Expose `--schedule {iid,shift}` in `run_all.py`; record in saved config.

7. **[P1] Publication-ready figure layout in `make_plots.py`.**
   Per-problem multi-panel: (a) NRMSE vs stream size, (b) median NLPD with IQR, (c) CRPS, (d) calibration curve (empirical coverage at 50/68/90/95 levels), (e) update time, (f) predict time, (g) frac_pathological_std.

8. **[P2] Fix river_knn's uncertainty or mark it unsupported.**
   Return `max(std(neighbour-y), floor)` with `floor = 1e-3 * y_range` so river_knn's NLPD/coverage are meaningful. (Option (a) from the review.)

9. **[P2] Test-set sizing and independence.**
   Use separate rng (`default_rng(seed + 10_000)`) for test set; set default `--n-test` to 1000.

## Explicit out-of-scope items (do NOT touch this iteration)

- Do not modify `pygptreeo/gptree.py` or `pygptreeo/gpnode.py` — the MoE-variance cancellation and calibrated-sigma explosion are real pygptreeo bugs, but fixing them is a separate track. The benchmark must *report* them honestly, not patch them.
- Do not change the `OnlineRegressor` abstract interface in `benchmarks/adapters/base.py`.
- Do not add GPU code paths; keep everything CPU to preserve timing fairness.
- Do not add new methods this iteration.

## Acceptance criteria

- `benchmarks/data/*.npz` contains new fields `median_nlpd`, `nlpd_trimmed`, `crps`, `frac_pathological_std` for every run.
- ≥ 3 seeds per (method, problem) saved; plots show median + IQR shading.
- `comparison.png` has calibration curves and a `frac_pathological_std` panel where pygptreeo visibly shows >0 on rosenbrock_2d while other methods show 0.
- `borehole_8d` and `friedman1_5d` appear in `problems.py` and are run by default.
- SVGP `n_inducing=256`, random_forest `n_estimators=300`.
- README documents per-method compute budget.
- `run_all.py` completes end-to-end in well under one hour for the new default config.
- No run emits `NLPD = 3.9e12` on saved data with the new floor.
