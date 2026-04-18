# Iteration 05 review

*Written by the critical reviewer. Reference: iter-04 summary + figures.*

## What iter 04 left on the table

1. `borehole_8d` row empty for `pygptreeo` and `gpytorch_svgp`.
2. Shift story still 6 runs on one problem with the `_A` baseline set.
3. No paper-ready single figure consolidating the story.
4. `plot_wilcoxon_per_problem` is pygptreeo_A-centric and mixes variants with alternatives on the same x-axis.
5. `sklearn_gp_B` on `friedman1_5d` has zero completed seeds (n_restarts=1 at d=5 exceeds 300s/fit).

## Prioritised punch-list

### P0 — must land

1. **Close borehole_8d.** `--methods pygptreeo_A gpytorch_svgp_A --problems borehole_8d --seeds 0 1 2 --max-wall-time 900 --n-stream 1500 --checkpoint-every 200`. 6 new runs, ~40 min.

2. **Rescue sklearn_gp_B d≥5.** In `_make_sklearn_gp_B` (`run_all.py`): when `d >= 5`, force `n_restarts_optimizer=0`; keep max_train=600. Re-run `--methods sklearn_gp_B --problems friedman1_5d --seeds 0 1 2 --max-wall-time 600`. 3 new runs.

3. **Shift sweep for _A variants.** `--methods pygptreeo_A gpytorch_svgp_A random_forest_A river_knn_A --schedules shift --seeds 0 1 --n-stream 2000 --problems rosenbrock_2d friedman1_5d`. 16 new shift runs; existing iid skipped by `exists` check.

4. **Paper-headline figure `headline.png`.** New `plot_headline` in `make_plots.py`: 1 row × 3 panels, width 12":
   - A: NRMSE bars, median + IQR, `_A` baselines + pygptreeo_B + pygptreeo_C, 4 problem-groups × 7 methods, log y.
   - B: median CRPS same layout, log y.
   - C: empirical coverage at nominal 0.95, linear y, dashed line at 0.95.
   Shared legend below.

5. **`plot_wilcoxon_variants`.** 3-col subplot: each panel uses a different pygptreeo variant (A/B/C) as the Wilcoxon baseline. Alternatives: `_A` baselines of the 4 non-pygptreeo methods + their `_B` variants (pygptreeo variants excluded as alternatives). Shows pygptreeo wins regardless of which variant you anchor on.

### P1 — should land

6. **Calibration markdown table in `summary.md`** — render `calibration_table.npz` (cov at nominal 0.95) as a per-(method, problem) table.

7. **Mean ± std companion to `wilcoxon_per_problem`.** Add `plot_ratio_bars_mean_std`, output `iteration_05/wilcoxon_per_problem_mean_std.png`.

### P2 — defer

- 3rd seed for `pygptreeo_B` on friedman1_5d.
- Shift sweep for `_B` variants.
- 3rd seed for `sklearn_gp_B` rosenbrock_2d.
- Any new variant on smooth_sines_2d.

## Out-of-scope

- Do NOT modify `pygptreeo/gptree.py`, `pygptreeo/gpnode.py`, `benchmarks/harness.py`, `benchmarks/adapters/base.py`, or `benchmarks/adapters/pygptreeo_adapter.py`.
- Do NOT add new methods, problems, or pygptreeo variants beyond A/B/C.
- Do NOT change theta, sigma_rel, aggregation, splitting_strategy, max_n_pred_leaves, std floor, or cap.
- Do NOT re-run existing iid `.npz` — `exists` guard in `run_all.py` must stay in place; no `--force`.
- Do NOT change `--n-stream > 3000` or `--n-test > 1000` for the default problems; borehole is the documented exception (1500).
- Do NOT change panel layout of `comparison.png` / `calibration.png` / `summary.png` / `pareto.png` beyond labels; `headline.png` and `wilcoxon_variants.png` are the only new figures.

## Acceptance criteria

- `borehole_8d` row in `iteration_05/comparison.png` has ≥ 2/3 seeds for pygptreeo_A and gpytorch_svgp_A; all pygptreeo runs have `frac_pathological_std[-1]==0`.
- `sklearn_gp_B__friedman1_5d__seed{0,1,2}.npz` exist.
- ≥ 16 new `*__shift__*.npz` files covering 4 methods × 2 problems × 2 seeds.
- `iteration_05/headline.png` exists, 1×3 layout.
- `iteration_05/wilcoxon_variants.png` exists, 3 sub-panels (A/B/C baselines), every non-pygptreeo alternative is above parity on ≥ 3/4 problems in all three panels.
- `iteration_05/calibration_table.npz` populated; `summary.md` renders it as markdown.
- `iteration_05/wilcoxon_per_problem_mean_std.png` exists.
- Total iter-05 runtime < 90 min.
- `summary.md` explicitly states whether pygptreeo remains the winner when the Wilcoxon baseline is `_B` or `_C`.
