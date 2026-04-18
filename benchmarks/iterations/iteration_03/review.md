# Iteration 03 review

*Written by the critical reviewer (Plan agent), for the implementer.
Reference: iteration_02 summary / review / plots.*

## Summary of what iter 02 left on the table

Iteration 02 landed the P0 items: upstream fix verified
(`frac_pathological_std = 0.0` on all 9 completed pygptreeo runs), hard
subprocess timeout, Wilcoxon p-values. Four gaps remain:

1. **Empty `borehole_8d` row for pygptreeo and SVGP.**
2. **Distribution-shift plot still has no data.**
3. **`sklearn_gp` only completes 2/12 default cells** — current per-dim
   budget (`max_train=1500 if d<=5 else 500`) does not rescue it.
4. **Wilcoxon is pooled only** — no per-problem ratio view.

Runtime budget: 30 min coding + 60 min runtime.

## Prioritised punch-list

### P0 — must land

1. **Close the `borehole_8d` hole.**
   Re-run
   `--methods pygptreeo gpytorch_svgp --problems borehole_8d --seeds 0 1 2 --max-wall-time 600 --n-stream 2000 --checkpoint-every 200`.
   Expected: 6 new runs in ~35 min. Acceptance: all 6 finish
   `aborted=False` with `frac_pathological_std[-1]=0`.

2. **Dial `sklearn_gp` down so it actually finishes.**
   In `run_all.py` set `max_train_points = 400 if d<=5 else 250` AND
   pass `n_restarts_optimizer=0` through `SklearnGPAdapter.__init__`
   (new kwarg, default 1 preserved for library callers). Re-label
   method in `make_plots.py METHOD_LABEL` as `"sklearn GP (N≤400)"`.
   Re-run `--methods sklearn_gp --problems rosenbrock_2d friedman1_5d
   borehole_8d --seeds 0 1 2`.

3. **Distribution-shift sweep.**
   `--methods pygptreeo gpytorch_svgp random_forest river_knn
   --schedules iid shift --seeds 0 1 --n-stream 2000
   --problems rosenbrock_2d friedman1_5d` → 32 runs. Exclude
   `sklearn_gp` from this sweep. `shift_vs_iid.png`: 2 subplots (one
   per problem), grouped bars, x = method, two bars per method (iid
   solid / shift hatched), log-y on final NRMSE.

4. **Add `wilcoxon_per_problem.png`.**
   Grouped bar chart: x = problem (4 groups), bars per alternative,
   height = median NRMSE ratio vs pygptreeo, log-y, horizontal line at
   `ratio=1`. No p-values. Add `plot_wilcoxon_per_problem` in
   `make_plots.py` next to `plot_wilcoxon_table`.

### P1 — should land

5. **5 seeds for critical problems.**
   `--methods pygptreeo gpytorch_svgp --problems rosenbrock_2d
   friedman1_5d --seeds 0 1 2 3 4`. Seeds 0-2 already exist → only 8
   new runs (seeds 3-4 × 2 methods × 2 problems).

6. **Calibration-summary `.npz` table.**
   `write_calibration_table(results, problems, out_path)` in
   `make_plots.py` that saves `iteration_03/calibration_table.npz`
   with `methods`, `problems`, `nominal_levels = [0.50, 0.6827, 0.90,
   0.95]`, `empirical_coverage[method, problem, level]` (median over
   seeds), `n_seeds[method, problem]`. One row per (method, problem);
   NaN where missing.

7. **NLPD regression assertion in `run_all.py`.**
   In `_run_one` after the final print block, add
   `if mednlpd > 1e3: print("[WARN] NLPD sanity: ...")`. Don't abort.

8. **Scaling panel.**
   New `plot_scaling(results, problems, out_path, method="pygptreeo")`
   at `iteration_03/scaling.png`: 1×4 subplots, x = checkpoint, y =
   per-point update time in ms = `np.diff(cum_update_time) /
   np.diff(checkpoints) * 1000`. Seed-median + IQR shade. Reference
   `log N` line for visual comparison.

### P2 — nice to have

9. Add one paragraph to `benchmarks/README.md` stating which plots
   directory is canonical.
10. Defer `piston_7d` and `gpytorch_exact_gp` baseline to iteration 04.

## Explicit out-of-scope

- Do NOT modify `pygptreeo/gptree.py`, `pygptreeo/gpnode.py`,
  `benchmarks/harness.py`, or `benchmarks/adapters/base.py`.
- Do NOT raise `--n-stream` above 3000 or `--n-test` above 1000.
- Do NOT add a GPU code path or new methods.
- Do NOT rerun existing `.npz` files with `--force` unless necessary.
- Do NOT change the `coverage_95` / `nlpd_trimmed` panel layout.
- Do NOT re-run `river_knn` on the shift schedule in addition to the
  32-run budget — it's included.

## Acceptance criteria

- `benchmarks/data/` gains ≥ 50 new iid runs and 32 new shift runs;
  total ≥ 125 files.
- `pygptreeo` and `gpytorch_svgp` each have 3 completed `borehole_8d`
  runs with `aborted=False` and `frac_pathological_std[-1]=0`.
- `sklearn_gp` completes ≥ 2 seeds on each of `rosenbrock_2d` and
  `friedman1_5d`.
- `benchmarks/iterations/iteration_03/` contains: `comparison.png`,
  `summary.png`, `pareto.png`, `calibration.png`, `wilcoxon_nrmse.png`,
  `wilcoxon_per_problem.png`, `shift_vs_iid.png`, `scaling.png`, and
  `calibration_table.npz`.
- `summary.md` reports: per-problem median NRMSE ratios for all 3
  alternatives, shift/iid NRMSE ratio per method on the 2 shift
  problems, wall-time table for borehole_8d, empirical coverage at
  nominal 0.95 per (method, problem), any slipped item with cause.
- Total iter-03 runtime < 90 min wall-clock on one CPU thread.
- Every pygptreeo run emits `[WARN] NLPD sanity …` zero times.
