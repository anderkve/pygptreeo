# Iteration 02 review

*Written by the critical reviewer (Plan agent), for the implementer.*

## Summary of iteration 01 gaps and what changed under the hood

- The MoE variance-cancellation bug that caused `frac_pathological_std > 0` and NLPD=O(10^12) on rosenbrock_2d is **fixed upstream** (`pygptreeo/gptree.py` now uses the numerically stable `sum w * (sigma^2 + (mu - mean_total)^2)` form; `pygptreeo/gpnode.py` also floors the sigma at `sqrt(alpha)` to prevent the sklearn neg-variance leak). The benchmark must now *verify* the fix end-to-end rather than defend against it.
- Iteration 01 only got 1-2 seeds per (method, problem) cell (19 `.npz` files, see `iteration_01/summary.md`). IQR bands in `comparison.png` are therefore placebo — visually single-line on most panels.
- `pygptreeo` and `sklearn_gp` are **absent from the borehole_8d row** because the outer `max_wall_time_s` check in `harness.py` only fires between checkpoints — a single `GaussianProcessRegressor.fit()` with 1500 training points in 8-D hangs for the entire run. Hard timeout is missing.
- `shift` schedule is wired into `problems.py` and `run_all.py`, but no saved run ever exercised it. Zero evidence for the locality story that the paper's thesis rests on.
- `comparison.png` has one dead panel (`frac_pathological_std` flat at zero given the 1e-6 floor) and one deceptive panel (`median_nlpd` dominated by river_knn's ~10^4 on the linear axis, compressing every other method to a flat line).
- Iteration plots currently only exist under `benchmarks/plots/`. Per-iteration copies in `benchmarks/iterations/iteration_01/` were done by hand. Iteration 02 must make this automatic.

## Prioritised punch-list

### P0 — must land this iteration

1. **Hard per-run timeout at the process level.**
   Sklearn's `.fit()` does not respect the harness loop timeout, so a single borehole_8d refit can swallow the whole `max_wall_time_s`. In `benchmarks/run_all.py`, wrap each `(method, problem, seed, schedule)` in `multiprocessing.Process(target=_child, args=...)` and `join(timeout=max_wall_time_s + 30)`. The child writes the `.npz` itself (reuse `save_result`); on timeout the parent calls `terminate()` and synthesises a minimal `.npz` with `aborted=True` plus whatever partial checkpoints the child already flushed (the child should save after every checkpoint, not only at the end). This is the single highest-value fix this iteration.

2. **Re-run pygptreeo on the problems where the bug was worst.**
   Rationale: prove the upstream fix. Invoke:
   `python benchmarks/run_all.py --methods pygptreeo --problems rosenbrock_2d friedman1_5d --seeds 0 1 2 3 4 --n-stream 3000 --checkpoint-every 300 --force`.
   Acceptance: `frac_pathological_std` at the final checkpoint is exactly 0.0 across all 10 runs; `median_nlpd` is finite and monotone-ish; no checkpoint has NLPD > 10^3.

3. **3 seeds per (method, problem) for the default 4 problems.**
   5 methods × 4 problems × 3 seeds = 60 runs. With per-method wall-time caps per item 5, this fits in ~75 minutes on one core.

4. **Write iteration-02 plots to both `benchmarks/plots/` AND `benchmarks/iterations/iteration_02/`.**
   `make_plots.py` accepts a new `--iter-dir` argument and writes each of the four PNGs to both locations. The `summary.md` at the end of the iteration must link to the per-iteration copies, not the moving target in `plots/`.

### P1 — should land this iteration

5. **Sklearn GP per-dim compute budget.**
   The asymmetric budget is the proximate cause of the borehole timeout. In `run_all.py`, if `d <= 5`, `max_train_points=1500`; if `d >= 6`, `max_train_points=500`. Raise `--max-wall-time` default from 900 to 1200. Document both in `benchmarks/README.md`.

6. **SVGP training sanity check.**
   `gpytorch_svgp_adapter.py` loops `n_epochs=60` passes through the whole buffer with `bs=128`. At `max_buffer=5000`, that's ~2340 gradient steps per refit. Cap the inner loop by total gradient steps (e.g. `min(n_epochs * n_batches, 500)`) — this also shaves wall-time for the borehole run.

7. **Statistical test panel.**
   Add a `plot_wilcoxon_table(results, problems, out_path, baseline='pygptreeo', metric='nrmse')` in `make_plots.py` that computes a Wilcoxon signed-rank p-value comparing `pygptreeo` against each alternative's final-checkpoint NRMSE, paired by seed. With only 3 seeds per cell a signed-rank over 3 pairs is underpowered — so pool over (problem × seed) pairs (12 pairs). Report per-method median NRMSE ratio alongside. Render as a small table figure at `iteration_02/wilcoxon_nrmse.png`. Do the same for `crps`.

8. **Fix the median-NLPD and pathological-std panels in `comparison.png`.**
   - Drop `frac_pathological_std` from the metrics grid (provably zero post-fix).
   - Change `median_nlpd` to `nlpd_trimmed` with `yscale="symlog"` and `linthresh=1.0` to keep river_knn visible without flattening everyone else.
   - Replace the removed panel with a **`coverage_95`** panel (ylim 0-1, horizontal line at 0.95). Over-coverage on SVGP/RF is the most interesting finding from iteration 01 — give it dedicated space.
   - In `plot_calibration`, keep one panel per problem but add a faint vertical guide at each nominal level.

9. **Distribution-shift experiment.**
   One dedicated figure at `iteration_02/shift_vs_iid.png`. Run:
   `python benchmarks/run_all.py --schedules iid shift --seeds 0 --n-stream 3000` for all 5 methods on all 4 default problems (40 runs, ~40 min with item 1 in place). Plot: 4 subplots (one per problem), grouped bars, x=method, y=final NRMSE, two bars per method (iid vs shift). If pygptreeo has locality benefit, its shift-bar should rise less than the refit GPs' do. This is the only experiment in this iteration that supports the paper's core claim.

10. **River kNN in primary plots.**
    Decision: **keep river_knn in `comparison.png` but visually de-emphasise** — `linewidth=1.0`, grey, `alpha=0.6`, and drop entirely from `pareto.png` and the Wilcoxon table.

### P2 — nice to have, okay to defer

- Add `piston_7d` to `DEFAULT_PROBLEMS` in `run_all.py` only if item 1 ships cleanly; otherwise defer.
- Add a `gpytorch_exact_gp` baseline. Defer.

## Explicit out-of-scope

- Do NOT modify `pygptreeo/gptree.py` or `pygptreeo/gpnode.py`.
- Do NOT modify `benchmarks/adapters/base.py` — the `OnlineRegressor` interface is stable.
- Do NOT change `_metrics` in `harness.py` — the metric definitions are now the paper's metrics.
- Do NOT add GPU code paths.
- Do NOT add new methods.
- Do NOT change the std floor from `1e-6 * y_range` — loosening it would hide any bug regression.
- Do NOT raise `--n-stream` above 3000. Longer streams will blow the 90-minute runtime budget.
- Do NOT raise `--n-test` above 1000.
- Do NOT silently regenerate `benchmarks/plots/` without also writing to `iteration_02/`.

## Acceptance criteria for iteration 02

- [ ] `benchmarks/data/` contains at least 60 `.npz` files covering 5 methods × 4 problems × 3 seeds under `iid`, plus 40 files for the `shift` schedule.
- [ ] Every `pygptreeo` run has `frac_pathological_std[-1] == 0.0`; this is a hard assertion the implementer must print and include in `summary.md`.
- [ ] No pygptreeo or sklearn_gp run has `aborted == True` on borehole_8d with the per-dim budget in item 5; if any still abort, `summary.md` must explain which step the child process died at.
- [ ] `benchmarks/iterations/iteration_02/` contains `comparison.png`, `summary.png`, `pareto.png`, `calibration.png`, `shift_vs_iid.png`, and `wilcoxon_nrmse.png`, all freshly written in this iteration.
- [ ] The `frac_pathological_std` panel is removed from `comparison.png` and replaced by `coverage_95`.
- [ ] `summary.md` reports: (a) the Wilcoxon p-value for pygptreeo-vs-best-alternative on pooled NRMSE, (b) the shift-vs-iid NRMSE ratio per method, (c) wall-time per (method, problem) in a table, (d) any acceptance criterion that slipped and why.
- [ ] Total end-to-end runtime under 90 minutes on one CPU thread.
