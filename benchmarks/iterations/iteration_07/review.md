# Iteration 07 review

## Status after iter 06

All P0/P1/P2 items from iter 06 landed. Reliability is 57/57, pygptreeo_A has n=5 on rosenbrock_2d and friedman1_5d with non-overlapping 1.96·SE CIs vs SVGP_A, `_B/_C` shift ratios within 2× of `_A` on friedman, the `_C` rosenbrock coverage anomaly confirmed real at 0.78 with n=5, asymptote at 5000 points ~1.1e-5, MoE ≈ PoE documented.

The benchmark is at "press go and commit". Iter 07 must not risk destabilising that — pick polish tasks only.

## Prioritised punch-list

### P0 — paper-ready table exporter (Candidate A)

Add `write_paper_tables(results, problems, out_dir, schedule="iid")` in `make_plots.py`. Writes two files per iteration directory:

- **`paper_table.tex`** — `booktabs` tabular with problems as columns and 7 paper-relevant methods (pygptreeo_A/B/C + sklearn_gp_A + gpytorch_svgp_A + random_forest_A + river_knn_A) as rows. Each cell shows `mean ± 1.96·SE (n=k)` for NRMSE. A second block reports coverage_95 per cell. Optional third block: CRPS.
- **`paper_table.md`** — same in GitHub-flavoured markdown for README / summary.

When `n < 3` omit SE and emit "n=k" only; when `n == 1` emit the bare value. Hook into `main()` next to `write_calibration_table`.

### P1 — close thin-cell gaps (Candidate B, trimmed)

Only the two runs that actually matter:

1. `pygptreeo_A` borehole_8d seed 4 (iter-06 aborted). `--methods pygptreeo_A --problems borehole_8d --seeds 4 --max-wall-time 900 --n-stream 2000`. 1 run, ~5 min. Gets borehole to n=5.
2. `sklearn_gp_A` borehole_8d seeds 0/1/2 at N≤400. 3 runs, ~4 min each. Fills the empty sklearn_gp_A bar on `headline.png`.

Skip all other "extra seeds" — every paper-relevant cell has n≥4 on problems that matter, and more seeds where pygptreeo already wins 60× is decorative.

### P2 — defer

| Candidate | Reason to defer |
| --------- | --------------- |
| C (README) | Low-priority; deferred to iter 08 if P0/P1 finish under budget. |
| D (redesign `comparison.png`) | `headline.png` is already the paper headline; `comparison.png` is a supplementary multi-panel artefact. |
| E (per-seed scatter) | `mean ± 1.96·SE` in the new paper table conveys the same seed-variance info. |
| F (Wilcoxon power analysis) | Low-n p-values already flagged in text; decorative numbers without scientific content. |

## Out-of-scope

- Do NOT modify `pygptreeo/gptree.py`, `pygptreeo/gpnode.py`, `benchmarks/harness.py`, `benchmarks/adapters/*`, or any method hyperparameters.
- Do NOT re-run existing `.npz` files (no `--force`).
- Do NOT add new problems, methods, or schedule types.
- Do NOT touch `headline.png` / `wilcoxon_variants.png` / `shift_vs_iid.png` panel layouts — only let them re-render from new data.
- Do NOT reformat existing iter-06 artefacts.

## Acceptance criteria

- `benchmarks/data/pygptreeo_A__borehole_8d__seed4.npz` exists with `frac_pathological_std[-1] == 0`.
- `benchmarks/data/sklearn_gp_A__borehole_8d__seed{0,1,2}.npz` exist.
- `headline.png` has no empty sklearn_gp_A bar on borehole_8d.
- `make_plots.write_paper_tables` defined; `main()` writes `paper_table.tex` and `paper_table.md` into both `benchmarks/plots/` and the `--iter-dir`.
- Both tables include every (method, problem) cell populated by existing `.npz` data and report "n=k" for k<5. At least the pygptreeo_A × {rosenbrock_2d, friedman1_5d, borehole_8d} cells carry "± 1.96·SE".
- `.tex` file compiles under a minimal `\documentclass{article}\usepackage{booktabs}` preamble (scratch-test).
- `summary.md` cites the reliability one-liner verbatim and quotes at least one row from the new table.
- Total iter-07 runtime < 45 min.
