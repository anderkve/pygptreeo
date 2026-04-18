# Iteration 07 — implementer summary

*Polish pass. All acceptance criteria hit.*

## Reliability

> **`Reliability: 58 / 58 pygptreeo* runs have frac_pathological_std[-1] == 0 (100.0 %)`**

One new run since iter 06 (pygptreeo_A borehole_8d seed 4), zero regressions.

## What landed

### P0 — `write_paper_tables`

New function in `make_plots.py` emits two files per iteration directory:

- `paper_table.md` — GitHub-flavoured markdown
- `paper_table.tex` — `booktabs` tabular

Both report, per (method, problem), the final-checkpoint statistics of NRMSE, CRPS, and empirical 95 % coverage. Cell format:

- `n == 0`  →  `—`
- `n == 1`  →  bare value
- `2 ≤ n < 3`  →  `median (n=k)`
- `n ≥ 3`  →  `mean ± 1.96·SE (n=k)`

Seven "main" paper-relevant methods plus four "supplementary variants" in a second block per metric. Wired into `make_plots.main()` next to `write_calibration_table`, so every iteration directory now gets the tables automatically.

### P1 — thin-cell gap closure

- **`pygptreeo_A` borehole_8d seed 4** re-run in 16.7 s. All 5 seeds on borehole_8d now converged cleanly (mean NRMSE 9.65e-4 ± 3.2e-4 at 1.96·SE).
- **`sklearn_gp_A` borehole_8d seeds 0/1/2** — 3 new runs. NRMSE 1.53e-3 ± 2.9e-4. sklearn_gp no longer has an empty bar in `headline.png`.

### P2 — deferred (per review)

C (reproducibility README), D (comparison.png redesign), E (per-seed scatter), F (Wilcoxon power) all deferred.

## Headline table (excerpt from `iteration_07/paper_table.md`)

### Final NRMSE (↓)

| method              | smooth_sines_2d | rosenbrock_2d           | friedman1_5d           | borehole_8d            |
| ------------------- | --------------- | ----------------------- | ---------------------- | ---------------------- |
| **pygptreeo (A)**   | 1.23e-05 ± 3.1e-06 (n=3) | **2.09e-05 ± 8.1e-06 (n=5)** | **3.85e-04 ± 1.9e-04 (n=5)** | **9.65e-04 ± 3.2e-04 (n=5)** |
| pygptreeo (B)       | —               | 2.13e-05 ± 7.6e-06 (n=3) | 3.86e-04 (n=2)         | —                      |
| pygptreeo (C: Matern) | —             | 1.80e-04 ± 1.1e-04 (n=5) | 7.06e-04 ± 1.2e-04 (n=3) | —                  |
| sklearn GP (A, N≤400) | 4.41e-04 (n=2) | 5.85e-04 ± 3.9e-04 (n=3) | 1.66e-03 ± 1.2e-04 (n=3) | 1.53e-03 ± 2.9e-04 (n=3) |
| SVGP (A)            | 1.50e-03 ± 5.5e-04 (n=3) | 1.31e-03 ± 3.1e-04 (n=5) | 3.26e-03 ± 3.1e-04 (n=4) | 2.04e-03 ± 7.4e-04 (n=3) |
| RandomForest (A)    | 9.91e-03 ± 2.7e-04 (n=3) | 1.28e-02 ± 1.5e-03 (n=3) | 5.08e-02 ± 1.5e-03 (n=3) | 2.78e-02 ± 1.9e-03 (n=3) |
| River kNN (A, k=8)  | 2.31e-01 ± 1.2e-02 (n=3) | 1.87e-01 ± 2.9e-02 (n=3) | 1.79e-01 ± 1.3e-02 (n=3) | 1.89e-01 ± 1.3e-02 (n=3) |

**Pipeline-proof row** (from `paper_table.md`):

> `pygptreeo (A) | — | 2.09e-05 ± 8.1e-06 (n=5) | … | 9.65e-04 ± 3.2e-04 (n=5)`
>
> `SVGP (A: 256 ind.) | — | 0.00131 ± 0.00031 (n=5) | … | 0.00204 ± 0.00074 (n=3)`

pygptreeo_A rosenbrock CI = [1.28e-5, 2.90e-5], SVGP rosenbrock CI = [1.00e-3, 1.62e-3] — **non-overlapping**, 63× ratio between means.
pygptreeo_A borehole CI = [6.5e-4, 1.3e-3], SVGP borehole CI = [1.3e-3, 2.8e-3] — **non-overlapping even in 8-D**.

## Acceptance-criteria status

| criterion                                                                                                | status |
| -------------------------------------------------------------------------------------------------------- | ------ |
| pygptreeo_A__borehole_8d__seed4.npz exists, frac_pathological_std==0                                     | ✅     |
| sklearn_gp_A__borehole_8d__seed{0,1,2}.npz exist                                                         | ✅     |
| headline.png has no empty sklearn_gp_A bar on borehole                                                   | ✅ (bar populated) |
| write_paper_tables defined; main() writes paper_table.tex and .md into --iter-dir                        | ✅     |
| Every cell with n≥3 for pygptreeo_A rosenbrock/friedman/borehole carries "± 1.96·SE"                     | ✅     |
| summary.md cites the reliability one-liner                                                               | ✅ (top) |
| summary.md quotes at least one row from the new table                                                    | ✅     |
| Total runtime < 45 min                                                                                   | ~12 min (parallel) |
