# Iterative benchmark development log

Each numbered subdirectory corresponds to one round of
**reviewer → implementer → run → plot** refinement of the benchmark. Later
iterations strictly supersede earlier ones, but we keep everything for audit
and provenance.

Layout of each iteration directory:

```
iteration_NN/
├── review.md                # the reviewer's instructions for this iteration
├── summary.md               # the implementer's summary + observations
├── comparison.png           # main multi-panel figure at iteration end
├── summary.png              # final-step bar charts
├── pareto.png               # accuracy vs compute scatter
├── calibration.png          # reliability diagrams
└── data/*.npz               # snapshot of the raw results used for the plots
```

## Roles

**Reviewer (Plan agent).** Reads the previous iteration's code + figures +
summary, writes `review.md` with a short prioritised punch-list and explicit
out-of-scope items, and states acceptance criteria.

**Implementer (this agent).** Applies the reviewer's punch-list, runs the
benchmark, writes `summary.md` with honest observations, and snapshots the
resulting plots + data to the iteration directory.

## Iteration log

| #  | Focus |
| -- | ----- |
| 00 | Baseline + NLPD bug investigation. |
| 01 | Robust metrics (median/trimmed NLPD, CRPS, multi-level coverage), multi-seed, fair per-method budgets, emulation-community problems (borehole, Friedman-1), distribution-shift schedule option. |
| 02 | Subprocess-level hard timeout, per-dim sklearn_gp budget, paired Wilcoxon, new plot layout. Upstream MoE-variance + sigma=0 fix verified end-to-end. |
| 03 | Per-problem Wilcoxon, structured `calibration_table.npz`, pygptreeo scaling plot, NLPD sanity trip-wire in driver, first `*__shift__*.npz` data. |
| 04 | **Algorithm settings critique + 10 method variants** (`pygptreeo_A/B/C`, `sklearn_gp_A/B`, `gpytorch_svgp_A/B`, `random_forest_A`, `river_knn_A/B`). Kernel ablation proves pygptreeo still wins apples-to-apples. |
| 05 | Borehole closure for pygptreeo + SVGP; sklearn_gp_B rescue; distribution-shift sweep over `_A` variants; `headline.png` + `wilcoxon_variants.png`. |
| 06 | 5 seeds on `_A`; shift for `_B/_C`; `pygptreeo_poe` ablation; long-stream asymptote on rosenbrock; reliability one-liner wired in. |
| 07 | Paper-ready table exporter (`paper_table.md`, `paper_table.tex`, `booktabs`). Thin-cell gap closures (sklearn_gp_A borehole, pygptreeo_A borehole seed 4). **Pinned canonical snapshot** for paper citations. |
| 08 | Reproducibility scaffolding: rewritten `benchmarks/README.md`, `benchmarks/regenerate_paper.sh`, long-stream on friedman1_5d and borehole_8d. |
| 09 | Endcap. PoE n=3, `run_summary.txt` generator, README cites the iter-01 fix commit SHA. Benchmark frozen after this commit. |
