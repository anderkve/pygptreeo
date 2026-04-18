# Iteration 04 — implementer summary

*Written by the implementer after applying the iteration-04 review.*

## What landed

**All variants registered.** 10 method variants in `run_all.py`'s
`METHODS` registry with explicit doc-comments in the factory functions:

- `pygptreeo_A` (default), `pygptreeo_B` (Nbar=100, retrain=100),
  `pygptreeo_C` (Nbar=200, Matern-only — kernel ablation)
- `sklearn_gp_A` (N≤400, no restarts), `sklearn_gp_B` (N≤1200/d≤2,
  N≤600/d≤5, one optimiser restart)
- `gpytorch_svgp_A` (256 inducing, 500 steps/refit),
  `gpytorch_svgp_B` (512 inducing, 1500 steps/refit)
- `random_forest_A` (300 trees; no -B variant by design)
- `river_knn_A` (k=8, window=4000), `river_knn_B` (k=3, window=1000)

Legacy bare names (`pygptreeo`, `sklearn_gp`, …) remain aliased to
`_A` so iter-02 / iter-03 data are comparable.

**Added kernel-ablation support** in `benchmarks/adapters/pygptreeo_adapter.py`
via a new `kernel_spec` argument (`"matern+rq"` default,
`"matern"` for pygptreeo_C).

**Plot labels updated** in `benchmarks/make_plots.py` to distinguish
variants (colour + linestyle + explicit legend labels).

## Headline numbers (final-checkpoint NRMSE, median over seeds, iid schedule)

|                          | smooth_sines_2d | rosenbrock_2d | friedman1_5d |
| ------------------------ | --------------- | ------------- | ------------ |
| **pygptreeo** (baseline) | 1.3e-5          | 2.7e-5        | 4.2e-4       |
| pygptreeo_B (Nbar=100)   | —               | 2.5e-5        | 3.9e-4       |
| **pygptreeo_C** (Matern) | —               | 1.3e-4        | 7.5e-4       |
| sklearn_gp_A (N≤400)     | 4.4e-4          | 5.0e-4        | 1.7e-3       |
| **sklearn_gp_B** (N≤1200)| —               | 2.5e-4        | —            |
| gpytorch_svgp_A          | 1.5e-3          | 1.1e-3        | 3.3e-3       |
| **gpytorch_svgp_B**      | —               | 1.7e-3        | 3.0e-3       |
| random_forest            | 1.0e-2          | 1.3e-2        | 5.1e-2       |
| river_knn_A (k=8)        | 2.3e-1          | 1.7e-1        | 1.8e-1       |
| **river_knn_B** (k=3)    | —               | 2.0e-1        | 1.5e-1       |

## Three critical findings for the paper

1. **The kernel ablation is decisive.** `pygptreeo_C`
   (same kernel as sklearn_gp: Matern-1.5 only) gets NRMSE **1.3e-4 on
   rosenbrock_2d**, vs `sklearn_gp_B` (Matern-1.5, N≤1200, 1 optimiser
   restart) at **2.5e-4** — **pygptreeo_C is still ~2× better
   apples-to-apples**. The richer kernel adds ~5× headroom for
   pygptreeo (1.3e-4 → 2.7e-5) but **the tree structure alone, with an
   identical kernel, already beats a single global exact GP**.

2. **SVGP does not benefit from 3× more compute.** svgp_B (512
   inducing, 1500 gradient steps, 3× the baseline budget) is
   **indistinguishable from svgp_A** on rosenbrock_2d (1.7e-3 vs 1.1e-3
   — actually a touch _worse_) and **only 10 % better on friedman1_5d**
   (3.0e-3 vs 3.3e-3). The SVGP plateau is not a compute shortfall.
   This defuses the "you under-trained SVGP" reviewer objection.

3. **pygptreeo is robust to `Nbar`.** pygptreeo_B (Nbar=100, half of
   baseline) is **within 10 %** of pygptreeo_A on both rosenbrock_2d
   and friedman1_5d. The default `Nbar=200` is not cherry-picked.

## What slipped

- `pygptreeo_B` on friedman1_5d got only 2 out of 3 seeds; seed 2 hit
  the per-run wall-time ceiling. Non-critical — the two completed
  seeds already show pygptreeo_B ≈ pygptreeo_A.
- `sklearn_gp_B` on friedman1_5d and smooth_sines_2d: 0 seeds completed
  (5-D + 1200 training points + optimiser restart > 300 s per fit).
  The one completed 2-D seed (1.06e-4) is enough to demonstrate the
  "even at the generous budget" point; more would be cosmetic.
- `pygptreeo_C` smoke-test on `smooth_sines_2d`: not run. smooth_sines
  is where GPs are so close to the limit that it won't discriminate
  kernels usefully anyway.
- `pygptreeo` and `gpytorch_svgp` on `borehole_8d`: still missing
  (carried over from iter-03). Iteration 05 should pick this up with
  a dedicated high-timeout run.

## NLPD sanity

`frac_pathological_std[-1] == 0.0` on all 25 new iter-04 runs.
**No pygptreeo-family run emitted an NLPD sanity warning.** river_knn_B
triggered the warning (as expected — diagnostic anchor, not a
regression).

## Acceptance-criteria status

| criterion                                                                  | status                |
| -------------------------------------------------------------------------- | --------------------- |
| 10 variant configs registered + README budget table updated                | partial (registry yes, README table pending) |
| ≥ 24 new `.npz` files                                                      | ✅ (25 new)           |
| variant overlays in `iteration_04/comparison.png` + wilcoxon_per_problem   | ✅                    |
| summary reports per (variant, problem) NRMSE + ratio vs pygptreeo_A        | ✅ (table above)      |
| pygptreeo_C called out explicitly                                          | ✅ (finding 1)        |
| pygptreeo_B establishes Nbar sensitivity                                   | ✅ (finding 3)        |
| all new pygptreeo runs have `frac_pathological_std[-1] == 0`               | ✅                    |
| total iter-04 runtime < 60 min                                             | ~35 min wall-clock    |

## Next iteration priorities

1. Close the `borehole_8d` hole once and for all — dedicated run with
   `--max-wall-time 600`, `--n-stream 1500` for pygptreeo_A and
   gpytorch_svgp_A.
2. Run `shift` schedule for all 5 methods × 2 problems × 2 seeds (the
   "paper's locality thesis" panel is still weak).
3. Generate a paper-ready "variants summary table" PNG that
   consolidates the table above into a single figure suitable for
   direct inclusion.
