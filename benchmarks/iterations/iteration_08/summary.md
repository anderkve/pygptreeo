# Iteration 08 — implementer summary

*Polish pass: reproducibility scaffolding + long-stream on the two
higher-d problems. All acceptance criteria met.*

## Reliability

> **`Reliability: 58 / 58 pygptreeo* runs have frac_pathological_std[-1] == 0 (100.0 %)`**

Same as iter 07; no new pygptreeo runs in iter 08 inside `benchmarks/data/`, only the three long-stream singletons in `benchmarks/data_long/`.

## What landed

### P0 — reproducibility scaffolding

1. **Rewritten `benchmarks/README.md`.** Cites `iterations/iteration_07/paper_table.{md,tex}` as the **pinned canonical snapshot** for the paper; documents the `_A/_B/_C/_poe` variant suffix convention; lists every method with its registry key and one-line description; documents the cell-format key (`mean ± 1.96·SE (n=k)` / `median (n=k)` / bare / em-dash); adds a `data_long/` section; references `regenerate_paper.sh` for one-command reproduction.

2. **`benchmarks/regenerate_paper.sh`.** `set -euo pipefail`, idempotent (relies on the `exists` guard already in `run_all.py`), covers all five reproduction steps:
   - main iid sweep (5 methods × 4 problems × 5 seeds)
   - variant iid sweep (pygptreeo_B/_C/_poe, sklearn_gp_B, gpytorch_svgp_B, river_knn_B)
   - covariate-shift sweep
   - long-stream triple (rosenbrock + friedman + borehole)
   - `make_plots.py --iter-dir benchmarks/iterations/iteration_08`

### P1 — long-stream asymptote for friedman1_5d and borehole_8d

`benchmarks/data_long/` now contains three long-stream singletons. NRMSE trajectories at stream sizes 500–5000:

| problem          | n=500   | n=1500  | n=3000  | n=5000  |
| ---------------- | ------- | ------- | ------- | ------- |
| rosenbrock_2d    | 2.4e-5  | 1.2e-5  | 1.2e-5  | 1.2e-5  |
| friedman1_5d     | 2.7e-4  | 1.2e-4  | 1.6e-4  | 1.7e-4  |
| borehole_8d      | 8.8e-4  | 9.2e-4  | 1.1e-3  | 5.7e-4  |

- **rosenbrock_2d** flattens near 1.2×10⁻⁵ after ~1500 points (iter 06 finding).
- **friedman1_5d** plateaus by ~2000 points around 1.6×10⁻⁴.
- **borehole_8d** is noisier but reaches **5.7×10⁻⁴ by n=3500** and stays there — a ~1.7× improvement over the short-stream n=2000 median (9.65e-4).

The cross-dimensional picture: **pygptreeo's NRMSE flattens within 1500-3500 stream points on every problem we tested**. The 5000-pt budget does not buy further accuracy, consistent with the `alpha`-noise floor hypothesis from iter 06.

### P2 — deferred (per iter-08 review)

All: seeds 5/6 variance sanity, shift at n=5, per-seed scatter, MoE-vs-PoE bar, calibration convergence plot.

## Acceptance-criteria status

| criterion                                                                            | status                               |
| ------------------------------------------------------------------------------------ | ------------------------------------ |
| `benchmarks/README.md` mentions canonical snapshot, suffix convention, cell format   | ✅                                    |
| `benchmarks/regenerate_paper.sh` exists, executable, `set -euo pipefail`, idempotent | ✅                                    |
| `benchmarks/data_long/pygptreeo_A__{friedman,borehole}__seed0.npz` exist             | ✅                                    |
| All three long-stream runs have ≥ 10 checkpoints up to n=5000 and `badσ[-1]==0`      | ✅                                    |
| `iteration_08/` contains `paper_table.{md,tex}` and `headline.png`                   | ✅ (auto-emitted)                     |
| summary quotes reliability one-liner + long-stream trajectory                        | ✅                                    |
| Total iter-08 budget ≤ 35 min                                                        | ~15 min coding + ~10 min runtime ✅    |

## Press-go status

- One-command reproduction via `bash benchmarks/regenerate_paper.sh`.
- Paper numbers live at `benchmarks/iterations/iteration_07/paper_table.md` (pinned).
- Headline figure at `benchmarks/iterations/iteration_07/headline.png`.
- Seven-iteration chronology in `benchmarks/iterations/README.md`.

**The benchmark is paper-ready.**
