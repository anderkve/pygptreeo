# Iteration 09 — implementer summary

*Endcap. Paper-ready.*

> **`Reliability: 60 / 60 pygptreeo* runs have frac_pathological_std[-1] == 0 (100.0 %)`**

Two runs added, both clean. Reliability string auto-regenerates — see `run_summary.txt` for the per-iteration copy with commit SHA.

## What landed

1. **PoE n=3.** `pygptreeo_poe__{rosenbrock_2d,friedman1_5d}__seed2.npz` produced. The MoE-vs-PoE block in `iteration_09/paper_table.md` now reads:

   | method | rosenbrock_2d | friedman1_5d |
   | --- | --- | --- |
   | pygptreeo (A) | **2.09e-05 ± 8.1e-06 (n=5)** | **3.85e-04 ± 1.9e-04 (n=5)** |
   | pygptreeo (PoE) | 2.48e-05 ± 1.2e-05 (n=3) | 3.33e-04 ± 2.5e-04 (n=3) |

   MoE and PoE are statistically indistinguishable on both problems at n=3; the 1.96·SE intervals overlap heavily. The paper can now report this finding at n=3 with honest uncertainty rather than as an n=2 footnote.

2. **`run_summary.txt` generator** in `make_plots.py:write_run_summary`. Writes a grep-able, diff-able plain-text dump of every (method, problem) cell plus the reliability one-liner and the HEAD commit SHA into `<iter-dir>/run_summary.txt`. Hooked into `main()` next to `write_paper_tables`. Future regression-checking across iterations is now a `diff` away.

3. **`regenerate_paper.sh`** step 2 updated: pygptreeo_poe now sweeps seeds `0 1 2`.

4. **README Reliability section** cites the iter-01 upstream-fix commit `3b79da6` — answers the "when was the fix?" reviewer question.

## Acceptance

All criteria met. Total iter-09 runtime: ~3 min parallel runtime + ~15 min coding.

## Status

The benchmark is **frozen** after this iteration. The pinned canonical paper snapshot remains **`benchmarks/iterations/iteration_07/`**. Iteration 09 is the last paper-ready artefact; any future work goes in a new branch.
