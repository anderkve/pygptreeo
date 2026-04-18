# Iteration 08 review

## Status after iter 07

Iter 07 landed the paper-ready snapshot: `paper_table.md`/`.tex` auto-emit into each iteration directory, 58/58 pygptreeo reliability is asserted, and the sklearn_gp_A × borehole_8d empty bar is closed. `pygptreeo_A` main cells carry n=5 with `± 1.96·SE`; headline NRMSE CIs for pygptreeo_A vs SVGP_A are non-overlapping on rosenbrock_2d (63× ratio) and borehole_8d.

The remaining gaps are about making the paper-ready artefact **reproducible and pinned** — not about adding new science.

## Prioritised punch-list

### P0 — must land

1. **Reproducibility README rewrite** (`benchmarks/README.md`).
   Current README still describes iter-01 budgets, lists bare legacy names, no mention of `paper_table.md`, the `_A/_B/_C/_poe` convention, per-iteration directories, `data_long/`, or which iteration is authoritative. Rewrite to include:
   - (a) canonical-snapshot pointer ("paper numbers live in `iterations/iteration_07/paper_table.{md,tex}`");
   - (b) single-command reproduction recipe matching the methods × problems × seeds grid used for the paper;
   - (c) per-method wall-time hints;
   - (d) paragraph on the cell format (`mean ± 1.96·SE (n=k)` / `median (n=k)` / bare / em-dash);
   - (e) short "what is in `data_long/`" note.
   No new runs.

2. **`benchmarks/regenerate_paper.sh`.**
   Thin shell script codifying the exact command sequence for the paper snapshot:
   (i) `run_all.py` over the 7 main methods × 4 problems × seeds 0-4,
   (ii) the shift sweep,
   (iii) `make_plots.py --iter-dir benchmarks/iterations/iteration_08`.
   `set -euo pipefail`; idempotent thanks to the `exists` guard. Referenced from the new README.

### P1 — should land

3. **Long-stream extension for friedman1_5d and borehole_8d.**
   `--n-stream 5000 --checkpoint-every 500 --out-dir benchmarks/data_long`, one seed each. Rosenbrock already has one long-stream trajectory; extending to friedman and borehole makes the "NRMSE keeps descending past n=3000" a cross-dimensional claim. ~12 min runtime, zero coding.

### P2 — defer

- **Seeds 5/6 variance sanity** — CI half-widths shrink by ~15 %; marginal.
- **Pygptreeo_A × shift × 5 seeds** — shift is a stress test, not headline.
- **Per-seed NRMSE-trajectory figure** — decorative; defer.
- **MoE-vs-PoE bar chart** — PoE is n=2; underpowered.
- **Calibration-vs-n-seeds convergence plot** — load bearing is already carried by `paper_table.md`.

## Out-of-scope

- Do NOT modify `pygptreeo/gptree.py`, `pygptreeo/gpnode.py`, `benchmarks/harness.py`, `benchmarks/adapters/*`, or plotting defaults.
- Do NOT add new methods or problems; do NOT change hyperparameters, kernel spec, `theta`, `sigma_rel`, or std-floor / cap.
- Do NOT re-run existing `.npz` files. Keep `exists` guard; no `--force`.
- Do NOT edit `iterations/iteration_07/` — it is the pinned canonical snapshot.
- Do NOT change `headline.png`, `paper_table.md`, `paper_table.tex` layouts.

## Acceptance criteria

- `benchmarks/README.md` mentions: `iterations/iteration_07/paper_table.md` as canonical, the `_A/_B/_C/_poe` suffix convention, the cell-format key, `data_long/`, and `regenerate_paper.sh`.
- `benchmarks/regenerate_paper.sh` exists, is executable, uses `set -euo pipefail`, and dry-run from a fresh `data/` would reproduce every `.npz` cited in `paper_table.md`.
- `benchmarks/data_long/pygptreeo_A__friedman1_5d__seed0.npz` and `benchmarks/data_long/pygptreeo_A__borehole_8d__seed0.npz` exist, both with checkpoints up to ≥ 5000 and `frac_pathological_std[-1] == 0`.
- `iterations/iteration_08/summary.md` quotes the reliability one-liner, shows the three-problem long-stream NRMSE trajectory (rosenbrock existing + two new), and links to the rewritten README plus `regenerate_paper.sh`.
- `iterations/iteration_08/` contains the usual auto-emitted `paper_table.md`/`.tex` and `headline.png`.
- Total iter-08 budget ≤ 35 min wall-clock.
