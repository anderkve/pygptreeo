# Iteration 09 review

## Status after iter 08

Paper-ready. One-command reproduction (`regenerate_paper.sh`), pinned iter-07 snapshot, headline + paper_table + calibration + shift + variant ablations + long-stream, 58/58 pygptreeo reliability.

Iter 09 is an **endcap**, not a feature round. Two light items, then freeze.

## Prioritised punch-list

### P0 — ship these two

**P0.1. PoE third seed (upgrades MoE-vs-PoE to n=3).**

```
python benchmarks/run_all.py \
    --methods pygptreeo_poe \
    --problems rosenbrock_2d friedman1_5d \
    --seeds 2 \
    --n-stream 2000 --checkpoint-every 200 --n-test 1000 \
    --max-wall-time 300
```

Budget: ~3 min. Then regenerate plots + tables so `pygptreeo (PoE)` row in `paper_table.md` flips from `median (n=2)` to `mean ± 1.96·SE (n=3)` on both problems. Update `regenerate_paper.sh` step 2 to use `--seeds 0 1 2` for `pygptreeo_poe`.

**P0.2. `run_summary.txt` generator in `make_plots.main()`.**

Plain-text, grep-able, diff-able. Written to `<iter-dir>/run_summary.txt`. Contents:

- `iter_dir:` + UTC timestamp
- exact reliability one-liner
- `method x problem` grid of mean NRMSE (median when n<3): `pygptreeo_A rosenbrock_2d 2.09e-05 n=5`
- commit SHA of HEAD (via `subprocess.check_output(["git","rev-parse","HEAD"])`, try/except for tarball runs)

Bonus in same commit: add one line to the README Reliability section citing the iter-01 fix commit SHA (answers the "when was the fix?" reviewer question at zero cost).

### P1 — defer / drop

- scaling.png overlay (decorative)
- LightGBM / XGBoost (new method family; out of scope)
- headline y-axis tweak (already legible)
- back-editing iter-08's summary.md (chronology lives in `iterations/README.md`)

## Out-of-scope

- New methods / new problems / new figures
- Any refit of iter-07's pinned numbers

## Acceptance criteria

- `benchmarks/data/pygptreeo_poe__{rosenbrock_2d,friedman1_5d}__seed2.npz` exist with `frac_pathological_std[-1]==0`.
- `iteration_09/paper_table.md` PoE row shows `(n=3)` on both problems.
- `iteration_09/run_summary.txt` exists; reliability line + HEAD commit SHA + method×problem grid.
- `regenerate_paper.sh` step 2 updated.
- README cites iter-01 upstream-fix SHA in Reliability section.
- `iteration_09/summary.md` is short (≤ 40 lines).
- Total budget ≤ 25 min.

## Commit message

```
iter 09: PoE n=3 + run_summary.txt — last paper-ready iteration

- Third seed for pygptreeo_poe on rosenbrock_2d and friedman1_5d;
  MoE-vs-PoE comparison now n=3 (was n=2).
- make_plots.py now emits run_summary.txt alongside paper_table.md:
  grep-able method × problem NRMSE grid, reliability line, HEAD SHA.
- regenerate_paper.sh: pygptreeo_poe sweep uses seeds 0 1 2.
- README: cite iter-01 upstream-fix commit SHA in Reliability section.

Benchmark is frozen after this commit; canonical paper snapshot
remains iteration_07/.
```
