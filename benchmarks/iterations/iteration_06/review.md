# Iteration 06 review

*Written by the critical reviewer. Reference: iter-05 summary + figures.*

## What iter 05 left on the table

Iter 05 closed the four biggest credibility gaps (borehole populated, shift over `_A` variants, sklearn_gp_B rescue on friedman1, headline + wilcoxon_variants figures, calibration table). Remaining gaps are "paper statistical honesty" rather than new science:

1. **Three-seed thin spots.** `pygptreeo_A` on rosenbrock, friedman1, borehole has only 3 seeds. With n=3 the IQR bars on `headline.png` are dominated by single-run variation — reviewers will ask for mean ± 1.96·SE. Empirically iid pygptreeo runs take 75–200 s, so 5 seeds is affordable.
2. **Shift limited to `_A`.** If `_C` (Matern-only) degrades under shift by roughly the same factor as `_A`, the "locality, not kernel" framing extends to the shift setting. 8 runs cover this.
3. **`pygptreeo_C` cov-at-0.95 = 0.78 on rosenbrock_2d vs 0.91 for `_A`** (iter-05 p.4). With n=3 this is 1-sigma at best. Two extra seeds will either confirm noise or flag the RQ kernel as a calibration contributor.
4. **No asymptote evidence.** All runs stop at n_stream ≤ 3000. One seed at 5000 points on rosenbrock_2d shows whether NRMSE keeps descending.
5. **Aggregation ablation absent.** pygptreeo supports PoE. `pygptreeo_poe` lets the paper assert "MoE ≥ PoE empirically". Cheap (4 runs) and the adapter already supports it trivially (run_all.py has a `_make_pygptreeo_poe` factory).
6. **`headline.png` legend** currently eats the bottom margin (ncol=7 single row). Compress to 2 rows.

Reliability one-liner is already wired in `main()` — just verify it runs and quote the number in `summary.md`.

## Prioritised punch-list

### P0 — must land

1. **Extra seeds on `pygptreeo_A`.**
   `--methods pygptreeo_A --problems rosenbrock_2d friedman1_5d borehole_8d --seeds 3 4 --n-stream 2000 --max-wall-time 600`. 6 runs. Gets `_A` to 5 seeds on three problems.

2. **Extra seeds on `gpytorch_svgp_A`.**
   `--methods gpytorch_svgp_A --problems rosenbrock_2d friedman1_5d --seeds 3 4`. 4 runs.

3. **Shift for `_B/_C`.**
   `--methods pygptreeo_B pygptreeo_C --problems rosenbrock_2d friedman1_5d --schedules shift --seeds 0 1 --n-stream 2000`. 8 runs. Summary must state whether the _B/_C shift-to-iid ratio is within 2× of `_A`.

4. **`pygptreeo_C` extra rosenbrock seeds.**
   `--methods pygptreeo_C --problems rosenbrock_2d --seeds 3 4`. 2 runs. Resolves whether the 0.78 coverage anomaly is noise.

### P1 — should land

5. **Long-stream asymptote.**
   `--methods pygptreeo_A --problems rosenbrock_2d --seeds 0 --n-stream 5000 --checkpoint-every 500`. Save under a different out-dir (`--out-dir benchmarks/data_long`) to avoid overwriting the short-stream seed-0 file. One extra panel in `summary.md` showing NRMSE trajectory to 5000.

6. **`headline.png` legend polish.** `plot_headline`'s `fig.legend(..., ncol=min(7, len(labels)) ...)` → `ncol=4` for a 2-row layout.

### P2 — defer

7. **`pygptreeo_poe` variant.** Factory already in `run_all.py` (iter 06 prep work). Run `--methods pygptreeo_poe --problems rosenbrock_2d friedman1_5d --seeds 0 1`. 4 runs, ~6 min.

8. Cite the reliability one-liner in `summary.md`.

## Out-of-scope

- Do NOT modify `pygptreeo/gptree.py`, `pygptreeo/gpnode.py`, `benchmarks/harness.py`, or `benchmarks/adapters/base.py`.
- Do NOT add new problems. `pygptreeo_poe` is the only new method.
- Do NOT change `theta`, `sigma_rel`, `max_n_pred_leaves`, kernel spec of `_A/_B/_C`, or any std-floor / cap values.
- Do NOT re-run existing `.npz` files (keep `exists` guard; no `--force`).
- Do NOT change `--n-stream > 3000` for anything except the single long-stream run (out-dir-separated).
- Do NOT touch `comparison.png` / `calibration.png` / `summary.png` / `pareto.png` / `wilcoxon_variants.png` panel layouts — only `headline.png` legend.

## Acceptance criteria

- `benchmarks/data/pygptreeo_A__{rosenbrock_2d,friedman1_5d,borehole_8d}__seed{3,4}.npz` exist (6 files), each with `frac_pathological_std[-1]==0`.
- `benchmarks/data/gpytorch_svgp_A__{rosenbrock_2d,friedman1_5d}__seed{3,4}.npz` exist (4 files).
- `benchmarks/data/pygptreeo_{B,C}__{rosenbrock_2d,friedman1_5d}__shift__seed{0,1}.npz` exist (8 files).
- `benchmarks/data/pygptreeo_C__rosenbrock_2d__seed{3,4}.npz` exist (2 files).
- Long-stream artefact exists (even in a separate out-dir) and is referenced in `summary.md`.
- `summary.md` explicitly answers: (a) does `_C` cov-at-0.95 on rosenbrock stay at ≈0.78 with 5 seeds? (b) is `_B/_C` shift degradation within 2× of `_A`? (c) asymptotic NRMSE at n=5000 vs n=3000?
- `summary.md` cites the reliability one-liner verbatim.
- `iteration_06/headline.png` has a 2-row legend.
- Total iter-06 runtime < 90 min.
