# Iteration 10 review — Latin hypercube sampling

## Goal

Add a Latin hypercube sampling (LHS) schedule for the stream of training points. Test every main method on every problem under LHS, alongside the existing iid schedule. Shows whether pygptreeo's advantage persists when the stream is space-filling by construction rather than random.

## Plan

1. **`benchmarks/problems.py` / `Problem.sample_schedule`.** New branch:
   `schedule="lhs"` → draw `n` points from `scipy.stats.qmc.LatinHypercube(d=self.dim, scramble=True, seed=rng.integers(...))`. Keep the natural LHS row order (no extra shuffle). Test set stays uniform-iid from the independent test rng.

2. **`benchmarks/run_all.py`.** No code change — the driver already accepts `--schedules` with arbitrary values. Simply invoke `--schedules lhs` on the main methods on the 4 default problems × 2 seeds. ~5 methods × 4 problems × 2 seeds = 40 runs.

3. **`benchmarks/make_plots.py`.** Add `plot_schedule_comparison(results, problems, out_path, schedules=("iid","lhs"))`: grouped bar chart per-problem, bars = methods, two bars per method (iid solid / lhs hatched), log-y final NRMSE. Wire into `main()`.

4. **Plot everything into `iteration_10/`** as usual.

## Hypotheses worth noting

- **Pygptreeo under LHS should be ≥ iid** — LHS gives more even coverage so per-leaf data is more balanced. Tree splits along quantile boundaries might benefit.
- **SVGP under LHS might also improve** — inducing points init from a random subsample; LHS means the subsample is already space-filling.
- **Random forest indifferent** — trees are rotation-invariant to uniform vs LHS for fixed N.
- **river_knn roughly unchanged** — kNN queries don't care about the stream history's spatial distribution.

## Out-of-scope

- No change to methods or hyperparameters.
- No change to test-set sampling (stays uniform iid).
- No new methods, no new problems.
- Don't re-run existing iid `.npz`.

## Acceptance criteria

- `benchmarks/data/<method>__<problem>__lhs__seed{0,1}.npz` exist for at least `pygptreeo_A`, `sklearn_gp_A`, `gpytorch_svgp_A`, `random_forest_A`, `river_knn_A` × 4 problems × 2 seeds (≈ 40 files).
- `iteration_10/plot_schedule_comparison.png` (or `lhs_vs_iid.png`) shows the 4-panel figure.
- `summary.md` reports, per (method, problem), the iid→LHS NRMSE ratio and whether pygptreeo's win ratio grows / shrinks / is unchanged.
- Reliability still 100%.
- Total runtime < 40 min.
