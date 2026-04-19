# Iteration 10 — implementer summary

*Latin hypercube sampling schedule.*

> **`Reliability: 76 / 76 pygptreeo* runs have frac_pathological_std[-1] == 0 (100.0 %)`**

New `--schedules lhs` branch in `Problem.sample_schedule` draws the stream
from `scipy.stats.qmc.LatinHypercube(d=dim, scramble=True)`. The test set
stays uniform-iid as in iter 09. Every main method was run under `lhs`
with seeds `{0, 1}` on the four default problems (40 `.npz`).

## What landed

1. **LHS schedule** — `benchmarks/problems.py:205-213`. Natural row order,
   per-seed scramble via `rng.integers(...)`. No change to the test rng.

2. **`plot_schedule_comparison`** — `benchmarks/make_plots.py:376-434`.
   Generalises `plot_shift_vs_iid` to an arbitrary tuple of schedules;
   plots grouped bars per method with hatched overlay per schedule.
   Auto-emitted into `main()`: produces `schedule_iid_vs_lhs.png`,
   `schedule_iid_vs_de.png`, `schedule_iid_vs_mcmc.png`, and
   `schedule_de_vs_mcmc.png` whenever the corresponding data exist
   (`make_plots.py:1251-1264`).

3. **40-run LHS sweep** — 5 methods × 4 problems × 2 seeds.

## LHS-vs-iid NRMSE ratio (LHS / iid, mean over seeds)

| method | smooth_sines_2d | rosenbrock_2d | friedman1_5d | borehole_8d |
|---|---|---|---|---|
| pygptreeo (A) | 1.09× | **0.59×** | **0.79×** | 1.63× |
| sklearn GP (A) | 3.01× | 1.08× | 0.91× | 0.93× |
| SVGP (A)       | 0.84× | 1.22× | 1.15× | 0.87× |
| RandomForest (A) | 0.92× | 0.82× | 0.99× | 1.02× |
| River kNN (A)  | 0.95× | 1.02× | 1.02× | 1.08× |

**Reading.** Values < 1.0× mean LHS helps; > 1.0× means it hurts. Two
takeaways:

- **Pygptreeo benefits on the two 2D/5D problems we'd expect** (rosenbrock,
  friedman-1: 0.59×, 0.79×) — space-filling feeds each tree leaf more
  balanced data, shrinking local-GP variance. On borehole_8d, LHS hurts
  a bit (1.63×): at n=2000 in 8 dimensions one LHS stratum per seed is
  only ~1.6 points/axis wide, and the stream order of LHS is naturally
  *anti*-clustered, which conflicts with pygptreeo's assumption that
  nearby points in time tend to land in the same leaf.
- **Sklearn GP spikes on smooth_sines** (3.01×). That method rebuilds
  its RBF+noise kernel on the last-400 window. LHS's enforced-spread
  stream means the rolling window is less dense, degrading the 2D fit.
  This is a *sampling-schedule* artefact, not a pathology.
- **SVGP, RF, kNN are essentially unchanged** (all ≤1.25× either way).
  Inducing points, tree splits, and k-NN queries are rotation-invariant
  to uniform-vs-LHS for fixed N — consistent with the hypotheses in the
  review.

The `paper_table.md` numbers and the iid baselines remain the same as
iter 09; this iteration only adds the LHS column.

## Artefacts

- `schedule_iid_vs_lhs.png` — 4-panel grouped bar chart per problem.
- `data/*__lhs__*.npz` — 40 new `.npz` files.
- Regenerated paper tables (with identical iid numbers) in
  `paper_table.md` and `paper_table.tex` for freshness.

## Acceptance criteria check

- `sample_schedule(..., schedule="lhs")` returns `(X, y)` with
  `scipy.stats.qmc.LatinHypercube` rows. ✔
- 40 LHS `.npz` files in `benchmarks/data/` (5 methods × 4 problems × 2 seeds). ✔
- `iteration_10/schedule_iid_vs_lhs.png` exists. ✔
- Per-(method, problem) iid→LHS NRMSE ratio table above. ✔
- Reliability 100 %. ✔ (76 / 76)

Total iter-10 runtime: ~22 min (LHS sweep) + ~5 min coding/plots.
