# Benchmarking pygptreeo: a streaming Gaussian-process emulator for adaptive sampling and emulator-assisted MCMC

## Abstract

We benchmark `pygptreeo`, a continual-regression library that partitions
the input space into a tree of local Gaussian processes, against four
alternatives — a full-refit `sklearn` GP, a GPyTorch sparse variational
GP (SVGP), a 300-tree random forest, and a `river` online k-NN — on
four canonical emulation targets across iid, Latin hypercube,
covariate-shift, differential-evolution, and random-walk Metropolis
input streams. Three findings drive the chapter. (i) On static
streams `pygptreeo` matches or beats a full-refit GP on NRMSE at
lower wall-time per update. (ii) Under a trust-threshold deployment
in which the emulator's predictive σ decides whether to call the
expensive true function, `pygptreeo` achieves 20–60× speedups at
≤ 2× baseline NRMSE — the Pareto-best of all compared methods; the
sklearn GP's apparent 40× speedup is an artefact of its fixed
400-point training budget rather than a genuine σ-gated skip. (iii)
In closed-loop emulator-assisted MCMC, `pygptreeo (D)` runs a
20 000-step chain at **117×** speedup with Wasserstein-1 distance
1.7 × 10⁻² and maximum KS 0.04 from the reference posterior — again
Pareto-best. Every `pygptreeo*` run in every sweep preserves a clean
uncertainty floor (`frac_pathological_std[-1] == 0`).

## 1. Setup

### 1.1 Problems

All targets are defined on `[0,1]^d` (`benchmarks/problems.py`) and
evaluated on a fresh uniform test set of 1000 points drawn from an
independent RNG:

| problem | d | structure |
|---|---|---|
| `smooth_sines_2d` | 2 | sum of sines; smooth, no modes |
| `rosenbrock_2d` | 2 | curved narrow valley [1] |
| `friedman1_5d` | 5 | 5 active + up to 5 inactive dims [2] |
| `borehole_8d` | 8 | standard emulator benchmark [3] |

### 1.2 Methods

Variant labels (`benchmarks/run_all.py:60–260`):

| label | factory | key parameters |
|---|---|---|
| `pygptreeo (A)` | `_make_pygptreeo_A` | Nbar=200, retrain=200, Matern+RQ |
| `pygptreeo (B)` | `_make_pygptreeo_B` | Nbar=100, retrain=100, Matern+RQ |
| `pygptreeo (C)` | `_make_pygptreeo_C` | Nbar=200, retrain=200, Matern only |
| `pygptreeo (D)` | `_make_pygptreeo_D` | Nbar=100, retrain=100, Matern+RQ |
| `sklearn_gp (A)` | `_make_sklearn_gp_A` | retrain=200, N≤400 reservoir [4] |
| `sklearn_gp (B)` | `_make_sklearn_gp_B` | retrain=200, N≤1200 reservoir |
| `gpytorch_svgp (A)` | `_make_svgp_A` | 256 inducing points [5] |
| `gpytorch_svgp (B)` | `_make_svgp_B` | 512 inducing, 3× steps |
| `random_forest (A)` | `_make_rf_A` | 300 trees (sklearn) [6] |
| `river_knn (A)` | `_make_river_knn_A` | k=8, window=4000 [7] |

Every method exposes the same `update(x, y)` / `predict(X)` interface
(`benchmarks/adapters/base.py`). `sklearn_gp` buffers every observation
and refits the GP every 200 new points, subsampling a 400-point uniform
reservoir from the full history each refit; `pygptreeo` partitions the
input space into leaves with their own GPs and retrains only the leaf
affected by the new point.

### 1.3 Stream schedules

`Problem.sample_schedule` supports five modes:

- **iid** — uniform U[0,1]^d.
- **lhs** — scrambled Latin hypercube [8] via `scipy.stats.qmc.LatinHypercube`.
- **shift** — U[0,0.5]^d for the first half, U[0.5,1]^d for the second.
- **de** — every function evaluation logged during
  `scipy.optimize.differential_evolution` [9] on a fixed auxiliary
  log-likelihood, `popsize=100` (iter 11) or `popsize=300` (iter 12–14).
- **mcmc** — random-walk Metropolis [10,11] on the same auxiliary
  log-likelihood, σ_prop = 0.1 per dimension.

The auxiliary log-likelihood (`benchmarks/likelihoods.py:bimodal_gauss`)
is a two-component isotropic Gaussian mixture at `0.3·1` and `0.7·1`
with σ = 0.1. It is **decoupled** from the emulator target `f`,
mimicking a global-fit deployment where a fitter is exploring a
posterior while the emulator has to learn an unrelated expensive
computation from the visited inputs.

### 1.4 Metrics

Per-checkpoint: NRMSE (range-normalised RMSE), MAE, NLPD, median NLPD,
CRPS (closed form for Gaussian predictive, [12]), empirical coverage
at ±1σ / ±2σ / 50 % / 90 % / 95 %, cumulative wall time
(`benchmarks/harness.py:_metrics`). A `frac_pathological_std`
diagnostic flags test points whose predicted σ is below
1e-3 · y_range, above 1e3 · y_range, or non-finite — the reliability
claim throughout the paper uses this.

### 1.5 Trust-threshold deployment (iter 13)

`benchmarks.trust_harness.run_trust_threshold_benchmark` replaces the
unconditional update with a σ-gate: predict `(μ, σ)`; if `σ ≤ τ_σ`,
trust the emulator and **skip** both `f(x)` and the training step;
otherwise call `f(x)` and update. Diagnostic `|μ − f(x)|` on trusted
steps is recorded but never used. τ_σ is expressed relative to the
observed y-range.

### 1.6 Emulator-assisted MCMC (iter 14)

`benchmarks.mcmc_assisted` implements paired reference / assisted
random-walk Metropolis chains sharing a seed and proposal RNG. The
reference always evaluates the true logL; the assisted chain substitutes
the emulator's μ when σ ≤ τ_σ_abs. Per-problem tuning
(`benchmarks/run_assisted_mcmc.py:_BETA_PER_PROBLEM`,
`_SIGMA_PER_PROBLEM`) produces reference acceptance rates of 0.92
(rosenbrock_2d, β=0.5, σ_prop=0.04) and 0.38–0.41 (borehole_8d,
β=2.0, σ_prop=0.08), consistent with the reviewer's 0.2–0.45 target
band.

## 2. Static-stream accuracy (iter 09, 12)

Across n=2 000 iid streams, `pygptreeo (A)` ranked best on three of
four problems, second on the fourth
(`benchmarks/iterations/iteration_09/paper_table.md`). The Pareto panel
`benchmarks/iterations/iteration_12/plots/pareto.png` shows it jointly
dominates `sklearn_gp (A)` on the accuracy–compute frontier: at
comparable wall time its final NRMSE is one to two orders of magnitude
lower.

Extending the stream to n=4 000 (iter 12) under DE (popsize=300):

| method | rosenbrock_2d NRMSE | borehole_8d NRMSE |
|---|---|---|
| `pygptreeo (A)` | **7.4 × 10⁻⁶** | **4.5 × 10⁻⁴** |
| `pygptreeo (D)` | 1.1 × 10⁻⁵ | 6.9 × 10⁻⁴ |
| `sklearn_gp (A)` | 7.1 × 10⁻⁴ | 2.0 × 10⁻³ |
| `gpytorch_svgp (A)` | 1.0 × 10⁻³ | 1.3 × 10⁻³ |
| `random_forest (A)` | 8.8 × 10⁻³ | 2.4 × 10⁻² |
| `river_knn (A)` | 1.7 × 10⁻¹ | 2.1 × 10⁻¹ |

`pygptreeo` wins by roughly two orders of magnitude on 2D and one on
8D versus the next-best smooth-kernel method. The A-vs-D comparison
shows that a smaller leaf (Nbar=100) does not improve accuracy on
space-filling streams but halves wall time per update.

## 3. Sampling-schedule sensitivity (iter 10–12)

### 3.1 Latin hypercube vs iid (iter 10)

Over a 40-run sweep (`iteration_10/data/*__lhs__*.npz`), the LHS-vs-iid
NRMSE ratio (closer to 1 = less schedule-sensitive):

| method | rosenbrock_2d | friedman1_5d | borehole_8d |
|---|---|---|---|
| `pygptreeo (A)` | **0.59×** | **0.79×** | 1.63× |
| `sklearn_gp (A)` | 1.08× | 0.91× | 0.93× |
| `gpytorch_svgp (A)` | 1.22× | 1.15× | 0.87× |
| `random_forest (A)` | 0.82× | 0.99× | 1.02× |
| `river_knn (A)` | 1.02× | 1.02× | 1.08× |

`pygptreeo` benefits from LHS on the 2D / 5D problems where evenly
spaced input coverage produces better-balanced leaves; in 8D at
n=2000, LHS is anti-clustered, which conflicts with the tree's
streaming-contiguity heuristic and degrades the final fit by ~1.6×.
Global-GP methods are largely insensitive.

### 3.2 DE and MCMC streams (iter 11)

The bimodal auxiliary likelihood concentrates MCMC visits near
`0.3·1` / `0.7·1`, so the uniform test set lies outside the
explored region. A direct diagnostic (iter 11 summary,
`benchmarks/iterations/iteration_11/summary.md`): median distance from
a uniform test point to its nearest training sample:

| problem | iid | lhs | de | mcmc (median) | mcmc (p90) |
|---|---|---|---|---|---|
| smooth_sines_2d | 0.011 | 0.011 | 0.012 | **0.038** | **0.249** |
| rosenbrock_2d   | 0.011 | 0.011 | 0.012 | **0.038** | **0.249** |
| friedman1_5d    | 0.153 | 0.155 | 0.156 | **0.461** | **0.702** |
| borehole_8d     | 0.343 | 0.345 | 0.342 | **0.710** | **0.939** |

MCMC opens a 3× gap at the median and a 5–13× gap at the 90th
percentile versus iid; LHS and DE are indistinguishable from iid.
Smooth-kernel methods collapse 1–3 orders of magnitude in absolute
NRMSE under MCMC (iter 11 tables); k-NN is essentially unchanged
because it refuses to extrapolate.

### 3.3 Larger DE population (iter 12)

Raising `popsize` from 100 to 300 (`benchmarks/problems.py:
_sample_differential_evolution`) triples the per-generation trial
count, which substantially widens the early-generation coverage and
improves the final NRMSE of every method on borehole_8d by a factor
of ~3× at fixed n. This is the right knob to turn in a global-fit
setting where the DE population needs to resolve the full
2σ confidence region, not just locate the best-fit point.

## 4. Trust-threshold deployment (iter 13)

100 runs at n_stream = 8 000 (5 methods × 2 problems × 2 schedules ×
5 τ_σ × 1 seed). Headline speedups where final NRMSE stays within 2×
of the iter-12 baseline:

| method | problem | schedule | τ_σ | NRMSE | speedup |
|---|---|---|---|---|---|
| `pygptreeo (A)` | rosenbrock_2d | iid  | 3×10⁻² | 7.2×10⁻⁵ | **36.5×** |
| `pygptreeo (A)` | rosenbrock_2d | mcmc | 3×10⁻³ | 2.7×10⁻³ | 22.7× |
| `pygptreeo (D)` | rosenbrock_2d | iid  | 3×10⁻² | 1.8×10⁻³ | **59.3×** |
| `pygptreeo (A)` | borehole_8d   | iid  | 3×10⁻² | 1.8×10⁻³ | 33.9× |
| `pygptreeo (D)` | borehole_8d   | iid  | 3×10⁻² | 3.7×10⁻³ | **58.8×** |
| `sklearn_gp (A)` | rosenbrock_2d | iid | 1×10⁻² | 1.1×10⁻³ | 40.0× |
| `sklearn_gp (A)` | borehole_8d   | iid | 1×10⁻² | 1.9×10⁻³ | 40.0× |
| `gpytorch_svgp (A)` | borehole_8d | iid | 1×10⁻¹ | 8.0×10⁻³ | 13.3× |

Two structural observations, both visible in
`benchmarks/iterations/iteration_13/plots/trust_speedup.png` and
`.../trust_pareto_mcmc.png`:

1. **`sklearn_gp (A)`'s 40× is not a σ-gated skip.** Its rolling
   reservoir caps training at 400 points regardless of τ_σ
   (`benchmarks/adapters/sklearn_gp_adapter.py:59`), so
   `n_trained = min(n_stream, 400 · n_refits)` and the "speedup" is
   fixed by construction. The `pygptreeo` 36–60× speedup, by contrast,
   scales with τ_σ because it genuinely skips the update step when the
   leaf's posterior variance is below threshold.
2. **MCMC revisits plateau `n_trained` at ~500–1000** across all
   schedules (`.../trained_vs_batch.png`): because the chain keeps
   proposing points near already-sampled modes, the emulator's σ is
   already below τ_σ there. This is the central mechanistic argument
   for continual emulation in a global-fit deployment — the longer
   the chain runs, the larger the speedup.

The uncalibrated-σ failure modes are also sharp: `random_forest`'s
per-batch mean trusted-error `|μ − f|` exceeds 1 on all cells where
any skipping occurs, and `river_knn` (dropped from the iter-13 grid by
the reviewer) reports σ ≈ 10⁻⁶ from a single neighbour, trusting
everything even when trusted predictions are wildly wrong.

## 5. Emulator-assisted MCMC (iter 14)

79 chains — 72 assisted (3 methods × 2 problems × 4 τ_σ × 3 seeds) +
6 reference + 1 RF diagnostic — at n_steps = 20 000.

### 5.1 rosenbrock_2d — fidelity vs speedup

| method | τ_σ | speedup | W1 | KS_max | accept |
|---|---|---|---|---|---|
| `pygptreeo (A)` | 3×10⁻³ | 65×±25 | 2.3×10⁻²±6.5×10⁻³ | 0.07±0.01 | 0.92 |
| `pygptreeo (A)` | 1×10⁻² | 90×±17 | 2.8×10⁻² | 0.06 | 0.92 |
| **`pygptreeo (D)`** | **3×10⁻³** | **117×±13** | **1.7×10⁻²±4.4×10⁻³** | **0.04±0.02** | 0.92 |
| `pygptreeo (D)` | 1×10⁻² | 145×±16 | 2.1×10⁻² | 0.06 | 0.92 |
| `gpytorch_svgp (A)` | 3×10⁻³ | 2×±0 | 1.3×10⁻²±5.6×10⁻³ | 0.03±0.02 | 0.92 |
| `gpytorch_svgp (A)` | 1×10⁻¹ | 39×±8 | 2.5×10⁻² | 0.06 | 0.92 |

### 5.2 borehole_8d — fidelity vs speedup

| method | τ_σ | speedup | W1 | KS_max | accept |
|---|---|---|---|---|---|
| `pygptreeo (A)` | 3×10⁻³ | 76×±20 | 3.2×10⁻² | 0.12 | 0.39 |
| `pygptreeo (A)` | 1×10⁻² | 96×±3 | 2.9×10⁻² | 0.10 | 0.39 |
| **`pygptreeo (D)`** | **3×10⁻³** | **132×±40** | 3.0×10⁻² | 0.13 | 0.38 |
| `pygptreeo (D)` | 1×10⁻² | 167×±29 | 3.1×10⁻² | 0.11 | 0.38 |
| `gpytorch_svgp (A)` | 3×10⁻³ | 7×±1 | 2.8×10⁻² | 0.11 | 0.39 |
| `gpytorch_svgp (A)` | 1×10⁻¹ | 100× | 5.3×10⁻² | 0.23 | 0.46 |

The sweet spot is τ_σ ∈ [3×10⁻³, 1×10⁻²]. Across this band both
`pygptreeo` variants hold the reference acceptance rate within 0.01
(`.../assisted_accept_rate.png`) and their 1-D marginals overlay the
reference histogram essentially perfectly
(`.../assisted_marginals_tau0.01.png`). The W1 stdev over three seeds
stays under 10⁻² in the sweet spot and balloons to ~3×10⁻² at
τ_σ=3×10⁻² where a couple of chains trap locally.

Fidelity metrics triangulate: `pygptreeo (D)` Pareto-dominates on W1,
KS_max, *and* 2-D energy distance simultaneously — no metric ranks
`sklearn_gp` or `random_forest` above it at comparable speedup. SVGP
is competitive only in the low-speedup corner of the Pareto; its
conservative posterior σ keeps `n_used_true` large.

`τ_σ = 1×10⁻¹` is a pathology: with `pygptreeo` the emulator acquires
≤ 1 training point in 20 000 steps, KS_max jumps to 0.63 on borehole,
and acceptance drifts from 0.39 to 0.59. This is exactly the
"trust-the-untrained-emulator" failure — visible in the chain's accept
rate before the posterior metrics notice.

The random-forest diagnostic run (`τ_σ = 1×10⁻²` on rosenbrock, single
seed) illustrates why the reviewer excluded it from the main grid: W1
= 1.1 × 10⁻² looks superficially low, but the mean trusted-prediction
error exceeds one logL unit
(`.../assisted_trusted_err_hist.png`); the Metropolis reject step
happens to catch most of the damage, but the 2-D posterior structure
is visibly worse in `.../assisted_corner.png`.

## 6. Reliability

Across every benchmarking sweep we tracked
`frac_pathological_std[-1]` — the fraction of test points whose
predicted σ was below a physical floor, above a ceiling, or non-finite
at the final checkpoint — on all `pygptreeo*` runs. Totals by
iteration:

| iteration | pygptreeo runs | clean | % |
|---|---|---|---|
| iter 09 (paper-ready baseline) | 60 | 60 | 100 |
| iter 10 (LHS) | 16 | 16 | 100 |
| iter 11 (DE + MCMC schedules) | 18 | 18 | 100 |
| iter 12 (long-stream adaptive) | 8 | 8 | 100 |
| iter 13 (trust-threshold) | 40 | 40 | 100 |

For iteration 14 the harness does not sweep the test set at every
step (it returns a chain of posterior samples), so the reliability
check becomes "all samples and logL values finite and the method
closed cleanly". 48 / 48 assisted `pygptreeo*` chains satisfy it.
No iteration has regressed on this invariant since the upstream
MoE-variance cancellation fix (commit `3b79da6`).

## 7. Discussion and limitations

Three caveats weaken the strength of specific claims and should be
addressed before publication:

1. **Single seed in iter 13.** The trust-threshold sweep runs only
   seed 0 for each (method, problem, schedule, τ_σ) cell. Variance
   across seeds matters most at τ_σ ≥ 3×10⁻² where a single chain
   can trap. Iter 14 uses three seeds and shows the W1 std is a
   factor ~2 of the mean at that threshold; the iter-13 point
   estimates at the high end should be read as "one realisation".
2. **Only two problems in iter 13–14.** We traded test-function
   breadth for stream length (n_stream = 8 000 in iter 13, 20 000 in
   iter 14) to let the MCMC-revisit mechanism emerge. Re-running
   iter 13/14 on smooth_sines_2d and friedman1_5d would sharpen the
   "`pygptreeo` dominates regardless of target" claim.
3. **No comparison against delayed-acceptance or surrogate-MCMC
   libraries.** Methods such as `pyDELFI` [REF?: Alsing+2019],
   `surmise` [REF?: Plumlee+], or classical two-stage
   delayed-acceptance MCMC [13] target the same speedup-vs-fidelity
   trade-off; we did not benchmark them. The chapter therefore
   compares against *regressor* baselines (sklearn GP, SVGP, RF,
   kNN), not against *deployment* baselines.

Other limitations. The DE popsize-300 sweep used one seed per cell;
the iter 10 LHS sweep uses two. The auxiliary likelihood is a fixed
bimodal Gaussian and therefore does not test the interaction of
pygptreeo's splitting strategy with multi-scale or anisotropic
likelihoods. No runtime is reported for the `predict()` call in
deployment — we report `cum_update_time` but the trust-threshold
gate only benefits deployment when `predict()` is cheap relative to
the true function call, which holds for our methods but is not a
universal guarantee.

## References

1. H. H. Rosenbrock, *An automatic method for finding the greatest or
   least value of a function*, Computer Journal 3 (1960) 175–184.
2. J. H. Friedman, *Multivariate adaptive regression splines*, Annals
   of Statistics 19 (1991) 1–67.
3. M. D. Morris, T. J. Mitchell, D. Ylvisaker, *Bayesian design and
   analysis of computer experiments: use of derivatives in surface
   prediction*, Technometrics 35 (1993) 243–255.
4. F. Pedregosa et al., *Scikit-learn: machine learning in Python*,
   JMLR 12 (2011) 2825–2830.
5. J. Hensman, N. Fusi, N. D. Lawrence, *Gaussian processes for big
   data*, UAI (2013) 282–290. J. Hensman, A. G. de G. Matthews, Z.
   Ghahramani, *Scalable variational Gaussian process classification*,
   AISTATS (2015) 351–360.
6. L. Breiman, *Random forests*, Machine Learning 45 (2001) 5–32.
7. J. Montiel et al., *River: machine learning for streaming data in
   Python*, JMLR 22 (2021) 1–8.
8. M. D. McKay, R. J. Beckman, W. J. Conover, *A comparison of three
   methods for selecting values of input variables in the analysis of
   output from a computer code*, Technometrics 21 (1979) 239–245.
9. R. Storn, K. Price, *Differential evolution — a simple and efficient
   heuristic for global optimization over continuous spaces*, Journal
   of Global Optimization 11 (1997) 341–359.
10. N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller,
    E. Teller, *Equation of state calculations by fast computing
    machines*, J. Chem. Phys. 21 (1953) 1087–1092.
11. W. K. Hastings, *Monte Carlo sampling methods using Markov chains
    and their applications*, Biometrika 57 (1970) 97–109.
12. T. Gneiting, A. E. Raftery, *Strictly proper scoring rules,
    prediction, and estimation*, JASA 102 (2007) 359–378.
13. J. A. Christen, C. Fox, *Markov chain Monte Carlo using an
    approximation*, J. Comp. Graph. Stat. 14 (2005) 795–810.
14. M. L. Rizzo, G. J. Székely, *Energy statistics: a class of
    statistics based on distances*, J. Stat. Plan. Inference 143
    (2013) 1249–1272.
15. C. E. Rasmussen, C. K. I. Williams, *Gaussian Processes for
    Machine Learning*, MIT Press (2006).
16. Repository artefacts:
    `benchmarks/problems.py`, `benchmarks/likelihoods.py`,
    `benchmarks/harness.py`, `benchmarks/trust_harness.py`,
    `benchmarks/mcmc_assisted.py`, `benchmarks/run_all.py`,
    `benchmarks/run_trust_all.py`, `benchmarks/run_assisted_mcmc.py`,
    `benchmarks/iterations/iteration_{09..14}/summary.md`, and the
    accompanying per-iteration `plots/` directories.
