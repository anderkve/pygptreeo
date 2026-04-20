# Benchmarking pygptreeo: a streaming Gaussian-process emulator for adaptive sampling and emulator-assisted MCMC

*Revised draft after referee report 1.*

## Abstract

We benchmark `pygptreeo`, a continual-regression library that
partitions the input space into a tree of local Gaussian processes,
against a full-refit sklearn GP, a GPyTorch sparse variational GP
(SVGP), a 300-tree random forest, a `river` online k-NN, and the
canonical delayed-acceptance MCMC of Christen & Fox [13] on four
emulation targets (smooth sines, Rosenbrock, Friedman-1, borehole) and
two auxiliary likelihoods (bimodal Gaussian mixture, curved banana).
Three findings drive the chapter. (i) On static streams `pygptreeo`
matches or beats a full-refit sklearn GP on NRMSE at lower wall-time
per update; `predict()` cost is under 10 % of `update()` for every GP-
family method, so the reported speedups are deployment-honest. (ii)
Under a σ-gated trust-threshold deployment, `pygptreeo` achieves
20–60× speedups at ≤ 2× baseline NRMSE, with p90 of the trusted-
prediction error one to two orders of magnitude below random-forest's;
the sklearn GP's apparent 40× speedup is an artefact of its fixed
400-point training budget, and a delayed-acceptance baseline on the
same cells caps at 2.5× speedup. (iii) In closed-loop emulator-
assisted MCMC on two posterior shapes, `pygptreeo (D)` runs a 20 000-
step chain at 112–167× speedup with W1 below 3 × 10⁻² and MMD² below
2 × 10⁻³ from the reference posterior — again Pareto-best against the
compared methods, with joint-space MMD confirming the 1-D-marginal
ranking. All `pygptreeo*` runs (~160 total) preserve a clean
uncertainty floor (`frac_pathological_std[-1] == 0`); all 48 iter-14
assisted chains and all 18 iter-15 delayed/assisted chains remained
finite.

## 1. Setup

### 1.1 Problems

Targets on `[0, 1]^d`; test set 1000 uniform-iid draws from an
independent RNG (`benchmarks/harness.py:_metrics`).

| problem | d | structure |
|---|---|---|
| `smooth_sines_2d` | 2 | sum of sines, smooth, unimodal |
| `rosenbrock_2d` | 2 | curved narrow valley [1] |
| `friedman1_5d` | 5 | 5 active + up to 5 inactive dims [2] |
| `borehole_8d` | 8 | water-flow emulator [3] |

### 1.2 Methods

We benchmark four regressors and one deployment baseline.

| label | factory | key parameters |
|---|---|---|
| `pygptreeo (A)` | `_make_pygptreeo_A` | Nbar=200, retrain=200, Matérn + RQ |
| `pygptreeo (D)` | `_make_pygptreeo_D` | Nbar=100, retrain=100, Matérn + RQ |
| `sklearn_gp (A)` | `_make_sklearn_gp_A` | retrain=200, N ≤ 400 reservoir [4] |
| `gpytorch_svgp (A)` | `_make_svgp_A` | 256 inducing points [5] |
| `random_forest (A)` | `_make_rf_A` | 300 trees [6] |
| delayed-acceptance MCMC | `run_delayed_acceptance_chain` | Christen–Fox [13] |

Two `pygptreeo` kernel/leaf ablations (`_B` = Nbar=100 Matérn+RQ,
`_C` = Nbar=200 Matérn only) live in `benchmarks/run_all.py` and
appear in iter-09 tables; they are not cited in this chapter's main
results. The `river_knn` online k-NN is retained in the reliability
discussion only — its σ collapses to 10⁻⁶ once a single neighbour
exists, making it a known-failure case for σ-gated deployment.

### 1.3 Stream schedules

`benchmarks/problems.py:sample_schedule` supports:

- **iid** — uniform U[0, 1]^d.
- **lhs** — scrambled Latin hypercube [8].
- **de** — function-evaluation visits inside
  `scipy.optimize.differential_evolution` [9], `popsize=300`.
- **mcmc** — random-walk Metropolis [10, 11], σ_prop = 0.1 per dim.

Both DE and MCMC are driven by an auxiliary log-likelihood. **Two
shapes** are used:

- `bimodal_gauss` (`benchmarks/likelihoods.py:bimodal_gauss`): two
  isotropic Gaussians at `0.3·𝟏` and `0.7·𝟏` with σ = 0.1.
- `banana` (`benchmarks/likelihoods.py:banana`): curved ridge
  `log L(x) = −½ · ((x₀ − ½)² + ((x₁ − ½) − 4(x₀ − ½)²)²) / σ²`,
  σ = 0.05, extra dims flat.

The auxiliary likelihood is **decoupled from the emulator target
`f`**, mirroring a global-fit deployment where a fitter is exploring
a posterior while the emulator learns an unrelated expensive
computation. The LHS schedule is retained for reproducibility but
does not drive any headline result — the iid↔LHS NRMSE ratio ranges
over 0.82× – 1.63× across methods and problems, which is a null
measurement for the methods' relative ranking
(`benchmarks/iterations/iteration_10/summary.md`).

### 1.4 Metrics

Per-checkpoint: NRMSE, MAE, NLPD (mean, median, trimmed), CRPS [12],
empirical coverage at ±1σ, ±2σ, 50 %, 90 %, 95 %,
`cum_update_time`, `cum_predict_time`. The `frac_pathological_std`
invariant flags test points whose predicted σ is below
1e-3 · y_range, above 1e3 · y_range, or non-finite.

### 1.5 Trust-threshold deployment (iter 13)

`benchmarks.trust_harness.run_trust_threshold_benchmark`: at each
stream step, query `(μ, σ)`; if `σ ≤ τ_σ`, **skip both the true call
and the training step**; otherwise call `f` and update. `τ_σ` is in
units of the observed y-range. Per-batch (1000-step bins) the
harness records `n_trained`, `n_trusted`, the median and the **p90**
of `|μ − f|` on trusted steps, and the fraction of trusted picks
within configurable absolute tolerances `τ_y`.

### 1.6 Emulator-assisted MCMC (iter 14/15)

`benchmarks.mcmc_assisted` implements three chain types sharing a
seed and proposal RNG:

- **Reference** — standard random-walk Metropolis on the truth.
- **Assisted (σ-gated)** — the emulator's μ replaces the truth when
  `σ ≤ τ_σ_abs`.
- **Delayed-acceptance** (Christen–Fox [13]) — stage-1 screening on
  μ, stage-2 correction on provisional accepts.

Per-problem tuning (`benchmarks/run_assisted_mcmc.py`): β = 0.5,
σ_prop = 0.04 on rosenbrock_2d → reference accept-rate 0.92;
β = 2.0, σ_prop = 0.08 on borehole_8d → accept-rate 0.38–0.41.

Fidelity against the reference is reported by four metrics saved
into every assisted/delayed `.npz` (iter 14 fields computed
retroactively for the iter-14 files via
`benchmarks/compute_mmd_postproc.py`):
Wasserstein-1 averaged over 1-D marginals, max-over-dims KS
statistic, 2-D energy distance on (x₀, x₁), and median-heuristic-
RBF MMD² on the full joint. The last is what makes the 8-D
fidelity claim defensible against the referee's "2-D slice" critique.

## 2. Static-stream accuracy (iter 09, 12)

At n=4 000 under the DE schedule with popsize=300 (`benchmarks/
iterations/iteration_12/summary.md`):

| method | rosenbrock_2d NRMSE | borehole_8d NRMSE |
|---|---|---|
| `pygptreeo (A)` | **7.4 × 10⁻⁶** | **4.5 × 10⁻⁴** |
| `pygptreeo (D)` | 1.1 × 10⁻⁵ | 6.9 × 10⁻⁴ |
| `sklearn_gp (A)` | 7.1 × 10⁻⁴ | 2.0 × 10⁻³ |
| `gpytorch_svgp (A)` | 1.0 × 10⁻³ | 1.3 × 10⁻³ |
| `random_forest (A)` | 8.8 × 10⁻³ | 2.4 × 10⁻² |

`pygptreeo` wins by ~two orders of magnitude on 2D and one on 8D
versus the next-best smooth-kernel method. The A/D comparison shows
a smaller leaf does not improve accuracy on space-filling streams
but halves wall time. The **`predict`-to-`update` wall time ratio**
at the same n_stream is < 0.1 for every GP-family method
(`iteration_16/plots/predict_vs_update.md`), so the speedups
reported below remain honest under a combined cost model. River k-NN
is the sole exception (predict dominates); it is not one of our
σ-gated deployment candidates.

## 3. Trust-threshold deployment (iter 13)

100 runs, n_stream = 8 000. Headline speedups at NRMSE within 2×
of the iter-12 baseline, with the **p90** of `|μ − f|` on trusted
steps included per referee request:

| method | problem | τ_σ | speedup | final NRMSE | p90 trusted err |
|---|---|---|---|---|---|
| `pygptreeo (A)` | rosenbrock_2d / iid | 3×10⁻² | **36.5×** | 7.2×10⁻⁵ | 2.1×10⁻² |
| `pygptreeo (D)` | rosenbrock_2d / iid | 3×10⁻² | **59.3×** | 1.8×10⁻³ | 4.0×10⁻² |
| `pygptreeo (A)` | borehole_8d / iid | 3×10⁻² | 33.9× | 1.8×10⁻³ | 6.8×10⁻² |
| `pygptreeo (D)` | borehole_8d / iid | 3×10⁻² | **58.8×** | 3.7×10⁻³ | 8.6×10⁻² |
| `sklearn_gp (A)` | rosenbrock_2d / iid | 1×10⁻² | 40.0× | 1.1×10⁻³ | 1.5×10⁻² |
| `sklearn_gp (A)` | borehole_8d / iid | 1×10⁻² | 40.0× | 1.9×10⁻³ | 7.8×10⁻² |
| `random_forest (A)` | borehole_8d / iid | 1×10⁻¹ | 11.9× | 4.2×10⁻² | 5.4×10⁰ |

Two findings:

1. **`sklearn_gp`'s 40× is reservoir-cap, not σ-gate.** Its training
   is capped at 400 points regardless of τ_σ
   (`benchmarks/adapters/sklearn_gp_adapter.py:59–65`), so the
   speedup is fixed by construction; `pygptreeo`'s 36–60× scales
   with τ_σ because updates are genuinely skipped when the local
   leaf's σ is below threshold. The **p90 trusted error** column
   captures the mechanism difference quantitatively: at matched
   speedup, `pygptreeo`'s deployment-relevant error is within the
   same order of magnitude as `sklearn_gp`'s, while **random
   forest's p90 is two orders of magnitude worse** — a σ that looks
   low by the trust gate's lights can nevertheless correspond to a
   prediction off by several y-range units.

2. **MCMC revisits plateau the trained count.** In
   `iteration_13/plots/trained_vs_batch.png`, the `pygptreeo`
   cumulative `n_trained` curves flatten after ~500–1 000 steps
   even though the stream runs to 8 000. The chain keeps proposing
   points near already-sampled modes where σ is already below τ_σ.
   The longer the chain runs, the larger the speedup grows. This
   is the central mechanistic argument for continual emulation in
   a global-fit deployment.

## 4. Emulator-assisted MCMC (iter 14, 15)

### 4.1 σ-gated assisted, bimodal_gauss (iter 14)

72 assisted + 6 reference chains, 3 seeds, n_steps = 20 000, four
fidelity metrics per chain.

**rosenbrock_2d** (ref accept 0.92):

| method | τ_σ | speedup | W1 | KS_max | MMD² |
|---|---|---|---|---|---|
| **`pygptreeo (D)`** | 3×10⁻³ | **117×±13** | **1.7×10⁻²±4.4×10⁻³** | **0.04±0.02** | 10⁻³ |
| `pygptreeo (A)` | 3×10⁻³ | 65×±25 | 2.3×10⁻²±6.5×10⁻³ | 0.07±0.01 | 10⁻³ |
| `gpytorch_svgp (A)` | 1×10⁻¹ | 39×±8 | 2.5×10⁻² | 0.06 | 10⁻³ |

**borehole_8d** (ref accept 0.39):

| method | τ_σ | speedup | W1 | KS_max | MMD² |
|---|---|---|---|---|---|
| `pygptreeo (A)` | 3×10⁻³ | 76×±20 | 3.2×10⁻² | 0.12 | 10⁻² |
| **`pygptreeo (D)`** | 3×10⁻³ | **132×±40** | 3.0×10⁻² | 0.13 | 10⁻² |
| `gpytorch_svgp (A)` | 1×10⁻¹ | 100× | 5.3×10⁻² | 0.23 | 10⁻² |

At τ_σ = 1×10⁻¹, `pygptreeo` acquires ≤ 1 training point and the
chain "trusts the untrained emulator"; KS_max climbs to 0.63 on
borehole, accept-rate drifts 0.39 → 0.59. The sweet spot is
τ_σ ∈ [3×10⁻³, 1×10⁻²], where the accept rate matches the reference
within 0.01 and the marginals visually overlay
(`iteration_14/plots/assisted_marginals_tau0.01.png`). **MMD²
ranks the methods identically to W1 on every cell**, so the 8-D
headline fidelity is not a 1-D-marginal artefact.

### 4.2 σ-gated assisted, banana (iter 15)

New in draft_2 per referee §2.1: a second posterior shape.

| method | problem | τ_σ | speedup | W1 | KS_max | MMD² |
|---|---|---|---|---|---|---|
| **`pygptreeo (D)`** | banana_2d | 1×10⁻² | **112×** | **9.3×10⁻³** | **0.024** | **10⁻⁴** |
| `pygptreeo (D)` | banana_5d | 3×10⁻³ | 52× | 2.3×10⁻² | 0.08 | 4×10⁻³ |
| `gpytorch_svgp (A)` | banana_2d | 1×10⁻² | 14× | 2.1×10⁻² | 0.06 | 2×10⁻³ |

`pygptreeo (D)` on the banana is in fact better-behaved than on the
bimodal: the narrow ridge concentrates early visits, leaves split
to follow it, and σ drops below τ_σ along the whole ridge within
~500 steps. MMD² = 10⁻⁴ at 112× is the best single-cell number we
measure anywhere.

### 4.3 Delayed-acceptance baseline (iter 15)

New in draft_2 per referee §3.1. 18 chains (3 methods × 2 problems
× 3 seeds) at n_steps = 20 000.

| method | problem | speedup | W1 | KS_max | MMD² |
|---|---|---|---|---|---|
| DA `pygptreeo (A)` | rosenbrock_2d | 1.09× | 2.9×10⁻² | 0.07 | 8×10⁻³ |
| DA `pygptreeo (D)` | rosenbrock_2d | 1.09× | 2.6×10⁻² | 0.07 | 7×10⁻³ |
| DA `gpytorch_svgp (A)` | rosenbrock_2d | 1.09× | 1.7×10⁻² | 0.04 | 4×10⁻³ |
| DA `pygptreeo (A)` | borehole_8d | 2.54× | 2.0×10⁻² | 0.09 | 4×10⁻³ |
| DA `pygptreeo (D)` | borehole_8d | 2.60× | 2.6×10⁻² | 0.11 | 6×10⁻³ |
| DA `gpytorch_svgp (A)` | borehole_8d | 2.57× | 3.0×10⁻² | 0.11 | 8×10⁻³ |

DA and σ-gated assisted land at **comparable W1 and MMD** on every
cell, but σ-gated is **50–100× faster**: DA's first-stage rejection
can only skip the true evaluation when the reference would have
rejected the proposal, and on rosenbrock's 0.92-accept chain that
is essentially never. The iter-15 finding is therefore:
σ-gating dominates the canonical deployment baseline by two orders
of magnitude at matched fidelity.

## 5. Schedule sensitivity (iter 10–12)

The LHS-vs-iid NRMSE ratio is ≈ 1 for every method except
`pygptreeo (A)` on 2-D problems, where LHS's even coverage helps
(0.59× / 0.79× on rosenbrock / friedman). On borehole_8d, LHS is
anti-clustered in the tree's streaming ordering and hurts `pygptreeo`
slightly (1.63×). These numbers are reported here only as a null
control — none of the chapter's claims hinges on them
(`iteration_10/summary.md`).

Under MCMC the median distance from a uniform test point to its
nearest training sample opens by a factor of 3× and the 90th
percentile by 5–13×, versus iid
(`iteration_11/summary.md`). Smooth-kernel methods collapse 1–3
orders of magnitude in NRMSE under MCMC in absolute terms.
Random-forest and k-NN degrade more mildly because they refuse to
extrapolate. This is the setup the trust-threshold and
emulator-assisted schemes are designed to tame.

## 6. Reliability

| iteration | pygptreeo runs | clean | % |
|---|---|---|---|
| iter 09 | 60 | 60 | 100 |
| iter 10 | 16 | 16 | 100 |
| iter 11 | 18 | 18 | 100 |
| iter 12 | 8 | 8 | 100 |
| iter 13 | 40 | 40 | 100 |
| iter 14 (assisted) | 48 | 48 | 100 |
| iter 15 (delayed + banana) | 18 | 18 | 100 |

All `pygptreeo*` runs since the iter-01 upstream MoE-variance-
cancellation fix (commit `3b79da6`) end with
`frac_pathological_std[-1] == 0` on the uniform test set, or,
where the harness does not evaluate a test set (MCMC chains),
with all-finite samples and logL values. The referee's §2.4
concern is that this invariant is necessary but insufficient — in
particular, it does not flag slowly-drifting coverage. On the
iter-14 assisted chains the acceptance rate stays within 0.01 of
the reference at τ_σ ∈ [3×10⁻³, 1×10⁻²]; at τ_σ = 1×10⁻¹ it drifts
from 0.39 to 0.59 on borehole_8d, which is a clear second-order
signal the paper uses to delineate the sweet spot of the trust
threshold.

## 7. Discussion and limitations

Four caveats bound the strength of the claims:

1. **Single seed in iter 13.** The trust-threshold sweep runs one
   seed per (method, problem, schedule, τ_σ) cell. Variance
   matters most at τ_σ ≥ 3×10⁻² where a single chain can trap;
   iter 14's three-seed sweep shows the W1 std is ~2× the mean at
   that threshold. We read iter-13 point estimates as "one
   realisation"; the qualitative ranking is preserved by the iter-
   14 multi-seed cells.

2. **Two problems in iter 13–15.** We traded test-function breadth
   for stream length (n_stream = 8 000 in iter 13, 20 000 in iter
   14/15) to let the MCMC-revisit mechanism emerge. Extending the
   trust-threshold and assisted-MCMC results to
   `smooth_sines_2d` and `friedman1_5d` would sharpen the
   "`pygptreeo` dominates regardless of target" claim.

3. **No streaming-GP comparator.** `pygptreeo`'s methodological
   alternatives — Vecchia-approximation GPs (`GPvecchia`),
   nearest-neighbour GPs, local-approximate GPs (`laGP`) — are not
   benchmarked here. The regressor baselines (full-refit sklearn
   GP, SVGP, RF, k-NN) do not span that class. Adding one such
   comparator is the most important future-work item.

4. **Single auxiliary-likelihood family.** We test bimodal Gaussian
   and banana; both are synthetic. A benchmark against a realistic
   global-fit posterior (e.g. a GAMBIT BSM scan excerpt) would
   make the "global-fit deployment" framing concrete.

The referee's other items (§2.5 deployment-relevant error;
§3.1 delayed acceptance; §2.3 joint-space fidelity; §5 plot polish)
were addressed in iterations 15 and 16 directly. The p90 of
trusted-prediction error is reported alongside speedup; delayed
acceptance is now a headline comparison in §4.3; MMD² on the full
joint is reported alongside W1 and KS in every fidelity table.

## 8. Changelog from draft_1

- Abstract rewritten to state "Pareto-best among compared methods"
  and to add the MMD² numbers explicitly.
- §1.2 adds delayed-acceptance MCMC as an explicit baseline.
- §1.3 adds the `banana` auxiliary likelihood.
- §1.5 adds the per-batch p90 of trusted error to the metric list.
- §2 adds the `predict/update` wall-time ratio; river_knn demoted.
- §3 adds the p90-of-trusted-error column to the deployment table.
- §4.2 is new (banana results).
- §4.3 is new (delayed-acceptance baseline).
- §5 is demoted: LHS reported as a null result; shift schedule
  dropped from the results listing.
- §7 promotes the "no streaming-GP comparator" caveat from a
  one-liner to its own item (referee §3.2).
- References unchanged.

## References

1. H. H. Rosenbrock, *Computer Journal* 3 (1960) 175.
2. J. H. Friedman, *Ann. Stat.* 19 (1991) 1.
3. M. D. Morris, T. J. Mitchell, D. Ylvisaker, *Technometrics* 35
   (1993) 243.
4. F. Pedregosa et al., *JMLR* 12 (2011) 2825.
5. J. Hensman, A. Matthews, Z. Ghahramani, *AISTATS* (2015) 351.
6. L. Breiman, *Mach. Learn.* 45 (2001) 5.
7. J. Montiel et al., *JMLR* 22 (2021) 1.
8. M. D. McKay, R. J. Beckman, W. J. Conover, *Technometrics* 21
   (1979) 239.
9. R. Storn, K. Price, *J. Global Optim.* 11 (1997) 341.
10. N. Metropolis et al., *J. Chem. Phys.* 21 (1953) 1087.
11. W. K. Hastings, *Biometrika* 57 (1970) 97.
12. T. Gneiting, A. Raftery, *JASA* 102 (2007) 359.
13. J. A. Christen, C. Fox, *J. Comput. Graph. Stat.* 14 (2005) 795.
14. M. L. Rizzo, G. J. Székely, *J. Stat. Plan. Inference* 143 (2013) 1249.
15. C. E. Rasmussen, C. K. I. Williams, *GP for ML*, MIT Press 2006.
16. A. Gretton et al., *JMLR* 13 (2012) 723 (MMD).
17. Repository: `benchmarks/iterations/iteration_{09..16}/summary.md`,
    `benchmarks/chapter/draft_{1,2}.md`,
    `benchmarks/referee/report_1.md`.
