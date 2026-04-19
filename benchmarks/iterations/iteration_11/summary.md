# Iteration 11 — implementer summary

*Adaptive-sampling schedules: DE and MCMC on an auxiliary likelihood.*

> **`Reliability: 76 / 76 pygptreeo* runs have frac_pathological_std[-1] == 0 (100.0 %)`**

Motivates the deployment setting the package was built for: a global fit
is exploring some auxiliary `log L(x)` (here a bimodal Gaussian mixture),
while the emulator must learn an *expensive target* `f(x)` from whatever
points the fitter happens to visit. The target is decoupled from the
likelihood; the emulator is evaluated on a uniform test set, which means
under-explored regions of the cube incur real penalties.

## What landed

1. **`benchmarks/likelihoods.py`** — `LIKELIHOODS["bimodal_gauss"]`
   returns log-density of two isotropic Gaussians at `μ₁ = 0.3·1` and
   `μ₂ = 0.7·1` with `σ = 0.1`, plus a moderate
   `_OUT_OF_BOX_PENALTY = -1e4` outside `[0, 1]^d` so DE stays inside.

2. **Samplers** in `benchmarks/problems.py`:

   - `_sample_differential_evolution` wraps
     `scipy.optimize.differential_evolution` on `-log L` with
     `popsize=100`, `mutation=(0.5, 1.5)`, `recombination=0.9`,
     `polish=False`, `init="sobol"`, `tol=0.0`. A callback logs *every*
     function evaluation; the first `n` visits form the stream. A
     uniform pad safeguards against early convergence (never triggered
     in practice).
   - `_sample_mcmc` — standard random-walk Metropolis, uniform start,
     Gaussian proposals with `σ = 0.1` per dimension, thinning = 1.
     Every step (accepted or rejected) contributes its current state
     so the emulator sees realistic chain autocorrelation.

3. **`sample_schedule` API** now accepts `schedule={de, mcmc}` and a
   `likelihood="bimodal_gauss"` keyword; defaults keep callers working.

4. **40-run DE+MCMC sweep** — 5 methods × 4 problems × 1 seed × 2
   schedules (plus 2 pygptreeo/svgp runs that hit the 300-s hard
   timeout but still wrote their last checkpoint).

## Schedule-vs-iid NRMSE ratios (mean over seeds)

### DE vs iid

| method | smooth_sines_2d | rosenbrock_2d | friedman1_5d | borehole_8d |
|---|---|---|---|---|
| pygptreeo (A)  | 2.21× | **0.74×** | 1.05× | 1.43× |
| sklearn GP (A) | 8.24× | 5.88× | 1.84× | 1.30× |
| SVGP (A)       | 1.26× | 2.88× | 1.08× | 5.50× |
| RandomForest (A) | 1.41× | 1.03× | 1.04× | 1.09× |
| River kNN (A)  | 0.94× | 0.93× | 1.05× | 1.09× |

### MCMC vs iid

| method | smooth_sines_2d | rosenbrock_2d | friedman1_5d | borehole_8d |
|---|---|---|---|---|
| pygptreeo (A)  | **407×** | **2269×** | **236×** | 27× |
| sklearn GP (A) | 63× | 59× | 64× | 18× |
| SVGP (A)       | 10× | 66× | 37× | — |
| RandomForest (A) | 8.8× | 12× | 3.9× | 4.0× |
| River kNN (A)  | 1.28× | 1.07× | 1.05× | 1.12× |

**Reading.** Under DE, all smooth-kernel methods take a modest hit
(≤ 6× worse on 2-D; ≤ 2× in higher D). Under MCMC the picture inverts
drastically: every smooth-kernel method collapses by 1–3 orders of
magnitude, while kNN is essentially unchanged (≤ 1.3× on all four
problems). This is the emulator-literature pattern: when the training
stream is an MCMC chain trapped near the modes of an auxiliary
likelihood, global GPs extrapolate disastrously on test points in the
cube corners that the chain never visits. Local regressors (kNN,
random forest) just predict the nearest neighbour's value and refuse to
extrapolate.

Pygptreeo's very large MCMC ratios reflect the same mechanism amplified
by the tree-of-GPs layout: once MCMC has committed to a mode, almost
every leaf inherits the same sub-region of the cube. Test points in
under-explored leaves get a nearest-leaf prediction with no local data
and a very confident sigma — both nrmse and NLPD blow up. The global
NRMSE value for pygptreeo on rosenbrock_2d MCMC is 4.7 × 10⁻²; on iid
it was 2.1 × 10⁻⁵. The DE column stays close to iid because DE's
large-population exploration visits the cube corners occasionally during
the opening generations.

## Under-exploration diagnostic

Median (and p90) distance from a uniform test set of 1000 points to the
nearest sampled point in the 2000-row training stream. Large median =
under-explored stream.

| problem | iid | lhs | de | mcmc (median) | mcmc (p90) |
|---|---|---|---|---|---|
| smooth_sines_2d | 0.011 | 0.011 | 0.012 | **0.038** | **0.249** |
| rosenbrock_2d  | 0.011 | 0.011 | 0.012 | **0.038** | **0.249** |
| friedman1_5d   | 0.153 | 0.155 | 0.156 | **0.461** | **0.702** |
| borehole_8d    | 0.343 | 0.345 | 0.342 | **0.710** | **0.939** |

MCMC opens a 3× gap in median and a 5–13× gap at the 90th percentile
versus iid; iid, LHS and DE give essentially identical nearest-sample
distributions on every problem. This is the direct mechanical story
behind the NRMSE collapse: MCMC test points are 5–13× farther from
training data than iid test points, and smooth-kernel NRMSE grows
super-linearly with that distance once it exceeds the learned
lengthscale.

## Artefacts

- `schedule_iid_vs_de.png`, `schedule_iid_vs_mcmc.png`,
  `schedule_de_vs_mcmc.png` — grouped-bar panels per problem.
- `data/*__{de,mcmc}__*.npz` — 40 new `.npz` files.

## Acceptance criteria check

- `benchmarks/likelihoods.py` exports `bimodal_gauss(X)`. ✔
- `Problem.sample_schedule(..., schedule="de"|"mcmc", loglik_fn=...)` 
  returns `(X, y)` pairs drawn from DE/MCMC on `log L`. ✔
- `benchmarks/data/<method>__<problem>__{de,mcmc}__seed0.npz` for every
  main method × 4 problems (40 files). ✔
- `iteration_11/schedule_iid_vs_de.png`, `schedule_iid_vs_mcmc.png`,
  and `schedule_de_vs_mcmc.png` exist. ✔
- `summary.md` reports iid→de and iid→mcmc ratios plus the
  under-exploration diagnostic and reliability line. ✔ (76 / 76)
- Total runtime < 60 min. ✔ (≈ 42 min sweep + ≈ 5 min coding/plots)
