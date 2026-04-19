# Iteration 14 review — Emulator-assisted MCMC posterior fidelity

## Goal

Iteration 13 measures the speedup the trust-threshold deployment can
deliver, but with the input stream **fixed in advance**. In a real
GAMBIT-style global fit the chain's *next* point depends on the
*previous* accept/reject decision, which itself depends on whether the
likelihood at the current step came from the truth or from the
emulator. So the emulator's mistakes can in principle steer the chain
into regions it would otherwise have rejected — biasing the posterior.

This iteration measures that bias directly. We run two MCMC chains side
by side on the **target function as a likelihood**:

- A **reference chain** that always evaluates `f` truthfully.
- An **emulator-assisted chain** that uses the trust-threshold
  deployment from iter 13: `σ ≤ τ_σ` ⇒ accept the emulator's
  prediction `μ` as the (negative-log-)likelihood and skip the true
  evaluation; otherwise call `f` and update the emulator.

Both chains start at the same seed and use the same proposal kernel.
The difference between their posterior samples is then a direct
measurement of "how much fidelity did we lose for the speedup?"

## Plan

1. **Treat `f` as a (negative-log) likelihood.** Most of our targets
   are bounded; pick a temperature so the corresponding distribution
   has interesting structure on the unit cube. We use
   `log L(x) = -β · (f(x) - min_f) / scale_f` with `β = 1` and
   `scale_f = 0.1 · (max_f - min_f)` estimated from a 200-point Sobol
   pre-pass. This is purely for posterior-comparison purposes —
   nothing about the emulator itself changes.

2. **`benchmarks/mcmc_assisted.py`** — implementation of both chains.
   - Reference: standard random-walk Metropolis, σ_prop=0.05, n=20000.
   - Assisted: same proposal, but at each likelihood evaluation, query
     the emulator first; if `σ_emu ≤ τ_σ`, use `μ_emu` as the log-L
     and skip the true call. Otherwise call f, update the emulator,
     and use the truth.
   - Both chains record every sample (no thinning) plus the per-step
     decision flag.

3. **`benchmarks/run_assisted_mcmc.py`** — driver. One run per
   `(method, problem, τ_σ, seed)`. Pairs each assisted run with a
   single shared reference chain per `(problem, seed)`.

4. **`benchmarks/make_assisted_plots.py`** — three plots per problem:
   - `assisted_marginals.png` — overlay of 1-D marginals (reference
     vs each emulator method, one row per dimension at most 4 dims).
   - `assisted_corner.png` — 2-D scatter of the first two dims for
     reference and one chosen method.
   - `assisted_fidelity_vs_speedup.png` — Wasserstein-1 distance
     between reference and assisted chain marginals (averaged over
     dims) on the y-axis vs measured speedup on the x-axis. Each
     point is one (method, τ_σ).

5. **Sweep**.
   - 2 problems: `rosenbrock_2d`, `borehole_8d`.
   - 4 methods only (the ones that gave non-trivial speedup in iter
     13's diagnostic): `pygptreeo_A`, `pygptreeo_D`,
     `gpytorch_svgp_A`, `random_forest_A`.
   - 3 trust thresholds: `τ_σ ∈ {1e-3, 1e-2, 1e-1}` (relative).
   - 1 seed.
   - 1 reference chain per problem (shared across methods at a given
     seed).
   - 24 assisted runs + 2 reference chains.

## Out-of-scope

- No new kernels / methods.
- No comparison with off-the-shelf delayed-acceptance MCMC libraries.
- Multi-chain convergence diagnostics (R-hat) deferred — single chain
  is sufficient for the marginal-distance argument.

## Acceptance criteria

- `iteration_14/data/reference__<problem>__seed0.npz` and
  `iteration_14/data/assisted__<method>__<problem>__tau{τ}__seed0.npz`
  for the 24 + 2 grid.
- Three plots in `iteration_14/plots/`.
- `summary.md` reports: speedup (n_accepted_emu / n_total per method),
  Wasserstein-1 distance between assisted and reference marginals
  averaged over dims, and the same trust-threshold pareto as iter 13.
- Reliability still 100 % on `pygptreeo_*`.
