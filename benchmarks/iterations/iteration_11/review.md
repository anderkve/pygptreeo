# Iteration 11 review — Adaptive sampling (DE + MCMC)

## Goal

Benchmark the emulators under **realistic "global-fit" streams**: the emulator is learning some expensive target function `f(x)`, but the points `x` it sees are not uniform or LHS. Instead they are produced by a sampler exploring a *separate* auxiliary log-likelihood `log L(x)` that lives on the same input space. This is the deployment setting that motivates continual emulation in the first place — the fit is exploring parameter space for reasons unrelated to `f`, and the emulator has to learn `f` from whatever the fitter happens to visit.

Two samplers, chosen to bracket the "exploration-vs-exploitation" spectrum at the slow end of both:

- **Differential evolution with a large population.** `scipy.optimize.differential_evolution`, `popsize=100`, early convergence disabled. Produces structured, population-based exploration — each generation of 100·d trial points is broadly scattered, narrowing as iterations proceed.
- **Random-walk Metropolis MCMC.** Standard Metropolis with Gaussian proposals, step size tuned so the acceptance rate is roughly 0.2–0.4. Produces correlated, chain-ordered visits that concentrate around modes of `log L`.

The auxiliary log-likelihood is decoupled from `f`: a bimodal Gaussian mixture in the unit cube at `μ₁ = 0.3·𝟏` and `μ₂ = 0.7·𝟏` with `σ = 0.1`. This shape is chosen so that:
- both samplers actually have to *move*,
- the unit cube corners and centre are under-explored,
- the target functions (Rosenbrock, Friedman-1, borehole) have support away from those modes as well as near them — i.e. the emulator is not always evaluated where it has data.

## Plan

1. **New `benchmarks/likelihoods.py`.** Registry of auxiliary log-likelihoods. Start with `bimodal_gauss` (bimodal Gaussian mixture). Defined on `[0, 1]^d` for any `d`.

2. **Samplers in `benchmarks/problems.py`.** New schedules `de` and `mcmc`:
   - `de`: wraps `scipy.optimize.differential_evolution` with a callback that logs every function evaluation. Runs until the log has ≥ `n` entries, then truncates to the first `n`. `popsize=100`, `mutation=(0.5, 1.5)`, `recombination=0.9`, `polish=False`. Starts from a uniform population on `[0, 1]^d`.
   - `mcmc`: random-walk Metropolis from a uniform start, Gaussian proposals with `σ = 0.1` in every dim, standard accept/reject. Every step (accepted or not) gets logged to the stream. Thinning = 1.

3. **`benchmarks/run_all.py`.** Thread a `--likelihood` CLI argument (default `bimodal_gauss`). `sample_schedule` grows to accept the likelihood name.

4. **Plots.** `plot_schedule_comparison` from iter 10 already handles arbitrary schedule lists — call it for `("iid", "de", "mcmc")` to get a three-bar panel per problem.

5. **`summary.md` must include**:
   - per-problem iid→de and iid→mcmc NRMSE ratios per method,
   - a short diagnostic showing the emulator's test-set residuals *correlate* with distance to the nearest sampled point (more than under iid), confirming the sampler really is an under-explorer,
   - reliability count (must stay 100 %).

## Out-of-scope

- No new methods.
- No change to method hyperparameters.
- No alternative likelihoods for this iteration; keep only `bimodal_gauss`.
- No adaptive scheduling where the sampler uses the emulator's uncertainty to decide what to visit next — that's a *very* different experiment.
- Don't re-run the existing iid / lhs `.npz` files.

## Acceptance criteria

- `benchmarks/likelihoods.py` exports `bimodal_gauss(X)`.
- `Problem.sample_schedule(..., schedule="de", loglik_fn=...)` returns an `(X, y)` pair of `n` rows where the `X` visits are those produced by DE on `log L`. Same for `schedule="mcmc"`.
- `benchmarks/data/<method>__<problem>__de__seed0.npz` and `__mcmc__seed0.npz` exist for every main method × 4 problems (40 files total).
- `iteration_11/schedule_iid_vs_de.png`, `schedule_iid_vs_mcmc.png`, and `schedule_de_vs_mcmc.png` exist.
- `summary.md` reports the iid→de and iid→mcmc NRMSE ratio per (method, problem) and the reliability line (expected 100 %).
- Total runtime < 60 min.
