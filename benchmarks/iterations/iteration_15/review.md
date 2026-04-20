# Iteration 15 review — Delayed-acceptance baseline + second likelihood + joint-space fidelity

## Goal

Three referee points from `benchmarks/referee/report_1.md` are
tractable in one iteration and together fix the largest weaknesses
of the chapter: (i) the absence of a deployment baseline, (ii) the
single fixed bimodal auxiliary likelihood, (iii) the 2-D-slice
fidelity metric in 8-D. Iter 16 will pick up the remaining referee
items (streaming-GP alternative, plot polish, variant pruning).

## Plan

1. **Delayed-acceptance MCMC baseline** (refs [13] in the draft).
   Add `run_delayed_acceptance_chain` next to `run_assisted_chain`
   in `benchmarks/mcmc_assisted.py`. Two-stage accept:
   - Propose `x'`.
   - Query emulator → `(μ_emu, σ_emu)`. If
     `log U₁ ≥ (μ_emu − lp_current)` **reject provisionally**
     without evaluating the truth (first-stage rejection).
     Otherwise (provisional accept), evaluate the truth
     `lp_true = logL(x')`, perform the *second-stage* Metropolis
     accept with correction ratio
     `(lp_true − μ_emu) − (lp_current_true − μ_emu_current)`
     (Christen–Fox). Train the emulator on every true evaluation.
   - Save per-chain: `n_first_stage_rejects`, `n_true_evals`,
     `speedup = n_steps / n_true_evals`, same fidelity metrics as
     assisted.
   Run the same 3 methods × 2 problems × 3 seeds as iter 14 (drop
   the τ_σ axis: delayed acceptance has no threshold). **18 runs**.

2. **Second auxiliary likelihood — banana** (`benchmarks/
   likelihoods.py`). Add `banana(X)`: a curved 2-D ridge in the
   first two dimensions, flat in the rest. Concretely
   `log L(x) = -0.5 · ((x_0 − 0.5)² + ((x_1 − 0.5) − 4·(x_0 − 0.5)²)²) / σ²`
   with σ = 0.05. Out-of-cube penalty `-1e4` as in `bimodal_gauss`.
   Register in `LIKELIHOODS`.

3. **MCMC-schedule and assisted runs on `banana`**. Two problems
   (rosenbrock_2d, borehole_8d) × three methods (pygptreeo_A,
   pygptreeo_D, gpytorch_svgp_A) × two τ_σ (3e-3, 1e-2) × 1 seed
   assisted + 1 reference seed per problem = **12 + 2 = 14 runs**.
   Thread `--likelihood banana` CLI through
   `run_assisted_mcmc.py` (default stays `bimodal_gauss`).

4. **Whole-space fidelity — MMD**. Add `mmd_rbf_joint` to
   `mcmc_assisted.py`: unbiased MMD with median-heuristic RBF on
   the full joint distribution (subsampled to 2000 points per
   chain). Closed-form, O(n²), no new dependency. Save alongside
   `w1_marginals`, `ks_marginals_max`, `energy_2d_01` in every
   iter-14 / iter-15 `.npz`. Retroactively compute MMD for the
   existing iter-14 `.npz` via a small post-processing script
   `benchmarks/compute_mmd_postproc.py` so we don't have to rerun.

5. **Plot additions**.
   - `plots/delayed_vs_assisted.png` — fidelity-vs-speedup Pareto
     with delayed-acceptance points overlaid on the iter-14 assisted
     Pareto (one panel per problem).
   - `plots/banana_marginals.png` — reference vs assisted 2-D
     scatter on (x_0, x_1) with the banana ridge visible.
   - `plots/fidelity_mmd_vs_speedup.png` — same axes as
     `assisted_fidelity_vs_speedup` but with MMD on the y-axis;
     two panels (rosenbrock_2d, borehole_8d).

6. **Summary.md requirements**.
   - Reliability invariant (extended): all `pygptreeo*` assisted
     chains end with `coverage_1sigma` in [0.6, 0.76] at the final
     checkpoint **and** finite samples/logL.
   - Delayed-acceptance speedup / W1 / MMD table per problem next
     to the assisted numbers.
   - Banana-likelihood speedup / W1 / MMD table per method.
   - MMD vs W1 comparison paragraph: does the whole-space metric
     agree with the 1-D-marginal ranking?

## Out-of-scope (defer to iter 16)

- Streaming-GP alternative (GPvecchia / NNGP / laGP) — needs new
  dependency research, not a one-iter change.
- Plot-quality polish (referee §5 items 1–8) — cleanup pass.
- pygptreeo (B) / (C) citations or removal — writing/scope
  decision, not a sweep.
- Neural-network surrogate — cross-cutting, defer.
- Adaptive MCMC (Haario) — defer.
- Predict/update wall time split — requires harness instrumentation.

## Acceptance criteria

- `benchmarks/iterations/iteration_15/data/` contains 18 delayed
  + 14 banana = **32 new `.npz`**, plus the post-processed MMD
  values merged into iter-14 files (in place, noted in commit).
- `mcmc_assisted.py` has `run_delayed_acceptance_chain` and
  `mmd_rbf_joint` with one-line docstrings.
- `likelihoods.py` has `banana` and is listed in `LIKELIHOODS`.
- `iteration_15/plots/` has the three plots named above.
- `summary.md` contains the extended reliability invariant and two
  referee-addressing tables.
- Reliability: 100 % of `pygptreeo*` assisted and delayed chains
  finite + coverage_1sigma in [0.6, 0.76].
- Sweep wall-time < 90 min.
