# Iteration 14 review — Closed-loop emulator-assisted MCMC: posterior bias vs speedup

## Goal

Iter 13 showed that a σ-gated trust threshold buys 20–60× fewer true-function
calls at < 2× baseline NRMSE (open-loop: emulator predictions are scored but
do not feed back into the stream). That result is silent on the question a
global-fit user actually asks: *if I let the emulator's mean replace the true
likelihood inside the MCMC accept/reject step, how wrong does my posterior
get?* Iter 14 runs paired reference and emulator-assisted random-walk
Metropolis chains (same seed, same proposal RNG) and reports posterior
fidelity (W1 and KS on 1-D marginals, energy distance on 2-D marginals)
against the reference-chain speedup, on the same two problems as iter 13.
This closes the loop: a trusted prediction now changes which x's the chain
visits, so emulator error compounds rather than washing out.

## Plan

1. **Likelihood mapping (`benchmarks/mcmc_assisted.py:38`).** Keep the form
   `log L(x) = -β · (f(x) - f_min) / (0.1 · f_range)` but **sweep β ∈ {0.5,
   2.0}** and pick one per problem from the reference-chain diagnostics
   below. Rationale:
   - rosenbrock_2d: f ∈ [0, 20] on [0,1]², mode at a narrow valley.
     β=1 with f_scale_eff=0.1·f_range=2 gives logL ∈ [−10, 0], i.e. a
     factor-e² concentration per unit f — already fairly peaked. Start at
     β=0.5 so the valley is resolved but σ_prop=0.05 still moves along it.
   - borehole_8d: f ∈ [0, ~310] (sample-dependent), broadly quadratic.
     β=1 with the 0.1·f_range scale produces a nearly-Gaussian posterior.
     Try β=2 to sharpen it, otherwise the "posterior" is essentially the
     prior and fidelity becomes uninformative.
   Decision rule: pick β so the reference chain's acceptance rate is in
   [0.2, 0.45] and the posterior std on x[0] is ≤ 0.25 (i.e. the marginal
   is narrower than the prior). Record the chosen β in each saved `.npz`.

2. **Sweep grid (driver: `benchmarks/run_assisted_mcmc.py`).**
   - **Methods:** drop `sklearn_gp_A` (iter-13 rolling-400-point window is
     a well-understood artefact and its "fast" column is already explained;
     including it here would just reproduce the same artefact at higher
     cost per sample). Drop `random_forest_A` (iter-13 trusted-error
     regularly > 1.0 → its assisted chain will trivially be wrong; a
     single diagnostic run is enough, see step 3). **Keep**
     `pygptreeo_A`, `pygptreeo_D`, `gpytorch_svgp_A`. Three methods.
   - **Problems:** `rosenbrock_2d`, `borehole_8d`. Same as iter 13.
   - **τ_σ grid:** `{3e-3, 1e-2, 3e-2, 1e-1}` — four points spanning
     "almost never trust" to "almost always trust", matching iter 13's
     informative range. Drop 1e-3 (iter 13 speedup ≈ 1×, posterior
     indistinguishable from reference) and drop the trust-everything
     1e0 extreme (iter 13 already mapped it).
   - **Seeds:** `{0, 1, 2}` (three seeds per cell). Needed because the
     posterior fidelity metric is noisy and the user's headline claim —
     "how wrong does the posterior become" — must not ride on a single
     seed. Reference chains are also seeded at `{0, 1, 2}`.
   - **n_steps:** `20000` (prototype default). Burn-in 2000, no thinning.
   - **proposal_sigma:** problem-specific. `0.04` for rosenbrock_2d
     (narrow valley), `0.08` for borehole_8d (wider cube). Set via a
     per-problem CLI map in a thin wrapper script
     `benchmarks/run_iter14.sh`; retain the single `--proposal-sigma`
     arg in `run_assisted_mcmc.py`.
   - **Total runs:** 3 methods × 2 problems × 4 τ_σ × 3 seeds = **72
     assisted chains** + 2 problems × 3 seeds = **6 reference chains**.
   - **Budget.** Reference chains at n=20000: rosenbrock ≈ 10 s,
     borehole ≈ 20 s → < 2 min total. Assisted chains: at τ_σ=3e-3
     (few trusted calls) the wall is dominated by `method.update` and
     `method.predict`; pygptreeo_D is ~3× faster than _A (iter 12),
     and at τ_σ=1e-1 the wall collapses (few updates). Empirical cap:
     pygptreeo_A rosenbrock n=8000 @ τ_σ=3e-3 was ~500 s in iter 13;
     scale linearly to n=20000 → ~1250 s worst cell. With 72 cells the
     naive bound is 25 h but the τ_σ=1e-1 and pygptreeo_D cells are
     order-of-magnitude cheaper, so realistic total ≈ 60–90 min. Enforce
     `--max-wall-time 1500` per cell; time-out cells report their
     partial-chain diagnostic and are logged as DNF.

3. **One diagnostic random-forest run** (not in the grid). Run
   `random_forest_A` × rosenbrock_2d × τ_σ=1e-2 × seed 0 only, to sanity-
   check the uncalibrated-σ regressor's assisted-chain failure mode.
   Store under `data/` like any other run; exclude from the fidelity
   plots via an allow-list. Keeps the qualitative claim "uncalibrated σ
   destroys the posterior" visible without polluting the Pareto.

4. **Fidelity metrics (`benchmarks/mcmc_assisted.py:180`).** The
   prototype's `wasserstein1_marginals` is adequate as the headline.
   Extend with two more metrics, saved into each assisted `.npz`:
   - **`ks_marginals_max`**: max over dims of the two-sample KS
     statistic (scipy.stats.ks_2samp). Catches tail mismatch that W1
     averages away.
   - **`energy_2d_01`**: energy-distance between reference and
     assisted chains on the (x[0], x[1]) 2-D marginal, subsampled to
     2000 points from each post-burn-in chain (closed-form O(n²);
     4e6 pairwise distances per call ≈ 0.1 s). Captures the 2-D
     correlation structure the corner plot shows. Use
     `scipy.spatial.distance.cdist`-based computation; do not add a
     new dependency.
   Implement both in `mcmc_assisted.py` alongside
   `wasserstein1_marginals`; keep the main sweep driver interface.

5. **Plots (`benchmarks/make_assisted_plots.py`).** Three existing
   panels plus two additions:
   - `assisted_marginals_tau{1e-2,3e-2}.png` — render at **two τ_σ
     picks**, not one, so the reader sees the fidelity/speedup trade
     visually. Row = dim, col = problem. Reference overlaid as black
     step histogram (unchanged).
   - `assisted_corner.png` — kept; add a second panel row for
     `method_pick=pygptreeo_D` so _A/_D differences are visible at a
     glance.
   - `assisted_fidelity_vs_speedup.png` — keep the log-log Pareto;
     **overlay both metrics** by using the marker shape for W1 vs KS
     (circle = W1, triangle = KS) on the same axes, annotated with τ_σ.
     Include the reference (speedup=1×, fidelity=0) as an anchor
     marker.
   - **New `assisted_trusted_err_hist.png`** — histogram of the
     per-accepted trusted-prediction error `trusted_err` (already
     recorded in the prototype) stratified by (method, problem) at
     τ_σ=1e-2. Lets the reader tie posterior bias to per-step bias.
   - **New `assisted_accept_rate.png`** — bar/line plot of acceptance
     rate (method, problem) × τ_σ; a collapse in accept-rate is a
     tell-tale for emulator-induced pathology independent of the
     fidelity metrics.

6. **Burn-in / thinning / aggregation.** Burn-in = 2000 (10 %), no
   thinning. Fidelity metrics computed on samples `[2000:]` of each
   chain. Multi-seed aggregation: report mean ± std across the 3 seeds
   in the summary tables; each per-seed value lives in its own `.npz`.

7. **Output destinations.** `.npz` into
   `iterations/iteration_14/data/`; `.png` into
   `iterations/iteration_14/plots/` and mirror to global
   `benchmarks/plots/`. Naming: keep the prototype's
   `reference__<problem>__seed<s>.npz` and
   `assisted__<method>__<problem>__tau<τ>__seed<s>.npz`.

8. **`summary.md` requirements.**
   - Reliability one-liner for `pygptreeo_*` runs (target 100 %; no
     `frac_pathological_std` stream here, so define it as "no
     `NaN`/`inf` in `samples` or `logL` and `method.close()` returns
     cleanly" — report as M / N).
   - β and proposal_σ chosen per problem, with the reference-chain
     acceptance rate and marginal-std diagnostic that justified each.
   - Per-(method, τ_σ) table per problem: speedup (n_steps /
     n_used_true), W1, KS_max, energy_2d_01, accept rate, mean
     trusted_err, wall time. Mean ± std across 3 seeds. Two tables,
     3 methods × 4 τ_σ rows each.
   - One paragraph per problem reading the Pareto: best method at the
     sweet spot (e.g. W1 within 2× of reference-noise floor at what
     speedup), and the failure mode at the aggressive end.
   - A short subsection on the random-forest diagnostic run stating
     the W1 it produced and why it is excluded from the main panels.
   - Wall-time table.

## Out-of-scope

- Multi-chain R-hat convergence diagnostics (3 seeds is enough for a
  noise bar on the fidelity metrics; full R-hat is deferred).
- Problems beyond rosenbrock_2d and borehole_8d (keep the iter-13
  contrast; more problems belong to the post-referee expansion).
- Comparison against off-the-shelf delayed-acceptance or surrogate-
  MCMC libraries (pyDELFI, surmise, etc.); that is a referee-prompted
  follow-up, not an in-scope basic calibration.
- Retraining on the rejected proposals (we only `method.update` on
  proposals we actually evaluated the truth for; leave the
  counterfactual "update on every proposal" question for iter 15/16).
- Tuning `proposal_sigma` adaptively during the chain (Haario-style).
  Fixed per-problem σ is enough for the bias-vs-speedup story.
- sklearn GP and river kNN methods (explained in step 2).

## Acceptance criteria

- `benchmarks/iterations/iteration_14/data/` contains 6 reference
  `.npz` and 72 assisted `.npz` (plus 1 diagnostic RF file), each with
  `w1_marginals`, `ks_marginals_max`, `energy_2d_01` scalar fields.
- `benchmarks/iterations/iteration_14/plots/` contains:
  `assisted_marginals_tau1e-2.png`, `assisted_marginals_tau3e-2.png`,
  `assisted_corner.png` (with _A and _D panels),
  `assisted_fidelity_vs_speedup.png`,
  `assisted_trusted_err_hist.png`, `assisted_accept_rate.png`.
  Mirror to `benchmarks/plots/`.
- `benchmarks/iterations/iteration_14/summary.md` includes the
  reliability line, the chosen β / σ_prop table, the two per-problem
  fidelity-vs-speedup tables (mean ± std over 3 seeds), the two
  Pareto paragraphs, the RF diagnostic subsection, and the wall-time
  table.
- `mcmc_assisted.py` gains `ks_marginals_max` and `energy_2d_01`
  functions with one-line docstrings and is used by the driver.
- Reliability: 100 % of `pygptreeo_*` chains end cleanly (see §8).
- Total sweep wall-time ≤ 2 hours on the benchmark box; any DNF cell
  is explicitly logged in `summary.md`.
