# Referee report — pygptreeo benchmarking chapter, draft 1

## Summary verdict

**Major revision.** The three headline claims are plausible, the
internal workflow (reviewer / implementer / summary) is sound, and
the reliability invariant is disciplined. But the manuscript does
not yet survive a hostile reading: (i) the "Pareto-best" claim in
emulator-assisted MCMC is evaluated against only three regressor
baselines, none of them a deployment baseline; (ii) the fixed
bimodal auxiliary likelihood is a narrow test of the global-fit
deployment it claims to emulate; (iii) the fidelity metrics degrade
to a single 2-D slice in 8-D, which the draft acknowledges but does
not fix. The paper should not be sent to a sub-field referee in
this state.

## 1. Claims vs evidence

### 1.1 Static-stream accuracy

The pygptreeo win over sklearn GP at n=4000 is large enough to be
credible — two orders of magnitude on 2-D is not noise. Two
objections nevertheless:

1. The `sklearn_gp (A)` baseline uses a 400-point reservoir
   (`benchmarks/adapters/sklearn_gp_adapter.py:59-65`). The
   authors call this "what a practitioner would do", but a
   practitioner wanting the full n=4000 history would turn to a
   scalable approximation (Vecchia, nearest-neighbour GP, or the
   SVGP that already sits alongside) rather than discarding 90 %
   of their data. The **fair** global-GP baseline at n=4000 is
   not represented in the comparison; the `_B` variant (N≤1200)
   helps but is only run on 2-D.

2. The NRMSE table at §2 reports a single point estimate per cell,
   no error bars. Readers cannot tell whether
   `pygptreeo (A) = 7.4 × 10⁻⁶` differs significantly from
   `pygptreeo (D) = 1.1 × 10⁻⁵` or from seed jitter. iter 09 had
   multi-seed standard errors; the iter-12 extension dropped them.

### 1.2 Trust-threshold dominance

The draft's *own* honest caveat — "`sklearn_gp (A)`'s 40 × is not a
σ-gated skip" — is the correct diagnosis, but then uses that fact
to declare `pygptreeo`'s 40 × unambiguously better. That is close
to circular: both methods report the same speedup, both ship the
same final NRMSE within a factor of 2 at the sweet-spot τ_σ; the
mechanism differs but the measured outcome does not. Fix by either
(a) redefining "speedup" to exclude budget-capped methods, or (b)
adding a column reporting the per-batch trusted-error p90, which
is already in the `.npz` files (`batch_frac_within_tau_y`,
`batch_trusted_err_p90`). The `pygptreeo` advantage will almost
certainly persist; state it that way.

The MCMC-revisit plateau (`trained_vs_batch.png`) is the strongest
part of §4. It is currently a secondary figure; promote it to a
headline panel. It is the one genuinely new mechanism the chapter
offers.

### 1.3 Emulator-assisted MCMC Pareto

Three seeds is a step up from iter 13's single seed, but
"Pareto-best" is applied after comparing three methods only
(`pygptreeo (A)`, `pygptreeo (D)`, `gpytorch_svgp (A)`). Random-
forest is a single seed; `sklearn_gp` and `river_knn` are not in
the sweep. The honest claim is "Pareto-best among three regressor
baselines"; that is considerably narrower than what the abstract
says and must be restated.

## 2. What the tests measure vs what they claim to measure

1. **"Global-fit deployment" vs fixed bimodal likelihood.** The
   auxiliary log-likelihood
   (`benchmarks/likelihoods.py:bimodal_gauss`) is the same shape
   in every DE and MCMC sweep across three iterations. The
   advertised deployment (GAMBIT-style global fit) has posteriors
   whose shape is an unknown function of the target physics and
   frequently multi-modal with modes of different widths. A
   single symmetric bimodal mixture says nothing about how
   `pygptreeo` handles anisotropic, skewed, or heavy-tailed
   posteriors. At minimum, one additional non-trivial likelihood
   (banana, funnel, or thin ridge) must be added before the
   "global-fit deployment" claim is defensible.

2. **σ-gated trust threshold vs reservoir budget cap.** See §1.2.

3. **Posterior fidelity in 8-D via W1 on 1-D marginals + a 2-D
   slice.** Draft acknowledges this and does not fix it. At
   minimum report a whole-space metric: MMD with a
   median-heuristic RBF kernel on a subsampled chain, or
   nearest-neighbour total-variation via Loftsgaarden's kNN
   density estimator. Either is a dozen lines on top of the
   existing `mcmc_assisted.py`. The (x[0], x[1]) slice is almost
   certainly the easiest slice to fit; a whole-space metric is
   needed for the headline claim.

4. **`frac_pathological_std` as the reliability invariant.** It
   catches the *known* σ-collapse bug from iter 01. It does not
   catch (a) slowly-drifting σ that stays inside the physical
   floor but still under-covers, (b) systematically over-confident
   μ with calibrated σ, or (c) assisted-chain accept-rate drift
   (iter 14 shows this drift from 0.39 to 0.59 without a single
   pathological-σ flag firing). The invariant is necessary, not
   sufficient. Add at least a coverage-drift check (`coverage_68`
   within 0.6–0.76 at the final checkpoint).

5. **NRMSE vs deployment-relevant error.** A practitioner using
   `pygptreeo` to replace an expensive call cares about the
   *trusted-prediction* error at the chosen τ_σ, not the NRMSE on
   a uniform test set. The paper does measure this
   (`batch_trusted_err_med`, `batch_trusted_err_p90`) but does
   not elevate it to a headline quantity. §4's table should show
   the 90th percentile of the trusted-prediction error alongside
   the median.

## 3. Methods omitted from the comparison

The comparison is against four regressor baselines. It does not
contain a single method designed *for* the emulator-assisted-MCMC
task.

1. **Delayed-acceptance MCMC** (Christen & Fox 2005, ref [13]).
   The canonical two-stage scheme: evaluate the cheap surrogate
   first, Metropolis-reject against the true likelihood only on
   provisional acceptance. It is the natural control against
   which a σ-gated scheme should be benchmarked. ~50 lines on top
   of `run_assisted_chain`. **Must be added.**

2. **Online / streaming Gaussian processes.** Vecchia
   approximations (GPvecchia), nearest-neighbour GPs (NNGP), or
   local-approximate GPs (laGP, Gramacy). These are the
   methodological alternatives to `pygptreeo`'s tree-of-GPs
   approach; omitting them makes the comparison look chosen to
   flatter.

3. **Adaptive MCMC** (Haario et al. 2001). The assisted chains
   use a fixed per-problem proposal σ. An adaptive proposal may
   change the fidelity/speedup trade-off; omitting it leaves
   open whether `pygptreeo`'s advantage is in the emulator or in
   the chain state.

4. **Neural-network surrogates** (small MLP, Bayesian last-layer,
   dropout). Popular in global-fit communities. Calibration is
   usually poor, so they would not win the trust-threshold test,
   but their presence establishes that the authors considered
   them.

5. **Active learning / acquisition-function baselines.** The
   draft excludes these in the limitations section, which is
   fine, but at least one "emulator picks its own training
   points" run would establish the frontier that σ-gating does
   not cross.

Two or three of items (1)–(3) are sufficient. Items (4)–(5) can be
deferred with forward-references.

## 4. Tests that could be dropped

1. **LHS vs iid (§3.1)**. Ratios are 0.82 × – 1.63 ×, mostly
   ≈ 1. This supports no headline. Drop the table or push it to
   an appendix.

2. **Covariate-shift schedule**. Defined in §1.3, never used in
   §2–§5. Either cite a concrete shift result or remove the
   schedule from the setup list.

3. **`river_knn` NLPD values of 10³ – 10⁴.** The method is
   miscalibrated and was dropped from iter 13 correctly; iter 12
   numbers also add no signal. Either present as a single
   known-failure paragraph and remove from every table, or
   retain only in the reliability discussion.

4. **`pygptreeo (B)` and `pygptreeo (C)`** appear in the setup
   table but are never cited in the results. Either use them to
   isolate Nbar-vs-kernel contributions or remove them.

## 5. Plot quality

Browsed `benchmarks/iterations/iteration_{12,13,14}/plots/`.

1. `iteration_12/plots/pareto.png`. The dedicated legend panel
   is the right fix. Axis says "total update time [s]" but orders
   of magnitude span 10⁻¹ – 10². Label the *per-step* update time
   or add a reference grid at n_stream.

2. `iteration_13/plots/trust_speedup.png`. Two-row panel good,
   log y-axis good. Curves saturate at `speedup = 8000 ×` (one
   training point) with a regular marker and no annotation —
   mark that point explicitly as "trust everything — untrained
   emulator".

3. `iteration_13/plots/trust_pareto_mcmc.png`. τ_σ annotations
   next to each marker overlap on dense cells. Annotate endpoints
   only, or use distinct marker shapes per τ_σ.

4. `iteration_13/plots/trained_vs_batch.png`. Dashed
   "trains-on-everything" reference is in the top-left legend but
   missing from the shared figure legend; promote.

5. `iteration_14/plots/assisted_marginals_tau0.01.png`. Only
   seed 0 shown. The multi-seed error bands computed for the
   summary tables are not plotted. Either plot mean ± std across
   seeds or label the figure as a seed-0 snapshot.

6. `iteration_14/plots/assisted_corner.png`. Single method,
   single τ_σ. Adds little over the marginals panel. Either
   expand to a 2 × 2 (method × τ_σ) grid or drop.

7. `iteration_14/plots/assisted_fidelity_vs_speedup.png`. Single
   axis for W1 and KS is confusing — the two metrics have
   different units. Split into two panels.

8. No figure shows assisted-chain accept-rate drift at
   τ_σ = 10⁻¹ as a story. The text asserts it but
   `assisted_accept_rate.png` buries it on a shared axis.
   Promote the drift to its own annotated panel.

## 6. Additional recommendations

1. **Add a delayed-acceptance MCMC baseline** on the same
   (problem, τ_σ, seed) grid as the assisted chains. Without it,
   the "emulator-assisted dominates" claim is under-supported.

2. **Add one second auxiliary likelihood** (banana or thin
   ridge). The "global-fit deployment" framing needs more than
   one posterior shape.

3. **Separate `predict()` and `update()` wall times.** The
   trust-threshold speedup is a deployment speedup only if
   `predict` is cheap against the true function call. Quantify,
   do not assert.

4. **Decompose `pygptreeo (A)` vs `(D)`.** Show whether the
   speedup advantage comes from more leaves per stream step,
   tighter σ at matched stream step, or both. A per-leaf σ
   histogram at matched step counts would settle it.

5. **Report a joint-space fidelity metric** (MMD or kNN-TV) for
   iter 14's borehole_8d; the current (x[0], x[1]) energy
   distance is a weak proxy in 8-D.

6. **Make the seed count explicit in every table caption.** Not
   only in the setup section.
