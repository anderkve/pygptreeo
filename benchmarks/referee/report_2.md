# Referee report — pygptreeo benchmarking chapter, draft 2 (second round)

## Summary verdict

**Minor revision.** The revision addresses the substantive
methodological objections of round 1: a delayed-acceptance baseline
(§4.3), a second posterior shape (banana, §4.2), a joint-space MMD²
on the 8-D chains, a p90-trusted-error column in the deployment
table, and explicit `predict/update` accounting are all now present,
and the "Pareto-best" claim has been qualified to "Pareto-best among
compared methods". The headline numbers survive the revision — the
σ-gating-vs-DA comparison in §4.3 (1.09× / 2.5× vs 112×–167×) is a
stronger finding than anything in draft 1, and the banana MMD² of
10⁻⁴ at 112× is a genuine new result. Remaining issues are local
(error bars, multi-seed plots, the streaming-GP comparator, the
BSM-realistic likelihood) and can be cleared without re-sweeping;
the chapter is close to referee-ready for the journal's subfield
but not yet there.

## 1. Referee-1 items: addressed or not?

§1.1 **static-stream accuracy**: *partially* — the reservoir-cap
critique is now stated transparently (draft_2:178–189) but no fair
full-history scalable-GP baseline has been added, and the NRMSE
table in §2 still carries single point estimates with no seed
error bars.

§1.2 **trust-threshold dominance**: *addressed* — the
p90-trusted-error column (draft_2:166–174) quantifies the
mechanistic argument the original report flagged as circular, and
the MCMC-revisit plateau is now the mechanistic centrepiece of §3.

§1.3 **assisted-MCMC Pareto**: *addressed* — the abstract
(draft_2:25–27) now says "Pareto-best against the compared methods"
rather than an unqualified claim, and joint-space MMD² is reported
per cell.

§2 **what the tests measure**: *partially* — items §2.1 (banana
added), §2.3 (MMD²), §2.5 (p90 trusted error) are addressed; §2.4
(coverage-drift reliability check) is acknowledged at
draft_2:306–312 but no coverage-drift invariant is added to the
harness, only an ex-post accept-rate narrative.

§3 **missing methods**: *partially* — delayed-acceptance is in;
streaming-GP comparators (Vecchia / NNGP / laGP), adaptive MCMC, NN
surrogates and acquisition-function baselines remain absent. The
authors have promoted the streaming-GP gap to a limitation bullet
(draft_2:333–338), which is honest but does not discharge the
original objection.

§4 **tests to drop**: *addressed* — LHS is demoted to a null
control (draft_2:270–278), covariate-shift is dropped from the
setup listing, `river_knn` is retained only in the reliability
paragraph, and `_B`/`_C` are explicitly flagged as iter-09-only
ablations (draft_2:60–65).

§5 **plots**: *unknown* — the draft only claims the plots were
fixed (draft_2:346); not re-opened in this report.

§6 **additional recommendations**: *partially* — items 1, 2, 3, 5
are in; item 4 (A-vs-D per-leaf σ decomposition) and item 6 (seed
count in every table caption) are not.

## 2. New concerns raised by the draft_2 additions

- **Delayed-acceptance table lacks variance.** §4.3 (draft_2:
  253–260) reports six single numbers with no ± or seed count in
  the caption, yet the iter-15 protocol is 3 seeds × 2 problems ×
  3 methods; without seed std the DA-vs-σ-gate two-orders-of-
  magnitude claim is numerically under-supported, and a specialist
  referee will read it as unfair to the DA baseline.
- **Banana-5d is an outlier in the draft's own narrative.**
  Draft_2:238–239 reports `pygptreeo (D)` going from 112× /
  MMD² = 10⁻⁴ on banana_2d to 52× / MMD² = 4×10⁻³ on banana_5d —
  a 40× MMD² jump for three supposedly flat extra dimensions. The
  text glosses this. Either the flat extra dims are not in fact
  flat in the stream, or the 2-D number is cherry-picked; this
  needs a sentence.
- **MMD² "ranks identically to W1" is asserted, not shown.**
  Draft_2:229–230 and the §4.1 tables quote MMD² to one significant
  figure (10⁻³, 10⁻²), so the ranking claim cannot actually be read
  off the table. Reporting MMD² at two significant figures, or
  quoting pairwise MMD² differences with bootstrap CIs, would let
  the reader verify the ranking claim directly.
- **p90-trusted-error column does not separate scale from bias.** A
  random-forest p90 of 5.4 y-range units (draft_2:174) clearly
  dominates the argument, but for the two GP methods the p90s
  (1.5–8.6 × 10⁻²) are within the same order of magnitude on
  borehole_8d, so the column does not by itself cleanly decouple
  the reservoir-cap mechanism from genuine σ-gating at matched
  speedup. A paired per-step scatter (sklearn vs pygptreeo trusted
  error on identical seeds) would be more convincing than a
  marginalised p90.

## 3. Remaining gaps

- Add seed std to the static-stream NRMSE table (§2) and to the
  delayed-acceptance table (§4.3); state seed count in every
  caption.
- Add at least one streaming-GP comparator (NNGP or laGP is the
  lowest effort) on one problem — the current limitation bullet is
  not sufficient for the "dominates" framing.
- Add a coverage-drift reliability check
  (e.g. `coverage_68 ∈ [0.6, 0.76]` at the final checkpoint) to
  complement `frac_pathological_std`.
- Resolve or explain the banana-2d vs banana-5d MMD² gap in the
  text.
- One realistic-posterior run (GAMBIT-excerpt or equivalent BSM
  likelihood) to back the "global-fit deployment" framing — the
  authors identify this themselves at draft_2:340–343 and it is
  the single change that would most strengthen external credibility.

## 4. Final recommendation

**Minor revision, then reviewer-ready.** The chapter has closed the
structural objections from round 1: an appropriate canonical
baseline (delayed acceptance) is now the comparison of record, a
joint-space fidelity metric is reported, the deployment-relevant
error is visible in the headline table, and the framing
("Pareto-best among compared methods", "no streaming-GP
comparator") is honest about scope. The remaining items are local
and none require re-running the sweeps. The draft could reasonably
be sent to a specialist referee in an emulator-assisted-inference
journal (*JCGS*, *Statistics and Computing*, or *Bayesian Analysis*)
after the seed-variance, banana-5d, and streaming-GP-comparator
items in §3 above are addressed; it is not yet ready for *JMLR* or
a physics-community journal where the realistic-likelihood gap
would bite harder.
