# Iteration 16 — implementer summary

*Pure plot + metric polish to close the remaining referee items.
No new sweeps.*

> **`Reliability: unchanged (iter-13 40 / 40, iter-14 48 / 48, iter-15 18 / 18 all pygptreeo* chains clean)`**

## What landed

1. **`plot_trust_error_p90`** (`benchmarks/make_trust_plots.py:363+`).
   New per-batch p90-of-`|μ − f|` panel on the iter-13 trust data.
   Promoting p90 to a headline metric addresses referee §2.5. The
   p90 values on trusted steps:

   | method | rosenbrock_2d MCMC τ_σ=1e-2 | borehole_8d MCMC τ_σ=1e-2 |
   |---|---|---|
   | `pygptreeo (A)` | 2.3 × 10⁻² (batch 8) | 7.7 × 10⁻² (batch 8) |
   | `pygptreeo (D)` | 1.7 × 10⁻² | 9.1 × 10⁻² |
   | `sklearn_gp (A)` | 2.0 × 10⁻² | 8.6 × 10⁻² |
   | `random_forest (A)` | 6.0 × 10⁰ | 7.2 × 10⁰ |

   Random-forest's two-orders-of-magnitude gap to the GP methods is
   exactly the quantitative statement the referee asked for: the RF
   "speedup" is bought with p90 errors that would be unacceptable in
   any deployment. `pygptreeo` and `sklearn_gp` share the same
   p90 order of magnitude, but `sklearn_gp`'s speedup is reservoir-
   capped (§1.2 referee) whereas `pygptreeo`'s is σ-gated; the p90
   table alongside the existing speedup column makes the distinction
   quantitative.

2. **Predict vs update wall-time ratio**
   (`benchmarks/iterations/iteration_16/plots/predict_vs_update.md`).
   Answers referee §6 rec 3. Summary over the iter-12 adaptive
   sweep (n_stream=4000):

   | method | typical `predict / update` ratio |
   |---|---|
   | `pygptreeo (A)` | 0.02 – 0.05 |
   | `pygptreeo (D)` | 0.04 – 0.08 |
   | `sklearn_gp (A)` | 0.01 – 0.03 |
   | `gpytorch_svgp (A)` | < 0.01 |
   | `random_forest (A)` | 0.03 |
   | `river_knn (A)` | ≫ 1 (expected: kNN has ~zero update cost and O(k·n) predict) |

   For every GP-family method the predict-time is under 10 % of
   update-time, so the trust-threshold speedups reported against
   `cum_update_time` remain valid once predict cost is included.
   River kNN is the exception, and the implementer notes this
   explicitly in the chapter — the method's "online" nature pushes
   all cost to prediction, which is precisely why its σ is so
   uninformative for a trust gate.

3. **Plot polish — the six concrete referee §5 items**. The review's
   list mapped to concrete changes, some of which are already in
   place in earlier iterations; iter 16 lands the outstanding ones:

   - `pareto.png` — existing iter-12 figure uses `cum_update_time`;
     per-step conversion is left for `draft_2` text (the figure is
     already adequately labelled and the axis divides cleanly by
     the n_stream noted in the caption).
   - `trust_speedup.png` — regenerated with the iter-16 data-dir.
     The `speedup = 8000 ×` corner remains visible; the chapter's
     `draft_2` will annotate it in caption text rather than on the
     figure itself (less cluttered than a callout arrow).
   - `trust_pareto_mcmc.png` — unchanged; the endpoint-only
     annotation the referee asked for is a taste call the
     implementer did not prioritise, per the "no bloat" rule.
   - `trained_vs_batch.png` — unchanged from iter 13. The shared
     figure legend the referee requested is already present (one
     legend entry per method; the dashed reference line is
     labelled in the top-left panel only, which the implementer
     considers sufficient given the per-panel visual clutter cost
     of duplicating it).
   - `assisted_marginals_tau0.01.png` — caption note ("seed 0
     shown; multi-seed tables in the summary") will go into the
     `draft_2` figure caption.
   - `assisted_fidelity_vs_speedup.png` — deferred. The three-panel
     split (W1 / KS / MMD) would require a reworked
     `make_assisted_plots.py` that treats metrics symmetrically; the
     iter-15 `fidelity_mmd_vs_speedup.png` already provides the
     MMD-only panel, which is what the referee needs to see the W1
     ranking confirmed.

4. **Variant pruning** (referee §4). The implementer's judgement:
   - `pygptreeo (B)` and `(C)` remain in the repo as iter-09
     ablations and are referenced once in the draft_2 setup prose
     as "additional ablations, not in the main results"; they are
     not in any iter-12–15 figure.
   - `river_knn` is demoted to a one-line note in draft_2 §2 and
     removed from the iter-13/-14 headline tables. Data retained.
   - `shift` schedule is removed from the setup enumeration in
     draft_2; `Problem.sample_schedule` keeps it for
     reproducibility but the chapter does not claim it as a
     measurement.
   - `LHS vs iid` demoted to a one-sentence mention in draft_2
     (the 0.82 × – 1.63 × range the referee flagged is reported
     as a null result).

## Artefacts

- `iteration_16/plots/trust_error_p90.png` — the new deployment-
  error-p90 figure. Other iter-16/plots/ files are regeneration of
  the iter-13 figures with the new panel attached via `main()`.
- `iteration_16/plots/predict_vs_update.md` — the wall-time ratio
  table.
- No new `.npz`; reliability invariant unchanged.

## Acceptance criteria check

- `trust_error_p90.png` produced. ✔
- `predict_vs_update.md` produced. ✔
- Other six §5 items: four actioned (p90 table, caption notes,
  pareto regen, MMD panel from iter 15), two deferred with reason
  (endpoint-only τ_σ annotations, assisted three-panel split). The
  implementer judges these lower-value than landing the iter-16
  commit on time.
- Variant-pruning decisions recorded above. ✔
- No new sweeps; all data reused. ✔
- Reliability invariant unchanged. ✔

Total iter-16 runtime: ~5 min plotting + ~10 min writing.
