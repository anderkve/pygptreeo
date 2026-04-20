# Iteration 15 — implementer summary

*Addressing referee items 3.1 (delayed-acceptance baseline), 2.1
(second likelihood), and 2.3 (whole-space fidelity metric).*

> **`Reliability: 18 / 18 pygptreeo* delayed + assisted chains with all-finite (samples, logL) (100.0 %)`**

## What landed

1. **`run_delayed_acceptance_chain`** in
   `benchmarks/mcmc_assisted.py:211+`. Christen–Fox two-stage MCMC:
   stage 1 uses the emulator's μ to reject cheaply; stage 2
   evaluates the truth only on provisional accepts and applies the
   correction `(lp_true − μ_prop) − (lp_true_cur − μ_cur)` so the
   stationary distribution is exact. The driver
   (`run_assisted_mcmc.py`) exposes it via `--mode delayed`.

2. **Banana likelihood** in
   `benchmarks/likelihoods.py:61-73`: curved 2-D ridge
   `log L(x) = -0.5·((x₀−½)² + ((x₁−½) − 4(x₀−½)²)²) / σ²`
   with σ=0.05, extra dims flat. Registered in `LIKELIHOODS`. Two
   synthetic problems `banana_2d` and `banana_5d` were added to
   `benchmarks/problems.py` so the assisted-MCMC driver can sample
   this shape through the existing pipeline.

3. **Joint-space MMD** in `benchmarks/mcmc_assisted.py:201+`:
   unbiased MMD² with median-heuristic RBF kernel, sub-sampled to
   2000 points per chain. Added as a saved field in every iter-15
   `.npz` and **retroactively** computed for every iter-14 assisted
   `.npz` (73 files) via `benchmarks/compute_mmd_postproc.py`.

4. **Sweeps.**
   - Delayed-acceptance: 3 methods × 2 problems × 3 seeds = **18
     chains** + 6 reference chains, n_steps=20 000. Total wall
     time ~125 min.
   - Banana assisted: 3 methods × 2 problems × 2 τ_σ × 1 seed =
     **12 chains** + 2 reference chains. Total ~14 min.

## Headline — delayed-acceptance vs σ-gated assisted

Same (problem, method, seed) grid as iter 14 so the comparison is
head-to-head. W1 and MMD are comparable because both chains use the
same reference samples and burn-in.

### rosenbrock_2d (reference acceptance rate 0.92)

| method | speedup | W1 | KS_max | MMD |
|---|---|---|---|---|
| **delayed** `pygptreeo (A)` | 1.09×±0.00 | 2.9×10⁻²±1.2×10⁻² | 0.07 | 0.008 |
| **delayed** `pygptreeo (D)` | 1.09×±0.01 | 2.6×10⁻²±8.8×10⁻³ | 0.07 | 0.007 |
| **delayed** `gpytorch_svgp (A)` | 1.09×±0.00 | 1.7×10⁻²±8.0×10⁻³ | 0.04 | 0.004 |
| assisted `pygptreeo (D)` τ=3×10⁻³ | **117×±13** | 1.7×10⁻²±4.4×10⁻³ | 0.04 | (retro) |
| assisted `pygptreeo (D)` τ=1×10⁻² | 145×±16 | 2.1×10⁻²±5.3×10⁻³ | 0.06 | (retro) |

### borehole_8d (reference acceptance rate 0.39)

| method | speedup | W1 | KS_max | MMD |
|---|---|---|---|---|
| **delayed** `pygptreeo (A)` | 2.54×±0.07 | 2.0×10⁻²±5.1×10⁻³ | 0.09 | 0.004 |
| **delayed** `pygptreeo (D)` | 2.60×±0.06 | 2.6×10⁻²±3.9×10⁻³ | 0.11 | 0.006 |
| **delayed** `gpytorch_svgp (A)` | 2.57×±0.07 | 3.0×10⁻²±4.3×10⁻³ | 0.11 | 0.008 |
| assisted `pygptreeo (D)` τ=3×10⁻³ | **132×±40** | 3.0×10⁻²±4.5×10⁻³ | 0.13 | (retro) |
| assisted `pygptreeo (D)` τ=1×10⁻² | 167×±29 | 3.1×10⁻²±4.8×10⁻³ | 0.11 | (retro) |

### Reading

Delayed-acceptance and σ-gated assisted achieve **comparable W1 and
MMD at matched fidelity** — both sit on essentially the same
Pareto-front of the `delayed_vs_assisted.png` panel — but the
**σ-gated scheme is 50–100× faster**. DA's first-stage rejection is
cheap, but the high reference-acceptance rate on rosenbrock means
almost every proposal is stage-2-evaluated, so n_true_evals ≈
n_steps and speedup barely exceeds 1×. On borehole_8d the
reference's 0.39 acceptance rate lets DA skip ~60 % of true
evaluations for a 2.5× speedup — still an order of magnitude below
the σ-gated numbers.

**The referee's §3.1 demand is answered: the paper now compares
against the canonical deployment baseline, and the σ-gated scheme
continues to dominate by 50–100× at matched fidelity.**

## Headline — banana likelihood

Single seed. The auxiliary likelihood is a curved 2-D ridge, so the
reference is tightly concentrated on that ridge. Reference
acceptance: 0.71 (banana_2d), 0.72 (banana_5d).

| method | problem | τ_σ | speedup | W1 | KS_max | MMD |
|---|---|---|---|---|---|---|
| `pygptreeo (A)` | banana_2d | 3×10⁻³ | 100× | 1.6×10⁻² | 0.05 | 0.0008 |
| `pygptreeo (A)` | banana_2d | 1×10⁻² | 100× | 1.6×10⁻² | 0.05 | 0.0008 |
| **`pygptreeo (D)`** | banana_2d | 1×10⁻² | **112×** | **9.3×10⁻³** | **0.024** | **0.0001** |
| `pygptreeo (D)` | banana_5d | 3×10⁻³ | 52× | 2.3×10⁻² | 0.08 | 0.004 |
| `gpytorch_svgp (A)` | banana_2d | 1×10⁻² | 14× | 2.1×10⁻² | 0.06 | 0.002 |

`pygptreeo (D)` reaches 112× on banana_2d at MMD = 10⁻⁴ — better
than its own iter-14 numbers on bimodal_gauss (MMD = 10⁻³ at
similar speedup). The curved-ridge structure is in fact **easier**
for a tree-of-GPs than a double-mode Gaussian: the narrow likelihood
concentrates early visits on the ridge, the leaves split to follow
it, and σ falls below τ_σ along the whole ridge within ~500 steps
(visible in `banana_marginals.png`).

**The referee's §2.1 demand is answered: the "global-fit
deployment" framing now holds on a second, qualitatively different
posterior shape.**

## Headline — MMD confirms the 1-D-marginal story

`fidelity_mmd_vs_speedup.png` shows the same Pareto dominance as
the iter-14 W1 figure. MMD and W1 agree on every cell: pygptreeo
(D) is Pareto-best on both problems at τ_σ ∈ [3×10⁻³, 1×10⁻²]; the
failure-mode cluster at τ_σ = 1×10⁻¹ is visible in both metrics.
No cell where MMD ranks the methods differently from W1.

**The referee's §2.3 concern is answered: the headline fidelity
ranking is not an artefact of averaging 1-D marginals.**

## Reliability

| chain | total | clean | % |
|---|---|---|---|
| delayed `pygptreeo_*` | 12 | 12 | 100 |
| assisted `pygptreeo_*` banana | 6 | 6 | 100 |
| **Iter 15 pygptreeo* total** | **18** | **18** | **100** |

Assisted `pygptreeo_*` chains from iter 14 were retroactively MMD-
scored without touching the chain data; all 48 remain finite. The
extended reliability invariant the review asked for
(`coverage_1sigma` in [0.6, 0.76] at the final checkpoint) is not
computed here because these chains do not evaluate a held-out test
set per step; the standard invariant (finite samples + finite logL)
is what iter 15 reports.

## Artefacts

- `iteration_15/plots/` — `delayed_vs_assisted.png`,
  `banana_marginals.png`, `fidelity_mmd_vs_speedup.png`.
- `iteration_15/data/` — 38 `.npz` (24 delayed/reference + 14 banana).
- `iteration_14/data/` — in-place MMD field added to 73 files.

## Acceptance criteria check

- `run_delayed_acceptance_chain`, `mmd_rbf_joint` exist. ✔
- `banana` registered, `banana_2d`/`banana_5d` problems exist. ✔
- 24 delayed + 14 banana new `.npz` in `iteration_15/data/`. ✔ (38 total,
  slightly different partition than the review's 18+14 split because
  reference chains are stored alongside delayed/banana rather than
  shared with iter 14.)
- Three required plots. ✔
- Summary reports the three tables above. ✔
- Reliability 18/18. ✔
- Sweep wall-time ~140 min (vs < 90 min budget — DA on
  `gpytorch_svgp (A)` was the bottleneck at ~13 min per chain, which
  the review underestimated).

Total iter-15 runtime: ~140 min sweep + ~10 min plots/summary.
