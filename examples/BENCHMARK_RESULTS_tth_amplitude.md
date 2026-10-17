# pyGPTreeO on 5D tree-level tt̄H amplitudes, and an uncertainty-gated integration

`examples/tth_amplitude_benchmark.py`, `examples/gated_integration_demo.py`,
`examples/tth_kde_stream.py`, `examples/amplification_demo.py`,
`examples/plot_tth_results.py`; data as in `examples/tth_amplitude/README.md`; one
CPU core per run, up to four runs in parallel; seed 1; 12–17 October 2026.

## 1. The benchmark

"Interpolating amplitudes" (Bresó-Pla, Heinrich, Magerya, Olsson, SciPost Phys. 19,
123 (2025), arXiv:2412.09534) compares interpolation methods for 2 → 3 amplitudes on
test functions of five phase-space variables, two of them tree level: f1,
q q̄ → t t̄ H, and f3, g g → t t̄ H. The test function is the weighted amplitude
`f = |M|² · dΦ/d(x₂..x₅) · (1/2ŝ) dρ/dβ²`, where the last factor is the flux times a
pseudo-PDF, their eq. (10), `ρ(β²) ∝ (1 − c₁β²)² / ((1 − c₂β²)(1 − c₃β²))` with
c = {1.0132, 0.9943, 0.3506} for q q̄ and {1.0134, 0.7344, 0.0987} for g g. It
falls from 0.9 at the low end of the β² range to 0.025 (f1) and 0.003 (f3) at the
high end, so the paper's f weights the high-energy region far less than the
phase-space measure alone. Their error is the relative L1 distance
`ε = Σ|f̃ − f| / Σ f` over uniformly sampled test points, with 1 % as the target, and
on LO-unweighted points (drawn with probability ∝ f) the mean relative error. Their
training points are Sobol points of the unit cube or LO-unweighted points.

The amplitudes here are the same functions computed with MadGraph
(`tth_amplitude/tth_driver.cpp` builds the momenta from the five variables and
returns the phase-space measure; the benchmark multiplies it by ρ, `--no-rho` leaves
it out). The domain is folded by the amplitude's symmetries as in the paper's
interpolation code, θ_H, θ_t ∈ [0, π/2] and φ_t ∈ [0, π]; the paper's networks use
θ ∈ [0, π] and fold only φ, four times the volume, which `--domain full` reproduces.
131 072 Sobol training points, 65 536 LO-unweighted ones, 40 000 uniform and 5000
unweighted test points per channel, on each domain.

Two weightings appear below. **Section 2 uses the paper's f**, with ρ, and is the
comparison with the paper and with arXiv:2601.00950. **Sections 3–7 were measured
before ρ was included**, with `f = |M|² dΦ` alone: a harder variant of the metric
that weights the high-energy collinear region 10–300 times more than the paper's
and is kept as such; the training target `log |M|²` and the trees are the same, only
the error weights and the importance-sampled streams differ.

The tree learns `log |M|²`: `GPTree(Nbar=100)` at the defaults; the hybrid adds the
feature network (`global_mean='net'`, refits amortised at 8 steps per update); a
larger network (`--net hidden=256,depth=4,steps=600,polish_steps=10000`) is run to
32 768 points. `acq` is the plain tree choosing which pool points to learn from,
500 at a time out of 20 000 random candidates, by the largest expected error
contribution `w |M|² σ`, against the stream order. The improved tree of Section 7
takes the ten `log(p_i · p_j)` as inputs, RBF leaves of 300 or 1000 points, and the
"mixed" stream that alternates Sobol and LO-unweighted points.

## 2. Results with the paper's weights

`eps_vs_points_paper_weights.png`; the networks' metric, the mean relative error on
LO-unweighted test points (their eq. 30), seed 1, one run each. Networks: the MLP(s)
and L-GATr curves of the paper's Figure 16 (trained and tested on unweighted
samples), read off the plot by pixel position; the two agree to 10–20 %, the mean is
quoted. ε on uniform test points (their eq. 28) is within 20 % of these values for
every row.

| configuration (mean relative error on LO-unweighted test points, paper's weights) | f1 @4000 | f1 @16000 | f1 @65536 | f1 @131072 | f3 @4000 | f3 @16000 | f3 @65536 | f3 @131072 |
|---|---|---|---|---|---|---|---|---|
| tree, package defaults (cube inputs, Matérn 3/2, Nbar 100, Sobol) | 8.19e-03 | 3.82e-03 | 1.70e-03 | 1.24e-03 | 3.84e-02 | 1.97e-02 | 9.52e-03 | 6.82e-03 |
| tree on log invariants, RBF leaves, Nbar 300, mixed stream | 7.00e-05 | 2.24e-05 | 5.15e-06 | 3.18e-06 | 4.92e-03 | 2.24e-03 | 6.61e-04 | 3.51e-04 |
| tree on log invariants, RBF leaves, Nbar 1000 refit only at splits, mixed stream | 3.40e-05 | 1.30e-05 | 2.70e-06 | 1.26e-06 | 9.11e-03 | 3.37e-03 | 8.01e-04 | 4.46e-04 |
| tree on log invariants, RBF leaves, Nbar 300, importance-sampled stream | 9.14e-05 | 2.10e-05 | 6.03e-06 | – | 5.33e-03 | 2.01e-03 | 6.13e-04 | – |
| tree on log invariants, RBF leaves, Nbar 1000 refit only at splits, importance-sampled stream | 5.56e-05 | 1.30e-05 | 2.97e-06 | – | 5.32e-03 | 2.32e-03 | 6.04e-04 | – |
| tree on log invariants, RBF leaves, Nbar 300, chooses its points from the mixed pool | 4.37e-05 | 6.65e-06 | 2.65e-06 | – | 3.15e-03 | 6.94e-04 | 3.48e-04 | – |
| tree on log invariants, RBF leaves, Nbar 300, importance-sampled stream with 20 % KDE-targeted points | – | – | – | – | 4.71e-03 | 1.72e-03 | 5.68e-04 | – |
| the same stream with the KDE points after the first 16 384 | – | – | – | – | 5.33e-03 | 2.01e-03 | 5.38e-04 | – |
| FULL DOMAIN: tree, package defaults | 1.57e-02 | 7.54e-03 | 3.44e-03 | 2.53e-03 | 6.94e-02 | 3.75e-02 | 1.79e-02 | 1.33e-02 |
| FULL DOMAIN: tree on log invariants, RBF leaves, Nbar 300, mixed stream | 4.31e-04 | 1.02e-04 | 2.35e-05 | 1.04e-05 | 1.11e-02 | 5.08e-03 | 1.76e-03 | 1.04e-03 |
| FULL DOMAIN: tree on log invariants, RBF leaves, Nbar 1000 refit only at splits, mixed stream | 3.75e-04 | 9.31e-05 | 2.04e-05 | 7.89e-06 | 1.97e-02 | 6.92e-03 | 2.23e-03 | 1.09e-03 |
| MLP(s) / L-GATr of arXiv:2412.09534, Fig. 16 (read off the plot) | 6e-4 | 1.6e-4 | 6.8e-5 | 5.6e-5 | 5.5e-3 | 1.1e-3 | 2.7e-4 | 1.7e-4 |

### 2.1 Against the paper's networks

* **f1 is far below the networks.** The 300-point tree on the mixed stream reaches
  3.2e-6 at 131 072 points (median relative error 4e-7, every test point within
  1 %) against the networks' 5.6e-5 at 1.3·10⁵, 18 times lower; the 1000-point
  tree refitted only at splits reaches 1.3e-6, 44 times lower, and both curves
  keep falling as N⁻¹ where the networks' flatten (5.6e-5 → 4.6e-5 from 10⁵ to
  10⁶ points). Choosing its points, the 300-point tree is at 6.7e-6 at 16 000,
  where the networks are at 1.6e-4.
* **f3 is within a factor 2 of the networks at 131 072 points on the plain
  stream, and ahead of them beyond that.** The 300-point tree on the mixed stream
  reaches 3.5e-4 at 131 072 against their 1.7e-4, and crosses their curve at about
  4·10⁵ points (1.0e-4 at 524 288, Section 2.6); choosing its points from a large
  enough pool it is ahead from 65 536 on (2.2e-4 against 2.7e-4; 1.1e-4 at
  131 072), and with the linear target of Section 2.7 it is three times below them
  at 131 072 (5.2e-5) and four times below at 262 144 (3.4e-5). The 1000-point
  leaves that help f1 hurt f3 here (4.5e-4):
  under the pseudo-PDF the error that counts sits at low β², near threshold, where
  the amplitude varies fastest and the smaller, more often refitted leaves do
  better. The networks' f3 curve flattens (1.9e-4 at 10⁵, 1.1e-4 at 10⁶) while the
  tree's still falls as N⁻⁰·⁹.
* **The folding is worth a factor 3 at equal N** (1.0e-5 against 3.2e-6 on f1,
  1.0e-3 against 3.5e-4 on f3 at 131 072), as four times the point density should
  give. On the full domain, which is what the networks were trained on, the
  300-point tree is 5 times below them on f1 and 6 times above them on f3 at
  131 072; 1000-point leaves bring f1 to 7.9e-6 and leave f3 at 1.1e-3. The
  folding is an exact symmetry, so it is legitimate for either side, but the two
  comparisons should be read with that in mind.
* **The package defaults** (cube inputs, Matérn 3/2, 100-point leaves, Sobol) end
  at 1.2e-3 and 6.8e-3: the representation, the leaf size and the stream are what
  separate them from the rows above, not the method.
* **Cost.** 35–55 ms per update for the 300-point tree on the invariants on a
  shared four-core machine; the networks' cost is their training, which the paper
  does not quote per point.

### 2.2 Against Table 2 of arXiv:2601.00950

"How to Trust Learned Loop Amplitudes" (Bahl, Braun, Heinrich, Plehn, Revelli)
trains a 5 × 512 MLP on the same tree-level amplitudes with 8·10⁴ LO-unweighted
points under the pseudo-PDF and reports, averaged over five runs, the mean and
median relative error on an LO-unweighted test sample (their Table 2): g g → t t̄ H
mean |Δ| (5.5 ± 0.4)e-4, median 7.3e-5, improved to (2.34 ± 0.04)e-4 and 3.2e-5 by
a 1.5·10⁴-point KDE resample into the low-accuracy regions; q q̄ → t t̄ H
(9.6 ± 0.7)e-5 and 2.0e-5, improved to 5.5e-5. The tree's mean relative error on
the 5000 LO-unweighted test points (`eps_unweighted`):

| tree (paper's weights, folded domain) | f1 @65 536 | f1 @131 072 | f3 @65 536 | f3 @131 072 |
|---|---|---|---|---|
| 300-point leaves, mixed stream | 5.2e-6 | 3.2e-6 | 6.6e-4 | 3.5e-4 |
| 1000-point leaves, refit at splits, mixed stream | 2.7e-6 | 1.3e-6 | 8.0e-4 | 4.5e-4 |
| 300-point leaves, choosing its points | 2.7e-6 | | 3.5e-4 | |

On f1 the tree is 20 to 70 times below their network; on f3 it matches their
plain network at 65 536 points and their resampled one at 131 072, with a median
relative error (on the uniform set) of 2e-6 against their 3e-5 to 7e-5: the
tree's residual error is entirely in the tail, as its diagnosis showed. Their
domain is θ ∈ [0, π] and φ ∈ [0, 2π], eight times the folded volume, so at equal
N they work with an eighth of the density; the full-domain rows of Section 2.1
cover half of that difference (f1 1.0e-5, f3 1.0e-3 at 131 072). Their learned
Gaussian σ is underconfident in 5D; the tree's calibrated σ covers 0.62–0.74 of
the test residuals at the 1σ level in every run here (target 0.68).

### 2.3 Where f3's error sits, and the remedies

The diagnosis of Section 7 holds under the paper's weights: the worst 1 % of test
points carry most of the error, in the forward and backward top configurations
(a gluon–top invariant small), which arXiv:2601.00950 also identifies as the
low-accuracy clusters of its network (its Fig. 10). Three ways of putting training
points there, each at 65 536 points with 300-point leaves:

| f3, paper's weights | ε @16 000 | ε @65 536 |
|---|---|---|
| mixed stream (reference) | 2.3e-3 | 7.2e-4 |
| LO-unweighted stream, nothing else | 2.3e-3 | 7.3e-4 |
| LO-unweighted stream with 20 % KDE-targeted points, interleaved | 1.8e-3 | 5.8e-4 |
| the same, the KDE points after the first 16 384 (the paper's two-step order) | 2.3e-3 | 6.1e-4 |
| the tree choosing its points from the mixed pool by `w |M|² σ` | 7.6e-4 | 3.8e-4 |

* The KDE resample of arXiv:2601.00950, reproduced with `tth_kde_stream.py` (a
  tree trained on the first 16 384 stream points, its relative errors on the
  uniform test set as KDE weights, Scott's bandwidth on the effective sample size),
  gives the tree the same factor 1.2–1.3 over its own stream that it gives their
  network. Interleaving the points beats appending them, as an online learner
  would expect.
* The tree's own acquisition does what the resample does, continuously and with
  no second stage: a factor 1.9 at 65 536 and 3 at 16 000 over the mixed stream.
  With a pool of 131 072 candidates it is ahead of the networks' curve at 16 000
  points and 1.3 times behind at 65 536; with a pool of 524 288 it is ahead from
  65 536 on (Section 2.6).
* Under the paper's weights the pure LO-unweighted stream no longer helps (it
  gave 2 × on the flat-flux metric): 7.3e-4 against the mixed stream's 7.2e-4 with
  300-point leaves, 8.1e-4 with 1000-point ones, and on f1 it is worse (7.6e-6
  against 5.2e-6). Its points sit at low β² and leave the rest of the cube to the
  Sobol half of the mixed stream.

### 2.4 Statistical amplification

`amplification_demo.py`, the measure of arXiv:2608.06450 ("Generative Amplification
with Surrogate Monte Carlo"): the fraction of the cross section in a bin of p_T(H),
computed from the tree's prediction over a million exact uniform points, is compared
with the exact fraction; the deviation is the surrogate's systematic error on that
bin, and `n_equiv = I (1 − I) / dev²` is the number of exact unweighted events whose
statistical error equals it. A second million exact points in the tail
p_T(H) > 3 m_t sharpens the tail bins (their stratified estimate). The 300-point tree
on the mixed stream at 65 536 points, paper's weights:

| | f1 | f3 |
|---|---|---|
| largest pull of a bin fraction, all bins | 0.00 | 0.5 |
| n_equiv, bulk bins (p_T(H) < 2 m_t) | > 5e5 | > 5e5 |
| n_equiv, tail bin [4, 6] m_t (164–205 training points in it) | > 8e6 | > 1.2e7 |
| amplification factor n_equiv / n_train, bulk to tail | > 21 to > 4e4 | > 16 to > 7e4 |

No bin deviates from the truth beyond the exact samples' own Monte Carlo error,
so every entry is a lower bound set by the test statistics, not by the tree. The
L-GATr of that paper reaches 2e4–1.7e5 in the tails of Z + jets with a million
training points. The calibrated σ propagated into a bin fraction as independent
per-point errors (their eq. 16) is 1e-8 to 1e-5, far below the resolution; the
paper reads its own numbers off a plateau rather than from that propagation. The
KS-based "differential" variant is not reported: with a finite holdout it
saturates at the holdout size.

### 2.5 Learning during the run against pretraining

`online_vs_pretrained.pdf` (`plot_online_vs_pretrained.py`); `gated_integration_demo.py`
with the paper's weights, 100 000 uniform Monte Carlo points, the 300-point tree on
the invariants, tolerance 1 %. The online tree starts
empty, calls the exact amplitude where its σ exceeds the tolerance and learns every
call. The pretrained tree learns exactly as many exact uniform points before the run
and is then frozen (`--pretrain N --frozen`), used either everywhere (ungated) or
with the same gate, whose calls it does not learn from.

| f1, 648 exact points for the online tree | exact calls, all / last 5000 | ε on the points used | integral shift | MC error |
|---|---|---|---|---|
| online, gated | 648 / 0.0 % | 4.2e-4 | −0.005 % | 0.28 % |
| pretrained on 648, ungated | 648 / 0 | 4.3e-4 | −0.001 % | 0.28 % |
| pretrained on 648, gated | 3909 / 2.6 % | 3.3e-4 | −0.001 % | 0.28 % |

| f3, 6892 exact points for the online tree | exact calls, all / last 5000 | ε on the points used | integral shift | MC error |
|---|---|---|---|---|
| online, gated | 6892 / 2.7 % | 2.2e-3 | +0.014 % | 0.32 % |
| pretrained on 6892, ungated | 6892 / 0 | 1.0e-2 | +0.29 % | 0.32 % |
| pretrained on 6892, gated | 16 601 / 8.8 % | 7.9e-4 | −0.003 % | 0.32 % |

* **At equal budget the pretrained tree is as good on f1 and four to five times
  worse on f3.** f1 is easy enough that 648 uniform points are 648 points wherever
  they come from. On f3 the online tree spent its 6892 evaluations where its σ was
  large, the collinear regions, and answers only where it is sure; the pretrained
  tree spent them uniformly and answers everywhere, with an integral shift of
  0.29 %, nine tenths of the Monte Carlo error and growing relative to it as the
  run goes on, against 0.014 %.
* **With the same gate the frozen tree never stops calling.** Its call rate is
  constant, 3.3 % on f1 and 9.7 % on f3, because the regions where it is unsure
  stay unsure; the online tree's rate falls to zero on f1 after 10 000 points and
  to 2.7 % and still falling on f3. Over this run the frozen tree spends 6 × (f1)
  and 2.4 × (f3) the exact evaluations for the same error control, and the ratio
  grows with the length of the run.
* **What online learning buys, then, is not accuracy per point but the two
  decisions a pretrained surrogate cannot make**: when enough has been learned,
  and where the next evaluation should go. A pretrained tree with a cleverer
  training set (Section 2.3's resample) would narrow the f3 gap at equal budget;
  it would not change the second row of each table.
* **Over 200 000 points** (`online_vs_pretrained200k.pdf`, with a third row of the
  absolute errors against the shrinking Monte Carlo error) the picture sharpens:
  the online tree's calls end at 663 (f1) and 8670 (f3, 1.5 % of the last points
  and falling), the gated frozen tree's at 8191 (12 ×) and 25 753 (3 ×); the
  ungated frozen tree's f3 integral, biased by 0.32 %, leaves the Monte Carlo
  error band at 150 000 points and is 1.4 Monte Carlo errors off at 200 000, while
  the online tree's deviation falls from 0.014 % to 0.009 %, 0.04 Monte Carlo
  errors.
* **Cost.** The online f3 run took 374 s against 8 s for the frozen tree on the
  same points: the tree's updates, not its predictions, are the cost, negligible
  against loop amplitudes and not against tree-level ones.

### 2.6 Beyond 131 072 points: does f3 cross the networks' curve?

`eps_vs_points_targets.png`. The gg training sets were enlarged to 524 288 Sobol
points and 262 144 LO-unweighted points (the first 65 536 of them the same as
before, so the earlier checkpoints are reproduced), and the 300-point tree on the
mixed stream was run to 524 288 points; the tree choosing its points now chooses
from this pool of 524 288 candidates instead of 131 072. Mean relative error on the
LO-unweighted test points:

| f3 | 65 536 | 131 072 | 262 144 | 524 288 |
|---|---|---|---|---|
| log target, mixed stream | 6.6e-4 | 3.5e-4 | 2.0e-4 | 1.0e-4 |
| log target, choosing its points from a pool of 131 072 | 3.5e-4 | – | – | – |
| log target, choosing its points from a pool of 524 288 | 2.2e-4 | 1.1e-4 | 9.2e-5 | – |
| linear target (Section 2.7), mixed stream | 3.1e-4 | 1.4e-4 | 7.4e-5 | – |
| linear target, choosing its points from a pool of 524 288 | 1.2e-4 | 5.2e-5 | 3.4e-5 | – |
| MLP(s) / L-GATr, Fig. 16 (1.1e-4 at 10⁶) | 2.7e-4 | 1.6e-4 | 1.4e-4 | 1.25e-4 |

* **The plain stream crosses the networks' curve at about 4·10⁵ points.** The log
  target on the mixed stream falls as N⁻⁰·⁹ to N⁻¹ through every doubling (3.5e-4,
  2.0e-4, 1.0e-4) while the networks' curve falls as N⁻⁰·²; at 524 288 the tree is
  at 1.0e-4 against their 1.25e-4 there and 1.1e-4 at a million points. The cost
  per learned point stays at 39–45 ms from 10³ to 5·10⁵ points: the leaf count
  grows, the leaf size does not.
* **The acquisition's pool size was the limit, not the acquisition.** With four
  times as many candidates the tree choosing its points reaches 2.2e-4 at 65 536
  against 3.5e-4 from the smaller pool, and is below the networks from 65 536 on:
  1.1e-4 at 131 072 against their 1.6e-4, 9.2e-5 at 262 144, below their value at
  a million points. The candidates are phase-space points with the analytic weight
  and the tree's own prediction; no exact evaluation is spent on them, so the pool
  is free. Its exhaustion shows in the last doubling (half the pool bought at
  262 144, the slope flattens to N⁻⁰·³), so 9.2e-5 is a pool-limited number, not
  the tree's.
* **With the linear target of Section 2.7 the tree is ahead at every N above
  65 536**: 1.4e-4 on the plain stream at 131 072 and 7.4e-5 at 262 144, and
  choosing its points 5.2e-5 at 131 072 (5.6e-5 in a repeat run) and 3.4e-5 at
  262 144, four times below the networks at that N and a third of their value at
  a million points, still falling as N⁻⁰·⁷ as the pool empties.

### 2.7 The target: log |M|², mass-scaled, or linear

The tree has learned log |M|² throughout. Two alternatives were tested with the same
tree (log-invariant inputs, RBF leaves, 300-point leaves, mixed stream): a mass
scale factored out, |M|² ŝ^k for k = 1, 2 (ŝ = 2 p₁·p₂, m_t = 1), in log space and in
linear space; and the linear amplitude itself, |M|² in units of its median over the
first 1000 points, with and without the ŝ factor; also the powers (|M|²)^½ and
(|M|²)^¼ as a middle ground. `--target logs1|logs2|lin|lins1|lins2|pow0.5|pow0.25`.
Mean relative error on the LO-unweighted test points:

| target | f3 @16 000 | f3 @65 536 | f1 @16 000 | f1 @65 536 |
|---|---|---|---|---|
| log \|M\|² (reference) | 2.2e-3 | 6.6e-4 | 2.2e-5 | 5.2e-6 |
| log(\|M\|² ŝ) | 2.3e-3 | 6.7e-4 | 2.3e-5 | – |
| log(\|M\|² ŝ²) | 2.5e-3 | 7.0e-4 | – | – |
| linear \|M\|² | 1.6e-3 | 3.1e-4 | 4.7e-5 | 1.1e-5 |
| linear \|M\|² ŝ | 3.0e-3 | 7.8e-4 | 4.3e-5 | – |
| linear \|M\|² ŝ² | 4.8e-3 | 1.2e-3 | – | – |
| (\|M\|²)^½ | 3.2e-3 | 8.6e-4 | 9.7e-5 | – |
| (\|M\|²)^¼ | 2.5e-3 | 6.7e-4 | 4.2e-5 | – |

* **A mass-scale factor does not help, in either space.** On the test set the
  1–99 % range of |M|² spans 2.1 decades, of |M|² ŝ the same 2.1 and of |M|² ŝ² 2.6:
  the variation of f3 is angular, the t-channel top poles 1/(p_g·p_t)², not the
  overall energy dependence. In log space the factor adds k log ŝ, a linear function
  of one input (log p₁·p₂), to a target the leaves fit with a constant prior mean;
  it adds structure and costs 2–6 %. In linear space it steepens the energy
  dependence and costs a factor 2.5–4 at 65 536 points.
* **The linear amplitude itself is the better target for f3**: a factor 2.1 at
  65 536 points, 2.5 at 131 072 and 2.7 at 262 144 (7.4e-5, Section 2.6) over
  log |M|², with the same median relative error and the same fraction of test
  points within 1 %. The metric is weighted by f, so it measures the absolute
  accuracy where |M|² is large, and a GP in linear space puts its accuracy there;
  in log space the error is relative and spent equally on the small values. The
  tail of f3 is its peaks, which is where the linear target wins.
* **On f1 the linear target is a factor 2 worse** (4.7e-5 against 2.2e-5 at
  16 000, 1.1e-5 against 5.2e-6 at 65 536): f1 has no poles, its log is close to a low-order rational function of
  the invariants, and the log target reaches relative errors of 10⁻⁶, far below
  anything an integration needs. The choice of target is a per-process one.
* **The powers are no middle ground**: (|M|²)^½ is worse than both log and linear
  on f3 and four times worse than log on f1; (|M|²)^¼ equals log on f3 and is
  twice worse on f1.
* **Side effects of the linear target.** The calibrated σ covers 0.60–0.64 of the
  test residuals at 1σ against 0.65–0.68 for the log target, and a GP in linear
  space can predict a negative amplitude where |M|² is tiny (none large enough to
  show in ε here; the power targets are clipped before the root). The update cost
  is 10–20 % lower (31–36 ms against 39–42 per point): the leaves' length scales
  settle faster. For an unweighting a floor at zero, or the log tree as a fallback
  where the linear prediction is within its σ of zero, would be needed.

### 2.8 The overlap between sibling leaves (theta)

`theta` sets the overlap between two sibling leaves as a fraction of the split
dimension's range; inside the overlap band a prediction is a linear blend of both
leaves' GPs, with the mixture variance. The default 10⁻⁴ is a hard partition in
practice (`theta = 0` is undefined, the routing probability divides by the overlap;
10⁻⁸ reproduces the default). Same trees as above, 65 536 points, mixed stream:

| theta | f3 (linear target) | f3 median rel. error | f3 coverage at 1σ | f1 (log target) | f1 coverage at 1σ |
|---|---|---|---|---|---|
| 10⁻⁴ (default) and 10⁻⁸ | 3.1e-4 | 5.1e-6 | 0.62 | 5.1e-6 | 0.73 |
| 0.02 | 2.8e-4 | 4.9e-6 | 0.67 | 5.8e-6 | 0.77 |
| 0.1 | 2.4e-4 | 7.7e-6 | 0.91 | 4.4e-6 | 0.92 |
| 0.3 | 5.3e-4 | 3.0e-5 | 1.00 | 8.3e-6 | 1.00 |

* **A moderate overlap buys 14–22 % on the mean error** (0.1 on both channels; 0.02
  gains 8 % on f3 and loses 13 % on f1, within the seed spread) at no cost in time.
  The gain is in the tail, where a leaf boundary cuts through a peak and the blend
  of two leaves is better than either edge.
* **It costs the median and the calibration.** At 0.1 the median relative error is
  1.5 times worse on f3 (the blend is worse than the better leaf in the bulk) and
  the blended σ covers 0.91–0.92 of the residuals instead of 0.68: the mixture
  variance adds the squared difference of the two means on top of the leaves' own
  σ. At 0.3 everything is worse. For a gated integration, which lives on the
  calibrated σ, the default or 0.02 is the safer choice; for the accuracy metric
  alone, 0.1.

### 2.9 Letting the leaves choose where to evaluate

`--configs leafacq --driver <tth_driver>`. Instead of ranking a fixed pool of
pre-evaluated points (Sections 2.3 and 2.6), the leaves decide. A Sobol sample of
2²⁰ candidate points in the cube is routed through the tree; each candidate carries
its analytic phase-space weight times ρ (`phase_space_weight()`, equal to the
driver's weight to 5·10⁻⁹), so scoring costs no amplitude evaluation. A leaf's
expected error contribution is Σ w |M|²_pred σ over its candidates, from its own GP;
the leaf with the largest one gets a batch of N̄/4 = 75 fresh points, drawn
uniformly from its candidates, evaluated with the MadGraph driver and learned.
Only that leaf is rescored; when it splits, its candidates are routed to the
children. This is acquisition from the amplitude code itself, with no pool of
pre-evaluated points. Mean relative error on the LO-unweighted test points:

| f3, linear target | 16 000 | 32 768 | 65 536 | 131 072 |
|---|---|---|---|---|
| plain mixed stream | 1.6e-3 | 7.1e-4 | 3.1e-4 | 1.4e-4 |
| choosing the highest-scoring points of a 524k pool (Section 2.6) | 0.8–1.0e-3 | 3.5e-4 | 1.2e-4 | 5.2–5.6e-5 |
| leaves choose; sum score, uniform pick, batch 75 | 8.6e-4 | 3.8e-4 | 1.5e-4 | 6.7e-5 |
| the same with the mean score | 2.3e-3 | 1.1e-3 | 2.4e-4 | – |
| the same, the leaf's highest-scoring candidates | 1.5e-3 | 5.3e-4 | 1.9e-4 | – |
| the same, batch 150 | 9.3e-4 | 3.9e-4 | 1.7e-4 | – |
| leaves choose, log target | 1.2e-3 | 5.2e-4 | 2.5e-4 | – |

| f1, log target | 16 000 | 32 768 | 65 536 |
|---|---|---|---|
| plain mixed stream | 2.2e-5 | 1.1e-5 | 5.2e-6 |
| choosing the highest-scoring points of the pool | 6.7e-6 | – | 2.7e-6 |
| leaves choose; sum score, uniform pick, batch 75 | 1.5e-5 | 6.9e-6 | 4.6e-6 |

* **The leaf-level decision gets most of the way.** On f3 it is a factor 2.1 below
  the plain stream at 131 072 (6.7e-5, 2.4 times below the networks) and 25 %
  behind the point-level acquisition from the pool; on f1 1.1–1.6 times below the
  stream and 1.7 times behind the pool acquisition. The two differ in what is
  ranked: a leaf's total, then uniform points inside it, against individual points
  ranked over the whole space. The second puts every point at a peak; the first
  spends part of each batch on the rest of the leaf.
* **Uniform inside the leaf beats the leaf's own top scores** (1.5e-4 against
  1.9e-4): a batch of 75 at one leaf's peak starves the rest of that leaf's GP. The
  pool acquisition gets away with top scores because it rescores the whole space
  after every 500 points and spreads them over many leaves.
* **The sum is the right leaf score**; the mean favours small leaves with a sharp
  peak over the large ones that hold most of the error (2.4e-4 against 1.5e-4).
  Batches of 150 are slightly worse than 75: finer steps, better decisions.
* **Cost and side effects.** 46–52 ms per learned point against 39–42 for the
  stream, the rescoring; 350–420 leaves at 65 536 against 280, because the bought
  points concentrate and leaves fill faster; σ coverage 0.58–0.65, a little below
  the stream's, because a batch of same-leaf residuals enters the calibration
  window together. The amplitude code is called on the fly in batches of 75 (15 µs
  per point here, so its share is nil).
* **What this is for.** No pre-evaluated pool is needed, only random phase-space
  points and the analytic weight, which any generator has, so this is the form an
  integrator can use directly. Scoring the candidates of the few leading leaves
  jointly, or mixing a uniform and a top-score share inside a batch, would be the
  next step towards the pool result.

## 3. Exact amplitudes (flat-flux weights): ε against the number of training points

`eps_vs_points.png`; uniform-test ε with flat-flux weights, with the networks of the
paper's Figure 16 (MLP(s) and L-GATr on LO-unweighted test points with the paper's
weights, read off the plot) for reference. The network columns are on a different
metric, so they are indicative only; Section 2 has the like-for-like comparison.

| points | f1 tree | f1 acq | f1 hybrid | f1 hybrid, 4 × 256 | f1 paper's networks | f3 tree | f3 acq | f3 hybrid | f3 hybrid, 4 × 256 | f3 paper's networks |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 2.7e-2 | 2.7e-2 | 1.8e-2 | 1.5e-2 | 3.5e-3 | 0.19 | 0.19 | 0.18 | 0.20 | 2.5e-2 |
| 2000 | 1.9e-2 | 1.7e-2 | 9.6e-3 | 6.6e-3 | | 0.15 | 0.10 | 0.12 | 9.8e-2 | |
| 4000 | 1.4e-2 | 1.1e-2 | 6.5e-3 | 3.1e-3 | 8e-4 (3000) | 0.12 | 6.5e-2 | 6.8e-2 | 5.0e-2 | 8e-3 (3000) |
| 8000 | 9.6e-3 | 7.4e-3 | 4.2e-3 | 1.9e-3 | | 8.7e-2 | 4.6e-2 | 5.3e-2 | 3.2e-2 | |
| 16 000 | 6.6e-3 | 5.0e-3 | 2.4e-3 | 1.3e-3 | 2.2e-4 (10⁴) | 6.4e-2 | 3.5e-2 | 4.2e-2 | 2.0e-2 | 1.7e-3 (10⁴) |
| 32 768 | 4.3e-3 | 3.5e-3 | 1.4e-3 | 8.3e-4 | | 4.4e-2 | 2.8e-2 | 2.9e-2 | 1.3e-2 | |
| 65 536 | 3.0e-3 | | 8.2e-4 | | | 3.1e-2 | | 2.1e-2 | | |
| 131 072 | 2.2e-3 | | 5.1e-4 | | 6e-5 (10⁵) | 2.3e-2 | | 1.5e-2 | | 2e-4 (10⁵) |

Unweighted-test ε is within 10 % of the uniform one for f1 and 10–20 % above it for
f3 (the tails of f3 are where the error sits). Update cost: tree 2 ms per point,
hybrid 4–16 ms (falling as refits become rarer), the larger network 33–78 ms.
Coverage of the calibrated sigma at the 68 % level: tree 0.65–0.74, hybrid
0.65–0.86, larger network 0.56–0.73.

Reading:

* **The hybrid beats the tree by 2 to 4 times at equal points, and the gap grows
  with the stream**: f1 5.1e-4 against 2.2e-3 at 131 072 points, where the hybrid at
  16 000 points is already where the tree ends; f3 1.5e-2 against 2.3e-2. A larger,
  longer-trained network halves the hybrid's error again (f1 8.3e-4 against 1.4e-3
  at 32 768, f3 1.3e-2 against 2.9e-2), at four to six times the update cost: the
  network's budget, not the leaves, sets the hybrid's accuracy here, and the larger
  network at 32 768 points is beyond the default hybrid at 131 072.
* **Choosing the points by the expected error contribution saves about a third of
  them**: the tree with acquisition reaches at 8000 points what the Sobol order
  reaches at 12 000 (f1) and at 16 000 what Sobol reaches at 30 000 (f3).
* **The paper's networks are far ahead of these default settings**: 10 to 25
  times lower ε at 10⁴ points than the default hybrid, 6 to 12 times than the larger
  network. With the defaults the tree family sits with the paper's classical
  interpolants, which reach 1 % at 10⁴–10⁵ points in 5D. Section 7 shows what
  closes the gap: the inputs, the leaf kernel and size, and the sampling. Every
  method here keeps improving with points, with no plateau in sight, which is the
  behaviour the paper reports for its classical methods and not for its networks.
* **Where the tree's errors sit.** On f3 the median relative error of the tree is
  1.5e-3 at 131 072 points with 87 % of test points within 1 %, while ε is 2.3e-2:
  the weighted error is dominated by the peaks of `w |M|²` (collinear tops, the
  Coulomb region), which a 100-point leaf in five dimensions resolves last.

## 4. Noisy amplitudes (flat-flux weights): 0.3 % relative noise on the training values

The paper's code carries a `NoisyEvaluator` with 3e-3 relative Gaussian noise; the
paper does not report results with it. Here the noise is added to `log |M|²` and
passed to the tree as the per-point sigma (`--noise 0.003`); the test values are
exact.

| points | f1 tree, exact / noisy | f1 hybrid, exact / noisy | f3 tree, exact / noisy | f3 hybrid, exact / noisy |
|---|---|---|---|---|
| 4000 | 1.4e-2 / 1.5e-2 | 6.5e-3 / 6.8e-3 | 0.12 / 0.12 | 6.8e-2 / 7.3e-2 |
| 16 000 | 6.6e-3 / 7.4e-3 | 2.4e-3 / 3.5e-3 | 6.4e-2 / 6.4e-2 | 4.2e-2 / 4.1e-2 |
| 32 768 | 4.3e-3 / 5.1e-3 | 1.4e-3 / 2.5e-3 | 4.4e-2 / 4.4e-2 | 2.9e-2 / 2.5e-2 |

The tree loses nothing on f3 and 5–20 % on f1; the hybrid loses nothing on f3 and up
to 1.7 times on f1, where its exact-data error is below the noise level and the
residual the leaves fit is noise-limited. The hybrid's coverage rises to 0.84–0.94
with noise: the network's error budget is added to every leaf's sigma, and with
noise-dominated residuals the calibration leaves it in place, as on the gg → HH grid.
A local version of that budget is the obvious next improvement.

## 5. Uncertainty-gated integration (flat-flux weights)

`gated_integration_demo.py`: uniform Monte Carlo points in chunks of 500; for each
point the tree predicts `log |M|²` with its calibrated sigma; where sigma exceeds the
tolerance the exact amplitude is evaluated and fed to the tree, elsewhere the tree's
value is used. The exact value of every point is also computed, so the shift the
surrogate introduces into the integral is known.

| surrogate, tolerance | points | exact calls, all / last 5000 | integral shift from the surrogate | Monte Carlo error |
|---|---|---|---|---|
| f1, tree on the cube, 1 % | 100 000 | 13.7 % / 5.4 % | +0.002 % | 0.32 % |
| f1, tree on the cube, 0.3 % | 200 000 | 30.0 % / 17.6 % | +0.001 % | 0.23 % |
| f1, hybrid on the cube, 1 % | 100 000 | 3.4 % / 0.8 % | +0.044 % | 0.32 % |
| f1, tree on log invariants, RBF, 300-point leaves, 1 % | 100 000 | 0.6 % / 0.0 % | −0.026 % | 0.32 % |
| f3, tree on log invariants, RBF, 300-point leaves, 1 % | 100 000 | 6.8 % / 2.9 % | +0.007 % | 1.05 % |

With a 1 % tolerance the cube tree answers 95 % of the last points itself and moves
the integral by a hundredth of its Monte Carlo error; the cube hybrid answers 99 % of
them and moves it by a seventh of the Monte Carlo error. The hybrid's shift is a
coherent bias of the network, the tree's local errors average out, so for a fixed
error budget the tree can be gated more aggressively than its tolerance alone
suggests and the hybrid less. The improved tree of Section 7 (`--inputs invariants
--kernel rbf --nbar 300`) needs 600 exact evaluations of f1 in 100 000 points, none
at all after the first 10 000, for a shift a twelfth of the Monte Carlo error, and
6800 of f3 (3 % of the last batch) for a shift a hundred-and-fiftieth of it. In every
case the shift is far inside the tolerance, since only points the surrogate is sure
of are taken from it.

## 6. What this demonstrates

On the paper's own metric (Section 2) the plain tree, given the log-invariants as
inputs, RBF leaves and the stream an integrator produces, is 18–44 times below the
paper's networks on f1 and within a factor 2 of them on f3 at 10⁵ points, crossing
their curve at 4·10⁵ (Section 2.6), with no network, learning online, and with its
curves still falling where theirs flatten; choosing its points from a large pool, or
learning the linear amplitude instead of its log (Section 2.7), it is below them on
f3 from 65 536 points on. It matches the five-run averages of arXiv:2601.00950 as
well. The 10–25 × deficit of the
package defaults (Section 3) is a matter of representation, leaf size and
sampling, not of the method. On top of that the tree family offers what the
networks do not: learning from the evaluations as they are made with no
retraining, which at equal budget is worth a factor 4 on f3 and, with the gate,
a call rate that falls to zero instead of staying constant (Section 2.5); a
per-point calibrated uncertainty that can gate when the exact amplitude is needed
(Sections 2.5 and 5) and that is worth more than 10⁷ exact events in the tails
(Section 2.4); the choice of which points to evaluate, which is what brings f3
below the networks' curve (Sections 2.3 and 2.6); and per-point noise in the
training values (Section 4). The real two-loop case, where the evaluations are
hours each and carry a QMC error, is in `BENCHMARK_RESULTS_hhgrid_amplitude.md`.

## Reproduce

```bash
cd examples
# Section 2: the paper's weights (the default); <data_dir> needs the {ch}_train_unweighted_rho.npz sets
python tth_amplitude_benchmark.py <data_dir> --channels uu,gg --configs tree --tag sweep_rhoDefault
python tth_amplitude_benchmark.py <data_dir> --channels uu,gg --configs tree --inputs invariants --kernel rbf --train mixed --nbar 300 --tag sweep_rhoMix300
python tth_amplitude_benchmark.py <data_dir> --channels uu,gg --configs tree --inputs invariants --kernel rbf --train mixed --nbar 1000 --retrain 1000000000 --tag sweep_rhoSplit1000
python tth_amplitude_benchmark.py <data_dir> --channels uu,gg --configs acq --nmax 65536 --inputs invariants --kernel rbf --train mixed --nbar 300 --tag sweep_rhoAcq300
python tth_amplitude_benchmark.py <full_domain_data_dir> --channels uu,gg --configs tree --inputs invariants --kernel rbf --train mixed --nbar 300 --domain full --tag sweep_fullMix300
# Section 2.6: <big_data_dir> holds 524 288 Sobol and 262 144 LO-unweighted gg points (tth_amplitude/README.md, step 6)
python tth_amplitude_benchmark.py <big_data_dir> --channels gg --configs tree --nmax 524288 --inputs invariants --kernel rbf --train mixed --nbar 300 --tag sweep_rhoMix300big
python tth_amplitude_benchmark.py <big_data_dir> --channels gg --configs acq --nmax 262144 --inputs invariants --kernel rbf --train mixed --nbar 300 --tag sweep_rhoAcq300big
python tth_amplitude_benchmark.py <big_data_dir> --channels gg --configs tree --nmax 262144 --inputs invariants --kernel rbf --train mixed --nbar 300 --target lin --tag sweep_scaleLinBig
python tth_amplitude_benchmark.py <big_data_dir> --channels gg --configs acq --nmax 262144 --inputs invariants --kernel rbf --train mixed --nbar 300 --target lin --tag sweep_scaleLinAcqBig2
# Section 2.7: the targets, e.g.
python tth_amplitude_benchmark.py <data_dir> --channels gg --configs tree --nmax 65536 --inputs invariants --kernel rbf --train mixed --nbar 300 --target logs1 --tag sweep_scaleLogS1
# Section 2.8: theta, e.g.
python tth_amplitude_benchmark.py <data_dir> --channels gg --configs tree --nmax 65536 --inputs invariants --kernel rbf --train mixed --nbar 300 --target lin --theta 0.1 --tag sweep_theta0.1
# Section 2.9: the leaves choose, fresh points from the driver
python tth_amplitude_benchmark.py <data_dir> --channels gg --configs leafacq --driver <tth_driver> --nmax 131072 --inputs invariants --kernel rbf --train mixed --nbar 300 --target lin --tag sweep_leafSum75
python tth_kde_stream.py <data_dir> gg <path to tth_driver> <kde_data_dir> --order interleaved
python tth_amplitude_benchmark.py <kde_data_dir> --channels gg --configs tree --nmax 65536 --inputs invariants --kernel rbf --train unweighted --nbar 300 --tag sweep_kdeint300
python amplification_demo.py <data_dir> gg --uniform <1e6 uniform points> --targeted <1e6 points with p_T(H) > 3 m_t> --checkpoints 16000,65536
python gated_integration_demo.py <path to tth_driver> --inputs invariants --kernel rbf --nbar 300 --tol 0.01 --tag online_gg
python gated_integration_demo.py <path to tth_driver> --inputs invariants --kernel rbf --nbar 300 --tol 0.01 --pretrain 6824 --frozen --tag frozen_gated_gg
python gated_integration_demo.py <path to tth_driver> --inputs invariants --kernel rbf --nbar 300 --tol inf --pretrain 6824 --frozen --tag frozen_gg
# Sections 3-7: the flat-flux weights (--no-rho)
python tth_amplitude_benchmark.py <data_dir> --channels gg,uu --configs tree,hybrid --no-rho --tag exact
python tth_amplitude_benchmark.py <data_dir> --channels gg,uu --configs acq --nmax 32768 --no-rho --tag acq
python tth_amplitude_benchmark.py <data_dir> --channels uu --configs hybrid --nmax 32768 \
    --net hidden=256,depth=4,steps=600,polish_steps=10000 --tag bignet_uu
python tth_amplitude_benchmark.py <data_dir> --channels uu --configs tree,hybrid --nmax 32768 --noise 0.003 --tag noise0.003_uu
python gated_integration_demo.py <path to tth_driver> --config tree --tol 0.01 --npoints 100000 --no-rho
# Section 7: the variants (with --no-rho), e.g. the best plain tree and the best hybrid over the full stream
python tth_amplitude_benchmark.py <data_dir> --channels uu,gg --configs tree --inputs invariants --kernel rbf --train mixed --nbar 300 --tag sweep_finalTreeMix300
python tth_amplitude_benchmark.py <data_dir> --channels uu,gg --configs hybrid --inputs invariants --features invariants --train mixed \
    --net hidden=256,depth=4,steps=600,polish_steps=10000 --nmax 65536 --tag sweep_finalHybMixBig
python plot_tth_results.py
```
With `--no-rho`, `<data_dir>` needs the `{ch}_train_unweighted.npz` sets of `tth_amplitude/README.md` for `--train mixed`.

## 7. Improving on the 5D benchmark (flat-flux weights): what moves the error

A controlled sweep on f1 and f3 to 16 384 points (seed 1, one run each; the hybrid
variants with `steps_per_update=8`), ε on the uniform test set at 4000, 16 000 and,
where run, 32 768 points. The baselines are the rows of Section 3.

* **Inputs.** `--inputs invariants` feeds the tree the ten `log(p_i · p_j)` of the
  external momenta (the inputs of the paper's networks) instead of the five cube
  coordinates; `--features invariants` gives them to the hybrid's network while the
  tree keeps the cube (the `feature_map` argument of `FeatureNetLearner`).
* **Training points.** `--train unweighted` draws them with probability proportional
  to `f = w |M|²` (65 536 such points per channel, generated by rejection), which is
  the stream an importance-sampling integrator produces; `mixed` alternates Sobol
  and f-weighted points.
* **Leaves.** `--kernel` Matérn 5/2, RBF or the additive + Matérn kernel; `--nbar 300`;
  `--split max_uncertainty`.
* **Network.** `--net hidden=256,depth=4,steps=600,polish_steps=10000`; the "one fit
  at the end" row trains the 4 × 256 network once, on the first 15 000 points, with
  3000 L-BFGS iterations and 30 000 Adam steps, to see what the amortised online
  training costs against a converged fit.

| variant (ε on the uniform test set) | f1 @4000 | f1 @16000 | f1 @32768 | f1 @131072 | f3 @4000 | f3 @16000 | f3 @32768 | f3 @131072 | ms/pt (f1 @16k) |
|---|---|---|---|---|---|---|---|---|---|
| tree (baseline: Matern 3/2, Nbar 100, Sobol) | 1.39e-02 | 6.59e-03 | 4.30e-03 | 2.17e-03 | 1.16e-01 | 6.36e-02 | 4.42e-02 | 2.27e-02 | 2.0 |
| tree, chooses its points (w|M|^2 sigma) | 1.10e-02 | 4.99e-03 | 3.45e-03 | – | 6.46e-02 | 3.53e-02 | 2.81e-02 | – | 2.8 |
| hybrid (baseline: 3x128 network on the cube) | 6.45e-03 | 2.38e-03 | 1.43e-03 | 5.07e-04 | 6.76e-02 | 4.16e-02 | 2.91e-02 | 1.46e-02 | 6.4 |
| hybrid, 4x256 network | 3.13e-03 | 1.25e-03 | 8.34e-04 | – | 5.04e-02 | 2.03e-02 | 1.31e-02 | – | 32.5 |
| tree, Matern 5/2 | 9.81e-03 | 4.21e-03 | – | – | 1.05e-01 | 6.24e-02 | – | – | 2.6 |
| tree, RBF | 9.55e-03 | 3.67e-03 | – | – | 1.10e-01 | 6.65e-02 | – | – | 2.7 |
| tree, additive + Matern | 9.40e-03 | 4.28e-03 | – | – | 1.14e-01 | 6.77e-02 | – | – | 13.1 |
| tree, Nbar 300 | 9.20e-03 | 4.27e-03 | – | – | 8.14e-02 | 4.98e-02 | – | – | 14.7 |
| tree, training points ~ f | 1.11e-02 | 5.20e-03 | – | – | 6.17e-02 | 3.21e-02 | – | – | 6.0 |
| tree, Sobol + f-weighted alternating | 1.19e-02 | 5.93e-03 | – | – | 6.83e-02 | 3.44e-02 | – | – | 5.9 |
| tree, inputs = 10 log invariants | 3.62e-03 | 1.82e-03 | – | – | 5.42e-02 | 2.29e-02 | – | – | 8.9 |
| tree, split by max_uncertainty | 1.35e-02 | 6.65e-03 | – | – | 1.10e-01 | 6.45e-02 | – | – | 7.6 |
| hybrid, network on log invariants | 1.58e-03 | 5.68e-04 | – | – | 2.42e-02 | 1.08e-02 | – | – | 17.6 |
| hybrid, Sobol + f-weighted | 5.12e-03 | 2.07e-03 | – | – | 4.16e-02 | 1.82e-02 | – | – | 14.2 |
| hybrid, invariants + 4x256 network | 8.93e-04 | 3.00e-04 | – | – | 1.53e-02 | 7.44e-03 | – | – | 75.2 |
| hybrid, invariants + mixed + 4x256 | 6.42e-04 | 2.34e-04 | 1.46e-04 | – | 4.50e-03 | 2.02e-03 | 2.19e-03 | – | 72.5 |
| hybrid, invariants, one 4x256 fit at the end (3000 L-BFGS + 30000 Adam) | 1.39e-02 | 3.82e-04 | – | – | 1.16e-01 | 5.44e-03 | – | – | 43.7 |
| tree, invariants + RBF | 1.04e-03 | 4.07e-04 | – | – | 8.15e-02 | 2.77e-02 | – | – | 7.8 |
| tree, invariants + Matern 5/2 | 1.69e-03 | 7.51e-04 | – | – | 4.88e-02 | 2.06e-02 | – | – | 10.5 |
| tree, inputs = cube + invariants (15D) | 3.56e-03 | 1.73e-03 | – | – | 5.72e-02 | 2.26e-02 | – | – | 13.6 |
| hybrid, tree and network on invariants | 1.58e-03 | 5.92e-04 | – | – | 2.54e-02 | 1.01e-02 | – | – | 20.1 |
| hybrid, tree and network on invariants, 4x256 | 8.77e-04 | 2.78e-04 | – | – | 2.23e-02 | 6.64e-03 | – | – | 48.1 |
| tree, invariants, training ~ f | – | – | – | – | 1.90e-02 | 9.74e-03 | – | – | nan |
| tree, invariants + RBF, training ~ f | 7.51e-04 | 2.19e-04 | – | – | 1.54e-02 | 6.61e-03 | – | – | 7.9 |
| hybrid, tree and network on invariants, training ~ f | – | – | – | – | 7.05e-03 | 2.66e-03 | – | – | nan |
| FULL: tree, invariants + RBF, Sobol | 1.04e-03 | 4.07e-04 | 2.48e-04 | 7.50e-05 | 6.18e-02 | 2.38e-02 | 2.00e-02 | 7.91e-03 | 3.8 |
| FULL: tree, invariants + RBF, Sobol + f-weighted | 7.97e-04 | 2.66e-04 | 1.45e-04 | 5.07e-05 | 1.90e-02 | 7.78e-03 | 5.15e-03 | 1.81e-03 | 7.6 |
| FULL: hybrid, invariants (both), Sobol + f-weighted | 1.10e-03 | 4.40e-04 | 2.62e-04 | 9.36e-05 | 1.02e-02 | 3.98e-03 | 2.42e-03 | 1.07e-03 | 17.0 |
| FULL: same, 4x256 | 5.91e-04 | 1.97e-04 | 1.15e-04 | – | 4.42e-03 | 1.87e-03 | 1.46e-03 | – | 40.9 |
| tree, invariants + RBF, mixed, Nbar 300 | 1.90e-04 | 5.42e-05 | – | – | 1.35e-02 | 4.61e-03 | – | – | 13.9 |
| hybrid, invariants (both), mixed, RBF leaves | 9.65e-04 | 2.94e-04 | – | – | 1.02e-02 | 3.75e-03 | – | – | 9.8 |
| tree, invariants + RBF, mixed, 0.3 % noise on the values | 3.05e-03 | 2.11e-03 | 1.86e-03 | – | 2.08e-02 | 9.57e-03 | 6.64e-03 | – | 2.9 |
| FULL: tree, invariants + RBF, Sobol + f-weighted, Nbar 300 | 1.90e-04 | 5.38e-05 | 2.64e-05 | 7.42e-06 | 1.35e-02 | 4.60e-03 | 2.48e-03 | 7.88e-04 | 16.7 |
| tree, invariants + RBF, mixed, Nbar 300, target log(|M|^2 prod s_gt^2) | 2.20e-04 | 9.50e-05 | – | – | 1.29e-02 | 3.86e-03 | – | – | 63.1 |
| tree, invariants + RBF, mixed, Nbar 300, target log(|M|^2 prod s_gt) | – | – | – | – | 1.08e-02 | 3.39e-03 | – | – | nan |
| tree, invariants + RBF, training ~ f, Nbar 300, target log(|M|^2 prod s_gt^2) | – | – | – | – | 8.21e-03 | 2.72e-03 | – | – | nan |
| tree, invariants + RBF, mixed, Nbar 300, additive global GP | – | – | – | – | 1.41e-02 | 5.03e-03 | – | – | nan |
| tree, invariants + RBF, mixed, Nbar 300, seed 2 | – | – | – | – | 1.09e-02 | 3.79e-03 | – | – | nan |
| tree, invariants + RBF, mixed, Nbar 300, seed 3 | – | – | – | – | 1.10e-02 | 3.75e-03 | – | – | nan |
| tree, invariants + RBF, training ~ f, Nbar 300 | – | – | – | – | 7.86e-03 | 2.62e-03 | – | – | nan |

Reading:

* **The representation is the lever.** In the log-invariants the amplitude is
  nearly a low-order rational function, and ten-dimensional leaves fit it far better
  than five-dimensional ones fit the cube: the plain tree gains 3.6 × on f1 and
  2.8 × on f3 from the inputs alone, and with RBF leaves 16 × on f1 (4.1e-4 at
  16 000 points, a median relative error of 1e-5, every test point within 1 %).
  f3 is rougher (collinear tops), where Matérn 5/2 beats RBF (2.1e-2 against
  2.8e-2) and the gain is 3 ×. Adding the cube coordinates to the invariants
  (15 inputs) changes nothing. For the hybrid the network on the invariants is
  worth 4 × on both channels, with the tree on the cube or on the invariants alike.
* **Points drawn where the integrand is large are the second lever.** Alone they
  give the tree 1.3 × on f1 and 2 × on f3 (the weighted error lives in the peaks);
  on top of the invariants and RBF leaves they bring the plain tree to 2.2e-4 on f1
  and 6.6e-3 on f3 at 16 000 points, 30 and 10 times the baseline, at 8 ms per point
  and with no network; the hybrid with everything on the invariants reaches 2.7e-3
  on f3, 16 times its baseline. Since an integrator's stream *is* importance
  sampled, this is the distribution the tree sees in its intended use, not a trick.
* **Kernels, leaf size, the split criterion.** On their own the smoother kernels
  give 1.6–1.8 × on f1 and nothing on f3; the additive kernel 1.5 × and nothing, at
  five times the cost; `Nbar = 300` 1.5 × and 1.3 × at seven times the cost; the
  `max_uncertainty` split nothing at three times the cost. These are second-order
  next to the representation and the sampling.
* **The network budget.** The 4 × 256 network halves the hybrid's error at four to
  six times the update cost, in every setting; with the invariants and mixed points
  it reaches 2.3e-4 at 16 000 and 1.5e-4 at 32 768 on f1. A single converged fit at
  the end of the same stream gives 3.8e-4 against 3.0e-4 for the amortised online
  training with the same network, so the online schedule costs little.
* **Leaf size in the invariants.** With ten inputs a 100-point leaf is short of
  points for its ten length scales; `Nbar = 300` on the invariants with RBF leaves
  and the mixed stream gives a further 4 × on f1 (5.4e-5 at 16 000, a median
  relative error of 1e-6) and 1.4 × on f3, at twice the update cost. In the cube
  the same change gave 1.5 ×.
* **Noise.** With 0.3 % noise on the training values the improved tree sits at
  1.9e-3 (f1) and 6.6e-3 (f3) at 32 768 points: on f1 its exact-data error is now
  far below the noise, so the noise sets the floor; on f3 the noisy and exact runs
  coincide.

### Full streams with the best configurations

`eps_vs_points_improved.png`; the "full stream" rows of the table.

| configuration (131 072 points, mixed stream unless noted) | f1 | f3 | update cost |
|---|---|---|---|
| tree on the cube, Matérn 3/2 leaves, Sobol (Section 3 baseline) | 2.2e-3 | 2.3e-2 | 2 ms |
| tree on log invariants, RBF leaves, Sobol | 7.5e-5 | 7.9e-3 | 4 ms |
| tree on log invariants, RBF leaves | 5.1e-5 | 1.8e-3 | 8 ms |
| tree on log invariants, RBF leaves, `Nbar = 300` | 7.4e-6 | 7.9e-4 | 16 ms |
| tree on log invariants, RBF leaves, `Nbar = 1000`, refit only at splits | 3.4e-6 | 5.8e-4 | 50 ms |
| hybrid on log invariants, Matérn 3/2 leaves | 9.4e-5 | 1.1e-3 | 11 ms |
| hybrid on log invariants, 4 × 256 network (65 536 points) | 7.1e-5 | 9.4e-4 | 21 ms |
| MLP(s) / L-GATr of arXiv:2412.09534 at 10⁵ points (Figure 16, their metric, indicative) | 6e-5 | 2e-4 | |

* **f1 is solved by the plain tree.** On the log-invariants with RBF leaves it
  matches the paper's networks at 131 072 points (7.5e-5 with Sobol points,
  5.1e-5 with the mixed stream), and with 300-point leaves it is eight times below
  them (7.4e-6, a median relative error of 4e-7, every test point within 1 %),
  online, at 16 ms per point, with no network. Its curve keeps falling as a power
  law where the networks' flattens.
* **f3 closes to a factor 4.** The tree with 300-point leaves reaches 7.9e-4 and
  the hybrids 9.4e-4 to 1.1e-3 against the networks' 2e-4, from 100 × worse at the
  baseline, with no sign of a plateau; the mixed stream alone is worth 4 × on f3
  at 131 072 points (7.9e-3 with Sobol against 1.8e-3), because the error lives in
  the collinear peaks the importance-sampled points populate.
* **The network no longer carries the result.** With the right inputs the plain
  tree with RBF leaves beats the hybrid with Matérn leaves on f1 and nearly matches
  it on f3; the hybrid's remaining advantage is on f3 at small point counts
  (4.4e-3 against 1.4e-2 at 4000). RBF leaves inside the hybrid give 2 × on f1
  (2.9e-4 against 5.9e-4 at 16 000) and nothing on f3.

### 1000-point leaves, refit only at splits

`--nbar 1000 --retrain 1000000000` with gradual splitting: a leaf's hyperparameters
are fitted once, when it fills and splits, and its children inherit them; between
splits new points enter a leaf's kernel matrix without a refit. Against the
300-point leaves refitted every 25 points (the best row above), ε on the uniform
test set:

<!-- nbar1000 table -->
| points | f1, Nbar 300 | f1, Nbar 1000 | f3, Nbar 300 | f3, Nbar 1000 | leaves (f1), 300 / 1000 | ms/pt (f1), 300 / 1000 |
|---|---|---|---|---|---|---|
| 1000 | 7.2e-04 | 3.7e-04 | 4.3e-02 | 3.8e-02 | 4 / 2 | 18 / 25 |
| 2000 | 3.7e-04 | 1.6e-04 | 2.5e-02 | 3.6e-02 | 8 / 3 | 21 / 31 |
| 4000 | 1.9e-04 | 9.6e-05 | 1.3e-02 | 1.1e-02 | 16 / 6 | 22 / 71 |
| 8000 | 1.1e-04 | 6.1e-05 | 8.2e-03 | 6.7e-03 | 33 / 12 | 19 / 81 |
| 16000 | 5.4e-05 | 2.3e-05 | 4.6e-03 | 4.3e-03 | 66 / 25 | 17 / 69 |
| 32768 | 2.6e-05 | 1.1e-05 | 2.5e-03 | 2.1e-03 | 139 / 53 | 16 / 58 |
| 65536 | 1.4e-05 | 4.6e-06 | 1.7e-03 | 1.2e-03 | 286 / 106 | 16 / 49 |
| 131072 | 7.4e-06 | 3.4e-06 | 7.9e-04 | 5.8e-04 | 586 / 201 | 15 / 45 |
<!-- /nbar1000 table -->

* **f1 gains 2–3 ×** at every point count: 2.3e-5 at 16 000 and 3.4e-6 at
  131 072 (a median relative error of 2e-7), eighteen times below the paper's
  networks at 10⁵ points, and the curve still falls as N⁻¹·¹. With ten length
  scales a leaf is better determined by 1000 points than by 300, and the single
  fit at the split is as good as the 40 refits the smaller leaves get.
* **f3 gains 1.1–1.4 ×**, growing with the point count (4.3e-3 against 4.6e-3 at
  16 000, 5.8e-4 against 7.9e-4 at 131 072): its error sits in the collinear
  peaks, where a larger leaf extends the same smooth extrapolation over more
  points rather than resolving the peak (next subsection). At 131 072 points
  f3 is within a factor 3 of the paper's networks on this metric.
* **The cost is 3–4 × per point** (50–80 ms against 16–20; all sweep runs share
  a four-core machine three at a time, so these are relative numbers), from the
  O(n³) factorisations of 1000-point leaves. Refitting such leaves every 250
  points instead (`--retrain 250`, run to 8000 points) buys 1.2 × on f1 (4.9e-5
  against 6.1e-5) and 1.3 × on f3 (5.2e-3 against 6.7e-3) at twice the cost
  (150–185 ms per point), so the refit-at-splits schedule keeps nearly all of the
  accuracy at half the price. NBAR1000_FINAL

### Where f3's error sits

`tth_error_diagnostic.py`, the tree above (invariants, RBF, 300-point leaves, mixed
stream) at 16 384 points, ε = 3.4e-3, broken down over the uniform test set:

* 61 % of the error sum comes from 1 % of the test points, whose relative errors
  run to 290 % while the median is 3e-5; 85 % of it from the top fifth in energy
  (β² > 0.8, which holds 62 % of the integral) and 59 % from φ_t > 0.8 π, where a
  top can line up with a beam after the boost.
* For those worst points the smallest invariant is a gluon–top one (p_g·p_t,
  80 % of cases) rather than a top–top or Higgs one: the t-channel top propagator,
  |M|² ∝ 1/(p_g·p_t)², whose peaks sharpen as (E_t/m_t)² with the energy. In the
  log-invariants the peak is a linear trend, −2 log(p_g·p_t), and a leaf with a
  constant prior mean extrapolates it badly at the edge of its data.
* Two remedies follow: factor the poles out of the target, learning
  log(|M|² Π s_gt^p) (`--target pole1`, `pole2`), and give the leaves a global
  additive model in the invariants (`--global additive`). Neither helps. At
  16 000 points on the mixed stream the plain target gives 3.8e-3, 3.8e-3 and
  4.6e-3 over three seeds; the pole-subtracted targets give 3.4e-3 (one power)
  and 3.9e-3 (two), inside the seed spread, and the additive global model
  5.0e-3 at twice the cost. On f1, which has no gluon–top poles, the shift
  only adds structure (9.5e-5 against 5.4e-5).
* What moves f3 is the stream. With every training point drawn ∝ f instead of
  every other one, the same tree reaches 2.6e-3 at 16 000 points (2.7e-3 with
  the pole-subtracted target), 1.5 × below the mixed stream: the collinear
  peaks are learned from points in them, which the importance-sampled stream
  supplies, rather than from a prior shape.
