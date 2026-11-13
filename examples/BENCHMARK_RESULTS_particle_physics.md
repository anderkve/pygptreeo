# Particle-physics-style targets: GPTree with and without a global model

Question: on the kinds of functions that are emulated in particle physics (cross-sections
and spectra over many orders of magnitude, thresholds, resonances, oscillations, ratios,
Monte Carlo noise, likelihoods with nuisance parameters), what do the global models add
to the plain tree?

Short answer, from eleven test targets, streams of 10 000 points and three seeds:

* **The feature network helps on nine of the eleven targets.** With `global_mean='net'`
  the tree has 0.52 of the plain tree's test error (geometric mean over all
  targets and seeds) and a lower error in 28 of 33 runs: a tenth on the EFT signal
  strength, 0.35 to 0.49 on six targets, and about two thirds on the loop-induced width
  and on the signal yield with Monte Carlo noise.
* **Two targets gain nothing from a global model at 10 000 points.** On the oscillation
  probability `'net'` is level with the plain tree in two seeds and has 3.3 times its
  error in the third, and along the stream the plain tree is ahead throughout. On the jet
  spectrum the typical error is the same with and without the network (a median of 0.04%
  of the cross-section, a 95th percentile of 0.4 to 0.6%), but the RMSE, which a few test
  points at the kinematic edge dominate, is about twice the plain tree's.
* **With 40 000 points** (one seed) `'net'` has 0.47 of the plain tree's test
  error and is ahead on all eleven targets, these two included (0.79 on each): on the
  oscillation probability it overtakes the plain tree after about 20 000 points. For the
  same seed at 10 000 points the figure was 0.48.
* **The additive model does not help on these targets.** `global_mean='additive'` has
  1.09 of the plain tree's RMSE (6 to 14% lower on six targets, level on three,
  higher on the same two as the network), and its median error is two or more times the
  plain tree's on nine targets. Beside the network it ends with weight zero on every
  target (0.10 in one seed of one). That is the expected outcome for functions whose structure is in
  thresholds and resonances on oblique lines; the gains of the additive model are on
  targets with additive structure (`BENCHMARK_RESULTS_additive_sums.md`).
* **Held-out selection matters for the network, also without noise.** The network alone
  with its earlier training has 6.1 times the plain tree's error on the
  oscillation probability; publishing the weights with the lowest error on a held-out
  fifth of the points, as `'net'` does, brings that to 1.6, at a cost of at
  most 16% on the other targets.
* **On a sampler stream** (a Metropolis walk on the 10D likelihood, one seed) `'net'` has
  0.16 of the plain tree's error around the second half of the walk.
* The typical error is small with and without the network (a median of 0.04 to 1.2% of
  the quantity for the logarithmic targets); what the network improves is the tail at
  thresholds, dips and funnels, by factors of two to five in the 95th percentile.
* The run time of `'net'` is 1.9 times that of the plain tree at 10 000 points
  and 1.4 times at 40 000.

![Test error of every configuration on every target](results/particle_physics/results.png)

## The targets

`particle_physics_targets.py` has eleven functions with the structure of quantities that
are emulated in particle physics. Each is cheap (a closed form, a small quadrature or a
4 x 4 eigenproblem) and schematic: right in structure, not in normalisation. Inputs are
in the unit cube and are mapped to physical ranges, masses and energies logarithmically.

![Two-input slices of the targets](results/particle_physics/gallery.png)

| target | inputs | quantity | what makes it hard |
|---|---|---|---|
| `resonance_xsec` | 4 | log10 of a production cross-section through a mediator that decays invisibly | sixteen orders of magnitude (between the 1st and the 99th percentile); a square-root threshold on the oblique line `M = 2 m`, below which the rate drops by three orders or more; a branching ratio that is a ratio of couplings |
| `dilepton_spectrum` | 4 | log10 of a differential cross-section with a Z' | a peak of 0.5 to 12% relative width on the diagonal `m_ll = M`, an interference dip below it, a continuum falling over nine orders |
| `jet_spectrum` | 4 | log10 of the inclusive jet cross-section in p_T and rapidity | close to a power law over most of the cube; a steep fall towards the kinematic edge at large rapidity, with a kink where that edge starts to cut the rapidity range; fourteen orders of magnitude |
| `oscillation_probability` | 5 | P(nu_mu -> nu_e) at 1300 km in matter | oscillates in `1 / E` with a phase proportional to `dm31 / E`; periodic in delta_CP; amplitudes that are products of functions of single angles |
| `branching_ratio` | 5 | invisible branching ratio of a Higgs-like scalar | between 0 and 1; unchanged when all couplings are scaled together; thresholds for WW, tt and, on a diagonal, the dark fermion |
| `loop_decay_width` | 4 | log10 of a loop-induced diphoton width | cusps where the scalar crosses a pair threshold (one on a diagonal); valleys where the loops cancel |
| `relic_density` | 4 | log10 of a dark-matter relic density | a funnel several orders of magnitude deep along `m_A = 2 m`, asymmetric from the thermal average; a threshold; a coupling that enters as a power |
| `neutralino_mass` | 4 | mass of the lightest neutralino, GeV | follows the smallest of three mass parameters, with rounded kinks where two of them cross; flat in the others |
| `eft_signal_strength` | 6 | signal strength in an effective field theory | a ratio of quadratic polynomials in six Wilson coefficients; smooth; every pair of inputs coupled |
| `signal_yield_mc` | 4 | log10 of a signal yield estimated from 10 000 simulated events per point | Monte Carlo noise that the simulation declares: below a percent where the efficiency is high, of order one where a few events pass; an efficiency that turns on along a diagonal of the mass plane |
| `global_fit_loglike` | 10 | log-likelihood of a toy global fit, saturating at -40 | three model parameters and seven nuisance parameters; Poisson terms, one-sided limits, a curved boundary of the allowed region and a plateau where a point is excluded |

For the targets without Monte Carlo noise the observation is exact and the declared sigma
is the numerical precision a user would quote: 0.1% of the quantity for the targets given
as a logarithm (4.3e-4 dex), 1e-4 for the probability and the branching ratio, 0.01 GeV,
and 1e-3 for the signal strength and the log-likelihood.

## Setup

`benchmark_particle_physics.py` streams `N` points into a tree, predicting each point
before it is used, and then predicts 3000 test points. Errors are against the noise-free
target.

* Streams: `uniform`, a random scan of the unit cube with a uniform test set; and, for the
  likelihood target, `sampler`: every proposal of a Metropolis random walk on the
  likelihood (step 0.04 per input), accepted or not, with the test set drawn around the
  second half of the walk.
* Tree settings, the same for every configuration: `Nbar=100`, `theta=1e-4`, a retrain
  every 25 points, gradual splitting, calibrated sigma, the default leaf kernel with one
  optimiser restart. The network runs with `steps_per_update=8`.
* Configurations: the plain tree; `global_mean='additive'`; `global_mean='net'` (the
  network beside the additive model); the network alone as it was before the additive
  model was added (`additive=False`); and the network alone with the training it has in
  the combination (`additive=False, holdout=0.2, patience=None`: a fifth of the points
  held out, the weights with the lowest error on them published).
* 10 000 points and seeds 1 to 3 for the uniform stream, one seed for the sampler
  stream and for 40 000 points. A table entry is the geometric mean over the seeds.
* Run times are from four single-threaded runs side by side on a four-core machine.

## Results: uniform stream, 10 000 points

Test RMSE over the range of the target, geometric mean of three seeds:

| target | plain tree | `'additive'` | `'net'` | network alone | network alone, held-out selection |
|---|---|---|---|---|---|
| resonance_xsec (4D) | 0.0093 | 0.0086 | 0.0044 | **0.0038** | 0.0043 |
| dilepton_spectrum (4D) | 0.0179 | 0.0168 | 0.0069 | 0.0074 | **0.0067** |
| jet_spectrum (4D) | **8.90e-05** | 2.39e-04 | 2.02e-04 | 1.54e-04 | 1.59e-04 |
| oscillation_probability (5D) | **0.0070** | 0.0104 | 0.0109 | 0.0428 | 0.0110 |
| branching_ratio (5D) | 0.0393 | 0.0401 | 0.0192 | **0.0174** | 0.0182 |
| loop_decay_width (4D) | 0.0150 | 0.0141 | 0.0098 | **0.0087** | 0.0097 |
| relic_density (4D) | 0.0064 | 0.0065 | 0.0023 | **0.0022** | 0.0022 |
| neutralino_mass (4D) | 0.0014 | 0.0012 | 5.49e-04 | **4.98e-04** | 5.39e-04 |
| eft_signal_strength (6D) | 0.0024 | 0.0023 | **2.34e-04** | 2.40e-04 | 2.45e-04 |
| signal_yield_mc (4D) | 0.0022 | 0.0018 | 0.0015 | **0.0014** | 0.0015 |
| global_fit_loglike (10D) | 0.0320 | 0.0326 | 0.0117 | **0.0099** | 0.0115 |

The same relative to the plain tree, with the range over the seeds in brackets:

| target | `'additive'` | `'net'` | network alone | network alone, held-out selection |
|---|---|---|---|---|
| resonance_xsec | 0.93 (0.82 to 1.07) | 0.47 (0.41 to 0.51) | 0.41 (0.37 to 0.45) | 0.46 (0.39 to 0.51) |
| dilepton_spectrum | 0.94 (0.90 to 0.97) | 0.38 (0.33 to 0.42) | 0.41 (0.37 to 0.47) | 0.37 (0.32 to 0.42) |
| jet_spectrum | 2.68 (1.80 to 3.46) | 2.27 (1.56 to 2.99) | 1.74 (0.83 to 2.76) | 1.79 (1.05 to 2.48) |
| oscillation_probability | 1.49 (1.35 to 1.62) | 1.55 (0.98 to 3.27) | 6.11 (4.86 to 8.71) | 1.57 (0.97 to 3.46) |
| branching_ratio | 1.02 (0.93 to 1.07) | 0.49 (0.43 to 0.58) | 0.44 (0.40 to 0.50) | 0.46 (0.42 to 0.52) |
| loop_decay_width | 0.94 (0.84 to 1.17) | 0.66 (0.53 to 0.92) | 0.58 (0.53 to 0.70) | 0.65 (0.54 to 0.88) |
| relic_density | 1.01 (0.79 to 1.40) | 0.35 (0.29 to 0.46) | 0.34 (0.27 to 0.43) | 0.34 (0.28 to 0.45) |
| neutralino_mass | 0.90 (0.81 to 0.97) | 0.41 (0.36 to 0.44) | 0.37 (0.35 to 0.38) | 0.40 (0.34 to 0.45) |
| eft_signal_strength | 0.94 (0.92 to 0.96) | 0.10 (0.09 to 0.10) | 0.10 (0.10 to 0.10) | 0.10 (0.09 to 0.11) |
| signal_yield_mc | 0.86 (0.80 to 0.91) | 0.68 (0.57 to 0.88) | 0.64 (0.57 to 0.72) | 0.68 (0.59 to 0.91) |
| global_fit_loglike | 1.02 (0.85 to 1.37) | 0.36 (0.28 to 0.42) | 0.31 (0.22 to 0.43) | 0.36 (0.27 to 0.42) |
| **geometric mean** | **1.09** | **0.52** | **0.54** | **0.50** |
| runs with a lower error than the plain tree | 21 of 33 | 28 of 33 | 28 of 33 | 28 of 33 |

`'net'` and the network alone with held-out selection differ by 4% or less on nine
targets (1.03 in the geometric mean): at 10 000 points the combination has
given all the weight to the network, so what is left of the additive model is the cheap
refit of the leaves. The jet spectrum (1.27) and the branching ratio (1.06) are the
exceptions.

The error on each point before it is used, over the second half of the stream, relative
to the plain tree:

| target | `'additive'` | `'net'` | network alone | network alone, held-out selection |
|---|---|---|---|---|
| resonance_xsec | 0.84 | 0.48 | 0.45 | 0.47 |
| dilepton_spectrum | 0.95 | 0.39 | 0.39 | 0.39 |
| jet_spectrum | 1.67 | 1.50 | 1.12 | 1.09 |
| oscillation_probability | 1.85 | 3.54 | 8.47 | 3.97 |
| branching_ratio | 1.11 | 0.56 | 0.56 | 0.54 |
| loop_decay_width | 0.97 | 0.68 | 0.65 | 0.69 |
| relic_density | 0.88 | 0.33 | 0.31 | 0.32 |
| neutralino_mass | 1.12 | 0.47 | 0.40 | 0.45 |
| eft_signal_strength | 0.99 | 0.11 | 0.11 | 0.11 |
| signal_yield_mc | 0.87 | 0.76 | 0.70 | 0.76 |
| global_fit_loglike | 1.04 | 0.44 | 0.40 | 0.45 |
| **geometric mean** | **1.08** | **0.57** | **0.57** | **0.56** |
| runs with a lower error than the plain tree | 17 of 33 | 27 of 33 | 28 of 33 | 28 of 33 |

### In the unit of the target

RMSE over the range hides what an error means for the quantity. Median and 95th
percentile of the absolute test error (seed 1), for the logarithmic targets as a relative
error of the quantity itself:

| target | unit | plain tree | `'additive'` | `'net'` | network alone | network alone, held-out selection |
|---|---|---|---|---|---|---|
| resonance_xsec | relative error of the quantity | 0.43% / 78% | 1.2% / 45% | 0.68% / 20% | 0.69% / 17% | 0.74% / 22% |
| dilepton_spectrum | relative error of the quantity | 0.72% / 136% | 1.9% / 111% | 1.2% / 32% | 0.99% / 24% | 0.98% / 30% |
| jet_spectrum | relative error of the quantity | 0.039% / 0.46% | 0.12% / 0.89% | 0.045% / 0.42% | 0.042% / 0.34% | 0.043% / 0.37% |
| oscillation_probability | probability | 0.00014 / 0.0026 | 0.00033 / 0.0038 | 0.00014 / 0.0028 | 0.0003 / 0.0097 | 0.00013 / 0.0028 |
| branching_ratio | branching ratio | 0.00021 / 0.039 | 0.004 / 0.049 | 0.00078 / 0.022 | 0.00058 / 0.015 | 0.00081 / 0.02 |
| loop_decay_width | relative error of the quantity | 0.61% / 25% | 1.3% / 24% | 0.67% / 13% | 0.76% / 13% | 0.73% / 13% |
| relic_density | relative error of the quantity | 0.099% / 32% | 0.39% / 22% | 0.17% / 8.6% | 0.23% / 9.4% | 0.16% / 8.5% |
| neutralino_mass | GeV | 0.059 / 1.4 | 0.16 / 1.7 | 0.037 / 0.59 | 0.039 / 0.53 | 0.038 / 0.51 |
| eft_signal_strength | signal strength | 0.0028 / 0.016 | 0.0026 / 0.014 | 0.00033 / 0.0013 | 0.00038 / 0.0014 | 0.00035 / 0.0014 |
| signal_yield_mc | relative error of the quantity | 1.2% / 7.3% | 1.2% / 5.9% | 0.66% / 3.9% | 0.72% / 3.9% | 0.69% / 3.6% |
| global_fit_loglike | log-likelihood | 0.12 / 2.8 | 0.48 / 2.1 | 0.057 / 0.6 | 0.046 / 0.45 | 0.052 / 0.58 |

The typical error is small everywhere, and the tails are where the thresholds, dips and
funnels are: a point next to a cliff of three orders of magnitude is wrong by a large
factor if the cliff is misplaced by a fraction of the spacing of the data. The network
shortens these tails by factors of two to five (twelve on the EFT signal strength). It
lowers the median error as well on four targets (neutralino mass, EFT signal strength,
signal yield, likelihood), leaves it where it was on the jet spectrum and the oscillation
probability, and raises it on the five targets with the sharpest thresholds, by 10% on
the loop-induced width and by up to a factor of 3.7 on the branching ratio. The additive
model on its own raises the median error on nine targets, by factors of 2 to 19: the
RMSE, on which it is level with the plain tree, flatters it.

### Which model `'net'` uses

The weights that `'net'` gives the two models at the end of the stream, and the share of
the target's variance that the additive model explains on its own:

| target | additive share | weights (additive, network), per seed |
|---|---|---|
| resonance_xsec | 0.75 | (0.00, 1.00), (0.00, 1.00), (0.00, 1.00) |
| dilepton_spectrum | 0.81 | (0.00, 1.00), (0.00, 1.00), (0.00, 1.00) |
| jet_spectrum | 0.96 | (0.00, 1.00), (0.00, 1.00), (0.00, 1.00) |
| oscillation_probability | 0.58 | (0.00, 1.00), (0.10, 0.91), (0.00, 1.00) |
| branching_ratio | 0.55 | (0.00, 1.00), (0.00, 1.00), (0.00, 1.00) |
| loop_decay_width | 0.85 | (0.00, 1.00), (0.00, 1.00), (0.00, 1.00) |
| relic_density | 0.76 | (0.00, 1.00), (0.00, 1.00), (0.00, 1.00) |
| neutralino_mass | 0.64 | (0.00, 1.00), (0.00, 1.00), (0.00, 1.00) |
| eft_signal_strength | 0.88 | (0.00, 1.00), (0.00, 1.00), (0.00, 1.00) |
| signal_yield_mc | 0.99 | (0.00, 1.00), (0.00, 1.00), (0.00, 1.00) |
| global_fit_loglike | 0.76 | (0.00, 1.00), (0.00, 1.00), (0.00, 1.00) |

An additive model explains 55 to 99% of the variance of these targets, and still the
leaves gain little from having it subtracted (the `'additive'` column above): what is
left is the part with the thresholds and the oblique structure, which is as hard for a
leaf as the target itself. The network learns that part too, so the held-out points give
it the full weight.

### Uncertainty

Fraction of the test points whose error is within the predicted sigma (0.68 is nominal),
mean of three seeds:

| target | plain tree | `'additive'` | `'net'` | network alone | network alone, held-out selection |
|---|---|---|---|---|---|
| resonance_xsec (4D) | 0.66 | 0.66 | 0.71 | 0.78 | 0.77 |
| dilepton_spectrum (4D) | 0.67 | 0.68 | 0.69 | 0.68 | 0.70 |
| jet_spectrum (4D) | 0.69 | 0.62 | 0.63 | 0.66 | 0.60 |
| oscillation_probability (5D) | 0.66 | 0.61 | 0.56 | 0.69 | 0.59 |
| branching_ratio (5D) | 0.76 | 0.64 | 0.68 | 0.64 | 0.72 |
| loop_decay_width (4D) | 0.67 | 0.67 | 0.68 | 0.64 | 0.67 |
| relic_density (4D) | 0.64 | 0.62 | 0.69 | 0.71 | 0.69 |
| neutralino_mass (4D) | 0.67 | 0.68 | 0.66 | 0.72 | 0.70 |
| eft_signal_strength (6D) | 0.67 | 0.69 | 0.50 | 0.42 | 0.38 |
| signal_yield_mc (4D) | 0.97 | 0.96 | 0.98 | 0.85 | 0.87 |
| global_fit_loglike (10D) | 0.70 | 0.67 | 0.63 | 0.63 | 0.65 |

Two entries stand out. On the EFT signal strength the trees with a network cover only
0.38 to 0.50 on the test set: the error left after a very accurate network is small and
structured, and the leaves' sigma underestimates it. On the signal yield with Monte Carlo
noise every configuration covers more than nominal, the plain tree included (0.97): the
sigma is calibrated against the noisy observations and is scored here against the
noise-free yield.

### Cost

Seconds for the 10 000 points (prediction and update of every point, seed 1):

| target | plain tree | `'additive'` | `'net'` | network alone | network alone, held-out selection |
|---|---|---|---|---|---|
| resonance_xsec (4D) | 51 | 58 | 89 | 120 | 111 |
| dilepton_spectrum (4D) | 47 | 54 | 89 | 119 | 113 |
| jet_spectrum (4D) | 64 | 57 | 88 | 108 | 101 |
| oscillation_probability (5D) | 41 | 62 | 87 | 116 | 102 |
| branching_ratio (5D) | 42 | 61 | 91 | 126 | 112 |
| loop_decay_width (4D) | 43 | 60 | 84 | 116 | 102 |
| relic_density (4D) | 62 | 66 | 87 | 122 | 106 |
| neutralino_mass (4D) | 47 | 71 | 83 | 113 | 101 |
| eft_signal_strength (6D) | 44 | 72 | 94 | 131 | 122 |
| signal_yield_mc (4D) | 46 | 72 | 92 | 127 | 111 |
| global_fit_loglike (10D) | 52 | 92 | 111 | 159 | 150 |

Relative to the plain tree: `'additive'` 1.3 (0.9 to 1.7), `'net'` 1.9 (1.4 to 2.2), the network alone
2.5 (1.7 to 3.0), with held-out selection 2.3 (1.6 to 2.9) (geometric mean and range over the
targets). `'net'` is faster than the network alone because its leaves take up a new
version of the global model by a refit with their kernel hyperparameters kept.

## Results: a sampler stream

The 10D likelihood, 10 000 proposals of a Metropolis walk, test points around the second
half of the walk (one seed):

| | plain tree | `'additive'` | `'net'` | network alone | network alone, held-out selection |
|---|---|---|---|---|---|
| test RMSE / range | 0.0211 | 0.0192 | 0.0034 | 0.0042 | 0.0037 |
| relative to the plain tree | 1.00 | 0.91 | 0.16 | 0.20 | 0.18 |
| prequential RMSE / range, second half | 0.0617 | 0.0734 | 0.0578 | 0.0620 | 0.0422 |
| coverage on the test points | 0.71 | 0.73 | 0.77 | 0.65 | 0.59 |
| seconds | 62 | 81 | 124 | 142 | 145 |

The additive model does not help here either. The network does, by a factor of five to
six in the region the walk has explored. On the points at the front of the walk (the
prequential column) the differences are small, because every global model extrapolates
there; only the network alone with held-out selection is clearly ahead of the plain tree.

## Results: 40 000 points

One seed, uniform stream. Test RMSE over the range of the target:

| target | plain tree | `'additive'` | `'net'` | network alone | network alone, held-out selection |
|---|---|---|---|---|---|
| resonance_xsec (4D) | 0.0057 | 0.0050 | **0.0037** | 0.0039 | 0.0043 |
| dilepton_spectrum (4D) | 0.0127 | 0.0115 | 0.0058 | **0.0045** | 0.0055 |
| jet_spectrum (4D) | 5.05e-05 | 5.70e-05 | **4.01e-05** | 4.28e-05 | 4.26e-05 |
| oscillation_probability (5D) | 0.0025 | 0.0029 | 0.0020 | 0.0036 | **0.0020** |
| branching_ratio (5D) | 0.0208 | 0.0196 | 0.0119 | **0.0119** | 0.0121 |
| loop_decay_width (4D) | 0.0085 | 0.0099 | **0.0057** | 0.0065 | 0.0059 |
| relic_density (4D) | 0.0035 | 0.0029 | 0.0020 | 0.0021 | **0.0020** |
| neutralino_mass (4D) | 7.14e-04 | 0.0013 | 3.07e-04 | **2.42e-04** | 2.89e-04 |
| eft_signal_strength (6D) | 0.0015 | 0.0014 | **1.59e-04** | 1.61e-04 | 1.64e-04 |
| signal_yield_mc (4D) | 0.0017 | 0.0016 | **0.0010** | 0.0012 | 0.0011 |
| global_fit_loglike (10D) | 0.0219 | 0.0218 | 0.0057 | **0.0048** | 0.0055 |

Relative to the plain tree:

| target | `'additive'` | `'net'` | network alone | network alone, held-out selection |
|---|---|---|---|---|
| resonance_xsec | 0.87 | 0.65 | 0.69 | 0.75 |
| dilepton_spectrum | 0.91 | 0.46 | 0.35 | 0.43 |
| jet_spectrum | 1.13 | 0.79 | 0.85 | 0.84 |
| oscillation_probability | 1.16 | 0.79 | 1.44 | 0.78 |
| branching_ratio | 0.94 | 0.57 | 0.57 | 0.58 |
| loop_decay_width | 1.17 | 0.67 | 0.76 | 0.69 |
| relic_density | 0.81 | 0.57 | 0.60 | 0.56 |
| neutralino_mass | 1.77 | 0.43 | 0.34 | 0.40 |
| eft_signal_strength | 0.93 | 0.11 | 0.11 | 0.11 |
| signal_yield_mc | 0.94 | 0.58 | 0.67 | 0.64 |
| global_fit_loglike | 0.99 | 0.26 | 0.22 | 0.25 |
| **geometric mean** | **1.03** | **0.47** | **0.49** | **0.48** |
| runs with a lower error than the plain tree | 7 of 11 | 11 of 11 | 10 of 11 | 11 of 11 |

![Error along the stream](results/particle_physics/curves.png)

The curves are the error on each batch of 1000 points before the points are used.

* **The gain of the network holds at four times the data.** `'net'` has 0.47 of
  the plain tree's test error and is ahead on all eleven targets; for the same seed at
  10 000 points the figure was 0.48. Along the stream its error over the eleven
  targets is 0.59 of the plain tree's on the first 5000 points, 0.55 up to 10 000, 0.51
  up to 20 000 and 0.48 after that.
* **The gain moves between targets.** It shrinks on three, where the tree with the
  network improves little between 10 000 and 40 000 points while the plain tree keeps
  improving (resonance cross-section 0.41 to 0.65, relic density 0.33 to 0.57,
  loop-induced width 0.53 to 0.67). It stays within 0.05 on six, and it appears on the
  two targets that had none.
* **Oscillation probability.** `'net'` has 1.9 to 2.9 times the plain tree's error over
  the first 10 000 points, is level with it between 10 000 and 20 000, and has 0.74 of
  its error over the last 10 000 (0.79 on the test set). The network alone without
  held-out selection is at twice the plain tree's error to the end.
* **Jet spectrum.** With a network the tree is behind the plain tree by a factor of 1.1
  to 1.7 along the first 30 000 points and level with it on the last 10 000. On the test
  set `'net'` has 0.79 of the plain tree's RMSE: the median error is the same (0.02 to
  0.03% of the cross-section), and the largest error, at the kinematic edge, is now
  smaller with the network (0.015 dex) than without (0.027 dex).
* **`'net'` and the network alone with held-out selection are the same model here**
  (0.98 in the geometric mean, 0.86 to 1.06 from target to target); the
  additive model has weight zero on every target. The difference that remains is the run
  time, since the leaves of `'net'` take up a new version of the global model by the
  cheap refit.
* **The additive model on its own** stays level with the plain tree (1.03) and
  gets worse on one target late in the stream: on the neutralino mass its error doubles
  after 31 000 points. The fit of its hyperparameters at 30 720 points moves two inputs
  to length scales of 3 to 4% of the range. The additive model's own test error does not
  change (0.076 of the range before and after), but what it leaves for the leaves now has
  structure at their scale.
* **Uncertainty.** The one-sigma coverage on the test set is between 0.62 and 0.79, with
  three exceptions: the relic density with the additive model (0.49), the EFT signal
  strength with a network alone (0.55 and 0.57; 0.66 with `'net'`), and the signal yield
  with Monte Carlo noise (0.98 to 1.00, for the reason given above).
* **Cost.** 160 to 250 seconds for the plain tree, 261 to 353 for `'net'`.
  Relative to the plain tree: `'additive'` 1.2, `'net'` 1.4, the network
  alone 1.9, with held-out selection 1.8. The share of the global
  model in the run time falls with the length of the stream, because its refits come at
  doublings.


## Where a global model is slow to help

**Jet spectrum.** Close to a power law in the bulk, where the plain tree's median error
is 0.04% of the cross-section and its 95th percentile 0.4 to 0.5%. With the network both
are the same within a quarter. The RMSE is set elsewhere: at the largest rapidities,
near the transverse momentum above which the kinematic limit cuts the rapidity range,
the spectrum falls by orders of magnitude within a short distance. In a check at 4000
points the 30 worst of 3000 test points carried 94% of the squared error, for the plain
tree and for `'net'` alike. In that corner the largest error of the trees with a network
is 0.06 to 0.18 dex, against 0.02 to 0.03 dex for the plain tree, and that is the whole of
the factor in the RMSE. The additive model is worse throughout the cube here (in seed 1
its median error is three times the plain tree's): its fit compromises between the bulk
and the corner, which leaves a residual of about 0.5 dex that bends along p_T and
rapidity where the target itself is nearly straight. With 40 000 points the trees with
a network no longer lose in the corner, and `'net'` has 0.79 of the plain tree's RMSE.

**Oscillation probability.** The additive part of an oscillation whose phase is
`dm31 / E` is, along the energy, the oscillation averaged over `dm31`: a damped wave.
Subtracting it leaves the difference between a wave and its average, which oscillates as
much as the target and, unlike the target, changes sign. The network fits the wave incompletely,
and the error it leaves is again a wave. In both cases the residual is no easier for a
leaf than the target. Along the stream the plain tree is ahead of every global model, by
factors of 1.9 to 8.5 in the prequential error above. At the end `'net'` is level with
it in two seeds and at 3.3 times its error in the third; the network alone without
held-out selection is at 4.9 to 8.7 times in all three. With 40 000 points the order
changes: `'net'` is level with the plain tree between 10 000 and 20 000 points and has
0.74 of its error over the last 10 000.

There is no rule in the package yet that detects these cases: the weights of `'net'`
compare the two global models with each other on held-out points, not the tree with and
without them. A trial is the practical test, since the prequential error needs no test
set: stream the points into a plain tree and into one with `global_mean='net'`, and
compare the errors on the points before they were used. The order can change along the
stream, as on the oscillation probability, so a trial of a few thousand points settles
the question only for streams of about that length.

## Reproducing

    # one cell (a few minutes; add --N 40000 for the long streams); results as a line starting with RESULT
    OMP_NUM_THREADS=1 python examples/benchmark_particle_physics.py --targets relic_density --configs tree,additive,net,net_only,net_only_holdout --seeds 1
    # the sampler stream
    OMP_NUM_THREADS=1 python examples/benchmark_particle_physics.py --targets global_fit_loglike --stream sampler --configs tree,net
    # tables and figures from the stored lines
    python examples/benchmark_particle_physics.py --summarize examples/results/particle_physics/uniform_N10000_seed*.jsonl
    python examples/plot_particle_physics.py

`results/particle_physics/` has the result lines of every run.
