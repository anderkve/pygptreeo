# pyGPTreeO as an on-the-fly emulator for AM3 spectra

Can a `GPTree` with `output_model='pca'` learn the photon spectra produced by
[AM3](https://am3.readthedocs.io/) while a parameter scan is running, accurately
enough to replace AM3 calls? This document records the setup and the measurements
from the scripts in this directory.


## 1. Setup

### AM3

AM3 (Klinger et al. 2023, arXiv:2312.13371) solves the time-dependent kinetic
equations of photons, leptons and hadrons in a homogeneous magnetised region. The
PyPI package called `am3` is an unrelated process manager; AM3 itself is built
from the DESY GitLab source:

```bash
apt-get install libeigen3-dev libgsl-dev        # Eigen3 and GSL headers
git clone --depth 1 https://gitlab.desy.de/am3/am3.git
cd am3 && SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0 pip install .   # scikit-build-core + pybind11, ~2 min
```

(The version override is needed for a shallow clone without tags.)

### The model: one-zone synchrotron self-Compton (SSC) blazar

`am3_model.py` wraps AM3 as a function of six comoving parameters, the
classic leptonic blazar scan:

| parameter | meaning | scan range |
|---|---|---|
| log10 B | magnetic field [G] | -2 .. 1 |
| log10 R | radius of the region [cm] | 15 .. 17 |
| log10 L_e | comoving electron injection luminosity [erg/s] | 39.5 .. 43.5 |
| p | injection spectral index | 1.5 .. 3.0 |
| log10 gamma_min | minimum electron Lorentz factor | 0.5 .. 2.5 |
| log10 gamma_max | maximum electron Lorentz factor (exponential cutoff) | 3.5 .. 6.0 |

Processes: electron synchrotron with self-absorption, inverse Compton, gamma-gamma
pair production (internal absorption and the cascade), escape on the light-crossing
time; hadronic processes off. The photon grid is AM3's own log grid from 1e-7 eV
to 8e13 eV (482 bins, d ln E = 0.1). Each point is evolved for 10 light-crossing
times with a time step of 0.05 R/c, which is converged in the time step to 2e-4 dex
and steady to about 0.02 dex in the slowest bins.

The emulated quantity is the comoving `log10(E^2 n(E) + floor)` on that grid, with
a soft floor of 1e-2 eV cm^-3 (far below anything observable, see below). The
Doppler factor, redshift and luminosity distance only rescale energies and fluxes,
so they are applied analytically on top (`observed_sed`) and never enter the
emulator; the scan then varies seven parameters (the six above plus the Doppler
factor in 5..50) while the emulator input stays six-dimensional.

**Cost of one AM3 spectrum**: 0.24-0.26 s on one core (kernel initialisation is a
one-off 3 s; changing the magnetic field costs 2 ms). The leptonic SSC model is
close to the cheapest thing AM3 computes; hadronic runs are 10-100 times slower,
which makes emulation correspondingly more valuable there.

### The scan

`scan_emulator.py` fits a synthetic SED: 23 points from 3e-4 eV (sub-mm) to
300 GeV (the far-IR and MeV gaps removed, lower radio points dropped as
self-absorbed), generated from
`theta_true = (-0.7, 16.2, 41.8, 2.4, 1.3, 4.9, delta=20)` at z = 0.1 with 0.05 dex
Gaussian errors; chi^2 in log flux. The source peaks at 1.1e-11 erg cm^-2 s^-1
in the UV with a GeV flux of 2e-12, an ordinary bright BL Lac. The optimiser is
scipy's `differential_evolution` (popsize 12 x 7 = 84 per generation, 40
generations, 3444 evaluations, no polishing, fixed seed).

A `GPTree(n_outputs=482, output_model='pca')` learns the log-spectrum from every
AM3 call the scan makes (`sigma = 0.01 dex`). Two modes:

* **shadow**: AM3 is called at every point and the emulator's prediction, made
  before the call, is scored against it. The DE trajectory is independent of the
  emulator, so shadow runs with different tree settings see identical point
  streams, and the run doubles as the pure-AM3 reference.
* **active**: the emulator replaces AM3 whenever its propagated uncertainty on
  chi^2, `sum_i 2 |r_i| sigma_i / sigma_obs,i` (a conservative first-order bound),
  is below `tol_abs + tol_rel * chi2_emu`; otherwise AM3 is run and the tree
  updated. A fraction of the accepted points is also verified with AM3.

## 2. What the spectra look like (400-point Latin hypercube over the box)

`characterise_spectra.py`; the AM3 sample took 97 s.

* The peak of `E^2 n(E)` spans **10 decades** across the box (log10 peak from
  2.4 to 12.4 eV cm^-3). Below the self-absorption turnover and above the
  pair-production cutoff the log-spectrum falls off a cliff; the soft floor
  keeps those regions finite and smooth in theta. At the true parameters of the
  scan below, the floor corresponds to an observed 3e-21 erg cm^-2 s^-1.
* A global PCA of the log-spectra needs many components: 7 for 99 % of the
  variance, 16 for 99.9 %, 28 for 99.99 %. Held-out reconstruction error on the
  visible bins is 0.065 dex rms with 16 components, 0.032 dex with 24, 0.018 dex
  with 32 and 0.006 dex with 48. A "shape plus log-peak" representation does
  not reduce the rank. The whole-box rank is therefore high; the point of the
  scan test is that a scan concentrates, and the tree's reservoir basis follows
  the stream.
* **Batch baseline, 300 train / 100 test random points over the whole box**:
  0.15-0.18 dex rms, 0.07-0.08 dex median error, worst 2 dex, for every PCA
  setting from 16 to 56 components. The error is GP-limited (300 points in six
  dimensions), not basis-limited. `output_model='independent'` on every fourth
  bin gives the same 0.18 dex rms at 8 times the prediction cost (32 vs 4 ms
  per point) and 4 times the fit cost, so `'pca'` is the right leaf model here.
  The calibrated predicted sigma is very conservative at this stage
  (|err|/sigma median 0.07); in the scans it calibrates itself (section 3).

## 3. Scan results

The reference is a shadow run (16 components, Nbar 100, retrain every 10
points): AM3 at every evaluation, the emulator scored but never used. The DE
reaches chi^2 = 56.8 after 3444 evaluations (chi^2 = 33.9 at the true
parameters) with the usual SSC degeneracies (gamma_min and the Doppler factor
trade against each other); 40 generations do not fully converge this
seven-parameter problem, which is a property of the optimiser budget. Timings are
for one core with two or three scans running concurrently (AM3 then costs 0.3-0.6
s per call instead of 0.25 s); four concurrent AM3 processes on this 4-core VM
were 9 times slower per call, so never do that.

### 3.1 Learning curve along the scan (reference shadow run)

Errors are of the emulated observed log-flux at the 23 data energies, i.e. what
the likelihood sees. Blocks of five generations (420 evaluations):

| evaluations | median rms [dex] | p90 rms [dex] | worst max [dex] | median \|dchi2\| | \|err\|/sigma median | best true chi2 |
|---|---|---|---|---|---|---|
| 0-420 | 0.103 | 0.261 | 10.5 | 931 | 0.65 | 468 |
| 420-840 | 0.054 | 0.114 | 10.7 | 191 | 0.67 | 358 |
| 840-1260 | 0.039 | 0.075 | 1.03 | 148 | 0.68 | 147 |
| 1260-1680 | 0.032 | 0.059 | 1.17 | 84 | 0.70 | 105 |
| 1680-2100 | 0.025 | 0.040 | 0.30 | 52 | 0.71 | 82 |
| 2100-2520 | 0.023 | 0.033 | 0.57 | 39 | 0.69 | 74 |
| 2520-2940 | 0.021 | 0.032 | 0.16 | 30 | 0.71 | 66 |
| 2940-3360 | 0.020 | 0.028 | 0.19 | 28 | 0.71 | 57 |
| 3360-3444 | 0.015 | 0.026 | 0.06 | 16 | 0.68 | 60 |

* After about 1500 AM3 calls the emulator predicts the next spectrum the scan
  asks for to 0.03 dex (7 %) rms at the data energies, after 2500 to 0.02 dex,
  with a worst case of 0.2-0.6 dex over hundreds of points (several dex in the
  first 1000, when the scan still jumps across the box); compare the 0.05 dex
  data errors. Near the optimum it is slightly better: for the 56 points with
  true chi^2 < 100 the median rms is 0.018 dex.
* **The predicted uncertainty is well calibrated**: the median of |err|/sigma
  is 0.65-0.71 throughout, and the 90th percentile of the per-spectrum maximum
  z over 482 bins is 3-4, consistent with Gaussian tails. This is what makes a
  gate possible.
* The chi^2 error scales with chi^2 itself: a spectrum error of e dex at the
  data points shifts chi^2 by up to 2 chi2 e / sigma_obs when the errors align
  with the residuals, so 0.02 dex at chi^2 = 100 is worth tens of units. In
  practice the relative error has a 90th percentile of about 10 % wherever the
  gates below apply, and |dchi2| at the 56 points with chi^2 < 100 has a median
  of 3.5, a 90th percentile of 14 and a maximum of 23.
* The propagated bound `sig_chi2` (absolute terms summed) is exceeded by the
  actual |dchi2| in 10 % of the cases (ratio median 0.31, p90 0.90, p99 1.8,
  max 3.2), so it is an uncertainty scale, not a hard bound.

### 3.2 Tree settings

Measured on the same trajectory with the uncorrected spectra (section 2's
first version) and consistent with everything above: Nbar = 200 with retraining
every 20 points gives the same accuracy as Nbar = 100 (0.023 against 0.026 dex
late in the scan) at 40 % higher update cost; the adaptive `'noise'` component
rule keeps 50-70 components for this nearly noise-free target and makes a tree
update slower than an AM3 call (0.3-0.5 s), without a visible accuracy gain at
the point where it was stopped. All scans below use 16 components, Nbar 100,
retraining every 10 points: 0.11 s per update and 4 ms per prediction. With
the 0.25 s AM3 call this leptonic model is about the cheapest thing AM3
computes, so the tree update is a visible fraction of the budget (a third of
the AM3 time); for hadronic runs it would be negligible. Prediction is 60 times
cheaper than an AM3 call.

### 3.3 What-if gate analysis (reference shadow run)

For each candidate gate: the fraction of evaluations the emulator would have
answered, the chi^2 error of those answers, and whether any point that improved
on the running best would have been misranked (a "missed" improvement is one
the emulator ranks as no improvement; a "false" one the reverse). "far" means
only points whose emulated chi^2 exceeds `far` times the best AM3 chi^2 so far
are emulated.

| gate: sig_chi2 < abs + rel chi2 | far | emulated | rel. error p90 | improvements missed / false | emulated with chi2 < 60 (max \|dchi2\|) |
|---|---|---|---|---|---|
| 1 + 0.02 chi2 | - | 0.1 % | 3 % | 0 / 0 | 0 |
| 5 + 0.10 chi2 | - | 41 % | 7.5 % | 0 / 0 | 0 |
| 10 + 0.20 chi2 | - | 86 % | 10 % | 0 / 1 | 2 (3.2) |
| 10 + 0.30 chi2 | - | 97 % | 11 % | 0 / 3 | 3 (3.2) |
| 20 + 0.50 chi2 | - | 99 % | 11 % | 0 / 7 | 3 (3.2) |
| 10 + 0.20 chi2 | 1.5 | 85 % | 10 % | 0 / 0 | 0 |
| 10 + 0.30 chi2 | 1.5 | 95 % | 11 % | 0 / 0 | 0 |
| none | 1.5 | 98 % | 11 % | 0 / 0 | 0 |
| none | 3.0 | 90 % | 11 % | 0 / 0 | 0 |

The strict absolute gate that a "trust the emulator everywhere" strategy needs
(|dchi2| below 1) is never satisfied: it would require about 0.002 dex
precision at the data points. The useful regime is different: the emulator
answers the bulk of the exploration (points well above the current best, where
a 10 % chi^2 error cannot change a DE decision) and AM3 is reserved for
candidates near the optimum. Without the `far` criterion, loosening the gate
starts to produce false improvements (emulated points ranked better than the
best when they are not); with `far = 1.5` none occur at any looseness, and the
uncertainty gate on top mainly protects against the early, badly predicted
points. Section 3.4 runs `10 + 0.2 chi2` with `far = 1.5` for real.

### 3.4 Active mode with the calibrated gate

Gate: emulate when `sig_chi2 < 10 + 0.2 chi2_emu` and `chi2_emu > 1.5 x` the best
AM3 chi^2 so far; 10 % of the emulated answers are verified with AM3 anyway
(and fed to the tree). Everything else as in the reference.

| run | evaluations | AM3 calls | emulated | wall time | AM3 time | tree update | tree predict | best chi^2 |
|---|---|---|---|---|---|---|---|---|
| pure AM3 (reference shadow run) | 3444 | 3444 | 0 | 1595 s (1220 s without the tree) | 1203 s | 374 s | 15 s | 56.8 |
| active, calibrated gate | 3444 | 1007 | 2437 | **430 s** | 311 s | 104 s | 14 s | **49.3** |

* **71 % fewer AM3 calls and a factor 2.8 in wall time against a scan without
  any tree** (3.7 against the shadow run, which pays for a tree it does not
  use), and a better optimum (chi^2 49.3 against 56.8; 33.9 at the true
  parameters). The different trajectory is partly luck, but the gate cannot
  hurt the reported optimum: the running best is always an AM3 value because
  an emulated point must be 1.5 times worse than it to be emulated at all.
* The emulator answered 79 % of all evaluations. Of the 274 verified emulated
  answers, none was in truth an improvement on the running best and none was
  in truth within 1.5 x of it; their relative chi^2 error had a median of 4 %,
  a 90th percentile of 11 % and a maximum of 25 % (at chi^2 in the thousands).
* The emulator learns per AM3 call, not per evaluation: at 1000 evaluations
  the tree had 387 points and 0.11 dex median error where the reference had
  1000 points and 0.04 dex; by the end (1007 points) it was at 0.017 dex, as
  good as the reference's 3443-point tree, because the AM3 calls it did make
  were concentrated where the scan ended up. `plots/spectra_active_gate.png`
  shows its spectra against AM3 at the best point and at three emulated points.
* The remaining cost is 311 s of AM3 for the 1007 calls near the optimum plus
  104 s of tree updates. A cheaper gate near the optimum is not available from
  this emulator: at chi^2 < 100 its chi^2 error is 4-14 units (section 3.1),
  which is what AM3 must still resolve.

### 3.5 Carrying the emulator to a second source

The use case is a series of scans. A second synthetic source
(`theta_true = (-0.3, 16.5, 42.3, 2.1, 1.0, 4.5, delta = 12)`, 0.05 dex errors,
chi^2 = 21.8 at the truth) was scanned twice with the gate of section 3.4: from
an empty tree, and from the 1007-point tree saved by the first source's gated
scan (`--load-tree`). Both ran concurrently with the reference run, so the wall
times are inflated by the shared machine; the AM3-call counts are not.

| run | AM3 calls after 100 / 500 / 1000 / 2000 evaluations | total AM3 calls | emulated | best chi^2 | final emulator error (median rms at data) |
|---|---|---|---|---|---|
| cold start | 90 / 239 / 379 / 647 | 998 | 2446 | 41.0 | 0.027 dex |
| warm start (tree from source 1) | 18 / 110 / 250 / 524 | 945 | 2499 | 44.5 | 0.031 dex |

* The inherited tree pays off in the global phase: 72 of the first 100
  evaluations and 130 of the first 1000 were answered from what the first scan
  had learned. After that the two scans need the same AM3 calls, because the
  calls that remain are the ones near the new optimum, which the first scan
  never visited. Over the whole scan the saving is 5 %.
* The inherited tree was briefly overconfident on the new source (|err|/sigma
  median 1.0 and a 90th-percentile maximum z of 11 in the first 420
  evaluations, against 0.6 and 4.5 for the cold tree) before its per-leaf
  calibration caught up; its error stayed slightly above the cold tree's for the
  rest of the scan (0.031 against 0.027 dex at the end), presumably because its
  leaves were shaped by the first source's stream. No emulated answer was
  misranked in either run (0 of 275 and 0 of 278 verified).
* Both DE runs stopped well above the chi^2 of the truth (41 and 44 against
  22), as did the pure-AM3 reference on the first source: 40 generations of 84
  are not enough for this seven-parameter SSC problem with its gamma_min /
  Doppler degeneracy. That is a property of the optimiser budget, not of the
  emulator.


### 3.6 Leaf kernel and the hybrid tree (same trajectory as the reference)

Both measured as shadow runs on the reference's 3444 points (16 components,
Nbar 100, retrain every 10). Median and 90th-percentile rms error of the
emulated log-flux at the data energies per block of 420 evaluations, and the
chi^2 error at the 56 points with true chi^2 < 100:

| leaf model | blocks 1-3 median | blocks 7-9 median | blocks 7-9 p90 | \|dchi2\| at chi^2 < 100: median / max | update per AM3 call |
|---|---|---|---|---|---|
| Matern 3/2 (default) | 0.103 / 0.054 / 0.039 | 0.021 / 0.020 / 0.015 | 0.032 / 0.028 / 0.026 | 3.5 / 23 | 0.11 s |
| squared exponential | 0.103 / 0.054 / 0.042 | 0.026 / 0.022 / 0.019 | 0.053 / 0.052 / 0.044 | 7.6 / 104 | 0.08 s |
| hybrid (`global_mean='net'`, Matern leaves) | 0.146 / 0.130 / 0.127 | 0.137 / 0.145 / 0.163 | 0.150 / 0.164 / 0.180 | 224 / 408 | 0.13 s |

* **The squared-exponential kernel is not better.** Its medians match the
  Matern ones to within 20 %, but its tails are heavier (90th percentile 1.7
  times worse late in the scan) and its chi^2 error near the optimum is twice
  the Matern one with a far worse maximum. The log-spectra are smooth in theta
  but not infinitely so: the floor transitions and the changes of cooling
  regime are kinks that the infinitely smooth kernel overshoots around.
* **The hybrid is a clear loss on this target.** Its error does not decrease
  with the number of points at all (0.13 dex in the first blocks, 0.16 in the
  last), so the leaves are not learning the network's residual. The likely
  reason is structural: the leaves project the residual onto a PCA basis that
  is fitted to the raw spectra, and the residual's structure falls outside the
  16 kept components, which no amount of data repairs. The amortised refit
  (one optimiser step per update) may add to it; the full-refit check was
  stopped before completion. In any case the hybrid is meant for targets of
  many inputs; six smooth inputs is the regime where the plain tree already
  works.

## 4. Conclusions

* **Yes, with a qualification.** A `GPTree(output_model='pca')` learning the
  comoving log-spectrum on AM3's log-energy grid from the scan's own AM3 calls
  reaches a 0.02-0.03 dex (5-7 %) median error at the data energies after about
  1000-1500 concentrated AM3 calls, with a worst case of 0.1-0.3 dex and a
  well-calibrated predicted uncertainty. That is not accurate enough to replace
  AM3 at the points that decide the fit (a chi^2 error of 4-14 at chi^2 < 100
  against the 23-point data set), but it is accurate enough to answer the bulk
  of a global optimiser's evaluations, which are far from the optimum.
* **The working recipe** is the gate of section 3.4: trust the emulator when
  its propagated chi^2 uncertainty is below 10 + 20 % of chi^2 and the point is
  at least 1.5 times worse than the best AM3 point so far; verify a tenth of the
  emulated answers. On this model that replaced 71-73 % of the AM3 calls, cut the
  wall time by a factor of about 3, and never misranked a candidate improvement.
  The factor is bounded by the AM3 calls near the optimum that the gate
  deliberately keeps; a real gain in that regime would need a 0.005 dex emulator.
* **Costs**: 4 ms per prediction (60 times cheaper than this cheapest AM3
  model) and 0.10-0.12 s per tree update at 16 components, i.e. a third of a
  leptonic AM3 call and negligible against a hadronic one. The adaptive
  ('noise') component rule keeps 50-70 components for this nearly noise-free
  target and makes updates slower than AM3 itself; a fixed 16 components loses
  nothing measurable in a scan (and 24-32 would be the next thing to try near
  the end of a scan, where the 16-component truncation is of the same order as
  the GP error). Nbar = 200 does not help.
* **Transfer across sources** saves only the global phase (5 % of the AM3 calls
  of the second scan here) because an optimiser's expensive calls are local. An
  emulator meant to serve many sources would have to be trained on a
  space-filling design of thousands of points first, or the scan should be a
  sampler that keeps exploring (MCMC, nested sampling), where the local benefit
  compounds.
* **What AM3 made easy**: one `AM3()` instance and one kernel initialisation
  serve every point (3 s once; a magnetic field change costs 2 ms), the leptonic
  SSC model costs 0.25 s per spectrum, and the output is already on a
  logarithmic energy grid. Learning the log-spectrum as a function of log energy
  and log parameters is the natural representation and is what is used here;
  the only extra ingredient is a soft floor for the self-absorbed and
  pair-absorbed cliffs. The Doppler factor, redshift and distance stay analytic
  and outside the emulator.
* **Caveats**: one leptonic model, one synthetic data set per source, one
  optimiser configuration, one seed. The worst-case errors (0.1-0.3 dex late in
  a scan, several dex early) mean a gate is essential; the self-calibrated
  uncertainty made the gate work here, but a hadronic model with sharper
  features in theta should be checked the same way before trusting it.
