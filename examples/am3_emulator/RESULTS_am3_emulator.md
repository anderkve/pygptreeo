# pyGPTreeO as an on-the-fly emulator for AM3 spectra

Can a `GPTree` with `output_model='pca'` learn the photon spectra produced by
[AM3](https://am3.readthedocs.io/) while a parameter scan is running, accurately
enough to replace AM3 calls? This document records the setup and the measurements
from the scripts in this directory.


> **Status**: the measurements in sections 2 and 3 were made with an emulated
> quantity that carried one spurious power of E (AM3's `get_photons` is a density
> per unit ln E). The wrapper is fixed; the sections are being redone with the
> corrected, physical SEDs and will replace the numbers below.

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

`scan_emulator.py` fits a synthetic SED: 22 points from 0.1 eV to 1 TeV (the far-IR
and MeV gaps removed, radio points dropped as self-absorbed), generated from
`theta_true = (-0.7, 16.2, 41.8, 2.4, 1.3, 4.9, delta=20)` at z = 0.1 with 0.05 dex
Gaussian errors; chi^2 in log flux. The optimiser is scipy's
`differential_evolution` (popsize 12 x 7 = 84 per generation, 40 generations,
3444 evaluations, no polishing, fixed seed).

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

All scans below share the same DE trajectory (same seed; in shadow mode the
emulator never influences the optimiser, and the first active run replaced a
single AM3 call), so the emulator settings are compared on identical point
streams. The DE reaches chi^2 = 36.0 after 3444 evaluations (chi^2 = 33.1 at
the true parameters) with the usual SSC degeneracies (gamma_min and the Doppler
factor trade against each other). Timings are for one core with two scans
running concurrently, which costs nothing compared with a solo run; four
concurrent AM3 processes on this 4-core VM were 9 times slower per call, which
is the steep first 200 evaluations of the AM3 time curve in
`plots/scan_overview.png`.

### 3.1 Learning curve along the scan (shadow run, 16 components, Nbar = 200)

Errors are of the emulated observed log-flux at the 22 data energies, i.e. what
the likelihood sees; "visible" bins behave the same. Blocks of five generations
(420 evaluations):

| evaluations | median rms [dex] | p90 rms [dex] | worst max [dex] | median \|dchi2\| | \|err\|/sigma median | best true chi2 |
|---|---|---|---|---|---|---|
| 0-420 | 0.106 | 0.265 | 3.39 | 865 | 0.66 | 486 |
| 420-840 | 0.061 | 0.112 | 0.85 | 262 | 0.70 | 258 |
| 840-1260 | 0.046 | 0.087 | 4.23 | 143 | 0.65 | 124 |
| 1260-1680 | 0.037 | 0.055 | 0.82 | 66 | 0.69 | 134 |
| 1680-2100 | 0.031 | 0.056 | 0.53 | 62 | 0.70 | 122 |
| 2100-2520 | 0.026 | 0.044 | 0.31 | 51 | 0.68 | 65 |
| 2520-2940 | 0.026 | 0.042 | 0.26 | 43 | 0.76 | 55 |
| 2940-3360 | 0.025 | 0.039 | 0.18 | 31 | 0.75 | 36 |
| 3360-3444 | 0.023 | 0.039 | 0.15 | 31 | 0.74 | 43 |

* After about 1500 AM3 calls the emulator predicts the next spectrum the scan
  asks for to 0.03 dex (7 %) rms at the data energies, and after 2500 calls to
  0.025 dex with a worst case of 0.15-0.3 dex over hundreds of points; compare
  the 0.05 dex data errors. Near the optimum it is slightly better: for the 35
  points with true chi^2 < 100 the median rms is 0.022 dex.
* **The predicted uncertainty is well calibrated**: the median of |err|/sigma
  is 0.65-0.76 throughout, and the 90th percentile of the per-spectrum maximum
  z over 482 bins is about 4, consistent with Gaussian tails. This is what makes
  a gate possible.
* The chi^2 error scales with chi^2 itself: a spectrum error of e dex at the
  data points shifts chi^2 by up to 2 chi2 e / sigma_obs when the errors align
  with the residuals, so 0.03 dex at chi^2 = 100 is worth tens of units. In
  practice the relative error has a 90th percentile of 8-12 % wherever the gate
  below applies, and |dchi2| is 5-9 (median) at the 35 points with chi^2 < 100.
* The propagated bound `sig_chi2` (absolute terms summed) is exceeded by the
  actual |dchi2| in 10 % of the cases (ratio p90 0.78, p99 1.9, max 10), so it
  is an uncertainty scale, not a hard bound.

### 3.2 Tree settings (same trajectory)

| setting | median rms, last 1000 evals | update cost per AM3 call | predict |
|---|---|---|---|
| 16 components, Nbar 100, retrain every 10 | 0.026-0.027 dex | 0.12 s | 4.6 ms |
| 16 components, Nbar 200, retrain every 20 | 0.023-0.026 dex | 0.17 s | 4.8 ms |

The accuracy is the same; the larger leaves cost 40 % more per update. With the
0.25 s AM3 call this leptonic model is about the cheapest thing AM3 computes, so
the tree update is a visible fraction of the budget (a third); for hadronic runs
it would be negligible. Prediction is 50 times cheaper than an AM3 call.

### 3.3 What-if gate analysis (shadow run)

For each candidate gate: the fraction of evaluations the emulator would have
answered, the chi^2 error of those answers, and whether any point that improved
on the running best would have been misranked (a "missed" improvement is one
the emulator ranks as no improvement; a "false" one the reverse). "far" means
only points whose emulated chi^2 exceeds `far` times the best AM3 chi^2 so far
are emulated.

| gate: sig_chi2 < abs + rel chi2 | far | emulated | rel. error p90 | improvements missed / false | emulated with chi2 < 60 (max \|dchi2\|) |
|---|---|---|---|---|---|
| 1 + 0.02 chi2 | - | 0.1 % | 0.5 % | 0 / 0 | 0 |
| 5 + 0.10 chi2 | - | 42 % | 6 % | 0 / 0 | 0 |
| 10 + 0.20 chi2 | - | 81 % | 8 % | 0 / 0 | 1 (2.0) |
| 10 + 0.30 chi2 | - | 95 % | 10 % | 1 / 0 | 5 (19.6) |
| 20 + 0.50 chi2 | - | 99 % | 10 % | 5 / 1 | 10 (19.6) |
| 10 + 0.20 chi2 | 1.5 | 81 % | 8 % | 0 / 0 | 1 (2.0) |
| 10 + 0.30 chi2 | 1.5 | 95 % | 10 % | 0 / 0 | 3 (19.6) |
| none | 1.5 | 98 % | 10 % | 0 / 0 | 3 (19.6) |
| none | 3.0 | 93 % | 10 % | 0 / 0 | 0 |

The strict absolute gate that a "trust the emulator everywhere" strategy needs
(|dchi2| below 1) is never satisfied: it would require about 0.002 dex
precision at the data points. The useful regime is different: the emulator
answers the bulk of the exploration (points well above the current best, where a
10 % chi^2 error cannot change a DE decision) and AM3 is reserved for candidates
near the optimum. With `10 + 0.2 chi2` and `far = 1.5`, 81 % of the evaluations
would have been emulated with no misranked improvement, and only one emulated
point had chi^2 below 60. Section 3.5 runs exactly that gate for real.

### 3.4 Active mode with the calibrated gate

Gate: emulate when `sig_chi2 < 10 + 0.2 chi2_emu` and `chi2_emu > 1.5 x` the best
AM3 chi^2 so far; 10 % of the emulated answers are verified with AM3 anyway
(and fed to the tree). Everything else as in the reference (16 components,
Nbar 100, retrain every 10 points).

| run | evaluations | AM3 calls | emulated | wall time | AM3 time | tree update | tree predict | best chi^2 |
|---|---|---|---|---|---|---|---|---|
| pure AM3 (shadow, Nbar 100) | 3444 | 3443 | 0 | 1665 s | 1238 s | 410 s | 16 s | 36.0 |
| active, calibrated gate | 3444 | 1130 | 2314 | **467 s** | 332 s | 120 s | 14 s | **31.7** |

* **3.6 times faster in wall time, 67 % fewer AM3 calls**, and a better
  optimum (chi^2 31.7 against 36.0; 33.1 at the true parameters). The different
  trajectory is partly luck, but the gate cannot hurt the reported optimum:
  the running best is always an AM3 value because an emulated point must be
  1.5 times worse than it to be emulated at all.
* The emulator answered 75 % of all evaluations (after the first 60 AM3 calls,
  80 %). Of the 254 verified emulated answers, none was in truth an improvement
  on the running best and none was in truth within 1.5 x of it; their relative
  chi^2 error had a median of 3.6 %, a 90th percentile of 9.8 % and a maximum
  of 31 % (at chi^2 in the thousands).
* The emulator learns per AM3 call, not per evaluation: at 1000 evaluations
  the tree had 380 points and 0.064 dex median error where the shadow run had
  1000 points and 0.046 dex; by the end (1130 points) it was at 0.023 dex, the
  same as the shadow run with 3443 points, because the AM3 calls it did make
  were concentrated where the scan ended up.
* The remaining cost is 332 s of AM3 for the 1130 calls near the optimum plus
  120 s of tree updates. A cheaper gate near the optimum is not available from
  this emulator: at chi^2 < 100 its chi^2 error is 8-12 units (section 3.1),
  which is what AM3 must still resolve.

