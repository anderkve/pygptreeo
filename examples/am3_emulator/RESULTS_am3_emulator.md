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

* The peak of `E^2 n(E)` spans **17 decades** across the box (log10 peak from 4.3
  to 21.7 eV cm^-3), peaking between 3e7 and 3e11 eV (comoving). Below the
  self-absorption turnover and above the pair-production cutoff the log-spectrum
  falls off a cliff; the soft floor keeps those regions finite and smooth in theta.
* A global PCA of the log-spectra needs many components: 8 for 99 % of the
  variance, 18 for 99.9 %, 32 for 99.99 %. Held-out reconstruction error on the
  visible bins is 0.10 dex rms with 16 components, 0.03 dex with 32, 0.014 dex with
  48. A "shape plus log-peak" representation does not reduce the rank. The
  whole-box rank is therefore high; the point of the scan test is that a scan
  concentrates, and the tree's reservoir basis follows the stream.
* **Batch baseline, 300 train / 100 test random points over the whole box**:
  0.18-0.20 dex rms, 0.08-0.09 dex median error, worst 2 dex, for every PCA
  setting from 16 to 73 components. The error is GP-limited (300 points in six
  dimensions), not basis-limited. `output_model='independent'` on every fourth
  bin gives the same 0.20 dex rms at 8 times the prediction cost (36 vs 4 ms per
  point) and 2-3 times the fit cost, so `'pca'` is the right leaf model here.
  The calibrated predicted sigma is very conservative at this stage
  (|err|/sigma median 0.05).

## 3. Scan results

(see below)
