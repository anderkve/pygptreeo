# pyGPTreeO on the gg → HH two-loop virtual amplitude grid

`examples/hhgrid_amplitude_demo.py`, run with `--hybrid`, three seeds for the closure
test, seed 1 for the rest; one CPU core; 12 October 2026.

## 1. The data and the baseline

The NLO gg → HH calculation with full top-mass dependence (Borowka et al. 2016;
Heinrich, Jones, Kerner, Luisoni, Vryonidou, arXiv:1703.09252) evaluates the two-loop
virtual amplitude numerically, point by point, with quasi-Monte-Carlo integration of
the master integrals. The POWHEG-BOX ggHH code ships the result as 6320 points
(github.com/mppmu/hhgrid, `share/grids_sm/Virt_full.grid`): `β_s = sqrt(1 − 4m_H²/ŝ)`,
`|cos θ|`, the finite virtual part `V` and its integration error. The quoted
relative error has median 2.1 %, a 10–90 % range of 0.28–4.1 % and a tail to 100 %.
The points are the phase-space sample of a 14 TeV run, so their density is the
physical one: dense above the tt̄ threshold `β_s = 0.69`, sparse below it.

The code interpolates them by mapping `β_s` to a uniformising variable `x = cdf(β_s)`
(`events.cdf`), filling a regular 101 × 31 grid of nodes by weighted linear fits
of the 9 nearest points plus the points in each node's cell (weights `1/e⁴`, as
`creategrid.py` passes `e²` to `curve_fit`), and interpolating the node values with
Clough-Tocher. `PowhegGrid` in the script is a port of that pipeline without the
LHAPDF and YAML dependencies; it gives the baseline below. The per-point errors are
not used beyond the fit weights, and the grid carries no uncertainty.

The tree learns `log V` with the relative error of each point as its observation
noise, `GPTree(Nbar=100)` at the package defaults (Matérn 3/2 ARD leaves, gradual
splitting, `min_lengthscale` splits, noise-aware calibration). The hybrid adds the
feature network as global model.

## 2. Closure test: 80 % learned, 20 % held out

The held-out values carry their own QMC error, so the natural measures are the
agreement with them in units of that error: an exact surrogate gives 68 % / 95 %
within 1 / 2 errors, 0.3 % beyond 3, and a mean squared deviation (`chi2`) of 1.
The median absolute deviation and the fraction off by more than 5 % are what the
grid's closure test reports; both are limited by the test points' noise (2.1 %
median) and are given for comparison.

| method | median \|dev\| | off by > 5 % | within 1 / 2 QMC errors | beyond 3 | chi2 | median sigma | coverage 68 / 95 | time |
|---|---|---|---|---|---|---|---|---|
| POWHEG grid pipeline (cdf, 101 × 31 nodes, Clough-Tocher) | 1.50 % | 11.6 % | 55 % / 85 % | 6.9 % | 48.8 | – | – | 2.4 s |
| Clough-Tocher on the raw points | 2.06 % | 21.5 % | 41 % / 71 % | 15.1 % | 232 | – | – | 0.1 s |
| GPTree, Nbar = 100 | 1.23 % | 9.9 % | 60 % / 91 % | 1.5 % | 2.40 | 0.40 % | 0.64 / 0.93 | 8.5 s |
| hybrid, `global_mean='net'` | 1.22 % | 9.7 % | 61 % / 91 % | 1.5 % | 1.60 | 0.98 % | 0.75 / 0.96 | 59 s |

Reading:

* **The tree is at the floor the data set allows.** 60 % / 91 % of the held-out
  values lie within 1 / 2 of their own QMC errors of the prediction, against 68 % /
  95 % for an exact amplitude, and the pull distribution (figure `pulls.png`,
  `(prediction − value) / sqrt(σ_tree² + σ_QMC²)`) has mean −0.01 and width 1.11.
  The median deviation of 1.2 % is the test values' own noise.
* **Measured against the QMC errors, the grid pipeline is 20 times further from the
  truth than the tree** (`chi2` 49 against 2.4) and raw Clough-Tocher 100 times;
  6.9 % of the held-out values lie more than 3 errors from the grid against 1.5 %
  from the tree (0.3 % for an exact amplitude). `chi2` is dominated by the most
  precise points, which is where interpolating through the noise of their
  neighbours costs most.
* **The tree's uncertainty is six times smaller than a single two-loop evaluation**
  (0.40 % median against 2.1 %) and calibrated: 64 % / 93 % coverage at the 68 % /
  95 % levels with the QMC error in the interval. Before the noise-aware
  calibration the calibrated sigma inflated to the noise level (2.0 %, coverage
  0.83); the latent GP sigma was already calibrated (0.34 %, coverage 0.67).
* **The hybrid matches the tree here**, as the package's benchmarks predict for a
  rough 2D target without a large-scale shape the leaves miss, at seven times the
  cost. Its sigma is larger because the network's error budget is added to every
  leaf's sigma, and with noise-dominated residuals the calibration leaves that
  budget in place (coverage 0.75).

## 3. Accuracy against the number of two-loop points

Seed 1; the held-out 20 % as above. The grid is rebuilt on the first `N` points of
the random order (the phase-space sample's own order); the tree is streamed over the
same points; in the acquisition column the tree chooses which of the remaining
points to learn from, 50 at a time, by its largest calibrated sigma.

| points | grid: median \|dev\| / > 5 % / chi2 | tree, random order | tree, acquisition |
|---|---|---|---|
| 250 | 2.16 % / 22.1 % / 5.6e8 | 1.32 % / 12.3 % / 55 | 1.32 % / 12.3 % / 55 |
| 500 | 1.96 % / 19.5 % / 6.2e8 | 1.31 % / 11.6 % / 4.8 | 1.33 % / 12.1 % / 5.9 |
| 1000 | 1.70 % / 16.5 % / 5.2e6 | 1.29 % / 11.1 % / 3.1 | 1.28 % / 9.9 % / 5.3 |
| 2000 | 1.63 % / 13.4 % / 151 | 1.29 % / 10.0 % / 2.7 | 1.26 % / 9.7 % / 3.0 |
| 4000 | 1.48 % / 12.2 % / 49 | 1.24 % / 10.4 % / 2.6 | 1.26 % / 9.8 % / 2.6 |
| 5056 | 1.46 % / 11.7 % / 42 | 1.24 % / 10.4 % / 2.5 | 1.27 % / 10.0 % / 2.7 |

* **500 points streamed into the tree beat the grid pipeline built on all 5056**
  on every measure (`chi2` 4.8 against 42; 11.6 % against 11.7 % off by more than
  5 %, 1.31 % against 1.46 % median deviation). The grid's `chi2` at a few hundred
  points is astronomical because sparsely filled nodes get negative or zero fitted
  values, which the Clough-Tocher surface then carries into the sub-threshold region.
* **Acquisition does not help on this data set**: the random order already follows
  the phase-space density, and the tree is at the noise floor from a thousand points
  on, so there is nothing left for a better choice of points to buy. The 5D tt̄H
  benchmark (`BENCHMARK_RESULTS_tth_amplitude.md`) is where it matters.

## 4. Maps

`maps.png`: the points coloured by their QMC error, the tree's prediction of `log V`
and its calibrated sigma over the `(β_s, |cos θ|)` plane, the tt̄ threshold marked.
The amplitude rises by four decades in `β_s` and varies little in `|cos θ|`; the
sigma map is at 0.1–1 % where the points are dense, rises towards the sparse region
below threshold, and shows the leaf partition as vertical bands, because the
`min_lengthscale` criterion splits along `β_s`.

## 5. What this demonstrates, and what it does not

The tree reproduces the two-loop amplitude to the precision the data allow, from a
tenth of the points the grid pipeline needs, with a calibrated per-point uncertainty
the pipeline does not have, at 1.2 ms per point. It does not show an improvement in
the *amplitude* beyond the data's own noise, because the only reference here is the
noisy data themselves: the 2 % QMC errors on the held-out values floor every
accuracy measure except those taken in units of that error. A cleaner test needs a
reference the surrogate did not see, which the 5D tree-level tt̄H benchmark provides.

## Reproduce

```bash
cd examples
OMP_NUM_THREADS=1 python hhgrid_amplitude_demo.py --hybrid     # about 15 minutes
OMP_NUM_THREADS=1 python hhgrid_amplitude_demo.py --quick      # closure test only, one seed
```
