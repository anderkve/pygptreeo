# Benchmark results

Measurements behind the package's defaults and the claims in the README, for
the code as it ships. All runs: one core per process (`OMP_NUM_THREADS=1`) on a
4-core container.

Conventions:

* **NRMSE** is the RMSE divided by the range of the target.
* **Prequential** errors are those of predictions made for each stream point
  before the tree is updated with it.
* **Coverage** is the fraction of errors within one predicted sigma (target 0.68).
* Unless a table says otherwise the tree is `GPTree(Nbar=100, theta=1e-4,
  retrain_every_n_points=25, splitting_strategy='gradual')` with the default
  ARD Matérn 3/2 leaves and one optimiser restart, and each point's declared
  noise is `1e-3 |y|`.
* Run times predate the speed-ups of commit `575531e` (the leaf prediction and
  the L-BFGS phase of the network refit) and were not measured again.

The scripts and raw records that produced these numbers were removed from the
repository; they are in its history up to commit `cd27934`
(`examples/benchmark_*.py`, `examples/compare_tree_vs_hybrid.py`,
`examples/results/`). `examples/streaming_performance.py` runs the same kind of
comparison on one stream.

## 1. Split-dimension criterion

20 000 uniform points in 3D, `Nbar=200`, `retrain_every_n_points=50`.
NRMSE of the last 2000 predictions:

| criterion       | aniso_chirp | rosenbrock | rastrigin | eggholder |
|-----------------|------------:|-----------:|----------:|----------:|
| max_spread      | 0.0010      | 0.0001     | **0.0543** | 0.0431   |
| max_variance    | 0.0009      | 0.0001     | 0.0552    | 0.0444    |
| max_uncertainty | **0.0001**  | **0.0000** | 0.0604    | **0.0428** |
| min_lengthscale | **0.0001**  | **0.0000** | 0.0599    | 0.0446    |
| random          | 0.0021      | 0.0002     | 0.0620    | 0.0467    |

`aniso_chirp` is rough along one input and smooth along another with three
times the spread. There the two GP-aware criteria are ten times better than the
spread-based ones, which split the smooth, wide dimension. On isotropic or
separable targets all criteria are within about 0.01 of each other.
`min_lengthscale` matches `max_uncertainty` and is cheaper (it reads the fitted
length scales instead of probing the GP on a grid), so it is the default.
Coverage stays at about 0.68 for every criterion.

## 2. The stream benchmark

Sections 3 and 4 use the same setup: 4000 points in six dimensions, three
seeds, on two targets that couple every input through a fixed random rotation
and so have no exact low-order additive decomposition (`RotatedRosenbrock`,
`GaussianPeaks`), under four input streams:

| stream | inputs |
|---|---|
| uniform | independent uniform samples |
| focusing | 666 uniform points, then a Gaussian cluster around the target's minimum whose width shrinks from 0.3 to 0.02 (an optimiser converging) |
| sweeping | a Gaussian cluster of width 0.12 whose centre follows a smooth path through the cube |
| walker | every proposal of a Metropolis random walk on `exp(-f / T)` (an MCMC sampler) |

Three errors are reported: prequential on the stream after the first 666
points, on 3000 uniform test points (accuracy over the whole cube), and on 3000
"focus" test points drawn where the stream ended.

## 3. Additive global GP (`global_mean='additive_gp'`)

A GP with `AdditiveMaternKernel(d, order=2)` on a 500-point coverage reservoir,
two optimiser restarts per fit; the leaves model its residual.

On targets with an exact low-order additive structure (Rosenbrock, Rastrigin)
the gain over the plain tree was two to three orders of magnitude. On the two
rotated targets it is 15 to 50%. NRMSE of the plain tree and the error of the
tree with the global GP relative to it (mean of the per-seed ratios; below 1 is
better than the plain tree):

| target | stream | plain tree: prequential / uniform / focus | ratio: prequential / uniform / focus |
|---|---|---|---|
| rotated_rosenbrock | uniform | 0.0155 / 0.0109 / 0.0110 | 0.70 / 0.71 / 0.69 |
| rotated_rosenbrock | focusing | 0.0069 / 0.0301 / 0.00004 | 0.75 / 0.59 / at floor |
| rotated_rosenbrock | sweeping | 0.0060 / 0.0647 / 0.0053 | 1.11 / 0.82 / 1.94 |
| rotated_rosenbrock | walker | 0.0062 / 0.0923 / 0.0024 | 0.78 / 0.82 / 0.46 |
| gaussian_peaks | uniform | 0.0168 / 0.0139 / 0.0135 | 0.81 / 0.81 / 0.84 |
| gaussian_peaks | focusing | 0.0074 / 0.0232 / 0.0001 | 0.74 / 0.73 / 0.87 |
| gaussian_peaks | sweeping | 0.0264 / 0.0807 / 0.0117 | 0.50 / 0.52 / 0.69 |
| gaussian_peaks | walker | 0.0111 / 0.1063 / 0.0039 | 0.65 / 0.59 / 0.34 |

The one soft spot is the end of the sweep on the rotated Rosenbrock (focus
ratio 1.94: one seed at 3.4, the other two level with the plain tree).

One-sigma coverage with `use_calibrated_sigma=True`:

| target | stream | plain tree: prequential / uniform / focus | with global GP: prequential / uniform / focus |
|---|---|---|---|
| rotated_rosenbrock | uniform | 0.67 / 0.67 / 0.67 | 0.68 / 0.64 / 0.63 |
| rotated_rosenbrock | focusing | 0.69 / 0.33 / 0.60 | 0.74 / 0.36 / 0.68 |
| rotated_rosenbrock | sweeping | 0.66 / 0.39 / 0.68 | 0.67 / 0.41 / 0.76 |
| rotated_rosenbrock | walker | 0.65 / 0.28 / 0.78 | 0.67 / 0.25 / 0.93 |
| gaussian_peaks | uniform | 0.68 / 0.66 / 0.67 | 0.68 / 0.67 / 0.68 |
| gaussian_peaks | focusing | 0.70 / 0.54 / 0.63 | 0.74 / 0.34 / 0.65 |
| gaussian_peaks | sweeping | 0.65 / 0.50 / 0.77 | 0.67 / 0.42 / 0.73 |
| gaussian_peaks | walker | 0.68 / 0.20 / 0.78 | 0.66 / 0.16 / 0.87 |

* **On the stream both are calibrated** (0.65 to 0.74).
* **Far from the data neither is.** On the uniform test set of the three
  moving streams, mostly unvisited territory, coverage is 0.16 to 0.54 for
  both. The per-leaf calibration is fitted on the stream and says nothing about
  regions the stream has not reached.
* **Where the data are dense the tree with the global GP is conservative**
  (up to 0.93): its error budget for the global model is a stream-wide average,
  and the global model is better than average there.

**Cost.** 4.1 to 11 times the plain tree's run time on this benchmark (171 to
386 s against 31 to 44 s under the same machine load). One fit of the global GP
takes 25 to 34 s in 3 and 6 dimensions and 76 s in 10, inside a single
`update_tree` call. A prediction costs 1.3 ms on uniform streams in 3 and 6
dimensions and 7 to 9 ms in 10 dimensions and on the 6D walker stream, against
0.9 ms for the plain tree.

## 4. Feature network (`global_mean='net'`, the hybrid)

`FeatureNetLearner` at its defaults, with `steps_per_update` set so that
refits are spread over the stream.

### 4.1 Ten 40 000-point streams

One seed per run, uniform stream unless noted. The noisy runs add
heteroscedastic noise `sigma_i = std(y) * 10^U(-1.5, 0)`, declared to the tree;
errors are against the noise-free target. NRMSE of the last 10 000 predictions:

| run | plain tree | hybrid | ratio |
|---|---|---|---|
| eggholder, d = 3 | 0.0404 | 0.0402 | 1.00 |
| rotated_rosenbrock, d = 6 | 0.0045 | 0.00028 | 0.06 |
| gaussian_peaks, d = 10 | 0.0377 | 0.0027 | 0.07 |
| gaussian_peaks, d = 6, walker | 0.0146 | 0.0021 | 0.14 |
| active_peaks, d = 10 | 0.0361 | 0.0018 | 0.05 |
| step_ridge, d = 6 | 0.0434 | 0.0119 | 0.27 |
| michalewicz, d = 5 | 0.100 | 0.0863 | 0.86 |
| ackley, d = 6 | 0.0647 | 0.0542 | 0.84 |
| griewank, d = 6 | 0.0986 | 0.0990 | 1.00 |
| chirp, d = 4 | 0.0875 | 0.0699 | 0.80 |
| rotated_rosenbrock, d = 6, noise | 0.0127 | 0.0031 | 0.24 |
| gaussian_peaks, d = 6, walker, noise | 0.0293 | 0.0083 | 0.28 |
| gaussian_peaks, d = 10, noise | 0.0501 | 0.0107 | 0.21 |

* **An order of magnitude where the target has a global structure a network
  can learn**: coupled valleys, rotated peaks, a function living on a hidden
  2-plane of ten inputs (`active_peaks`). A sharp step across a rotated
  hyperplane (`step_ridge`) gains a factor of 3.6.
* **14 to 20% on rough, oscillatory targets** (`michalewicz`, `ackley`,
  `chirp`), where the network removes a bowl or a trend and the leaves still
  have to fit the fine structure.
* **Nothing where 40 000 points do not resolve the ripples** (`eggholder`,
  `griewank`).
* The hybrid's coverage on the ten noise-free runs was 0.65 to 0.70 (plain
  tree 0.66 to 0.68), measured before held-out selection became the default.
  It can dip to about 0.5 for one batch early in a stream, when a new version
  of the network shifts the residual under leaves calibrated against the old
  one.

### 4.2 The stream benchmark, without and with noise

The setup of section 2, once as it is and once with Gaussian noise of 0.2
standard deviations of the target. Error relative to the plain tree over the 72
combinations of target, stream, seed and test set:

| | hybrid | hybrid with `holdout=0` |
|---|---|---|
| no noise: geometric mean | 0.34 | 0.31 |
| no noise: better than the plain tree in | 69 of 72 | 68 of 72 |
| no noise: range | 0.07 to 1.16 | 0.09 to 1.34 |
| noise: geometric mean | 0.75 | 1.44 |
| noise: better than the plain tree in | 58 of 72 | 8 of 72 |
| noise: range | 0.32 to 1.79 | 0.66 to 4.66 |

The held-out fifth of the points is what keeps the hybrid ahead of the plain
tree on noisy targets, at a cost of 9% in error on short noise-free streams
(4000 points) and nothing at 40 000. It also helps where the network cannot fit
the target: on the 6D Rastrigin and Levy functions the hybrid has 0.097 and
0.081 of the range against the plain tree's 0.117 and 0.106, where a network
trained on every point has 0.151 and 0.137.

Off the stream the hybrid's sigma is no better than the plain tree's: on the
uniform test set of the moving streams its coverage was 0.11 to 0.54, measured
with an earlier training default of the network. On the noisy streams held-out
selection raised the coverage on the uniform test set from 0.53 to 0.75 (plain
tree 0.70).

### 4.3 Cost

Seconds per run, plain tree and hybrid side by side under the same load:

| run | plain tree | hybrid |
|---|---|---|
| gaussian_peaks, d = 6, uniform, 4000 points | 18 | 52 |
| rotated_rosenbrock, d = 6, walker, 4000 points | 22 | 49 |
| rotated_rosenbrock, d = 6, 40 000 points | 208 | 308 |
| gaussian_peaks, d = 10, 40 000 points | 253 | 398 |

So 1.5 to 1.6 times the plain tree on the long runs. A refit of the network
has a fixed length, and refits come at each doubling of the point count, so
their share falls along a stream. A prediction takes 1.1 to 1.2 ms on average
against 0.8 to 0.9 ms for the plain tree. The store of every point seen costs
`d + 2` floats per point, about 5% of the tree's own footprint (1.3 kB per
point at `Nbar=100` in 6D).

### 4.4 Leaf kernel, `Nbar`, retrain interval and splitting

Measured on the 40 000-point runs with an earlier training default of the
network (Adam on every point); NRMSE of the last 10 000 predictions.

**Leaf kernel.** Six kernels on the network's residual, on four runs:

| leaf kernel | eggholder, d = 3 | rotated_rosenbrock, d = 6 | gaussian_peaks, d = 10 | gaussian_peaks, d = 6, walker | update ms per point |
|---|---|---|---|---|---|
| Matérn 1/2 | 0.042 | 0.0008 | 0.0039 | 0.0032 | 4.9 to 9.0 |
| **Matérn 3/2** (default) | 0.039 | 0.0006 | 0.0039 | 0.0030 | 4.8 to 9.1 |
| Matérn 5/2 | 0.042 | 0.0005 | 0.0040 | 0.0029 | 4.6 to 9.4 |
| RBF | 0.076 | 0.0006 | 0.0039 | 0.0029 | 4.1 to 7.3 |
| rational quadratic | 0.043 | 0.0006 | 0.0038 | 0.0029 | 13 to 24 |
| additive + Matérn 3/2 | 0.031 | 0.0007 | 0.0039 | 0.0031 | 16 to 35 |

On the smooth targets the kernel does not matter. On the rough residual of the
Eggholder RBF fails, and the additive kernel gains 20%, because the Eggholder
is a sum of terms in pairs of coordinates. The rational quadratic and additive
kernels cost three to seven times the update time. Matérn 3/2 stays the
default; use `AdditiveMaternKernel` when the target is known to have low-order
additive structure.

**`Nbar` and the retrain interval**, paired for about equal retrain cost per
point, plain tree / hybrid:

| Nbar / retrain | eggholder, d = 3 | gaussian_peaks, d = 6, walker | rotated_rosenbrock, d = 6 |
|---|---|---|---|
| 50 / 6 | 0.043 / 0.039 | 0.020 / 0.0031 | 0.0065 / 0.0007 |
| **100 / 25** (default) | 0.040 / 0.039 | 0.015 / 0.0030 | 0.0045 / 0.0006 |
| 200 / 75 | 0.042 / 0.040 | 0.011 / 0.0028 | 0.0034 / 0.0006 |
| 400 / 200 | 0.059 / 0.046 | 0.010 / 0.0027 | 0.0028 / 0.0005 |

* **The hybrid is nearly flat in `Nbar`** on the 6D targets (within 15% of the
  default): the network supplies the large-scale shape, and a small, smooth
  residual is fitted as well by 50 points as by 400.
* **The plain tree on a smooth 6D target gains from large, rarely refitted
  leaves**: 30 to 38% at 400 / 200, at the price of fit spikes of 5.7 to 9 s
  against 0.4 s. On the rough 3D Eggholder the same setting is 45% worse.
* **Gradual splitting** gains the plain tree 2 to 8% over standard splitting
  and the hybrid nothing; standard splitting is 15 to 25% cheaper per update.

## 5. Cost along a 100 000-point stream (plain tree)

`GaussianPeaks` in six dimensions, walker stream. Times per window since the
previous checkpoint; the batch prediction is of 1000 uniform test points in
`mode='loop'`:

| points | update ms: mean / 99th percentile / max | predict ms per point: single / batch | leaves | elapsed s |
|---|---|---|---|---|
| 1000 | 5.17 / 131.7 / 301 | 0.67 / 0.02 | 17 | 5.9 |
| 4000 | 4.34 / 106.8 / 279 | 0.68 / 0.06 | 60 | 21.1 |
| 16000 | 4.39 / 101.6 / 399 | 0.71 / 0.22 | 237 | 84.2 |
| 64000 | 4.71 / 110.1 / 474 | 0.78 / 1.63 | 935 | 346.5 |
| 100000 | 4.63 / 106.7 / 494 | 0.78 / 3.57 | 1460 | 545.0 |

The update and the single-point prediction (the default `mode='recursive'`,
which visits only the leaves a point can belong to) cost the same at 1460
leaves as at 17. Batch prediction in `mode='loop'` calls every leaf once and
grows with the number of leaves; on a long stream predict with the default
mode. The hybrid in its shipped form was not run on this benchmark.
