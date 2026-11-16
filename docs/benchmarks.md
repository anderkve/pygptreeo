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
* Run times in sections 1 to 5 predate the speed-ups of commit `575531e` (the
  leaf prediction and the L-BFGS phase of the network refit) and were not
  measured again.
* Sections 1 to 6 were measured with `append_points_between_fits=False`,
  before the tree appended points between fits (section 7).
* Every coverage but those of the second table of section 7.5 was measured
  with `widen_sigma_with_distance=False`, before the sigma grew with the
  distance from a leaf's points.

The scripts and raw records that produced these numbers were removed from the
repository; they are in its history up to commit `cd27934`
(`examples/benchmark_*.py`, `examples/compare_tree_vs_hybrid.py`,
`examples/results/`). `examples/streaming_performance.py` runs the same kind of
comparison on one stream. Sections 6 and 7 were measured later, with scripts
that are not in the repository, and section 8 with those of
`research/partitioning`.

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
against 0.8 to 0.9 ms for the plain tree. The store of the Adam phase costs
`d + 2` floats per point, about 5% of the tree's own footprint (1.3 kB per
point at `Nbar=100` in 6D), and holds at most `store_size` points (100 000 by
default).

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

## 6. A memory limit (`max_memory_mb`)

`GaussianPeaks` in six dimensions, 40 000 points, plain tree, means of three
seeds. NRMSE and coverage are prequential, over the last 10 000 predictions.
"Left behind" is the NRMSE and coverage, at the end of the stream, of 2000
points of its first quarter. The sweeping stream does not come back to a region
it has left; the walker keeps returning to the same peaks. The tree without a
limit takes 66 to 68 MB by its own estimate, and the limits of 34.1, 11.4 and
3.4 MB are what 300, 100 and 30 fitted leaves take. "Removal only" is the same
tree with no leaf ever put to sleep (`FITTED_SHARE` above 1), which is what a
GP backend that cannot release a fit gives.

| stream | limit MB | at the limit | leaves (asleep) | points kept | NRMSE | coverage | left behind: NRMSE / coverage |
|---|---|---|---|---|---|---|---|
| walker | none | | 585 | 40 000 | 0.0119 | 0.67 | 0.0001 / 1.00 |
| walker | 34.1 | sleep | 568 (371) | 38 897 | 0.0120 | 0.67 | 0.0068 / 0.99 |
| walker | 34.1 | removal only | 302 | 20 654 | 0.0198 | 0.66 | 0.0275 / 0.77 |
| walker | 11.4 | sleep | 192 (127) | 13 393 | 0.0244 | 0.67 | 0.0317 / 0.70 |
| walker | 11.4 | removal only | 101 | 6814 | 0.0290 | 0.67 | 0.0419 / 0.65 |
| walker | 3.4 | sleep | 58 (39) | 3869 | 0.0354 | 0.67 | 0.0472 / 0.68 |
| walker | 3.4 | removal only | 30 | 2091 | 0.0464 | 0.66 | 0.0589 / 0.66 |
| sweeping | none | | 598 | 40 000 | 0.0168 | 0.67 | 0.0026 / 0.97 |
| sweeping | 34.1 | sleep | 568 (371) | 38 048 | 0.0168 | 0.67 | 0.0082 / 0.90 |
| sweeping | 34.1 | removal only | 302 | 20 494 | 0.0097 | 0.66 | 0.134 / 0.36 |
| sweeping | 11.4 | sleep | 192 (127) | 13 231 | 0.0110 | 0.66 | 0.238 / 0.59 |
| sweeping | 11.4 | removal only | 101 | 6955 | 0.0088 | 0.66 | 0.312 / 0.16 |
| sweeping | 3.4 | sleep | 58 (39) | 4002 | 0.0080 | 0.66 | 0.195 / 0.62 |
| sweeping | 3.4 | removal only | 30 | 2100 | 0.0081 | 0.66 | 0.293 / 0.20 |

* **Sleeping leaves fit 1.9 times as many leaves and points into a limit**:
  568 against 302 leaves at 34.1 MB, with two thirds of them asleep.
* **With half the memory of the unlimited tree the walker loses nothing.** At
  34.1 MB its error is that of the unlimited tree in every seed, where removal
  alone gives 1.7 times that error. At 11.4 and 3.4 MB the error with sleeping
  leaves is 16 and 24% below that of removal alone, and it is lower in all
  nine pairs of runs.
* **The sweeping stream gains from forgetting, and sleeping leaves forget
  less.** Removal alone lowers the error on this stream by 42 to 52%, at the
  same median error (0.0012 of the range). With sleeping leaves the
  error is that of the unlimited tree at 34.1 MB, 25% above removal alone at
  11.4 MB and the same at 3.4 MB.
* **What forgetting removes on the sweeping stream are predictions by leaves
  whose points lie elsewhere** (one seed). In the unlimited tree 0.7% of the
  predictions are made by a leaf whose tenth nearest point is more than three
  cluster widths from the point predicted, and they make two thirds of the
  squared error. The stream keeps entering territory that belongs to the open
  side of the region of a leaf it left long ago. Once such a leaf is removed
  its territory falls to a leaf with recent points: with removal alone at
  11.4 MB the squared error is three times smaller, and no prediction is made
  from beyond four widths. Where the tenth nearest point is within two widths,
  91% of the predictions, the error is the same with and without a limit
  (0.0046 to 0.0049). On the walker no prediction of the unlimited tree is
  made from beyond four widths, although 99% of them are made by leaves whose
  points are mostly more than 4000 updates old.
* **The regions left behind are remembered for longer.** With sleeping leaves
  the error there is 16 times lower at 34.1 MB on the sweeping stream and 4
  times lower on the walker, and 1.2 to 1.5 times lower at the two smaller
  limits. Its coverage is 0.59 to 0.99 against 0.16 to 0.77.
* **The on-stream coverage holds** at 0.66 to 0.67 for every limit.
* **Waking a leaf is cheap.** On the walker a leaf is woken for 0.12 to 0.20
  of the points, and the mean time of a prediction rises from 0.26 to 0.31 ms
  to 0.31 to 0.39 ms. The sweeping stream at 3.4 MB wakes a leaf for half of
  the points, at 0.50 ms per prediction against 0.29 ms. The update takes 2.2
  to 2.4 ms in every run.
* **The tree's estimate of its size holds.** With every 3000 points of a
  walker stream, the estimate (`memory_mb`) was between 0.98 and 1.05 times the
  memory that Python had allocated since the tree was created (`tracemalloc`),
  for `Nbar` of 50, 100 and 200, one and three outputs, without a limit and
  with limits that put leaves to sleep and remove them. In separate processes
  the resident memory grew by at most 6% more than the limit, and stayed there
  over the last 24 000 of 48 000 points.

The hybrid on the same streams (`steps_per_update=1`, one seed), with a limit
of 218.3 MB: the 206.9 MB of the network's reservoir, store and weights, and
the 11.4 MB of the table above for the leaves.

| stream | at the limit | leaves (asleep) | NRMSE | coverage | left behind: NRMSE / coverage |
|---|---|---|---|---|---|
| walker | no limit | 589 | 0.0014 | 0.68 | 0.0002 / 0.88 |
| walker | sleep | 192 (127) | 0.0016 | 0.68 | 0.0011 / 0.61 |
| walker | removal only | 101 | 0.0018 | 0.68 | 0.0013 / 0.58 |
| sweeping | no limit | 597 | 0.0081 | 0.68 | 0.0010 / 1.00 |
| sweeping | sleep | 192 (127) | 0.0067 | 0.68 | 0.014 / 0.82 |
| sweeping | removal only | 101 | 0.0062 | 0.69 | 0.042 / 0.25 |

* **The limit costs the hybrid less than the plain tree**: on the walker 1.2
  times the error without a limit with sleeping leaves and 1.3 times with
  removal alone, against 2.1 and 2.8 for the plain tree with the same seed and
  the same memory for its leaves.
* **The network keeps part of what the leaves forget.** The error in the
  regions left behind is 28 times (walker) and 14 times (sweeping) lower than
  that of the plain tree with sleeping leaves. On the sweeping stream the sigma
  there is too small without sleeping leaves (coverage 0.25).

`examples/streaming_performance.py --max_memory_mb` runs a stream under a limit.


## 7. Points appended between fits (`append_points_between_fits`)

"As before" is `append_points_between_fits=False`, the tree of sections 1 to 6.

### 7.1 `GaussianPeaks`

`GaussianPeaks` in six dimensions, 40 000 points, plain tree, prequential NRMSE
of the last 10 000 predictions, means of three seeds.

| tree | sweeping | walker |
|---|---|---|
| as before | 0.0168 | 0.0119 |
| points appended (the default) | 0.0086 | 0.0092 |

* **The appended points lower the error in all six runs.** The median error
  falls too, from 0.0012 to 0.0010 of the range on the sweeping stream and
  from 0.0019 to 0.0015 on the walker. 97% of the points are appended; the
  others arrive with a fit.
* **The coverage is 0.66 to 0.67** in every run.
* **Cost**, means of two seeds and both streams: an update takes 2.15 ms as
  before and 2.39 ms with the points appended, a prediction 0.30 and 0.33 ms.

One seed each:

| run | as before | points appended |
|---|---|---|
| uniform stream | 0.0055 | 0.0051 |
| focusing stream | 0.0023 | 0.0019 |
| `RotatedRosenbrock`, d = 6, sweeping | 0.0127 | 0.0070 |
| `Eggholder`, d = 3, sweeping | 0.0538 | 0.0503 |
| `Eggholder`, d = 3, walker | 0.0163 | 0.0145 |
| `retrain_every_n_points=100`, sweeping | 0.0230 | 0.0083 |
| `retrain_every_n_points=100`, walker | 0.0133 | 0.0099 |
| hybrid, sweeping | 0.0081 | 0.0069 |
| hybrid, walker | 0.0014 | 0.0012 |
| `max_memory_mb=11.4`, sweeping | 0.0128 | 0.0080 |
| `max_memory_mb=11.4`, walker | 0.0257 | 0.0175 |

* **The appended points lower the error in every run**, by 7% (uniform,
  `Eggholder` sweeping) to 64%.
* **With appended points the retrain interval matters little.** A fit every
  100 points then gives 0.0083 and 0.0099, against 0.0083 and 0.0098 with a
  fit every 25 points (same seed), at two thirds of the update time.
* **Under a memory limit the sweeping stream no longer loses to removal
  alone** (0.0091 in section 6). The appended points make the GPs larger, so
  the tree holds 185 leaves where it held 192.

### 7.2 The physics targets

The eleven targets of `examples/target_functions_physics.py`, 20 000 points,
one seed, plain tree. The table gives the NRMSE of the last 5000 predictions
with the points appended, divided by that of the tree as before.

| target | d | walker | sweeping | uniform | focusing |
|---|---|---|---|---|---|
| `dilepton_spectrum` | 4 | 0.91 | 0.51 | 0.94 | 0.93 |
| `resonance_xsec` | 4 | 0.68 | 0.86 | 0.92 | 0.85 |
| `jet_spectrum` | 4 | 0.33 | 0.77 | 0.99 | 0.85 |
| `oscillation_probability` | 5 | 0.65 | 0.84 | 0.85 | 0.94 |
| `branching_ratio` | 5 | 0.83 | 0.91 | 0.83 | |
| `loop_decay_width` | 4 | 0.92 | 0.59 | 0.95 | 0.89 |
| `relic_density` | 4 | 0.94 | 0.92 | 0.92 | 0.92 |
| `neutralino_mass` | 4 | 0.71 | 0.85 | 0.94 | 0.89 |
| `eft_signal_strength` | 6 | 0.61 | 0.33 | 0.86 | 0.95 |
| `signal_yield_mc` | 4 | 0.96 | 0.43 | 0.96 | 0.99 |
| `global_fit_loglike` | 10 | 0.98 | 0.80 | 0.95 | 0.71 |
| geometric mean | | 0.74 | 0.68 | 0.92 | 0.89 |

* **The error is lower in all 43 combinations.** `branching_ratio` on the
  focusing stream has no NRMSE: the target is constant over the last 5000
  points.
* **The coverage** changes by at most 0.012.
* **An update** takes 7% longer on the uniform and focusing streams.

### 7.3 The overlap (`theta`)

`GaussianPeaks` in six dimensions, 20 000 points, one seed, the default tree.
NRMSE of the last 5000 predictions, with the coverage in brackets.

| `theta` | sweeping | walker | uniform |
|---|---|---|---|
| 1e-4 (the default) | 0.0127 (0.66) | 0.0098 (0.68) | 0.0072 (0.68) |
| 0.05 | 0.0113 (0.73) | 0.0097 (0.77) | 0.0065 (0.73) |
| 0.2 | 0.0095 (0.87) | 0.0090 (0.92) | 0.0060 (0.88) |

* **A wide overlap lowers the error by 8 to 25%**, on every stream. The sigma
  of a blend of leaves is then too large, and a prediction takes 0.7 to
  1.3 ms where it took 0.30 to 0.35 ms.
* **The gain is in the predictions, not in where the points are stored.**
  With the points routed at `theta=1e-4` and the queries weighted with an
  overlap of 0.2, the errors are 0.0108, 0.0088 and 0.0059.

### 7.4 Another leaf than the owner for a prediction (not in the code)

Rules evaluated on recorded predictions: for every query of the last 5000 of
a run, the prediction of the leaf that owns it and of the leaves it would
reach beyond each of the five nearest cuts above that leaf (its family), with
each leaf's distance to the query. 140 runs: the eleven physics targets,
`GaussianPeaks` and `RotatedRosenbrock` (d = 6) on the four stream types with
three seeds, 20 000 points, points appended (without the focusing stream for
seed 3, and without three runs whose target is constant over the last 5000
points). All numbers are relative to the own leaf always.

| rule | NRMSE | worst run | runs more than 2% better | more than 2% worse | 99% quantile of the error | predictions off by more than 10 sigma |
|---|---|---|---|---|---|---|
| the own leaf | 1 | | | | 1 | 0.79% |
| the family fallback (removed) | 0.98 | 1.35 | 18 | 5 | 0.99 | 0.78% |
| the leaf with the nearest points, always | 2.2 | 75 | 23 | 111 | | |
| blend with the neighbour | 0.94 | 4.9 | 117 | 4 | 0.93 | 0.56% |
| blend, where the two agree within 5 sigma | 0.95 | 1.02 | 96 | 0 | 0.94 | 0.79% |

* **The NRMSE of a run rests on a few queries.** The five largest of the
  5000 errors of the own leaf carry 23% of the squared error in the median
  run and 71% in the tenth of the runs where they carry most. 2.2% of its
  predictions are off by more than 5 sigma, and 94.5% lie within 3 sigma.
* **The family fallback** (`far_threshold`, removed) gave a query more than 7
  standard deviations from the mean of its leaf's fitted points, along some
  input, to the leaf of its family with the nearest points. The threshold
  was chosen on seed 1, where the error on the walker and the sweeping stream
  was 0.90 times that of the own leaf, with no run more than 3% worse (4:
  0.89, with single runs up to 1.53; 5: 0.87, up to 1.26; 10: 0.97). With
  seeds 2 and 3 it is 1.02 on the walker and 0.96 on the sweeping stream,
  with single runs from 0.53 to 1.35.
* **The blend**: the own leaf and the family leaf beyond the nearest cut, with
  weights `z**-8`, where `z` is the root mean square over the inputs of the
  query's distance from the mean of a leaf's fitted points in units of their
  standard deviation. The neighbour's weight is multiplied by 0.3, and it
  takes part only where its `z` is at most 2. The sigma is the root of the
  sum of the squared weighted sigmas, with a coverage of 0.70 against 0.71
  for the own leaf. It predicts with 1.7 leaves per query.
* **The blend lowers the error in most runs, on every stream type** (0.93 on
  the uniform and focusing streams, 0.91 on the walker, 0.98 on the sweeping
  stream), and the largest error of a run in 71 runs, against 6 where it is
  larger. Its losses are single queries for which the neighbour is wrong by
  up to 29 times the target's range with a small sigma, close to its own
  points. The largest error of the own leaf is 2.9 times the range.
* **The agreement test** uses the blend only where the neighbour's prediction
  lies within 5 sigma of the own leaf's prediction. No run is then more than
  2% worse. The predictions off by more than 10 sigma stay: where the two
  leaves disagree, the test keeps the own leaf.
* **Distances to a leaf's points in units of the inputs' range** do not rank
  the leaves: the leaf with the nearest points is the worse one in most runs.

### 7.5 Errors by the distance from the leaf's points

The own leaf's predictions in the 140 runs of section 7.4, by the far score of
the query: its distance from the mean of the leaf's fitted points in units of
their standard deviation, the largest over the inputs. Means over the 39 runs
of a stream type. On the uniform stream 99.5% of the queries are below 2.

| far score | sweeping: queries | squared error | off by more than 5 sigma | within 1 sigma | walker: queries | squared error | off by more than 5 sigma | within 1 sigma |
|---|---|---|---|---|---|---|---|---|
| below 2 | 73% | 19% | 1.9% | 0.75 | 86% | 39% | 1.9% | 0.74 |
| 2 to 4 | 25% | 43% | 4.7% | 0.60 | 14% | 43% | 5.0% | 0.60 |
| 4 to 7 | 1.2% | 25% | 11% | 0.49 | 0.7% | 13% | 14% | 0.49 |
| above 7 | 0.3% | 13% | 16% | 0.44 | 0.1% | 5% | 13% | 0.39 |

* **On a stream that moves, most of the squared error is just outside the
  leaf's points.** 27% of the queries of the sweeping stream have a far score
  above 2 and carry 81% of the squared error, and the sigma covers fewer of
  them the farther they are.

The sigma with `widen_sigma_with_distance` (the default) against the sigma
without it, for the same predictions: the thirteen targets with another seed,
20 000 points. Each cell is without it, then with it.

| far score | sweeping: within 1 sigma | off by more than 5 sigma | walker: within 1 sigma | off by more than 5 sigma |
|---|---|---|---|---|
| below 2 | 0.75, 0.75 | 2.5%, 2.5% | 0.74, 0.74 | 2.5%, 2.5% |
| 2 to 4 | 0.60, 0.70 | 5.1%, 3.0% | 0.60, 0.69 | 5.5%, 3.6% |
| 4 to 7 | 0.46, 0.71 | 14%, 7.3% | 0.50, 0.76 | 14%, 2.7% |
| above 7 | 0.53, 0.78 | 15%, 10% | | |
| all queries | 0.70, 0.73 | 3.3%, 2.7% | 0.72, 0.73 | 2.9%, 2.6% |

* **The sigma grows by 10% on average on the sweeping stream and 5% on the
  walker**, all of it for the queries beyond 2. The slope at the end of a run
  is 0.9 on the sweeping stream and 0.7 on the walker (medians; at most 2.3
  and 5.5).
* **On the uniform stream** 0.5% of the queries are beyond 2, and the sigma
  grows by 0.2% on average. The slope is larger there (median 5.9, at most
  16), since it is learned from queries just beyond 2; 0.63 of these are
  then within 1 sigma, against 0.57.
* **A factor per band of the far score** in place of the one slope gives
  0.72, 0.62 and 0.67 within 1 sigma in the three bands above 2 on the
  sweeping stream, and 2.6% off by more than 5 sigma.

### 7.6 Points beyond a leaf's cuts (not in the code)

`GaussianPeaks`, `branching_ratio` and `loop_decay_width` on three stream types
with two seeds, 20 000 points, points appended. The NRMSE of the last 5000
predictions relative to the default tree, as geometric means; every query is
predicted by one leaf, chosen by the cuts without overlap.

| change | uniform | walker | sweeping | all runs (range) | runs more than 2% worse |
|---|---|---|---|---|---|
| points stored with an overlap of 0.05 | 0.96 | 1.00 | 0.94 | 0.97 (0.86 to 1.12) | 3 of 18 |
| points stored with an overlap of 0.2 | 0.99 | 1.04 | 0.90 | 0.98 (0.80 to 1.13) | 7 of 18 |
| points stored with an overlap of 0.5 | 1.46 | 1.47 | 0.95 | 1.27 (0.77 to 2.00) | 7 of 9 |
| a leaf keeps 25 of its sibling's points | 0.91 | 1.04 | 0.88 | 0.94 (0.75 to 1.20) | 4 of 18 |
| the same with `Nbar=80` | 0.97 | 1.10 | 0.87 | 0.98 (0.81 to 1.23) | 2 of 9 |
| a leaf keeps 10 of its sibling's points | 0.94 | 1.02 | 0.98 | 0.98 (0.85 to 1.14) | 2 of 9 |

* **An overlap for the stored points alone** gives a leaf points beyond its
  cuts and takes away some of those within them. A small overlap lowers the
  error a little on the uniform and sweeping streams and a large one raises
  it. The rows with 9 runs have one seed.
* **Kept points**: with gradual splitting a leaf starts with copies of its
  sibling's points and drops them as its own arrive, those nearest the cut
  last. Here the last 25 or 10 are never dropped. The GPs then hold up to 125
  points, and with `Nbar=80`, where they hold as many as in the default tree,
  most of the gain is gone.
* **The outliers stay.** With an overlap of 0.05 the 99% quantile of the
  error is 1.02 times that of the default tree and the largest error 1.00
  times, and 2.0% of the predictions are off by more than 5 sigma, against
  2.1%.

## 8. Other ways to partition the data

Sections 8.2 to 8.13 were measured with neighbours that all count alike
(`neighbour_fade=None`). Section 8.15 has the neighbours that fade out, which
are the default of `predict_from='neighbours'`.

Sections 8.1 to 8.3: eleven targets (`GaussianPeaks` and `RotatedRosenbrock`
in six dimensions, `Eggholder` in three, `Chirp` in four, a four-dimensional
target with a ridge and a kink on oblique hyperplanes, `resonance_xsec`,
`loop_decay_width`, `relic_density`, `eft_signal_strength`, `neutralino_mass`
and `oscillation_probability`) on the uniform, walker and sweeping streams,
5000 points, one seed: 33 runs. The tables give the NRMSE of the last 1250
predictions divided by that of the default tree in the same run, as geometric
means over the runs, and the mean coverage.

### 8.1 Rules for the cuts (not in the code)

One GP per leaf, as in the default tree. A hyperplane cut has a free normal
and is placed at the median of the leaf's points along it, unless the row
says otherwise.

| cuts, one GP per leaf | uniform | walker | sweeping | all runs (range) | runs more than 2% worse | coverage |
|---|---|---|---|---|---|---|
| along the input with the largest spread of the points (`max_spread`) | 1.34 | 1.20 | 1.25 | 1.26 (0.66 to 4.31) | 24 of 33 | 0.69 |
| hyperplane: largest spread of the points | 1.49 | 1.34 | 1.18 | 1.33 (0.27 to 4.33) | 24 of 33 | 0.69 |
| hyperplane: between the centroids of two clusters | 1.54 | 1.33 | 1.08 | 1.30 (0.30 to 4.88) | 23 of 33 | 0.69 |
| hyperplane: random direction | 1.61 | 1.46 | 1.05 | 1.35 (0.32 to 3.27) | 28 of 33 | 0.69 |
| hyperplane: largest change of the GP mean | 1.18 | 1.18 | 0.91 | 1.08 (0.54 to 2.30) | 15 of 33 | 0.68 |
| hyperplane: largest change of the GP mean's gradient | 1.07 | 0.93 | 1.14 | 1.04 (0.55 to 3.16) | 15 of 33 | 0.68 |
| the default cut, placed where the gradient changes most | 1.03 | 0.97 | 0.98 | 0.99 (0.48 to 1.80) | 15 of 33 | 0.68 |
| hyperplane of the gradient's change, placed where it changes most | 0.97 | 0.94 | 0.92 | 0.94 (0.43 to 1.80) | 11 of 33 | 0.68 |
| the default cut, decorrelated inputs for the GPs | 1.00 | 0.98 | 1.03 | 1.00 (0.70 to 1.43) | 15 of 33 | 0.68 |
| the axis of the decorrelated inputs with the shortest length scale | 1.06 | 1.23 | 1.09 | 1.13 (0.46 to 2.44) | 22 of 33 | 0.69 |
| hyperplane of the gradient's change, decorrelated inputs | 0.85 | 0.86 | 1.03 | 0.91 (0.21 to 1.89) | 15 of 33 | 0.69 |
| the same, placed where the gradient changes most | 0.87 | 0.82 | 0.90 | 0.86 (0.22 to 1.93) | 15 of 33 | 0.68 |
| a model in every node (multi-resolution) | 1.66 | 1.27 | 1.33 | 1.41 (1.01 to 6.39) | 32 of 33 | 0.70 |

* **No rule for the cuts is better than the default without being worse in
  a third of the runs or more.**
* **Rules that only look at the points** (their spread, two clusters, a
  random direction) are 26 to 35% worse, along the inputs or not.
* **Hyperplanes from the leaf's GP** are level with the default on average
  and far from it in single runs. On the target with the oblique ridge the
  hyperplane along the change of the gradient has 0.74 to 0.79 of the
  default's error, on `oscillation_probability` 0.55 to 0.98, and on the
  smooth `eft_signal_strength`, `GaussianPeaks` and `RotatedRosenbrock` 0.99
  to 1.70.
* **Decorrelated inputs** make an oblique cell usable for the leaf's kernel:
  0.21 to 0.74 on the oblique ridge and 0.39 to 1.11 on `relic_density`,
  whose funnel lies on a diagonal, against 1.32 to 1.58 on `GaussianPeaks`.
* **A model in every node** is worse in every run.
* **On 20 000-point streams** the same 33 runs give the same picture:

| 20 000 points | uniform | walker | sweeping | all runs (range) | runs more than 2% worse |
|---|---|---|---|---|---|
| hyperplane of the gradient's change, placed where it changes most | 1.02 | 0.99 | 0.85 | 0.95 (0.42 to 1.71) | 16 of 33 |
| the same with decorrelated inputs | 0.86 | 0.92 | 0.76 | 0.84 (0.18 to 2.69) | 11 of 33 |
| predictions from neighbours (section 8.4), default cuts | 0.72 | 0.66 | 0.69 | 0.69 (0.38 to 0.98) | 0 of 33 |

### 8.2 Predictions from a query's neighbours (`predict_from='neighbours'`)

The kernel and the calibration are those of the query's leaf, and the
prediction is conditioned on the query's nearest stored points. In this
table the search for them is exact and no leaf prefers its own GP, except
in the fourth row (the two defaults of sections 8.4 and 8.5 came later); the
rows below it change one setting of the third each.

| predictions from neighbours | uniform | walker | sweeping | all runs (range) | runs more than 2% worse | coverage |
|---|---|---|---|---|---|---|
| 100 neighbours, no test against the kernel | 0.86 | 0.72 | 0.71 | 0.76 (0.31 to 2.12) | 4 of 33 | 0.69 |
| the same, distances in fitted length scales | 1.08 | 0.69 | 0.81 | 0.84 (0.44 to 6.46) | 4 of 33 | 0.68 |
| **100 neighbours, tested against the kernel** | 0.76 | 0.69 | 0.79 | 0.74 (0.45 to 1.04) | 1 of 33 | 0.69 |
| the same as it ships: 16 leaves searched, a leaf may keep to its own GP | 0.76 | 0.70 | 0.79 | 0.75 (0.45 to 1.04) | 1 of 33 | 0.69 |
| 150 neighbours | 0.72 | 0.64 | 0.76 | 0.71 (0.40 to 1.01) | 0 of 33 | 0.69 |
| 200 neighbours | 0.71 | 0.62 | 0.75 | 0.69 (0.36 to 1.00) | 0 of 33 | 0.69 |
| limit of the test 3 | 0.78 | 0.71 | 0.81 | 0.77 (0.45 to 1.04) | 1 of 33 | 0.69 |
| limit of the test 30 | 0.75 | 0.71 | 0.77 | 0.75 (0.45 to 1.04) | 1 of 33 | 0.69 |
| distances in fitted length scales | 0.75 | 0.65 | 0.80 | 0.73 (0.44 to 1.01) | 0 of 33 | 0.69 |
| 30 of the 100 at growing ranks of distance (no test) | 1.08 | 0.68 | 0.82 | 0.85 (0.42 to 6.49) | 3 of 33 | 0.68 |
| 4 points per level above the leaf among the 100 | 0.77 | 0.72 | 0.79 | 0.76 (0.47 to 1.05) | 2 of 33 | 0.69 |
| `Nbar=50` | 0.88 | 0.83 | 0.95 | 0.88 (0.54 to 2.39) | 9 of 33 | 0.69 |
| `Nbar=50`, a fit every 12 points | 0.85 | 0.83 | 0.89 | 0.86 (0.47 to 2.05) | 6 of 33 | 0.69 |
| `Nbar=200`, a fit every 50 points | 0.74 | 0.74 | 0.78 | 0.75 (0.44 to 1.31) | 4 of 33 | 0.68 |
| a fit every 50 points or more (one fit per leaf) | 0.77 | 0.72 | 0.83 | 0.77 (0.45 to 1.03) | 4 of 33 | 0.69 |
| at most 10 leaves searched | 0.76 | 0.69 | 0.79 | 0.75 (0.47 to 1.04) | 1 of 33 | 0.69 |
| hyperplane cuts along the gradient's change | 0.82 | 0.68 | 0.83 | 0.77 (0.45 to 1.81) | 5 of 33 | 0.68 |
| the same, placed where the gradient changes most | 0.76 | 0.71 | 0.77 | 0.75 (0.44 to 1.15) | 2 of 33 | 0.69 |
| the tree without neighbours, `Nbar=200`, a fit every 50 points | 0.89 | 0.93 | 0.87 | 0.89 (0.58 to 1.27) | 7 of 33 | 0.69 |
| the tree without neighbours, `theta=0.2` | 0.91 | 0.97 | 0.96 | 0.95 (0.65 to 1.58) | 7 of 33 | 0.86 |

* **The neighbours lower the error by 26%**, on every stream type, with one
  run more than 2% worse (`resonance_xsec` on the sweeping stream, 1.04).
  The gain is 36 to 55% on the smooth six-dimensional targets and 5 to 24%
  on the rough `Eggholder` and `Chirp`.
* **The coverage is unchanged** (0.69).
* **More neighbours lower the error further**: 29% with 150 and 31% with
  200, at a cost per prediction that grows with the cube of their number.
* **Doubling the leaves of the tree without neighbours gains 11%** at 1.6
  times the update time, and a wide overlap 5% with a sigma that is too
  large.
* **The leaf size still matters.** With `Nbar=50` the kernels are fitted to
  half as many points and the error is 19% higher than with `Nbar=100`, for
  the same 100 neighbours. `Nbar=200` gains nothing over `Nbar=100`.
* **A search that stops after 10 leaves** costs 1% in error.
* **Nothing else in the table helps**: far points among the neighbours,
  distances in fitted length scales, or cuts that are not along the inputs.

### 8.3 The test of the neighbours against the kernel

Recorded for every query of the second half of the 33 runs: the prediction
of the query's leaf, and that of the leaf's kernel conditioned on the 100
nearest points, with `y^T K^-1 y / k` of those points (the consistency). All
numbers are relative to the leaf's own prediction.

| the neighbours predict where the consistency is | NRMSE | worst run | runs more than 2% worse | share of the queries |
|---|---|---|---|---|
| anything (no test) | 0.78 | 3.10 | 3 of 33 | 100% |
| at most 1.5 | 0.81 | 1.01 | 0 | 81% |
| at most 3 | 0.76 | 1.00 | 0 | 91% |
| at most 5 | 0.73 | 1.00 | 0 | 94% |
| at most 10 (the default) | 0.73 | 1.00 | 0 | 96% |
| at most 30 | 0.72 | 1.00 | 0 | 98% |
| at most 100 | 0.72 | 1.00 | 0 | 99% |
| whichever of the two is nearer the truth | 0.62 | 0.97 | 0 | 60% |

* **Without the test three runs are up to 3.1 times worse.** On
  `resonance_xsec`, whose value falls by orders of magnitude across an
  oblique line, a leaf on the smooth side has length scales of 30 to 600
  times its spread. Its 23 worst predictions from neighbours on a sweeping
  stream have a consistency of 350 to 13 000 and errors of up to 13 where
  the leaf's own are below 0.5; elsewhere the consistency is 0.1 to 2.
* **Any limit from 5 to 100 gives the same error**, and no run is worse than
  the leaf's own predictions.
* **Other tests are no better**: the leave-one-out errors of the 5, 10 or 20
  nearest neighbours in units of their sigma (0.72 to 0.73 at their best
  limits), the smaller of the two sigmas (0.75), the largest set of nearest
  neighbours that passes the test, or the neighbours up to the first one
  that the nearer ones predict badly (0.72).
* **The gain grows with the distance of the query from the centre of its
  leaf**, the far score of section 7.5:

| far score | below 0.75 | 0.75 to 1.25 | 1.25 to 1.75 | 1.75 to 2.5 | above 2.5 |
|---|---|---|---|---|---|
| share of the queries | 4% | 21% | 45% | 21% | 13% |
| NRMSE relative to the leaf's own | 0.98 | 0.88 | 0.77 | 0.74 | 0.68 |

* **The number of neighbours**, without the test: 1.27 with 10, 1.01 with
  20, 0.85 with 50, 0.78 with 100, 0.74 with 200.

### 8.4 Predictions from neighbours on 20 000-point streams

The option without fading (100 neighbours from the 16 nearest leaves, the
test against the kernel, and a leaf that keeps to its own GP while that
predicts better) on nineteen targets: the eleven physics targets, `GaussianPeaks`,
`RotatedRosenbrock`, `StepRidge` (d = 6), `ActiveSubspacePeaks` (d = 10),
`Michalewicz` (d = 5), `Eggholder` (d = 3), `Chirp` and the target with the
oblique ridge (d = 4), on four stream types with two seeds: 150 runs
(`branching_ratio` is constant at the end of the focusing stream). The NRMSE
of the last 5000 predictions divided by that of the default tree, for the two
seeds:

| target | d | uniform | walker | sweeping | focusing |
|---|---|---|---|---|---|
| `active_peaks` | 10 | 0.73 / 0.74 | 0.69 / 0.73 | 0.63 / 0.82 | 0.56 / 0.58 |
| `branching_ratio` | 5 | 1.11 / 0.89 | 0.93 / 0.99 | 1.00 / 0.98 |  |
| `chirp` | 4 | 0.92 / 0.91 | 0.72 / 0.67 | 0.67 / 0.70 | 0.32 / 0.33 |
| `diagonal_ridge` | 4 | 0.70 / 0.67 | 0.74 / 0.67 | 0.83 / 0.77 | 0.68 / 0.72 |
| `dilepton_spectrum` | 4 | 0.90 / 0.85 | 0.92 / 0.85 | 0.73 / 0.82 | 0.94 / 0.86 |
| `eft_signal_strength` | 6 | 0.39 / 0.40 | 0.38 / 0.35 | 0.58 / 0.44 | 1.04 / 1.04 |
| `eggholder` | 3 | 0.88 / 0.89 | 0.83 / 0.86 | 0.90 / 0.90 | 0.83 / 0.85 |
| `gaussian_peaks` | 6 | 0.64 / 0.66 | 0.40 / 0.46 | 0.51 / 0.46 | 0.54 / 0.62 |
| `global_fit_loglike` | 10 | 0.98 / 0.93 | 0.94 / 0.95 | 0.93 / 0.96 | 0.04 / 1.00 |
| `jet_spectrum` | 4 | 0.97 / 0.82 | 1.00 / 0.76 | 0.61 / 0.34 | 0.45 / 0.44 |
| `loop_decay_width` | 4 | 0.85 / 0.79 | 0.86 / 0.92 | 0.39 / 0.21 | 0.29 / 0.27 |
| `michalewicz` | 5 | 1.00 / 1.02 | 0.81 / 0.95 | 1.01 / 1.00 | 0.90 / 0.89 |
| `neutralino_mass` | 4 | 0.84 / 0.80 | 0.98 / 0.27 | 0.83 / 0.99 | 0.48 / 0.41 |
| `oscillation_probability` | 5 | 0.60 / 0.53 | 0.62 / 1.01 | 0.92 / 0.94 | 0.85 / 0.90 |
| `relic_density` | 4 | 0.90 / 0.86 | 0.92 / 0.96 | 0.86 / 0.71 | 0.92 / 0.95 |
| `resonance_xsec` | 4 | 0.92 / 0.96 | 0.61 / 1.00 | 0.97 / 0.88 | 0.67 / 0.69 |
| `rotated_rosenbrock` | 6 | 0.52 / 0.48 | 0.53 / 0.53 | 0.45 / 0.62 | 0.47 / 0.47 |
| `signal_yield_mc` | 4 | 0.84 / 0.84 | 0.86 / 0.85 | 0.56 / 0.53 | 0.95 / 0.93 |
| `step_ridge` | 6 | 0.91 / 0.86 | 0.96 / 0.92 | 0.87 / 0.90 | 0.92 / 0.92 |
| geometric mean | | 0.78 | 0.74 | 0.71 | 0.61 |

* **0.71 of the default tree's error** as the geometric mean over the 150
  runs, from 0.04 to 1.11. 135 runs are more than 2% better and 4 more than
  2% worse.
* **The gain is largest on the streams that move**: 22% on the uniform
  stream, 26% on the walker, 29% on the sweeping and 39% on the focusing
  stream.
* **Every part of the error distribution falls**: the mean absolute error to
  0.74, the median to 0.74, the 99% quantile to 0.72 and the largest error
  of a run to 0.67.
* **The four runs that are worse.** `eft_signal_strength` on the focusing
  stream, 4% with both seeds: its declared noise is large against the change
  of the target across the neighbours of a query in the contracted cluster,
  and without the choice of the leaf's own GP it was 26% worse.
  `branching_ratio` on the uniform stream, 11% with one seed and 11% better
  with the other: one prediction of a leaf's own GP beside a threshold,
  which the neighbours had failed the test for, is off by more than the
  target's range. `Michalewicz` on the uniform stream, 2%.
* **`global_fit_loglike` on the focusing stream, 0.04**: a single prediction
  of the default tree is off by twice the range of the last quarter.
* **Off the stream the gain is smaller**: 0.75 on points held out of the
  stream itself, predicted at its end, and 0.83 on uniform test points.
* **The coverage is unchanged**, 0.710 against 0.709, and 2.0% of the
  predictions are off by more than 5 sigma against 1.9%.

### 8.5 The leaves that are searched

`GaussianPeaks`, `Eggholder`, `resonance_xsec`, `eft_signal_strength`,
`global_fit_loglike` and `ActiveSubspacePeaks` on three stream types, 20 000
points, one seed (18 runs), with the test against the kernel and no choice
of the leaf's own GP. The neighbours are the 100 nearest points of the
leaves whose points' boxes are nearest to the query.

| leaves searched | NRMSE relative to the exact search (range) | relative to the default tree |
|---|---|---|
| 4 | 1.05 (0.94 to 1.28) | 0.71 |
| 8 | 1.01 (0.98 to 1.05) | 0.68 |
| 16 (the default) | 1.00 (0.95 to 1.02) | 0.68 |
| 32 | 1.00 (0.99 to 1.01) | 0.68 |
| every leaf that can hold a neighbour | 1 | 0.68 |

* **Sixteen leaves give the error of the exact search.**
* **The exact search looks at more leaves the more inputs there are**: at
  the end of the 20 000-point runs of section 8.4, 14 to 16 leaves per query
  in three dimensions, 8 to 31 in four, 7 to 46 in five, 35 to 79 in six and
  41 to 77 in ten, of about 285.

### 8.6 100 000 points

`GaussianPeaks` and `eft_signal_strength` (d = 6), `Eggholder` (d = 3),
`resonance_xsec` (d = 4) and `global_fit_loglike` (d = 10) on the uniform,
walker and sweeping streams, one seed: 15 runs of the default tree and of
the tree with `predict_from='neighbours'`. Times are means over the runs and
over the points of each part of the stream, with four runs at a time on four
cores.

| points | default tree: update ms mean / 99th percentile | predict ms | neighbours: update ms mean / 99th percentile | predict ms | leaves | deepest leaf of the cuts, mean (largest) | NRMSE relative to the default tree |
|---|---|---|---|---|---|---|---|
| 0 to 1000 | 2.40 / 71 | 0.20 | 2.98 / 66 | 0.50 | 15 | 5 (at most 6) | 0.99 |
| 1000 to 4000 | 2.70 / 76 | 0.25 | 3.50 / 80 | 0.66 | 58 | 8 (at most 11) | 0.76 |
| 4000 to 16000 | 2.52 / 67 | 0.24 | 3.18 / 64 | 0.65 | 235 | 12 (at most 17) | 0.69 |
| 16000 to 64000 | 2.53 / 70 | 0.24 | 3.41 / 73 | 0.76 | 936 | 19 (at most 34) | 0.62 |
| 64000 to 100000 | 2.52 / 67 | 0.26 | 3.40 / 64 | 0.81 | 1469 | 20 (at most 34) | 0.63 |

* **The update time does not grow with the stream** for either tree beyond
  the first 4000 points, where the leaves fill up. With neighbours an update
  takes 1.3 times as long, and a prediction three times. These runs had the
  first division of the index's groups (section 8.9), with which a
  prediction from neighbours went from 0.65 to 0.81 ms; section 8.12 has
  the times of the package as it is.
* **The search looks at 16 leaves per query** from the first split to the
  last, of 1469 at the end.
* **The gain from the neighbours grows with the stream** and then holds: the
  more points there are, the nearer a query's neighbours lie compared with
  the points of its leaf, whose number stays the same.

NRMSE of the last 25 000 predictions:

| target | d | stream | default tree | neighbours | ratio |
|---|---|---|---|---|---|
| `eft_signal_strength` | 6 | sweeping | 0.0018 | 0.0007 | 0.38 |
| `eft_signal_strength` | 6 | uniform | 0.0009 | 0.0003 | 0.37 |
| `eft_signal_strength` | 6 | walker | 0.0019 | 0.0006 | 0.33 |
| `eggholder` | 3 | sweeping | 0.0328 | 0.0283 | 0.86 |
| `eggholder` | 3 | uniform | 0.0213 | 0.0170 | 0.79 |
| `eggholder` | 3 | walker | 0.0232 | 0.0195 | 0.84 |
| `gaussian_peaks` | 6 | sweeping | 0.0060 | 0.0031 | 0.52 |
| `gaussian_peaks` | 6 | uniform | 0.0039 | 0.0022 | 0.57 |
| `gaussian_peaks` | 6 | walker | 0.0068 | 0.0027 | 0.40 |
| `global_fit_loglike` | 10 | sweeping | 0.0153 | 0.0117 | 0.77 |
| `global_fit_loglike` | 10 | uniform | 0.0165 | 0.0146 | 0.88 |
| `global_fit_loglike` | 10 | walker | 0.0174 | 0.0155 | 0.89 |
| `resonance_xsec` | 4 | sweeping | 0.0135 | 0.0128 | 0.94 |
| `resonance_xsec` | 4 | uniform | 0.0055 | 0.0051 | 0.94 |
| `resonance_xsec` | 4 | walker | 0.0078 | 0.0077 | 0.99 |

* **0.65 of the default tree's error** as the geometric mean, from 0.33 to
  0.99, and no run worse. The coverage is 0.66 to 0.78.

### 8.7 Cost with one process

Streams of 20 000 points, one run at a time on an otherwise idle machine,
with the first division of the index's groups (section 8.9). The update time
is the mean over the stream, the prediction time the mean over
1000 single-point predictions at the points that follow it, and the RMSE
that of those predictions, relative to the default tree. The four numbers of
a cell are for `GaussianPeaks` on the uniform and on the walker stream,
`resonance_xsec` and `global_fit_loglike` on the uniform stream.

| setting | update ms | predict ms | leaves searched | RMSE relative to the default tree |
|---|---|---|---|---|
| default tree | 1.77 / 1.92 / 2.95 / 3.38 | 0.14 / 0.14 / 0.14 / 0.14 | | 1 |
| **neighbours: 100 from 16 leaves** | 2.50 / 2.66 / 3.59 / 4.23 | 0.58 / 0.62 / 0.51 / 0.46 | 16 | 0.66 / 0.59 / 0.94 / 0.85 |
| exact search | 2.81 / 2.84 / 3.71 / 4.38 | 0.91 / 1.14 / 0.62 / 0.57 | 94 / 111 / 33 / 72 | 0.66 / 0.59 / 0.94 / 0.85 |
| 8 leaves | 2.48 / 2.59 / 3.60 / 4.08 | 0.62 / 0.57 / 0.46 / 0.42 | 8 | 0.69 / 0.59 / 0.94 / 0.83 |
| 50 neighbours | 2.45 / 2.50 / 3.58 / 4.09 | 0.46 / 0.44 / 0.39 / 0.36 | 16 | 0.78 / 0.72 / 1.02 / 1.00 |
| 200 neighbours | 2.99 / 2.97 / 4.13 / 4.70 | 1.07 / 1.09 / 0.89 / 0.78 | 16 | 0.61 / 0.47 / 0.93 / 0.87 |

* **A prediction from neighbours takes 0.46 to 0.62 ms** against 0.14 ms,
  3.3 to 4.4 times as long.
* **An update takes 0.6 to 0.9 ms longer**, 22 to 41%: the point is
  predicted from the neighbours and from the leaf's own GP before it is
  stored.
* **The exact search** takes up to 0.5 ms more per prediction for the same
  error, the more the higher the dimension of the cells that touch a query's
  neighbourhood.
* **With 50 neighbours** a prediction is 0.1 to 0.2 ms faster and the gain
  is gone on two of the four streams. **With 200** it takes 1.7 to 1.8 times
  as long and the error falls further on `GaussianPeaks` only.

### 8.8 A stream that drifts

A four-dimensional cluster of width 0.03 that moves along the first input by
its width every 120 points, 12 000 points, each predicted before it is added;
one process. The first pair of columns is the package before the index of
the leaves (commit `1a242db`), where a point reaches its leaf through the
cuts. The times are means over the 2000 points up to each row.

| points | leaves | depth of the cuts, mean (largest) | through the cuts: predict ms | update ms | leaf index: predict ms | update ms | leaf index, neighbours: predict ms | update ms |
|---|---|---|---|---|---|---|---|---|
| 2000 | 26 | 14 (25) | 0.24 | 3.16 | 0.18 | 3.28 | 0.44 | 3.73 |
| 4000 | 52 | 27 (51) | 0.37 | 3.46 | 0.18 | 3.52 | 0.51 | 4.18 |
| 6000 | 77 | 39 (76) | 0.51 | 3.77 | 0.18 | 3.41 | 0.51 | 4.03 |
| 8000 | 110 | 54 (95) | 0.64 | 4.06 | 0.19 | 3.53 | 0.51 | 4.03 |
| 10 000 | 145 | 66 (113) | 0.73 | 4.17 | 0.20 | 3.61 | 0.50 | 4.05 |
| 12 000 | 179 | 76 (131) | 0.83 | 4.43 | 0.19 | 3.78 | 0.51 | 4.17 |

* **Every split adds a level to the cuts**: the deepest leaf is 131 levels
  down, where a balanced tree of 179 leaves has 8.
* **Through the cuts the prediction time grows with the depth**, 3.4 times
  from the first 2000 points to the last, and the update time by 1.3 ms.
  With the index both stay as they are, and the predictions are the same.
* **A tree more than 1100 levels deep** (two inputs, `Nbar=8`, 9000 points
  of such a stream) predicts, is saved and loaded with the index
  (`tests/test_leaf_index.py`). Through the cuts the descent failed at about
  1000 levels and saving from about 150.
* **With neighbours the RMSE of the last 6000 predictions is 0.59 of the
  default tree's.**

### 8.9 The index of the leaves up to a million points

Without GPs (`research/partitioning/index_scaling.py`): leaves of 100 points,
cut at the median along a random input, on a uniform stream and on a random
walk in the unit cube. The groups of the index that the search for the 16
nearest leaves opens, as the mean over 500 queries near the latest points:
with groups divided at the median of their children's centres (the index of
sections 8.4 to 8.8), then with groups divided where their halves overlap
least (the index of the package). Both give the same leaves.

| points | leaves | uniform, d = 3 | d = 6 | d = 10 | walk, d = 3 | d = 6 | d = 10 |
|---|---|---|---|---|---|---|---|
| 5000 | 70 | 5.0 → 3.9 | 4.5 → 3.7 | 4.6 → 4.4 | 3.9 → 3.8 | 4.5 → 4.5 | 4.0 → 4.7 |
| 20 000 | 300 | 6.9 → 5.0 | 8.5 → 5.4 | 10.6 → 5.5 | 6.1 → 5.8 | 7.8 → 7.6 | 10.3 → 7.3 |
| 100 000 | 1500 | 10.9 → 6.8 | 19.3 → 7.4 | 25.2 → 8.2 | 12.3 → 7.9 | 25.7 → 10.1 | 30.3 → 12.3 |
| 400 000 | 6000 | 14.8 → 7.2 | 37.0 → 8.6 | 48.9 → 9.4 | 19.0 → 7.8 | 46.1 → 10.4 | 67.6 → 16.4 |
| 1 000 000 | 14 500 | 17.9 → 7.4 | 55.3 → 9.3 | 70.9 → 10.4 | | | |

* **Divided at the median of the centres, the groups opened grow with a
  power of the number of leaves**: twice as many for four times the leaves
  in six and ten dimensions.
* **Divided where the halves overlap least, they grow by one to four for
  four times the leaves**, and a point lies in the cells' box of one bottom
  group, against 5 in four dimensions and 9 in eight.
* **The predictions do not change**: on two 5000-point streams with
  `predict_from='neighbours'` all of them are identical with the two
  divisions.

The same on the default tree of `GaussianPeaks` on the uniform stream
(`research/partitioning/profile_search.py`, 1000 queries at the points that
follow each checkpoint, three processes at a time), first division → least
overlap:

| points | leaves | groups opened | `nearest` ms | the whole search for 100 neighbours, ms |
|---|---|---|---|---|
| 5000 | 70 | 5.0 → 3.9 | 0.06 → 0.05 | 0.14 → 0.14 |
| 20 000 | 291 | 8.8 → 5.9 | 0.10 → 0.07 | 0.19 → 0.16 |
| 50 000 | 728 | 9.1 → 7.8 | 0.12 → 0.09 | 0.23 → 0.20 |
| 100 000 | 1436 | 13.1 → 8.3 | 0.17 → 0.10 | 0.32 → 0.23 |

### 8.10 Neighbours with the feature network

`GaussianPeaks`, `RotatedRosenbrock`, `Eggholder`, `resonance_xsec`,
`eft_signal_strength` and `relic_density` on the uniform, walker and
sweeping streams, 20 000 points, one seed: 18 runs. NRMSE of the last 5000
predictions divided by that of the default tree, for the three streams in
that order. The network is `global_mean='net'` with one training step per
update.

| target | neighbours | feature network | both |
|---|---|---|---|
| `eft_signal_strength` | 0.39 / 0.38 / 0.58 | 0.12 / 0.15 / 0.44 | 0.11 / 0.14 / 0.34 |
| `eggholder` | 0.88 / 0.83 / 0.90 | 0.98 / 1.52 / 1.09 | 0.88 / 1.09 / 0.97 |
| `gaussian_peaks` | 0.64 / 0.40 / 0.51 | 0.18 / 0.11 / 0.72 | 0.17 / 0.10 / 0.61 |
| `relic_density` | 0.90 / 0.92 / 0.86 | 0.48 / 0.72 / 0.74 | 0.47 / 0.73 / 0.67 |
| `resonance_xsec` | 0.92 / 0.61 / 0.97 | 0.59 / 2.55 / 1.18 | 0.61 / 2.53 / 1.12 |
| `rotated_rosenbrock` | 0.52 / 0.53 / 0.45 | 0.08 / 0.07 / 0.50 | 0.07 / 0.06 / 0.43 |
| geometric mean | 0.64 | 0.43 | 0.39 |

* **With the network the neighbours lower the error by a further 10%**
  (0.72 to 1.03 of the error with the network alone): 14 of the 18 runs are
  more than 2% better and one is worse (`resonance_xsec` on the uniform
  stream, 3%).
* **The gain is smaller than without the network** (36% on these runs),
  since the network already carries what the points beyond a leaf say about
  the target.
* **The neighbours do not repair a run that the network makes worse**:
  `resonance_xsec` on the walker stream stays at 2.5 times the tree's error.
* **The coverage is unchanged**, 0.71 against 0.70 with the network alone,
  and 3.0% of the predictions are off by more than 5 sigma with both.
* **A prediction takes 1.44 ms against 0.50 ms** with the network alone
  (four runs at a time on four cores), and an update 4.7 ms against 3.3 ms:
  the leaf's snapshot of the network is evaluated at the neighbours.

### 8.11 Noisy observations

The eleven targets of sections 8.1 to 8.3 with Gaussian noise of 0.2
standard deviations of the target on every observation, declared to the
model, on the three stream types: 33 runs of 5000 points. NRMSE of the last
1250 predictions against the noise-free target, with
`predict_from='neighbours'` relative to the default tree.

| target | uniform | walker | sweeping |
|---|---|---|---|
| `chirp` | 0.98 | 0.82 | 0.88 |
| `diagonal_ridge` | 0.87 | 0.82 | 0.84 |
| `eft_signal_strength` | 0.73 | 0.77 | 0.75 |
| `eggholder` | 0.95 | 0.86 | 0.94 |
| `gaussian_peaks` | 0.91 | 0.70 | 0.79 |
| `loop_decay_width` | 0.84 | 0.81 | 0.83 |
| `neutralino_mass` | 0.97 | 1.02 | 0.74 |
| `oscillation_probability` | 0.85 | 0.81 | 0.97 |
| `relic_density` | 0.93 | 0.94 | 0.90 |
| `resonance_xsec` | 0.98 | 0.89 | 0.96 |
| `rotated_rosenbrock` | 0.80 | 0.79 | 0.81 |
| geometric mean | 0.89 | 0.83 | 0.85 |

* **0.86 of the default tree's error**, from 0.70 to 1.02, and no run more
  than 2% worse. Without the noise the same runs are at 0.75.
* **The coverage of the noise-free target** is 0.91 against 0.90, with a
  sigma that includes the declared noise.

### 8.12 Cost along a 200 000-point stream

`GaussianPeaks` (d = 6) on the uniform stream, one process on an otherwise
idle machine, the package as it is. The update time is the mean since the
previous row, the prediction time the mean over 1000 single-point
predictions at the points that follow, and the RMSE that of those
predictions (`research/partitioning/time_scaling.py`).

| points | leaves | default tree: update ms | predict ms | RMSE | neighbours: update ms | predict ms | RMSE | RMSE ratio |
|---|---|---|---|---|---|---|---|---|
| 5000 | 71 | 1.86 | 0.13 | 0.459 | 2.50 | 0.55 | 0.327 | 0.71 |
| 20 000 | 289 | 1.86 | 0.17 | 0.282 | 2.59 | 0.61 | 0.175 | 0.62 |
| 50 000 | 730 | 1.88 | 0.14 | 0.237 | 2.53 | 0.63 | 0.139 | 0.59 |
| 100 000 | 1465 | 1.96 | 0.15 | 0.162 | 2.65 | 0.62 | 0.103 | 0.64 |
| 200 000 | 2917 | 1.92 | 0.14 | 0.129 | 2.78 | 0.59 | 0.068 | 0.53 |

* **The prediction time does not grow with the stream**: 0.55 to 0.63 ms
  with neighbours and 0.13 to 0.17 ms without, from 71 leaves to 2917.
* **The update time grows by 3% for the default tree and by 11% with
  neighbours** from the first 5000 points to the last 100 000.
* **The gain from the neighbours grows with the stream**, to 0.53 of the
  default tree's error at 200 000 points.

`global_fit_loglike` (d = 10) on the walker stream, 100 000 points, with
the first division of the index's groups and with the package as it is,
side by side (two processes at a time; the predictions are the same):

| points | default tree, predict ms: first division → least overlap | neighbours, update ms | neighbours, predict ms |
|---|---|---|---|
| 5000 | 0.12 → 0.12 | 3.81 → 3.63 | 0.43 → 0.38 |
| 20 000 | 0.17 → 0.18 | 3.74 → 3.59 | 0.55 → 0.50 |
| 50 000 | 0.17 → 0.13 | 3.81 → 3.59 | 0.66 → 0.42 |
| 100 000 | 0.21 → 0.13 | 3.84 → 3.33 | 0.97 → 0.53 |

* **With the first division a prediction from neighbours took twice as long
  at 100 000 points as at 5000**, and one of the default tree 1.7 times.
  With the division at the least overlap neither grows beyond the spread of
  the measurement.

### 8.13 Two clusters per split, and the steps of the predictions (not in the code)

**A split into two clusters.** The points of a leaf are divided into two
clusters (2-means), and a point goes to the child whose centroid is nearer:
a cut by the hyperplane half-way between the centroids. The 33 runs of
5000 points of sections 8.1 to 8.3, one GP per leaf, relative to the default
tree.

| two clusters of the leaf's points | uniform | walker | sweeping | all runs (range) | runs more than 2% worse |
|---|---|---|---|---|---|
| nearest centroid | 1.48 | 1.31 | 1.03 | 1.26 (0.22 to 4.48) | 24 of 33 |
| nearest centroid, standardised inputs | 1.44 | 1.35 | 1.13 | 1.30 (0.35 to 4.83) | 26 of 33 |
| cut at the median along the line between the centroids (section 8.1) | 1.54 | 1.33 | 1.08 | 1.30 (0.30 to 4.88) | 23 of 33 |
| nearest centroid, distances in the GP's fitted length scales | 0.99 | 1.17 | 0.97 | 1.04 (0.65 to 2.43) | 15 of 33 |
| the same, cut at the median | 0.99 | 0.99 | 1.00 | 0.99 (0.61 to 1.71) | 14 of 33 |
| nearest centroid, fitted length scales, decorrelated inputs for the GPs | 1.08 | 1.31 | 1.14 | 1.17 (0.59 to 3.24) | 25 of 33 |

* **Clusters of the inputs alone are 26 to 30% worse than the default
  cuts**, as the other rules of section 8.1 that do not look at the target.
* **Clusters in the metric of the leaf's kernel are level with the default
  cuts** and differ from them by up to 2.4 times in single runs, as the
  other hyperplanes from the GP.

**Steps of the predicted mean.** After 20 000 points, along 60 straight paths
from a stored point to its 200th nearest stored point, in 1000 steps
(`research/partitioning/jitter.py`). A step counts by how far it departs
from the mean of the steps on either side of it, in units of the predicted
sigma, so that one discontinuity counts up to three times: itself and, at
half its size, the step before and after it. Means over the paths, which
pass through 2.4 to 2.9 leaves.

| target, stream | prediction from | steps above 0.01 sigma | above 0.1 sigma | above 1 sigma | their sum in sigma, mean (median) | largest step, median |
|---|---|---|---|---|---|---|
| `GaussianPeaks`, uniform | the leaf | 4.6 | 3.9 | 0.9 | 4.3 (2.3) | 0.89 |
| | the query's neighbours | 243 | 32.6 | 0.2 | 14.0 (13.1) | 0.45 |
| | the neighbours of one of 8 cells of the leaf | 8.6 | 7.0 | 1.7 | 5.7 (4.7) | 1.46 |
| | of one of 32 cells | 10.7 | 8.3 | 1.4 | 5.8 (5.0) | 1.21 |
| `Eggholder`, walker | the leaf | 4.9 | 3.9 | 1.4 | 4.8 (3.0) | 1.26 |
| | the query's neighbours | 185 | 39.0 | 1.4 | 16.9 (12.3) | 0.75 |
| | the neighbours of one of 8 cells of the leaf | 10.0 | 7.0 | 1.4 | 5.6 (4.6) | 1.21 |
| | of one of 32 cells | 14.5 | 10.0 | 1.4 | 5.8 (4.1) | 0.92 |
| `resonance_xsec`, uniform | the leaf | 6.1 | 4.8 | 2.0 | 13.7 (2.8) | 1.36 |
| | the query's neighbours | 175 | 35.1 | 2.1 | 25.8 (16.7) | 1.21 |

* **The leaf's prediction steps where the path enters another leaf**, 4 to
  5 times on a path, by about one sigma at the largest.
* **The neighbours' prediction steps whenever the set of neighbours
  changes**: 33 to 39 steps above 0.1 sigma on a path and 3 to 5 times the
  sum, with a largest step that is no larger than the leaf's.

**Neighbours of a cell.** Each leaf is divided into 4 to 32 cells by median
cuts of its points, and a prediction is conditioned on the 100 stored points
nearest to the mean of the points of the query's cell, so that the queries
of a cell share their neighbours (`research/partitioning/anchored.py`). The
33 runs of 5000 points, relative to the default tree.

| neighbours of | uniform | walker | sweeping | all runs (range) | runs more than 2% worse |
|---|---|---|---|---|---|
| the query (`predict_from='neighbours'`) | 0.76 | 0.70 | 0.79 | 0.75 (0.45 to 1.04) | 1 of 33 |
| one of 4 cells of its leaf | 0.89 | 1.05 | 1.04 | 0.99 (0.78 to 1.24) | 12 of 33 |
| one of 8 cells | 0.87 | 1.08 | 1.04 | 0.99 (0.78 to 1.55) | 13 of 33 |
| one of 16 cells | 0.84 | 0.96 | 1.04 | 0.94 (0.67 to 1.66) | 8 of 33 |
| one of 32 cells | 0.83 | 0.98 | 1.02 | 0.94 (0.64 to 2.02) | 10 of 33 |

* **Neighbours that are shared by a cell step as rarely as the leaf's
  prediction and keep 1 to 6% of the 25% that the query's own neighbours
  gain.** A leaf of 100 points in four to six dimensions is two or three
  points wide along each input: 32 cells halve it along five inputs, and
  the centre of a cell is still a quarter of the leaf away from most of its
  queries.

### 8.14 Neighbours that fade out: the prototype

The prediction from a query's neighbours with a noise that grows with the
distance of a neighbour (`research/partitioning/tapered.py`; section 8.15
has the version of the package). With `r` the
distance of a neighbour divided by that of the first stored point beyond the
100 neighbours, and `t = (r - start) / (1 - start)` where `r` exceeds
`start`, the noise variance of the neighbour is raised by
`e^2 t^2 / (1 - t)^2`, where `e^2` is the typical squared error of the
leaf's recent predictions (the median of their squares divided by 0.455). A
point enters and leaves the set of neighbours with no weight, so the
prediction is continuous in the query for as long as its leaf, and the leaves
that are searched, stay the same.

The 33 runs of sections 8.1 to 8.3, relative to the default tree:

| 100 neighbours unless stated | 5000 points: uniform | walker | sweeping | all runs (range) | runs more than 2% worse | 20 000 points: all runs (range) | runs more than 2% worse |
|---|---|---|---|---|---|---|---|
| as in the package (no fading) | 0.76 | 0.70 | 0.79 | 0.75 (0.45 to 1.04) | 1 of 33 | 0.69 (0.38 to 0.98) | 0 of 33 |
| fading from `start = 0.9` | 0.77 | 0.72 | 0.80 | 0.76 (0.47 to 1.04) | 1 of 33 | 0.70 (0.39 to 0.98) | 0 of 33 |
| fading from 0.8 | 0.78 | 0.73 | 0.81 | 0.77 (0.50 to 1.04) | 1 of 33 | 0.71 (0.38 to 0.99) | 0 of 33 |
| fading from 0.5 | 0.83 | 0.78 | 0.83 | 0.82 (0.57 to 1.07) | 2 of 33 | | |
| from 0.8, a tenth of the added noise | 0.77 | 0.72 | 0.80 | 0.76 (0.47 to 1.04) | 1 of 33 | | |
| from 0.8, ten times the added noise | 0.81 | 0.76 | 0.83 | 0.80 (0.53 to 1.04) | 2 of 33 | | |
| 150 neighbours, no fading | 0.72 | 0.65 | 0.76 | 0.71 (0.40 to 1.02) | 0 of 33 | | |
| 150 neighbours, fading from 0.8 | 0.74 | 0.68 | 0.78 | 0.73 (0.43 to 1.04) | 1 of 33 | | |
| 150 neighbours, fading from 0.5 | 0.76 | 0.71 | 0.80 | 0.76 (0.49 to 1.05) | 2 of 33 | | |

The steps of the predicted mean, measured as in section 8.13 after 20 000
points:

| target, stream | prediction from | steps above 0.01 sigma | above 0.1 sigma | above 1 sigma | their sum in sigma, mean (median) | largest step, median |
|---|---|---|---|---|---|---|
| `GaussianPeaks`, uniform | the leaf | 4.6 | 3.9 | 0.9 | 4.3 (2.3) | 0.89 |
| | neighbours | 243 | 32.6 | 0.2 | 14.0 (13.1) | 0.45 |
| | neighbours fading from 0.8 | 24.4 | 4.8 | 0.2 | 2.0 (1.5) | 0.27 |
| | fading from 0.9, 32 leaves searched | 21.0 | 2.6 | 0.2 | 1.5 (0.8) | 0.22 |
| `Eggholder`, walker | the leaf | 4.9 | 3.9 | 1.4 | 4.8 (3.0) | 1.26 |
| | neighbours | 185 | 39.0 | 1.4 | 16.9 (12.3) | 0.75 |
| | neighbours fading from 0.8 | 36.5 | 3.6 | 0.5 | 2.8 (1.4) | 0.35 |
| | fading from 0.9, 32 leaves searched | 51.9 | 4.4 | 0.5 | 3.5 (1.9) | 0.37 |
| `resonance_xsec`, uniform | the leaf | 6.1 | 4.8 | 2.0 | 13.7 (2.8) | 1.36 |
| | neighbours | 175 | 35.1 | 2.1 | 25.8 (16.7) | 1.21 |
| | neighbours fading from 0.8 | 32.9 | 5.5 | 1.6 | 12.8 (3.1) | 1.09 |
| | fading from 0.9, 32 leaves searched | 43.6 | 5.5 | 1.6 | 13.7 (3.1) | 0.98 |

* **With fading neighbours the prediction steps as often as the leaf's
  own**, 3 to 6 times above 0.1 sigma on a path against 33 to 39 without
  fading, and by less: the largest step of a path is 0.2 to 1.1 sigma
  against 0.9 to 1.4 for the leaves.
* **The fading costs 1% of the error from 0.9 and 3% from 0.8** at 20 000
  points (0.99 to 1.06 and 0.98 to 1.13 of the error without fading in
  single runs), and no run is worse than the default tree. The coverage is
  0.685 against 0.684.
* **The width of the fading hardly changes the steps**, and a wider one
  costs more error: 9% from 0.5.
* **The steps that remain** are where the path enters another leaf, whose
  kernel and metric differ, and where the 16 leaves that are searched
  change: with 32 leaves, or an exact search, the steps above 0.1 sigma on
  a 6000-point tree of `GaussianPeaks` fall from 3.9 to 2.3 per path.
* **Cost** (`GaussianPeaks`, 20 000 points, one process): a prediction
  takes 0.57 ms against 0.52 ms without fading, and 0.69 ms with 32 leaves
  searched; an update 2.5 ms against 2.3 ms.

### 8.15 Neighbours that fade out (`neighbour_fade`)

In the package the noise variance of a neighbour grows by `t^2 / (1 - t)^2`
times a unit, with `t` as in section 8.14 and `start = 0.9`. Candidates for
the unit, on the 33 runs of 5000 points relative to the default tree, with
the steps of the mean after 20 000 points as in section 8.13 (`GaussianPeaks`
on the uniform stream / `Eggholder` on the walker stream):

| unit of the added noise | all runs (range) | runs more than 2% worse | steps above 0.1 sigma | their sum in sigma |
|---|---|---|---|---|
| no fading | 0.75 (0.45 to 1.04) | 1 of 33 | 32.6 / 39.0 | 14.0 / 16.9 |
| the squared error of the leaf's predictions (section 8.14), plus the neighbour's own noise variance | 0.76 (0.48 to 1.04) | 1 of 33 | 4.8 / 3.5 | 2.2 / 2.9 |
| **the variance the leaf's GPs predict, plus the neighbour's own** | 0.77 (0.49 to 1.04) | 1 of 33 | 4.8 / 2.5 | 2.1 / 2.2 |
| a quarter of the predicted variance, plus the neighbour's own | 0.76 (0.47 to 1.04) | 1 of 33 | 5.0 / 2.9 | 2.2 / 2.4 |

* **The squared error of a leaf's predictions is 0.12 to 0.47 of the
  variance its GPs predict** (medians over the leaves of four noise-free
  targets after 5000 points), and 4.5 times that variance on a target with
  noise of 0.2 standard deviations, where it contains the noise.
* **Under a kernel that is not fitted** (a fixed kernel, declared noise ten
  times the size of the errors, 30 neighbours in two dimensions) the largest
  step along a path of 400 points across a leaf is 0.61 of that without
  fading with the squared error alone as the unit, 0.32 with the
  neighbour's noise added, and 0.26 with the predicted variance.
* **On `resonance_xsec`** the predicted variance gives 5.5 steps above 0.1
  sigma on a path with a sum of 12.7 sigma, as the squared error does.

The 150 runs of section 8.4 with the package as it is: the NRMSE of the last
5000 predictions divided by that of the default tree, for the two seeds.

| target | d | uniform | walker | sweeping | focusing |
|---|---|---|---|---|---|
| `active_peaks` | 10 | 0.76 / 0.78 | 0.72 / 0.75 | 0.66 / 0.86 | 0.63 / 0.65 |
| `branching_ratio` | 5 | 1.14 / 0.89 | 0.95 / 0.97 | 0.99 / 0.96 |  |
| `chirp` | 4 | 0.92 / 0.91 | 0.73 / 0.68 | 0.69 / 0.70 | 0.37 / 0.37 |
| `diagonal_ridge` | 4 | 0.70 / 0.67 | 0.76 / 0.68 | 0.83 / 0.78 | 0.75 / 0.73 |
| `dilepton_spectrum` | 4 | 0.89 / 0.85 | 0.95 / 0.90 | 0.73 / 0.81 | 0.94 / 0.88 |
| `eft_signal_strength` | 6 | 0.42 / 0.44 | 0.43 / 0.40 | 0.64 / 0.49 | 1.04 / 1.05 |
| `eggholder` | 3 | 0.88 / 0.89 | 0.82 / 0.86 | 0.91 / 0.90 | 0.84 / 0.85 |
| `gaussian_peaks` | 6 | 0.65 / 0.69 | 0.42 / 0.49 | 0.54 / 0.48 | 0.50 / 0.58 |
| `global_fit_loglike` | 10 | 0.99 / 0.96 | 0.92 / 0.96 | 0.95 / 0.97 | 0.20 / 1.00 |
| `jet_spectrum` | 4 | 0.73 / 0.83 | 1.00 / 0.77 | 0.63 / 0.75 | 0.44 / 0.44 |
| `loop_decay_width` | 4 | 0.87 / 0.79 | 0.86 / 0.93 | 0.39 / 0.18 | 0.29 / 0.28 |
| `michalewicz` | 5 | 1.01 / 1.02 | 0.81 / 0.96 | 1.01 / 1.00 | 0.91 / 0.91 |
| `neutralino_mass` | 4 | 0.84 / 0.79 | 0.99 / 0.26 | 0.83 / 0.99 | 0.52 / 0.44 |
| `oscillation_probability` | 5 | 0.63 / 0.55 | 0.63 / 1.02 | 0.93 / 0.94 | 0.87 / 0.93 |
| `relic_density` | 4 | 0.89 / 0.86 | 0.92 / 0.96 | 0.87 / 0.74 | 0.93 / 0.96 |
| `resonance_xsec` | 4 | 0.93 / 0.98 | 0.62 / 1.00 | 0.97 / 0.89 | 0.72 / 0.73 |
| `rotated_rosenbrock` | 6 | 0.54 / 0.52 | 0.56 / 0.53 | 0.49 / 0.68 | 0.50 / 0.51 |
| `signal_yield_mc` | 4 | 0.84 / 0.84 | 0.87 / 0.87 | 0.51 / 0.54 | 0.94 / 0.93 |
| `step_ridge` | 6 | 0.92 / 0.87 | 0.95 / 0.93 | 0.87 / 0.91 | 0.91 / 0.91 |
| geometric mean | | 0.79 | 0.75 | 0.73 | 0.65 |
| without fading (section 8.4) | | 0.78 | 0.74 | 0.71 | 0.61 |

* **0.73 of the default tree's error** as the geometric mean over the 150
  runs, from 0.18 to 1.14, against 0.71 without fading. 135 runs are more
  than 2% better than the default tree and 4 more than 2% worse, the four of
  section 8.4: `branching_ratio` on the uniform stream (14%),
  `eft_signal_strength` on the focusing stream (4 and 5%) and `Michalewicz`
  on the uniform stream (2%).
* **Run by run the fading costs 0.8% of error in the median** and 3.3% in
  the geometric mean (0.75 to 4.81 of the error without fading): 0.9% on the
  uniform stream, 1.9% on the walker, 3.3% on the sweeping and 7.4% on the
  focusing stream. Without `global_fit_loglike` on the focusing stream,
  where one prediction of the default tree is off by twice the range and the
  ratio is 0.20 with fading and 0.04 without, it is 2.2%.
* **The rest of the error distribution**: the mean absolute error is 0.77 of
  the default tree's, the median 0.76, the 99% quantile 0.74 and the largest
  error of a run 0.69 (without fading 0.74, 0.74, 0.72 and 0.67); 0.77 on
  points held out of the stream and 0.84 on uniform test points.
* **The coverage is 0.712** against 0.709 for the default tree, and 1.76% of
  the predictions are off by more than 5 sigma, against 1.95% for the
  default tree and 2.02% without fading.
* **Cost** (20 000 points, one process): on `GaussianPeaks` a prediction
  takes 0.57 ms with fading, 0.54 ms without and 0.14 ms for the default
  tree, and an update 2.53, 2.46 and 1.82 ms; on `resonance_xsec` 0.60, 0.46
  and 0.14 ms, and 3.61, 3.58 and 2.95 ms.
