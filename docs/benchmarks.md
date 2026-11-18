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
that are not in the repository. So was section 9, with the code as it ships;
its curve families are in `examples/target_functions_curves.py`, and
`examples/multioutput_physics_curves.py` runs the comparison of section 9.1
on one of them.

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


## 8. Leaf fits by other processes (`defer_fits`)

### 8.1 An MPI trainer with fit workers

`examples/mpi_fit_workers.py`: `GaussianPeaks` in six dimensions, 12 000
points, `Nbar=100`, a fit every 25 points, one seed. One MPI process holds the
tree (the trainer) and hands the leaf fits to the fit workers, on a machine
with four cores. The trainer takes the points as fast as it can, so a result
arrives after as many points as it ever would. Its time per point is that of
`update_tree`, of sending the tasks and of applying the results. NRMSE and
coverage are those of the last 3000 predictions, each made before the point
was added.

| stream | fit workers | trainer ms per point | points per second | fits | points until the result, mean | NRMSE | coverage |
|---|---|---|---|---|---|---|---|
| focusing | none, fits in place | 2.10 | 477 | | | 0.0033 | 0.69 |
| focusing | 1 | 0.76 | 1322 | 208 | 1101 | 0.0032 | 0.70 |
| focusing | 2 | 0.77 | 1293 | 390 | 94 | 0.0031 | 0.69 |
| focusing | 3 | 0.78 | 1282 | 397 | 53 | 0.0033 | 0.70 |
| sweeping | none, fits in place | 2.35 | 425 | | | 0.0119 | 0.70 |
| sweeping | 1 | 0.85 | 1171 | 242 | 541 | 0.0133 | 0.70 |
| sweeping | 2 | 0.82 | 1222 | 392 | 82 | 0.0147 | 0.70 |
| sweeping | 3 | 0.74 | 1349 | 357 | 51 | 0.0127 | 0.70 |

* **The trainer takes 2.7 to 3.2 times as many points per second** with fit
  workers, and the number of workers does not change this: the trainer does
  not wait for a fit. Of its 0.80 ms per point, 0.27 to 0.30 ms is the
  prediction of the point for the calibration, 0.18 ms the appended point,
  0.11 ms the results applied and 0.04 ms the tasks made.
* **The error is that of fits in place on the focusing stream and 7 to 24%
  higher on the sweeping stream.** The runs differ in the order in which the
  results arrive, so two runs do not give the same numbers. With every result
  applied 60 and 500 points late in a single process, the NRMSE is 0.0028 and
  0.0028 on the focusing stream and 0.0139 and 0.0151 on the sweeping stream.
* **With one worker the results arrive 500 to 1100 points late** and the
  leaves get about half as many fits. Nothing queues up: a leaf waits for its
  result before it is handed another task.
* **The fits take 2.3 to 2.5 ms of a worker's time per point of the stream**
  (65 to 77 ms each, in a single-process run with the same settings), so a
  trainer at 1000 points per second keeps two to three workers busy.
* **A task for a leaf of 100 points is a pickle of 7.3 kB, its result one of
  236 bytes.**
* **A result applied at once gives the leaf of a fit in place**: until the
  first split the predictions are equal to the last bit
  (`tests/test_deferred_fits.py`).

### 8.2 A batch of points

The same target and settings, one process: a tree of the first points of the
focusing stream takes the next points as one batch. The NRMSE is that of the
1000 points that follow the batch, predicted after it.

| tree | batch | how it is added | fits | seconds | NRMSE |
|---|---|---|---|---|---|
| 6000 points | 300 | not added | | | 0.0097 |
| 6000 points | 300 | `update_tree` per point | 4 | 0.35 | 0.0088 |
| 6000 points | 300 | `allow_training=False`, then a fit of every leaf that took a point | 53 | 2.56 | 0.0091 |
| 6000 points | 300 | `defer_fits=True`, the tasks solved and applied after the batch | 4 | 0.34 | 0.0088 |
| 6000 points | 2000 | not added | | | 0.0114 |
| 6000 points | 2000 | `update_tree` per point | 69 | 4.97 | 0.0053 |
| 6000 points | 2000 | `allow_training=False`, then a fit of every leaf that took a point | 88 | 5.05 | 0.0058 |
| 6000 points | 2000 | `defer_fits=True`, the tasks solved and applied after the batch | 30 | 2.84 | 0.0054 |
| 1000 points | 5000 | `update_tree` per point | 175 | 10.06 | 0.0106 |
| 1000 points | 5000 | `allow_training=False`, then a fit of every leaf that took a point | 88 | 7.37 | 0.0168 |
| 1000 points | 5000 | `defer_fits=True`, the tasks solved and applied after the batch | 15 | 4.97 | 0.0189 |
| 1000 points | 5000 | `defer_fits=True`, the tasks solved and applied every 250 points | 174 | 11.50 | 0.0110 |
| empty | 5000 | `update_tree` per point | 172 | 9.77 | 0.0144 |
| empty | 5000 | `fit` | 70 | 3.69 | 0.0148 |
| empty | 5000 | `defer_fits=True`, the tasks solved and applied after the batch | 1 | 1.35 | 0.0589 |

* **Point by point is the most accurate way into a tree that exists**, in
  every row, and what `batch_update_tree` does. On a uniform stream the tree of 1000 points gives 0.0128 point by
  point, 0.0161 with the leaves fitted after the batch and 0.0161 with the
  results applied after it.
* **Fitting the leaves once after the batch does not save fits on a large
  tree.** Without training a point is stored but not appended, so every leaf
  that took a point needs a fit, and the 300 points went to 53 leaves. Point
  by point a leaf is fitted with every 25th point and the others are appended.
* **Results applied after the batch are for a batch that is small compared
  with the tree.** A leaf and the leaves it splits into ask for one fit until
  its result is applied, so the batch costs at most one fit per leaf it
  started in. At a third of the tree's points that is half the fits at the
  same error. At five times its points every new leaf has the hyperparameters
  of a leaf of the old tree, and the error is 78% higher; on an empty tree
  there is one fit and four times the error.
* **`fit` is the way into an empty tree**: 2.6 times as fast as point by point
  at about the same error. Its sigma is not calibrated: it covers 0.91 of the
  errors (0.83 on the uniform stream) where point by point gives 0.71.

## 9. Curves as outputs (multi-output targets)

The twelve families of `examples/target_functions_curves.py`: a curve
`f(x; theta)` on a grid of 48 to 128 points in `x` is the multi-output target,
and three or four parameters `theta`, uniform in the unit cube, are the inputs.
The observations are exact, with a declared sigma of 0.1% of the quantity for
the curves given as `log10` and of 1e-4 to 1e-3 in the unit of the curve
otherwise. `peak_counts_mc` has Monte Carlo noise, declared bin by bin. The
NRMSE is the RMSE over all grid points divided by the range of the family, and
errors are against the noise-free curves. The tree is that of the conventions
above, with the scikit-learn leaves. `kinematic_edge` is also run on a grid of
128 points in place of 64 (section 9.4).

### 9.1 The output models on a stream

800 curves, each predicted before the tree is updated with it. NRMSE of the
last 400 predictions; for `'shared'` and `'aligned'` the geometric mean of
three seeds, with the ratio of the two and its range over the seeds, and one
seed for `'independent'` and `'pca'`:

| family | points in x | independent | pca | shared | aligned | shared / aligned |
|---|---:|---:|---:|---:|---:|---:|
| `matter_power` | 64 | 3.2e-5 | 6.8e-5 | 5.0e-5 | 5.0e-5 | 1 |
| `eft_distribution` | 48 | 1.4e-4 | 1.5e-4 | 1.5e-4 | 1.5e-4 | 1 |
| `jet_spectrum` | 64 | 4.1e-5 | 4.6e-5 | 4.9e-5 | 4.9e-5 | 1 |
| `dust_sed` | 64 | 1.1e-5 | 7.7e-6 | 1.1e-5 | 1.1e-5 | 1 |
| `z_lineshape` | 64 | 2.4e-4 | 2.2e-4 | 2.3e-4 | 6.3e-5 | 3.6 (3.2 to 4.1) |
| `zprime_spectrum` | 128 | 8.5e-3 | 7.9e-3 | 8.0e-3 | 2.0e-3 | 3.9 (3.1 to 4.8) |
| `kinematic_edge` | 64 | 1.3e-3 | 2.5e-3 | 1.2e-3 | 7.7e-4 | 1.6 (1.5 to 1.7) |
| `kinematic_edge`, 128 points | 128 | 1.4e-3 | 2.3e-3 | 1.2e-3 | 1.0e-4 | 12 (9.9 to 13) |
| `oscillation_probability` | 96 | 1.1e-3 | 9.2e-4 | 1.1e-3 | 1.1e-3 | 1 |
| `cmb_peaks` | 128 | 1.6e-4 | 1.9e-4 | 2.1e-4 | 1.1e-4 | 1.8 (1.6 to 2.1) |
| `form_factor` | 96 | 3.2e-3 | 6.3e-3 | 4.3e-3 | 4.3e-3 | 1 |
| `ringdown` | 128 | 2.7e-3 | 2.7e-3 | 2.4e-3 | 8.2e-4 | 3 (2.8 to 3.2) |
| `peak_counts_mc` | 60 | 3.2e-3 | 4.1e-3 | 3.9e-3 | 3.9e-3 | 1 |

Coverage of the same predictions, time of an update, and the share of the
leaves that hold an alignment at the end of the stream:

| family | coverage: independent | pca | shared | aligned | ms per update: independent | pca | shared | aligned | leaves aligned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `matter_power` | 0.67 | 0.67 | 0.66 | 0.66 | 240 | 20 | 7.8 | 11 | 0.00 |
| `eft_distribution` | 0.67 | 0.70 | 0.67 | 0.67 | 131 | 12 | 5.6 | 8.1 | 0.00 |
| `jet_spectrum` | 0.68 | 0.67 | 0.66 | 0.66 | 205 | 18 | 6.8 | 10 | 0.00 |
| `dust_sed` | 0.65 | 0.63 | 0.65 | 0.65 | 212 | 19 | 9 | 13 | 0.00 |
| `z_lineshape` | 0.65 | 0.66 | 0.66 | 0.66 | 174 | 34 | 7.2 | 15 | 1.00 |
| `zprime_spectrum` | 0.66 | 0.66 | 0.66 | 0.67 | 245 | 225 | 9.4 | 24 | 1.00 |
| `kinematic_edge` | 0.68 | 0.71 | 0.67 | 0.66 | 149 | 69 | 6 | 13 | 0.91 |
| `kinematic_edge`, 128 points | 0.68 | 0.66 | 0.67 | 0.67 | 295 | 96 | 9.4 | 23 | 1.00 |
| `oscillation_probability` | 0.69 | 0.70 | 0.66 | 0.66 | 203 | 25 | 8.2 | 15 | 0.00 |
| `cmb_peaks` | 0.68 | 0.69 | 0.67 | 0.66 | 313 | 53 | 10 | 27 | 1.00 |
| `form_factor` | 0.68 | 0.67 | 0.67 | 0.67 | 194 | 142 | 9.2 | 14 | 0.00 |
| `ringdown` | 0.69 | 0.69 | 0.67 | 0.67 | 221 | 38 | 9.4 | 20 | 1.00 |
| `peak_counts_mc` | 0.99 | 0.98 | 0.99 | 0.99 | 110 | 32 | 5 | 8.7 | 0.00 |

* **`'independent'`, `'pca'` and `'shared'` have the same error within a
  factor 2.1 on every family**, and `'shared'` within 1.6 of
  `'independent'`, at 3 to 5% of its time.
* **`'aligned'` lowers the error on the families with a feature that moves**:
  by a factor 3.6 for `z_lineshape`, 3.9 for `zprime_spectrum`, 3.0 for
  `ringdown`, 1.8 for `cmb_peaks`, and 12 for `kinematic_edge` on 128 grid
  points (1.6 on 64). On the other families its leaves decline the alignment,
  and the predictions are those of `'shared'`. On the test curves after the
  stream the factors are 4.2, 4.4, 3.1, 1.8, 10 and 1.2.
* **The coverage is 0.63 to 0.71 for the exact curves**, for every output
  model. For the noisy curves it is 0.98 to 0.99 (section 9.5).
* **An update with `'aligned'` takes 2.1 to 2.7 times as long as with
  `'shared'`** where the leaves hold alignments, and 1.4 to 1.8 times where
  they decline: a leaf that declined waits for 1, then 3, then 7 fits before
  it tries again. A prediction takes 0.3 to 0.5 ms with `'shared'` and 2 to 4
  ms from a leaf with an alignment, with four such runs on four cores.

### 9.2 Batch fits

`fit` on 200 and 800 curves, NRMSE on 400 test curves.

200 curves:

| family | independent | pca | shared | aligned | components of pca | fit time [s]: independent | pca | shared | aligned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `matter_power` | 5.4e-5 | 5.8e-5 | 7.8e-5 | 7.8e-5 | 6 | 25 | 1.6 | 0.31 | 3.1 |
| `eft_distribution` | 2.1e-4 | 2.0e-4 | 2.1e-4 | 2.1e-4 | 4 | 13 | 0.73 | 0.22 | 1.5 |
| `jet_spectrum` | 1.1e-4 | 1.1e-4 | 1.2e-4 | 1.2e-4 | 6 | 26 | 1.3 | 0.36 | 1.3 |
| `dust_sed` | 1.4e-5 | 9.5e-6 | 1.7e-5 | 1.7e-5 | 5 | 24 | 1.1 | 0.26 | 1.4 |
| `z_lineshape` | 3.8e-4 | 3.3e-4 | 3.7e-4 | 1.3e-4 | 15 | 17 | 2.2 | 0.21 | 1.4 |
| `zprime_spectrum` | 1.4e-2 | 1.4e-2 | 1.4e-2 | 4.3e-3 | 120 | 19 | 14 | 0.15 | 2.7 |
| `kinematic_edge` | 2.3e-3 | 2.4e-3 | 2.2e-3 | 6.8e-4 | 35 | 15 | 4.5 | 0.17 | 1.6 |
| `kinematic_edge`, 128 points | 2.4e-3 | 2.3e-3 | 2.3e-3 | 2.3e-4 | 48 | 30 | 6.2 | 0.17 | 2.9 |
| `oscillation_probability` | 3.1e-3 | 3.2e-3 | 3.5e-3 | 3.5e-3 | 10 | 21 | 1.3 | 0.17 | 4.1 |
| `cmb_peaks` | 4.1e-4 | 4.3e-4 | 4.4e-4 | 2.7e-4 | 24 | 22 | 4.3 | 0.23 | 2.8 |
| `form_factor` | 6.6e-3 | 1.1e-2 | 8.4e-3 | 8.4e-3 | 74 | 20 | 7.5 | 0.14 | 2 |
| `ringdown` | 4.8e-3 | 5.0e-3 | 5.3e-3 | 1.9e-3 | 22 | 17 | 3.4 | 0.16 | 3.7 |
| `peak_counts_mc` | 4.1e-3 | 4.4e-3 | 5.3e-3 | 5.3e-3 | 15 | 13 | 1.9 | 0.14 | 1.9 |

800 curves:

| family | independent | pca | shared | aligned | components of pca | fit time [s]: independent | pca | shared | aligned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `matter_power` | 5.5e-5 | 4.2e-5 | 6.3e-5 | 6.3e-5 | 6 | 89 | 6.8 | 1.1 | 8.2 |
| `eft_distribution` | 9.1e-5 | 9.1e-5 | 9.3e-5 | 9.3e-5 | 4 | 57 | 3.6 | 1.4 | 6.7 |
| `jet_spectrum` | 5.4e-5 | 6.7e-5 | 8.5e-5 | 8.5e-5 | 6 | 96 | 8.1 | 1.8 | 6.7 |
| `dust_sed` | 9.4e-6 | 6.5e-6 | 1.1e-5 | 1.1e-5 | 5 | 94 | 6 | 1.4 | 6.5 |
| `z_lineshape` | 1.8e-4 | 1.7e-4 | 1.8e-4 | 5.2e-5 | 15 | 83 | 14 | 1.2 | 6.6 |
| `zprime_spectrum` | 9.4e-3 | 8.7e-3 | 9.2e-3 | 3.1e-3 | 125 | 97 | 92 | 0.78 | 14 |
| `kinematic_edge` | 9.1e-4 | 1.7e-3 | 9.0e-4 | 6.7e-4 | 35 | 57 | 31 | 0.8 | 7.3 |
| `kinematic_edge`, 128 points | 9.2e-4 | 1.7e-3 | 9.0e-4 | 9.0e-5 | 56 | 111 | 44 | 0.81 | 12 |
| `oscillation_probability` | 1.4e-3 | 1.4e-3 | 1.6e-3 | 1.4e-3 | 11 | 57 | 8.1 | 0.74 | 13 |
| `cmb_peaks` | 2.3e-4 | 2.5e-4 | 2.6e-4 | 1.4e-4 | 24 | 105 | 16 | 0.94 | 10 |
| `form_factor` | 4.4e-3 | 8.0e-3 | 4.8e-3 | 4.8e-3 | 77 | 93 | 69 | 0.64 | 8.6 |
| `ringdown` | 1.8e-3 | 1.9e-3 | 1.9e-3 | 7.9e-4 | 22 | 82 | 15 | 0.82 | 13 |
| `peak_counts_mc` | 3.5e-3 | 3.4e-3 | 4.6e-3 | 4.6e-3 | 20 | 45 | 17 | 0.58 | 8.1 |

`fit` builds the tree before a leaf is fitted, so its splits follow the spread
of the points and not the fitted length scales (`min_lengthscale`), and the
sigma is not calibrated. The sigma of a shared-kernel leaf is set per output
from the leave-one-out errors of the leaf's curves, and covers 0.59 to 0.78 of
the test errors with 200 curves and 0.49 to 0.77 with 800. That of
`'independent'` covers 0.70 to 1.00 and that of `'pca'` 0.77 to 1.00. Before
the sigma per output, that of `'shared'` covered 0.81 to 1.00.

### 9.3 More curves

`fit`, NRMSE on the test curves:

| family | shared: 200 | 800 | 3200 | aligned: 200 | 800 | 3200 |
|---|---:|---:|---:|---:|---:|---:|
| `z_lineshape` | 3.7e-4 | 1.8e-4 | 5.3e-5 | 1.3e-4 | 5.2e-5 | 1.8e-5 |
| `zprime_spectrum` | 1.4e-2 | 9.2e-3 | 5.5e-3 | 4.3e-3 | 3.1e-3 | 1.3e-3 |
| `kinematic_edge`, 128 points | 2.3e-3 | 9.0e-4 | 4.3e-4 | 2.3e-4 | 9.0e-5 | 4.2e-5 |
| `cmb_peaks` | 4.4e-4 | 2.6e-4 | 1.3e-4 | 2.7e-4 | 1.4e-4 | 5.8e-5 |
| `ringdown` | 5.3e-3 | 1.9e-3 | 6.2e-4 | 1.9e-3 | 7.9e-4 | 2.2e-4 |
| `matter_power` | 7.8e-5 | 6.3e-5 | 2.4e-5 | 7.8e-5 | 6.3e-5 | 2.4e-5 |

With 200 curves `'aligned'` has the error that `'shared'` has with 800 on
`cmb_peaks` and `ringdown`, and a lower one on the three other families with
a feature that moves. On `matter_power` every leaf declines.

### 9.4 The grid has to resolve the feature

`kinematic_edge` has an edge that is smeared by a mass resolution of 1 to 4
GeV, on a grid from 20 to 140 GeV. The same family on four grids, `fit`, NRMSE
on the test curves:

| grid points | spacing [GeV] | 200 curves: shared | aligned | 800 curves: shared | aligned | leaves with an alignment, 800 curves |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 3.9 | 2.2e-3 | 2.2e-3 | 9.3e-4 | 9.3e-4 | 0.08 |
| 64 | 1.9 | 2.2e-3 | 6.8e-4 | 9.0e-4 | 6.7e-4 | 0.92 |
| 128 | 0.94 | 2.3e-3 | 2.3e-4 | 9.0e-4 | 9.0e-5 | 1.00 |
| 256 | 0.47 | 2.3e-3 | 2.3e-4 | 9.0e-4 | 9.1e-5 | 1.00 |

An alignment interpolates the curves between the grid points. With grid
points 0.9 GeV apart or closer the gain is a factor 10, at 1.9 GeV it is 1.4
to 3.3, and at 3.9 GeV nearly every leaf declines the alignment. A finer grid
costs `'shared'` nothing and `'aligned'` time in proportion. For
`zprime_spectrum` the gain was 2.8 to 3.4 on grids of 64 to 512 points.

### 9.5 Other measurements

* **The fit of a shared-kernel GP** (100 points, three inputs, one
  optimiser run), by scikit-learn and by the adapter from `Y Y^T`:

  | outputs | scikit-learn [ms] | adapter [ms] |
  |---:|---:|---:|
  | 8 | 113 | 93 |
  | 128 | 776 | 90 |
  | 1000 | 8087 | 152 |

  The log-likelihoods at the optimum agree to a relative 1e-7 and the
  predictions to 4e-5 of the scale of the targets.
* **The logarithm of a quantity over many decades.** `dust_sed` covers ten
  decades. With 800 curves given as `10^y`, with a declared sigma of 0.1% of
  the value, in place of `y`, relative errors of the predicted spectrum:

  | | independent | pca | shared |
  |---|---:|---:|---:|
  | median, log10 as the target | 6.0e-5 | 2.4e-5 | 1.9e-5 |
  | median, the value as the target | 1.7e-3 | 3.8e-2 | 2.9e-3 |
  | 99th percentile, log10 as the target | 7.8e-4 | 6.0e-4 | 9.0e-4 |
  | 99th percentile, the value as the target | 0.12 | 3.3 | 0.14 |
  | predictions below zero, the value as the target | 0.03% | 2.5% | 0.05% |

  (Measured before the changes of this section, which do not change the
  predicted means.)
* **`x` as one more input of a single-output tree**, with the 200 curves as
  `200 p` points: 2.3 to 83 times the NRMSE of `'shared'` (median 12), with
  135 to 381 leaves, and a `fit` of 12 to 65 seconds.
* **The number of components of `'pca'`**, 800 curves, `fit`:

  | family | 3 | 6 | 12 | 24 | 48 | noise rule | its components | fit time [s]: 24 | noise rule |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | `zprime_spectrum` | 3.8e-2 | 2.5e-2 | 1.3e-2 | 8.9e-3 | 8.7e-3 | 8.7e-3 | 125 | 17 | 89 |
  | `kinematic_edge` | 4.4e-2 | 2.2e-2 | 6.3e-3 | 1.8e-3 | 1.7e-3 | 1.7e-3 | 35 | 17 | 27 |
  | `form_factor` | 5.4e-2 | 3.2e-2 | 1.9e-2 | 1.1e-2 | 8.1e-3 | 8.0e-3 | 77 | 19 | 71 |
  | `cmb_peaks` | 1.0e-2 | 2.6e-3 | 4.4e-4 | 2.5e-4 | 2.5e-4 | 2.5e-4 | 24 | 14 | 12 |

  The declared sigma of these curves is 1e-4 of their scale or less, so the
  noise rule keeps every component above it. Beyond the error of the leaf GPs
  more components do not lower the error.
* **`'pca'` and the starts of the optimiser.** On `kinematic_edge` and
  `form_factor` the error of `'pca'` is twice that of `'shared'` whatever the
  number of components (800 curves, `fit`: 1.7e-3 and 8.0e-3 against 9.0e-4
  and 4.8e-3). With three restarts of the optimiser of each score GP
  (`Default_GPR(n_restarts_optimizer=3)`) it is 9.7e-4 and 4.5e-3, at three
  times the time: a score of a curve with a feature that moves oscillates
  with the parameters, and a single start ends in a local optimum of its
  likelihood.
* **Noise and the calibrated sigma.** On the stream of `peak_counts_mc` the
  calibrated sigma covers the noise-free curve in 0.98 to 0.99 of the cases, for
  every output model. For a single output, `sin(5 x1) cos(3 x2)` with a
  declared noise of 0 (1e-4), 0.05 and 0.2, 1500 points: the sigma of the
  last 750 predictions averages 0.0001, 0.052 and 0.207, the RMSE against
  the noise-free target is 0.0008, 0.016 and 0.048, and the coverage of the
  target is 0.65, 0.99 and 1.00. The coverage of the observations is 0.65,
  0.66 and 0.66.
* **A shared-kernel GP with the outputs in units of their noise**
  (`peak_counts_mc`, whose noise differs by a factor 3 between the bins,
  `fit`, two seeds): NRMSE 6.2e-3 against 5.3e-3 and 5.9e-3 with 200 curves,
  4.5e-3 and 4.6e-3 against 4.6e-3 and 4.7e-3 with 800, 3.7e-3 and 3.8e-3
  against 3.8e-3 with 3200.

### 9.6 Other warps (not in the code)

* **Single leaves.** 100 curves in a cube of side 0.8, 0.5 or 0.3 of the
  parameter range, 200 test curves in the same cube, three positions of the
  cube, one shared-kernel GP on the curves or on the aligned curves. NRMSE,
  geometric mean over the positions:

  | family | side | curves | warp that maps the grid onto itself | warp with free ends |
  |---|---:|---:|---:|---:|
  | `z_lineshape` | 0.8 | 5.2e-4 | 1.9e-4 | 5.8e-4 |
  | | 0.5 | 1.8e-4 | 6.0e-5 | 1.9e-4 |
  | | 0.3 | 5.9e-5 | 2.0e-5 | 6.7e-5 |
  | `zprime_spectrum` | 0.8 | 1.6e-2 | 7.7e-3 | 9.6e-3 |
  | | 0.5 | 8.8e-3 | 1.7e-3 | 3.1e-3 |
  | | 0.3 | 4.1e-3 | 4.2e-4 | 8.1e-4 |
  | `kinematic_edge` | 0.8 | 3.4e-3 | 8.7e-4 | 6.8e-4 |
  | | 0.5 | 1.0e-3 | 2.6e-4 | 2.8e-4 |
  | | 0.3 | 2.9e-4 | 1.0e-4 | 1.3e-4 |
  | `cmb_peaks` | 0.8 | 6.8e-4 | 3.1e-4 | 5.0e-4 |
  | | 0.5 | 2.4e-4 | 1.0e-4 | 1.9e-4 |
  | | 0.3 | 8.0e-5 | 3.3e-5 | 6.6e-5 |

  The warp with free ends is a shift and a stretch, and a curve is continued
  beyond the grid by its value at the edge. These runs fitted the warp from
  the curves themselves, without the start from blurred curves.
* **Knots, squares of the offsets and blurred starts**, 800 curves, `fit`.
  Error of `'shared'` over that of `'aligned'`:

  | family | 8 knots (as in the code then) | 16 knots | 8 knots, squares | 16 knots, squares | every fit from blurred curves |
  |---|---:|---:|---:|---:|---:|
  | `z_lineshape` | 3.6 | 3.1 | 2.1 | 2.3 | 3.4 |
  | `zprime_spectrum` | 2.0 | 1.4 | 2.1 | 2.1 | 3.2 |
  | `kinematic_edge`, 128 points | 9.3 | 6.7 | 8.1 | 2.0 | 9.3 |
  | `cmb_peaks` | 1.9 | 1.7 | 1.4 | 1.3 | 1.9 |
  | `ringdown` | 2.6 | 1.6 | 1.3 | 1.0 | 1.3 |

  The fit took 2 to 7 times as long with 16 knots or with the squares. The
  code runs a first fit both from the curves and from blurred curves and
  keeps the better result.
* **The aligned curve predicted and warped back**, in place of each curve
  warped into the frame of the query: on single leaves with an early form of
  the warp the NRMSE was 7.1e-5 against 4.6e-5 (`kinematic_edge`, side 0.3)
  and 9.1e-4 against 6.4e-4 (`zprime_spectrum`, side 0.5), and equal on
  `z_lineshape` and `cmb_peaks`.
