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
that are not in the repository, and section 8 with
`examples/mpi_fit_workers.py`. The scripts of section 9
(`research/partitioning/`) were removed from the repository as well; they are
in its history up to commit `25a4673`. Section 10 was measured with scripts
that are not in the repository; its curve families are in
`examples/target_functions_curves.py`, and
`examples/multioutput_physics_curves.py` runs the comparison of section 10.1
on one of them. Sections 11 and 12 were measured with scripts that are not in
the repository; `examples/trainer.py` runs the comparison of section 12.2 on
one target.

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

## 9. Other ways to partition the data

Sections 9.2 to 9.13 were measured with neighbours that all count alike,
with the kernel of the query's leaf alone and with every leaf awake
(`NEIGHBOUR_FADE`, `NEIGHBOUR_BLEND` and `NEIGHBOUR_AWAKE` at None).
Section 9.15 has the neighbours that fade out, section 9.17 the kernels
mixed across the cuts and section 9.19 the leaves asleep, which are the
defaults of `predict_from='neighbours'`. Sections 9.2 to 9.20 were measured
with a switch to the leaf's own GP at one limit (`NEIGHBOUR_TRANSITION` and
`NEIGHBOUR_CONSISTENCY_OUTSIDE` at None); section 9.21 has the gradual
change, which is the default as well. These settings are constants of
`pygptreeo/neighbours.py`; they were options of the tree when the sections
were measured, and `neighbour_tree` in `research/partitioning/variants.py`
sets them for one tree.

Sections 9.1 to 9.3: eleven targets (`GaussianPeaks` and `RotatedRosenbrock`
in six dimensions, `Eggholder` in three, `Chirp` in four, a four-dimensional
target with a ridge and a kink on oblique hyperplanes, `resonance_xsec`,
`loop_decay_width`, `relic_density`, `eft_signal_strength`, `neutralino_mass`
and `oscillation_probability`) on the uniform, walker and sweeping streams,
5000 points, one seed: 33 runs. The tables give the NRMSE of the last 1250
predictions divided by that of the default tree in the same run, as geometric
means over the runs, and the mean coverage.

### 9.1 Rules for the cuts (not in the code)

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
| predictions from neighbours (section 9.4), default cuts | 0.72 | 0.66 | 0.69 | 0.69 (0.38 to 0.98) | 0 of 33 |

### 9.2 Predictions from a query's neighbours (`predict_from='neighbours'`)

The kernel and the calibration are those of the query's leaf, and the
prediction is conditioned on the query's nearest stored points. In this
table the search for them is exact and no leaf prefers its own GP, except
in the fourth row (the two defaults of sections 9.4 and 9.5 came later); the
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

### 9.3 The test of the neighbours against the kernel

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

### 9.4 Predictions from neighbours on 20 000-point streams

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

### 9.5 The leaves that are searched

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
  the end of the 20 000-point runs of section 9.4, 14 to 16 leaves per query
  in three dimensions, 8 to 31 in four, 7 to 46 in five, 35 to 79 in six and
  41 to 77 in ten, of about 285.

### 9.6 100 000 points

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
  first division of the index's groups (section 9.9), with which a
  prediction from neighbours went from 0.65 to 0.81 ms; section 9.12 has
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

### 9.7 Cost with one process

Streams of 20 000 points, one run at a time on an otherwise idle machine,
with the first division of the index's groups (section 9.9). The update time
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

### 9.8 A stream that drifts

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

### 9.9 The index of the leaves up to a million points

Without GPs (`research/partitioning/index_scaling.py`): leaves of 100 points,
cut at the median along a random input, on a uniform stream and on a random
walk in the unit cube. The groups of the index that the search for the 16
nearest leaves opens, as the mean over 500 queries near the latest points:
with groups divided at the median of their children's centres (the index of
sections 9.4 to 9.8), then with groups divided where their halves overlap
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

### 9.10 Neighbours with the feature network

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

### 9.11 Noisy observations

The eleven targets of sections 9.1 to 9.3 with Gaussian noise of 0.2
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

### 9.12 Cost along a 200 000-point stream

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

### 9.13 Two clusters per split, and the steps of the predictions (not in the code)

**A split into two clusters.** The points of a leaf are divided into two
clusters (2-means), and a point goes to the child whose centroid is nearer:
a cut by the hyperplane half-way between the centroids. The 33 runs of
5000 points of sections 9.1 to 9.3, one GP per leaf, relative to the default
tree.

| two clusters of the leaf's points | uniform | walker | sweeping | all runs (range) | runs more than 2% worse |
|---|---|---|---|---|---|
| nearest centroid | 1.48 | 1.31 | 1.03 | 1.26 (0.22 to 4.48) | 24 of 33 |
| nearest centroid, standardised inputs | 1.44 | 1.35 | 1.13 | 1.30 (0.35 to 4.83) | 26 of 33 |
| cut at the median along the line between the centroids (section 9.1) | 1.54 | 1.33 | 1.08 | 1.30 (0.30 to 4.88) | 23 of 33 |
| nearest centroid, distances in the GP's fitted length scales | 0.99 | 1.17 | 0.97 | 1.04 (0.65 to 2.43) | 15 of 33 |
| the same, cut at the median | 0.99 | 0.99 | 1.00 | 0.99 (0.61 to 1.71) | 14 of 33 |
| nearest centroid, fitted length scales, decorrelated inputs for the GPs | 1.08 | 1.31 | 1.14 | 1.17 (0.59 to 3.24) | 25 of 33 |

* **Clusters of the inputs alone are 26 to 30% worse than the default
  cuts**, as the other rules of section 9.1 that do not look at the target.
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

### 9.14 Neighbours that fade out: the prototype

The prediction from a query's neighbours with a noise that grows with the
distance of a neighbour (`research/partitioning/tapered.py`; section 9.15
has the version of the package). With `r` the
distance of a neighbour divided by that of the first stored point beyond the
100 neighbours, and `t = (r - start) / (1 - start)` where `r` exceeds
`start`, the noise variance of the neighbour is raised by
`e^2 t^2 / (1 - t)^2`, where `e^2` is the typical squared error of the
leaf's recent predictions (the median of their squares divided by 0.455). A
point enters and leaves the set of neighbours with no weight, so the
prediction is continuous in the query for as long as its leaf, and the leaves
that are searched, stay the same.

The 33 runs of sections 9.1 to 9.3, relative to the default tree:

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

The steps of the predicted mean, measured as in section 9.13 after 20 000
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

### 9.15 Neighbours that fade out (`NEIGHBOUR_FADE`)

In the package the noise variance of a neighbour grows by `t^2 / (1 - t)^2`
times a unit, with `t` as in section 9.14 and `start = 0.9`. Candidates for
the unit, on the 33 runs of 5000 points relative to the default tree, with
the steps of the mean after 20 000 points as in section 9.13 (`GaussianPeaks`
on the uniform stream / `Eggholder` on the walker stream):

| unit of the added noise | all runs (range) | runs more than 2% worse | steps above 0.1 sigma | their sum in sigma |
|---|---|---|---|---|
| no fading | 0.75 (0.45 to 1.04) | 1 of 33 | 32.6 / 39.0 | 14.0 / 16.9 |
| the squared error of the leaf's predictions (section 9.14), plus the neighbour's own noise variance | 0.76 (0.48 to 1.04) | 1 of 33 | 4.8 / 3.5 | 2.2 / 2.9 |
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

The 150 runs of section 9.4 with the package as it is: the NRMSE of the last
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
| without fading (section 9.4) | | 0.78 | 0.74 | 0.71 | 0.61 |

* **0.73 of the default tree's error** as the geometric mean over the 150
  runs, from 0.18 to 1.14, against 0.71 without fading. 135 runs are more
  than 2% better than the default tree and 4 more than 2% worse, the four of
  section 9.4: `branching_ratio` on the uniform stream (14%),
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

### 9.16 Blending across the cuts: the prototypes

What is left of the steps of section 9.15 is where a query passes into
another leaf, whose kernel, scaling of the inputs and targets, metric of the
distances and calibration differ. Two ways to pass from one leaf to the next
gradually, within a zone around each cut whose width is a share of the
spread of the node's points along the cut, with weights that change linearly
across it (`research/partitioning/blended.py`); where a point is stored does
not change:

* **predictions blended**: the query is predicted from its neighbours with
  the hyperparameters of each leaf in turn, and the predictions are mixed;
* **kernels mixed**: one prediction, with the weighted sum of the leaves'
  kernels (in the units of the inputs and targets) as its kernel, and the
  weighted means of their metrics, target means, units of the fading and
  sigma scalers. A single output without a global model; a leaf that prefers
  its own GP, or neighbours that fail the test, give blended predictions.

The steps of the mean after 20 000 points, as in section 9.13, for
`GaussianPeaks` on the uniform stream / `Eggholder` on the walker stream /
`resonance_xsec` on the uniform stream:

| | leaves per prediction | steps above 0.1 sigma | their sum in sigma | largest step, median |
|---|---|---|---|---|
| the default tree | 1 | 3.9 / 3.9 / 4.8 | 4.3 / 4.8 / 13.7 | 0.89 / 1.26 / 1.36 |
| neighbours that fade out (the package) | 1 | 4.8 / 2.5 / 5.5 | 2.1 / 2.2 / 12.7 | 0.26 / 0.32 / 1.08 |
| predictions blended, zone 0.05 | 1.39 / 1.51 / 1.49 | 2.4 / 0.3 / 1.4 | 1.1 / 0.5 / 1.3 | 0.10 / 0.02 / 0.04 |
| predictions blended, zone 0.1 | 1.99 / 2.15 / 2.10 | 2.3 / 0.4 / 0.9 | 1.0 / 0.7 / 0.9 | 0.09 / 0.01 / 0.02 |
| kernels mixed, zone 0.05 | 1.39 / 1.51 / 1.49 | 2.9 / 0.5 / 2.8 | 1.3 / 0.7 / 2.9 | 0.11 / 0.03 / 0.07 |
| kernels mixed, zone 0.1 | 1.99 / 2.15 / 2.10 | 2.5 / 0.2 / 1.9 | 1.1 / 0.5 / 1.9 | 0.10 / 0.03 / 0.05 |

The 33 runs of sections 9.1 to 9.3, relative to the default tree, and
relative to the neighbours that fade out run by run:

| | 5000 points: all runs (range) | against fading alone, mean / median (range) | 20 000 points: all runs (range) | against fading alone, mean / median (range) | coverage at 20 000 | off by more than 5 sigma |
|---|---|---|---|---|---|---|
| neighbours that fade out (the package) | 0.77 (0.49 to 1.04) | 1 | 0.71 (0.39 to 0.99) | 1 | 0.686 | 1.52% |
| predictions blended, zone 0.05 | 0.76 (0.48 to 1.04) | 0.985 / 0.992 (0.91 to 1.02) | | | | |
| predictions blended, zone 0.1 | 0.75 (0.48 to 1.05) | 0.977 / 0.988 (0.80 to 1.05) | | | | |
| kernels mixed, zone 0.05 | 0.75 (0.48 to 1.04) | 0.976 / 0.992 (0.82 to 1.03) | 0.69 (0.38 to 0.97) | 0.977 / 0.985 (0.86 to 1.03) | 0.710 | 1.08% |
| kernels mixed, zone 0.1 | 0.72 (0.27 to 1.05) | 0.939 / 0.983 (0.31 to 1.05) | 0.67 (0.38 to 0.97) | 0.943 / 0.976 (0.51 to 1.08) | 0.733 | 0.80% |

* **Either way the steps at the cuts all but vanish**: the largest step of
  a path falls from 0.26 to 1.08 sigma to 0.01 to 0.11 sigma. On
  `GaussianPeaks` two to three steps of a tenth of a sigma remain, where
  the 16 leaves that are searched change.
* **The error falls too**: with the kernels mixed in a zone of 0.1 it is 6%
  lower than with fading alone at 20 000 points (2% in the median run, 22 of
  33 runs more than 2% better and one worse, by 8%), and 2% lower in a zone
  of 0.05. That is 0.67 and 0.69 of the default tree's error, where the
  neighbours without fading have 0.69 on these runs.
* **The coverage rises** to 0.71 and 0.73: a leaf's sigma scaler is
  calibrated on the predictions with its own kernel alone, which are less
  accurate in the zones.
* **Cost** (`GaussianPeaks`, 20 000 points, one process, the prototype
  walking the cuts for every query): a prediction with the kernels mixed
  takes 1.04 and 1.07 times as long as with fading alone in zones of 0.05
  and 0.1, and 1.24 and 1.86 times with the predictions blended.

### 9.17 Kernels mixed across the cuts (`NEIGHBOUR_BLEND`)

In the package the kernels are mixed as in section 9.16, with two
differences. The zone at a face of a leaf is a share of the leaf's own width
along that input, where the prototype took a share of the spread of the node
that was cut; and a leaf's sigma scaler is calibrated on the mixed
prediction. The 33 runs of sections 9.1 to 9.3, relative to the default tree
and, run by run, to the neighbours that fade out without mixing:

| | 5000 points: all runs (range) | against fading alone, mean / median (range) | 20 000 points: all runs (range) | against fading alone, mean / median (range) | coverage at 20 000 | off by more than 5 sigma |
|---|---|---|---|---|---|---|
| neighbours without fading or mixing | 0.75 (0.45 to 1.04) | | 0.69 (0.38 to 0.98) | 0.977 / 0.993 (0.88 to 1.01) | 0.686 | 1.52% |
| fading alone | 0.77 (0.49 to 1.04) | 1 | 0.71 (0.39 to 0.99) | 1 | 0.686 | 1.52% |
| prototype, zone 0.05 of the node | 0.75 (0.48 to 1.04) | 0.976 / 0.992 (0.82 to 1.03) | 0.69 (0.38 to 0.97) | 0.977 / 0.985 (0.86 to 1.03) | 0.710 | 1.08% |
| prototype, zone 0.1 of the node | 0.72 (0.27 to 1.05) | 0.939 / 0.983 (0.31 to 1.05) | 0.67 (0.38 to 0.97) | 0.943 / 0.976 (0.51 to 1.08) | 0.733 | 0.80% |
| package, zone 0.05 of the leaf | 0.76 (0.48 to 1.04) | 0.985 / 0.998 (0.82 to 1.02) | | | | |
| **package, zone 0.1 of the leaf** | 0.74 (0.38 to 1.04) | 0.962 / 0.992 (0.43 to 1.03) | 0.69 (0.38 to 0.98) | 0.983 / 0.990 (0.86 to 1.06) | 0.692 | 1.34% |
| package, zone 0.2 of the leaf | 0.73 (0.38 to 1.04) | 0.952 / 0.988 (0.44 to 1.02) | 0.68 (0.38 to 0.97) | 0.967 / 0.980 (0.80 to 1.07) | 0.700 | 1.21% |

The steps of the mean after 20 000 points, as in section 9.13, for
`GaussianPeaks` on the uniform stream / `Eggholder` on the walker stream /
`resonance_xsec` on the uniform stream:

| | steps above 0.1 sigma | their sum in sigma | largest step, median |
|---|---|---|---|
| the default tree | 3.9 / 3.9 / 4.8 | 4.3 / 4.8 / 13.7 | 0.89 / 1.26 / 1.36 |
| fading alone | 4.8 / 2.5 / 5.5 | 2.1 / 2.2 / 12.7 | 0.26 / 0.32 / 1.08 |
| **package, zone 0.1 of the leaf** | 3.0 / 0.5 / 3.6 | 1.3 / 0.9 / 4.4 | 0.11 / 0.04 / 0.11 |
| package, zone 0.2 of the leaf | 2.6 / 0.5 / 3.0 | 1.2 / 0.8 / 3.0 | 0.11 / 0.03 / 0.07 |

Cost after 20 000 points, one process (a slower machine than that of
sections 9.7, 9.12 and 9.15): milliseconds per prediction and per update
without and with the mixing, and the mean number of leaves in a prediction.

| target, stream | d | default tree: predict / update | fading alone | kernels mixed, zone 0.1 | leaves per prediction |
|---|---|---|---|---|---|
| `Eggholder`, walker | 3 | 0.22 / 1.91 | 0.70 / 2.66 | 0.78 / 2.76 | 1.28 |
| `resonance_xsec`, uniform | 4 | 0.17 / 3.24 | 0.62 / 4.15 | 0.75 / 4.56 | 1.32 |
| `GaussianPeaks`, uniform | 6 | 0.17 / 2.09 | 0.71 / 2.93 | 0.79 / 3.11 | 1.43 |
| `global_fit_loglike`, uniform | 10 | 0.16 / 3.83 | 0.50 / 4.76 | 0.69 / 5.16 | 1.34 |
| `ActiveSubspacePeaks`, uniform | 10 | 0.16 / 3.16 | 0.76 / 4.21 | 0.93 / 4.36 | 1.35 |

* **The steps at the cuts are gone as in the prototype**: the largest step
  of a path is 0.04 to 0.11 sigma, against 0.26 to 1.08 with fading alone.
  On `resonance_xsec`, where leaves keep to their own GP or the neighbours
  fail the test more often, 0.9 steps of more than a sigma remain on a path,
  against 1.6.
* **The error is 2% lower than with fading alone at 20 000 points** (1% in
  the median run; 8 runs more than 2% better and one worse, by 6%), which
  takes back most of what the fading costs: 0.69 of the default tree's
  error, as the neighbours without fading. The zones of the leaves are
  narrower than those of the prototype, whose 6% came with zones of old
  cuts that reach across many leaves.
* **The coverage stays at 0.69** with the sigma scaler calibrated on the
  mixed prediction, where the prototype had 0.73.
* **Twice the zone** gives another 2% at 20 000 points, with 1.9 leaves in
  a prediction on `GaussianPeaks` and 0.96 ms per prediction against 0.79.
* **A prediction takes 11 to 37% longer** than with fading alone and an
  update 4 to 10%. About 1.3 to 1.4 leaves take part in a prediction
  whatever the number of inputs: a leaf of a tree this size has seen eight
  or nine cuts, so most of its faces in ten dimensions are still open.

The 150 runs of section 9.4 with the package as it is (neighbours that fade
out, kernels mixed in zones of 0.1): the NRMSE of the last 5000 predictions
divided by that of the default tree, for the two seeds.

| target | d | uniform | walker | sweeping | focusing |
|---|---|---|---|---|---|
| `active_peaks` | 10 | 0.74 / 0.77 | 0.69 / 0.72 | 0.65 / 0.84 | 0.59 / 0.61 |
| `branching_ratio` | 5 | 1.11 / 0.84 | 0.89 / 0.95 | 1.11 / 0.93 |  |
| `chirp` | 4 | 0.91 / 0.90 | 0.72 / 0.68 | 0.69 / 0.70 | 0.37 / 0.37 |
| `diagonal_ridge` | 4 | 0.61 / 0.64 | 0.73 / 0.67 | 0.73 / 0.75 | 0.66 / 0.69 |
| `dilepton_spectrum` | 4 | 0.89 / 0.85 | 0.94 / 0.85 | 0.73 / 0.79 | 0.93 / 0.83 |
| `eft_signal_strength` | 6 | 0.42 / 0.44 | 0.43 / 0.39 | 0.65 / 0.48 | 0.95 / 0.95 |
| `eggholder` | 3 | 0.87 / 0.88 | 0.82 / 0.86 | 0.90 / 0.90 | 0.83 / 0.83 |
| `gaussian_peaks` | 6 | 0.64 / 0.67 | 0.41 / 0.48 | 0.53 / 0.48 | 0.49 / 0.58 |
| `global_fit_loglike` | 10 | 0.95 / 0.93 | 0.86 / 0.86 | 0.94 / 0.93 | 0.20 / 1.00 |
| `jet_spectrum` | 4 | 0.74 / 0.80 | 1.00 / 0.76 | 0.72 / 0.75 | 0.44 / 0.44 |
| `loop_decay_width` | 4 | 0.86 / 0.79 | 0.84 / 0.90 | 0.38 / 0.17 | 0.28 / 0.27 |
| `michalewicz` | 5 | 0.98 / 1.00 | 0.78 / 0.92 | 0.99 / 0.98 | 0.81 / 0.81 |
| `neutralino_mass` | 4 | 0.82 / 0.77 | 0.98 / 0.24 | 0.82 / 0.96 | 0.52 / 0.44 |
| `oscillation_probability` | 5 | 0.63 / 0.55 | 0.63 / 1.02 | 0.93 / 0.94 | 0.84 / 0.90 |
| `relic_density` | 4 | 0.88 / 0.80 | 0.92 / 0.96 | 0.86 / 0.74 | 0.91 / 0.95 |
| `resonance_xsec` | 4 | 0.88 / 0.96 | 0.65 / 0.96 | 0.97 / 0.85 | 0.71 / 0.72 |
| `rotated_rosenbrock` | 6 | 0.53 / 0.51 | 0.56 / 0.51 | 0.48 / 0.67 | 0.48 / 0.49 |
| `signal_yield_mc` | 4 | 0.82 / 0.82 | 0.86 / 0.86 | 0.50 / 0.54 | 0.94 / 0.93 |
| `step_ridge` | 6 | 0.90 / 0.85 | 0.93 / 0.89 | 0.85 / 0.89 | 0.84 / 0.86 |
| geometric mean | | 0.77 | 0.74 | 0.72 | 0.63 |
| fading alone (section 9.15) | | 0.79 | 0.75 | 0.73 | 0.65 |
| neither (section 9.4) | | 0.78 | 0.74 | 0.71 | 0.61 |

* **0.71 of the default tree's error** as the geometric mean over the 150
  runs, from 0.17 to 1.11, against 0.73 with fading alone. 142 runs are
  more than 2% better than the default tree and 2 more than 2% worse, both
  by 11%: `branching_ratio` with one seed on the uniform and on the sweeping
  stream. `eft_signal_strength` on the focusing stream, 4 to 5% worse
  before, is 5% better.
* **Run by run the error is 2.3% below that with fading alone** (1.7% in
  the median; 0.86 to 1.13), with 68 runs more than 2% better and 3 worse:
  `jet_spectrum` and `branching_ratio` on the sweeping stream by 13% and
  `resonance_xsec` on the walker by 6%. Against the neighbours without
  fading or mixing it is 0.7% lower in the median run and 0.9% higher in the
  mean.
* **The rest of the error distribution**: the mean absolute error is 0.75 of
  the default tree's, the median 0.78, the 99% quantile 0.71 and the largest
  error of a run 0.66; 0.74 on points held out of the stream and 0.83 on
  uniform test points.
* **The coverage is 0.719** against 0.709 for the default tree, and 1.51% of
  the predictions are off by more than 5 sigma, against 1.95% for the
  default tree, 1.76% with fading alone and 2.02% with neither.

Along a 200 000-point stream (`GaussianPeaks`, d = 6, uniform; one process,
the three settings one after the other on the machine of the cost table
above), as in section 9.12: milliseconds per update since the previous row
and per prediction at the points that follow, and the RMSE of those
predictions.

| points | leaves | default tree: update / predict | RMSE | fading alone: update / predict | RMSE | fading and mixing: update / predict | RMSE |
|---|---|---|---|---|---|---|---|
| 5000 | 71 | 2.13 / 0.17 | 0.459 | 3.00 / 0.71 | 0.328 | 3.07 / 0.82 | 0.328 |
| 20 000 | 289 | 2.08 / 0.17 | 0.282 | 3.05 / 0.72 | 0.177 | 3.32 / 0.90 | 0.175 |
| 50 000 | 730 | 2.03 / 0.16 | 0.237 | 3.22 / 0.81 | 0.134 | 3.47 / 0.83 | 0.134 |
| 100 000 | 1465 | 2.19 / 0.17 | 0.162 | 3.20 / 0.79 | 0.106 | 3.67 / 1.10 | 0.104 |
| 200 000 | 2917 | 2.26 / 0.18 | 0.129 | 3.26 / 0.79 | 0.069 | 3.79 / 1.06 | 0.067 |

The leaves of a prediction in the index alone, without GPs
(`research/partitioning/index_scaling.py` with zones of 0.1, uniform stream,
500 queries): their mean number, and the time to find them with their
weights.

| points | leaves | d = 3: leaves per prediction | ms | d = 6 | ms | d = 10 | ms |
|---|---|---|---|---|---|---|---|
| 5000 | 70 | 1.25 | 0.03 | 1.30 | 0.03 | 1.40 | 0.03 |
| 20 000 | 290 | 1.25 | 0.03 | 1.37 | 0.04 | 1.47 | 0.03 |
| 100 000 | 1450 | 1.30 | 0.04 | 1.41 | 0.05 | 1.52 | 0.04 |
| 400 000 | 5800 | 1.30 | 0.04 | 1.43 | 0.05 | 1.60 | 0.05 |
| 1 000 000 | 14 500 | 1.33 | 0.05 | 1.47 | 0.09 | 1.65 | 0.06 |

* **With the mixing the cost of a point grows slowly with the stream**: from
  the first 5000 points to 200 000, an update by 23% and a prediction by 29%
  (with a spread of a tenth between the rows), where fading alone grows by
  9 and 12% and the default tree by 6 and 4%.
* **The number of leaves in a prediction grows by a tenth to a fifth from
  5000 points to a million**, towards the `1.1^d` of leaves with two faces
  along every input, and the search for them stays below 0.1 ms.
* **The error at 200 000 points is 0.52 of the default tree's**, against
  0.54 with fading alone.

### 9.18 Leaves asleep by default with neighbours: the prototype

A prediction from neighbours needs the kernel and the scalers of a leaf, not
the factor of its kernel matrix, which is what a sleeping leaf releases
(section 6). In a prototype (`research/partitioning/sleepy.py`) every leaf
sleeps except the last few that fitted or were woken, for a point or for a
prediction from their own points. 20 000 points, three runs at a time; the
predictions are the same to the last digit.

| target, stream | leaves awake | memory MB (of it the GPs) | update ms | predict ms | leaves woken per 1000 updates | per 1000 predictions |
|---|---|---|---|---|---|---|
| `GaussianPeaks`, uniform | all 281 | 37.2 (29.8) | 3.36 | 0.86 | | |
| | 1 | 9.7 (2.3) | 3.55 | 0.79 | 956 | 37 |
| | 8 | 10.3 (2.9) | 3.49 | 0.85 | 850 | 13 |
| | 32 | 12.7 (5.3) | 3.51 | 0.81 | 621 | 10 |
| `GaussianPeaks`, walker | all 293 | 38.7 (30.9) | 3.54 | 0.88 | | |
| | 1 | 10.8 (3.1) | 3.60 | 0.85 | 525 | 33 |
| | 8 | 11.4 (3.7) | 3.50 | 0.86 | 236 | 4 |
| `resonance_xsec`, uniform | all 286 | 36.6 (29.5) | 4.58 | 0.98 | | |
| | 1 | 8.9 (1.8) | 4.73 | 0.85 | 1018 | 375 |

* **The tree takes a quarter of the memory**: about 34 kB per leaf in place
  of 132 kB.
* **An update takes 2 to 6% longer**, for the factor of the leaf that
  receives the point: its own prediction is compared with the neighbours'
  and the point is appended to it. The prediction time does not change
  beyond the spread of the measurement.
* **A leaf is woken for 1 to 4% of the predictions** on `GaussianPeaks`,
  where its own GP predicts because the neighbours fail the test or the leaf
  prefers it, and for 38% on `resonance_xsec`, where a fifth of the leaves
  prefer their own GP.

A memory limit with predictions from neighbours and every leaf awake, as the
package was (`research/partitioning/memory_limit.py`; `GaussianPeaks`,
20 000 points, four runs at a time): the RMSE of the predictions made before each of the
last 5000 points.

| stream | `max_memory_mb` | leaves (asleep) | points kept | from the leaves: RMSE | from neighbours: RMSE |
|---|---|---|---|---|---|
| uniform | none (37.3 MB) | 279 (0) | 20 000 | 0.309 | 0.198 |
| | 12 | 194 (139) | 13 700 | 0.374 | 0.240 |
| | 6 | 98 (70) | 6900 | 0.511 | 0.325 |
| walker | none (37.8 MB) | 289 (0) | 20 000 | 0.122 | 0.050 |
| | 12 | 195 (138) | 13 400 | 0.152 | 0.058 |
| | 6 | 97 (69) | 6700 | 0.196 | 0.082 |

* **The limit holds with neighbours as it does without**: the size the tree
  reports and the size counted from its leaves are at the limit, and as many
  leaves sleep and are removed. The error with neighbours is 0.42 to 0.64
  of that of the leaves at the same limit.
* **Leaves are removed while half of the leaves' memory is left to the GPs
  that are awake** (`FITTED_SHARE`), which predictions from neighbours do
  not need: at 12 MB a third of the points are gone and the error is 17 to
  21% higher, where the whole tree asleep takes 9.7 to 10.8 MB.

### 9.19 Leaves asleep by default (`NEIGHBOUR_AWAKE`)

In the package the leaves whose GPs were fitted or woken last stay awake, 8
by default, and so does a leaf that predicts from its own points. Two trees
side by side in one process, each point given to both in turn
(`research/partitioning/time_paired.py`), 20 000 points: the times with the
leaves asleep divided by those with every leaf awake.

| target, stream | leaves asleep | memory MB, every leaf awake | leaves asleep | update time | prediction time |
|---|---|---|---|---|---|
| `GaussianPeaks`, uniform | 272 of 281 | 37.2 | 10.1 | 1.11 | 1.00 |
| `GaussianPeaks`, walker | 282 of 293 | 38.7 | 10.5 | 1.03 | 1.01 |
| `resonance_xsec`, uniform | 226 of 286 | 36.6 | 14.1 | 1.06 | 1.06 |
| `global_fit_loglike`, uniform | 151 of 293 | 40.4 | 25.5 | | |

* **The tree takes 27 to 63% of the memory.** On `global_fit_loglike` half
  of the leaves predict from their own points and stay awake.
* **An update takes 3 to 11% longer**, most on a uniform stream, where
  nearly every point wakes its leaf. **A prediction takes up to 6% longer.**
  With the leaves that predict from their own points asleep too, a
  prediction on `resonance_xsec` took 17% longer and an update 9%, for 9.1
  MB in place of 14.1.
* **The predictions are the same**: on the 33 runs of 5000 points the RMSE
  agrees to four digits in every run, and the largest difference of a single
  prediction is 4e-7 of the target's range (a factor that is computed again
  differs in its last digits from one that grew point by point).
* **Against the default tree** (side by side as above, where the default
  tree's short predictions lose more to the other tree than they would
  alone): an update takes 1.3 to 1.7 times as long and a prediction 3.0 to
  3.4 times, on `Eggholder` (walker), `GaussianPeaks`, `resonance_xsec` and
  `global_fit_loglike` (uniform).

A memory limit, as in the table of section 9.18 (`GaussianPeaks`, 20 000
points):

| stream | `max_memory_mb` | leaves (asleep) | points kept | RMSE | with every leaf awake (section 9.18): points kept | RMSE |
|---|---|---|---|---|---|---|
| uniform | none (10.7 MB) | 279 (264) | 20 000 | 0.198 | 20 000 | 0.198 |
| | 12 | 279 (264) | 20 000 | 0.198 | 13 700 | 0.240 |
| | 6 | 151 (141) | 10 100 | 0.263 | 6900 | 0.325 |
| walker | none (10.7 MB) | 289 (276) | 20 000 | 0.050 | 20 000 | 0.050 |
| | 12 | 289 (276) | 20 000 | 0.050 | 13 400 | 0.058 |
| | 6 | 157 (149) | 10 800 | 0.066 | 6700 | 0.082 |

* **At 12 MB nothing is removed**, and at 6 MB half of the points are kept
  where a third was, with an error 19 to 20% lower.

### 9.20 Predictions from neighbours outside the data

The stream and tree of the README's animation
(`research/partitioning/animation_grid.py`: `Himmelblau` in two dimensions,
1500 points of a cloud that moves and contracts, `Nbar=100`, a fit every 2
points, the additive kernel with an RBF base), predicted on a 70 x 70 grid
over the unit square. The target is at most 77.9 there. "Away" are the
cells further than 0.05 from the nearest point of the stream, a quarter of
the grid. The declared noise is a share of the target value.

| seed | noise | leaves: largest prediction | RMSE away | near | neighbours: largest prediction | RMSE away | near |
|---|---|---|---|---|---|---|---|
| 4 | 1e-3 | 73.1 | 14.9 | 1.21 | 74.8 | 10.8 | 0.85 |
| | 1e-4 | 74.9 | 13.0 | 1.94 | 86.1 | 9.2 | 1.09 |
| | 1e-5 | 73.0 | 10.6 | 2.04 | 107.3 | 9.1 | 1.21 |
| | 1e-6 | 73.0 | 11.6 | 2.03 | 148.8 | 11.7 | 1.32 |
| 1 | 1e-3 | 74.9 | 11.2 | 1.24 | 86.4 | 12.6 | 0.97 |
| | 1e-4 | 77.1 | 11.0 | 1.30 | 92.0 | 10.2 | 1.02 |
| | 1e-5 | 73.8 | 16.5 | 1.34 | 75.9 | 14.4 | 0.98 |
| | 1e-6 | 237.8 | 36.5 | 2.70 | 239.3 | 33.3 | 2.17 |
| 2 | 1e-3 | 73.0 | 7.0 | 1.52 | 73.0 | 16.7 | 1.50 |
| | 1e-4 | 73.6 | 7.8 | 1.63 | 76.9 | 9.8 | 1.83 |
| | 1e-5 | 94.8 | 8.3 | 2.49 | 94.8 | 11.1 | 2.52 |
| | 1e-6 | 125.6 | 11.1 | 3.96 | 125.6 | 13.2 | 3.55 |

* **Away from the data the two are equally accurate on average**, 13.3 and
  13.5 over the twelve runs, **and fail in different places.** Near the
  data the neighbours have the lower error in 10 of the 12.
* **With a noise of 1e-6 of the value a leaf's own GP can overshoot**: to
  238 with seed 1 and 126 with seed 2, in both modes.
* **The neighbours add failures of their own.** With seed 4 and a noise of
  1e-6 they predict up to 149 at the upper edge of the square, where the
  target is 65 and the leaf's own GP predicts 40. With seed 2 and a noise
  of 1e-3 they predict -15 in the lower right corner, where the target is
  50 and the leaf's own GP predicts 51, and 114 cells are off by more than
  30 against 3.

The worst query of each of the two:

| | (0.275, 1.0), seed 4, noise 1e-6 | (0.91, 0.19), seed 2, noise 1e-3 |
|---|---|---|
| target | 65.2 | 49.8 |
| from the neighbours | 149.2 +- 21.1 | -14.8 +- 43.2 |
| the leaf's own GP | 40.4 +- 12.3 | 51.1 +- 7.8 |
| neighbours among the leaf's own points | 28 of 100 | 27 of 100 |
| extent of the neighbours along x1, of the leaf's points | 0.14 to 0.46, 0.21 to 0.35 | 0.39 to 1.00, 0.21 to 0.66 |
| targets of the neighbours, in the leaf's standard deviations from its mean | -6.4 to 1.3 | -9.1 to 2.2 |
| consistency with the leaf's kernel | 7.6 | 9.5 (13.2 before the fading) |
| a GP fitted to the 100 neighbours | 66.5 +- 7.7 | -42.1 +- 6.1 |

* **The leaf of the query supplies the kernel, other leaves most of the
  points.** Away from the data the 100 nearest points reach across several
  leaves, and their targets lie 6 to 9 of the leaf's standard deviations
  from its mean. The kernel carries their trend on to the query, where the
  leaf's own GP, given only its own points, returns to their mean. Both
  sets pass the test against the kernel, with a consistency just under the
  limit of 10.
* **In the first case the kernel is at fault.** A GP fitted to the same 100
  points predicts 66.5. With the leaf's kernel the matrix has a condition
  number of 7e12 and the weights of the neighbours reach 3e5, for targets of
  at most 6.4. The same conditioning with a noise of 1e-5, 1e-4 and 1e-3 of
  the value gives 122, 83 and 57, and with the 44 neighbours inside the
  leaf's own range of x1 it gives 91.
* **In the second the extrapolation itself is.** The targets of the
  neighbours fall towards the query, and a GP fitted to them predicts -42.

A stricter test for such queries (`research/partitioning/guarded.py`, not
in the code): where the centre of the neighbours lies further from the
query than 0.65 of the distance of the furthest neighbour, they predict only
with a consistency of at most 3. That ratio is at most 0.6 for 95% of the
stream's own points, in two dimensions and in six, and 0.67 to 0.78 for the
queries of the two patches.

| seed, noise | neighbours: largest prediction | RMSE away | stricter test: largest prediction | RMSE away | leaves: RMSE away |
|---|---|---|---|---|---|
| 4, 1e-6 | 148.8 | 11.7 | 72.9 | 10.5 | 11.6 |
| 4, 1e-3 | 74.8 | 10.8 | 74.8 | 11.0 | 14.9 |
| 1, 1e-6 | 239.3 | 33.3 | 239.3 | 36.8 | 36.5 |
| 1, 1e-3 | 86.4 | 12.6 | 86.4 | 12.5 | 11.2 |
| 2, 1e-6 | 125.6 | 13.2 | 125.6 | 12.1 | 11.1 |
| 2, 1e-3 | 73.0 | 16.7 | 73.0 | 7.3 | 7.0 |

* **Both patches go.** The error near the data is within 2% in 5 of the 6
  runs and 8% higher in the sixth (seed 1, noise 1e-3).
* **On the 33 runs of 5000 points the stricter test costs accuracy**: the
  error along the stream is 2.0% higher and that of uniform test points
  1.5% (6.0% and 3.6% on the sweeping stream), with 4 runs more than 2%
  worse and 2 better. With "the nearest neighbour is further than one of
  the leaf's spreads" as the mark of such a query the two numbers are 2.3%
  and 2.1%:
  in six dimensions the nearest neighbour of a point of the stream itself
  is 0.9 to 1.2 spreads away at the median.
* **On those targets the neighbours are the better choice outside the data
  as well.** For uniform test points after 5000 points of a stream, eight
  of the targets on three streams (`extrapolation_diag.py`), the leaf's own
  GP for every query has an RMSE 3% higher overall and 31 to 36% higher on
  `eft_signal_strength` and `GaussianPeaks`. On six targets after 3000
  points, with a noise of 1e-3 and of 1e-6 of the value and with the default
  and the additive kernel (72 runs), it is 1.2 to 5.7% higher. The largest
  single error is another matter with the additive kernel: 1.9 times the
  target's range with a noise of 1e-6 and 1.3 times with 1e-3, where the
  leaf's own GP stays at 0.8. With the default kernel it is 1.1 to 1.2
  times for both.

Two more variants on the same runs: the limit of the test at 3 for every
query (`NEIGHBOUR_CONSISTENCY` at 3), and the 100 nearest points together with
the rest of the leaf's own points (`research/partitioning/with_leaf.py`, not
in the code), 127 to 141 points per prediction on average. Largest
prediction and RMSE away from the data:

| seed, noise | neighbours | limit 3 for every query | with the leaf's points | the same without the test |
|---|---|---|---|---|
| 4, 1e-6 | 148.8, 11.7 | 72.9, 10.6 | 157.5, 12.7 | 620.8, 44.3 |
| 4, 1e-3 | 74.8, 10.8 | 74.8, 11.0 | 73.1, 14.5 | 73.1, 15.0 |
| 1, 1e-6 | 239.3, 33.3 | 239.3, 36.8 | 239.3, 34.6 | 581.0, 50.3 |
| 1, 1e-3 | 86.4, 12.6 | 86.4, 12.5 | 86.6, 13.1 | 88.8, 19.3 |
| 2, 1e-6 | 125.6, 13.2 | 125.6, 12.1 | 125.6, 13.5 | 249.2, 25.9 |
| 2, 1e-3 | 73.0, 16.7 | 73.0, 6.9 | 73.1, 19.0 | 73.0, 6.1 |

* **The leaf's remaining points do not remove the patches.** The two worst
  queries are predicted as 157 from 151 points and -14.8 from 161: the
  points of other leaves near the query are still there, and the leaf's own
  lie further away. The consistency falls to 6.0 and 6.4, since the leaf's
  own points fit its kernel.
* **Without the test the predictions reach 621**, and in one run the error
  near the data is 26.7 where it was 3.6.
* **On the 33 runs of 5000 points the leaf's points lower the error along
  the stream by 5.8%** (3.1% at the median), and 150 nearest neighbours by
  5.7% (2.8%): the gain is that of more points. Without the test it is
  4.7%.
* **A limit of 3 for every query costs 3.6%** along the stream and 2.1% at
  uniform test points, with 10 runs more than 2% worse and 1 better, against
  2.0% and 1.5% for the limit of 3 outside the neighbours only. On the grids
  the two do the same.

A gradual change in place of the two limits (the prototype
`research/partitioning/smooth_guard.py`; section 9.21 has it in the
package): the neighbours' prediction and the leaf's own are mixed, with a weight for the
neighbours that falls from 1 to 0 as their consistency rises from half the
limit to twice it, linearly in its logarithm. The limit slides from 10 to 3
as the centre of the neighbours moves from 0.5 to 0.8 of the distance of the
furthest one. Largest prediction and RMSE away from the data, and the steps
of the mean along 40 straight lines across the square, of 1000 steps each
(a step counts by how far it departs from the mean of the steps on either
side of it, in units of sigma):

| seed, noise | neighbours | limit 3 outside | gradual | gradual, without the slide |
|---|---|---|---|---|
| 4, 1e-6 | 148.8, 11.7 | 72.9, 10.5 | 74.0, 9.1 | 137.2, 10.1 |
| 4, 1e-3 | 74.8, 10.8 | 74.8, 11.0 | 74.8, 11.0 | 74.8, 10.8 |
| 1, 1e-6 | 239.3, 33.3 | 239.3, 36.8 | 239.3, 36.4 | 239.3, 32.7 |
| 1, 1e-3 | 86.4, 12.6 | 86.4, 12.5 | 86.4, 12.3 | 86.4, 12.6 |
| 2, 1e-6 | 125.6, 13.2 | 125.6, 12.1 | 125.6, 12.2 | 125.6, 13.1 |
| 2, 1e-3 | 73.0, 16.7 | 73.0, 7.3 | 73.0, 8.3 | 73.0, 13.2 |

| seed, noise | steps above 1 sigma per line: neighbours | limit 3 outside | gradual | leaves | largest step on the worst line: neighbours | limit 3 outside | gradual | leaves |
|---|---|---|---|---|---|---|---|---|
| 4, 1e-6 | 9.2 | 9.4 | 6.5 | 0.8 | 101 | 99 | 45 | 28 |
| 2, 1e-3 | 1.0 | 1.1 | 0.3 | 0.0 | 12.6 | 12.6 | 5.5 | 0.4 |

* **The patches go as with the limit of 3 outside the neighbours**, and
  the largest error of the first is 32 where that leaves 41. Without the
  slide of the limit the first patch stays.
* **On the 33 runs of 5000 points the error along the stream is 1.4%
  higher** and that of uniform test points 1.1%, against 2.0% and 1.5%
  with the limit of 3 outside. A change over a factor of 3 on either side
  of the limit in place of 2 gives 0.7% and 1.0%. Without the slide of the
  limit the error is that of the switch.
* **The large steps are fewer by a third to two thirds** and the largest
  about half as high. Steps above 0.1 sigma are as many as before, 53 and
  15 per line, where the leaves' own predictions have 10 and 1: across
  the whole square most steps of the neighbours' predictions have other
  causes.

### 9.21 A gradual change to the leaf's own GP (`NEIGHBOUR_TRANSITION`, `NEIGHBOUR_CONSISTENCY_OUTSIDE`)

In the package the centre of the neighbours counts a neighbour that fades
out by what is left of it, which makes the measure about 0.03 smaller than
in the prototype of section 9.20. The grids of that section with the
defaults, against the switch at one limit:

| seed, noise | switch: largest prediction | RMSE away from the data | gradual: largest prediction | RMSE away from the data |
|---|---|---|---|---|
| 4, 1e-6 | 148.8 | 11.7 | 85.0 | 8.9 |
| 4, 1e-3 | 74.8 | 10.8 | 74.8 | 11.0 |
| 1, 1e-6 | 239.3 | 33.3 | 239.3 | 36.2 |
| 1, 1e-3 | 86.4 | 12.6 | 86.4 | 12.4 |
| 2, 1e-6 | 125.6 | 13.2 | 125.6 | 12.4 |
| 2, 1e-3 | 73.0 | 16.7 | 73.0 | 8.8 |

* **Away from the data the mean RMSE is 14.9**, against 16.4 with the
  switch and 15.4 for the leaves. The error near the data is within 0.6% in
  5 of the 6 runs and 2% higher in the sixth.
* **The steps** along the lines across the square are those of the
  prototype: 6.5 and 0.3 above one sigma per line (seed 4 with a noise of
  1e-6, seed 2 with 1e-3), against 9.2 and 1.0 with the switch.
* **On the 33 runs of 5000 points** the error along the stream and that
  of uniform test points are both 1.0% higher than with the switch, and
  the same at the median: 3 runs are more than 2% worse, 2 better, and the
  mean is 2.8% higher on the sweeping stream and within 0.6% on the other
  two. 1.5% of the predictions are off by more than 5 sigma, against 1.7%.

The 150 runs of 20 000 points of section 9.17, as geometric means of the
NRMSE of the last 5000 predictions:

| stream | against the default tree: switch | gradual | gradual against the switch | uniform test points, gradual against the switch |
|---|---|---|---|---|
| uniform | 0.770 | 0.769 | 0.998 | 0.997 |
| walker | 0.735 | 0.733 | 0.997 | 1.010 |
| sweeping | 0.720 | 0.716 | 0.995 | 1.005 |
| focusing | 0.632 | 0.630 | 0.997 | 0.999 |
| all | 0.714 | 0.711 | 0.997 | 1.003 |

* **At 20 000 points the error along the stream is 0.3% lower than with
  the switch** and that of uniform test points 0.3% higher, with 4 runs more
  than 2% worse and 8 better.
* **One run of 150 is more than 2% worse than the default tree**
  (`branching_ratio` on the uniform stream, 9%), where two were, by 11%.
* **The coverage is 0.72 for both**, and 1.4% of the predictions are off
  by more than 5 sigma, against 1.5% with the switch and 2.0% for the
  default tree.

Cost, with the two settings side by side in one process
(`research/partitioning/time_paired.py`), 20 000 points, then 1000 points
that follow in the stream and 1000 uniform ones: the times with the gradual
change divided by those with the switch, and the share of the predictions
that needed the own GP of a leaf.

| target, stream | update time | prediction time, stream | uniform | own GP needed, stream: switch | gradual | uniform: switch | gradual | of them from a sleeping leaf |
|---|---|---|---|---|---|---|---|---|
| `GaussianPeaks`, uniform | 1.004 | 1.010 | 1.010 | 4.2% | 4.5% | 5.1% | 5.2% | 0.2% |
| `GaussianPeaks`, walker | 1.000 | 0.999 | 1.106 | 7.6% | 7.6% | 8.9% | 25.5% | 10.4% |
| `resonance_xsec`, uniform | 1.004 | 1.036 | 1.035 | 38.1% | 39.6% | 38.8% | 40.4% | 13.9% |
| `global_fit_loglike`, uniform | 1.005 | 1.035 | 1.012 | 72.6% | 73.3% | 65.6% | 65.9% | 6.8% |
| `Eggholder`, walker | 1.002 | 1.008 | 1.039 | 1.0% | 1.6% | 0.9% | 5.6% | 1.7% |
| `eft_signal_strength`, sweeping | 1.002 | 0.999 | 1.002 | 12.9% | 12.9% | 2.7% | 3.0% | 0.3% |

* **An update takes up to 0.5% longer and a prediction along the stream
  up to 4%.** Along the stream the leaf's own GP is needed for at most 1.5
  points of a hundred more than with the switch.
* **A prediction away from the stream takes up to 11% longer**, where a
  quarter of the uniform test points need the leaf's own GP in place of a
  tenth, and one in ten wakes a sleeping leaf for it.
* **The memory is the same within 5%**: 10.9 MB for 10.4 where most leaves
  are woken, and within 0.3 MB elsewhere.

## 10. Curves as outputs (multi-output targets)

The twelve families of `examples/target_functions_curves.py`: a curve
`f(x; theta)` on a grid of 48 to 128 points in `x` is the multi-output target,
and three or four parameters `theta`, uniform in the unit cube, are the inputs.
The observations are exact, with a declared sigma of 0.1% of the quantity for
the curves given as `log10` and of 1e-4 to 1e-3 in the unit of the curve
otherwise. `peak_counts_mc` has Monte Carlo noise, declared bin by bin. The
NRMSE is the RMSE over all grid points divided by the range of the family, and
errors are against the noise-free curves. The tree is that of the conventions
above, with the scikit-learn leaves. `kinematic_edge` is also run on a grid of
128 points in place of 64 (section 10.4).

### 10.1 The output models on a stream

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
  model. For the noisy curves it is 0.98 to 0.99 (section 10.5).
* **An update with `'aligned'` takes 2.1 to 2.7 times as long as with
  `'shared'`** where the leaves hold alignments, and 1.4 to 1.8 times where
  they decline: a leaf that declined waits for 1, then 3, then 7 fits before
  it tries again. A prediction takes 0.3 to 0.5 ms with `'shared'` and 2 to 4
  ms from a leaf with an alignment, with four such runs on four cores.

### 10.2 Batch fits

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

### 10.3 More curves

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

### 10.4 The grid has to resolve the feature

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

### 10.5 Other measurements

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

### 10.6 Other warps (not in the code)

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
* **The factor of the leave-one-out test**, streams of 800 curves, NRMSE of
  the last 400 predictions with a factor 1.5 (as in the code), 2 and 3, with
  the default and with Matérn 5/2 leaves (section 10.8):

  | family | leaves | 1.5 | 2 | 3 |
  |---|---|---:|---:|---:|
  | `cmb_peaks` | 3/2 | 1.2e-4 | 1.8e-4 | 1.9e-4 |
  | `cmb_peaks` | 5/2 | 2.2e-5 | 3.2e-5 | 3.3e-5 |
  | `ringdown` | 3/2 | 8.8e-4 | 8.8e-4 | 1.5e-3 |
  | `ringdown` | 5/2 | 4.7e-4 | 4.7e-4 | 1.3e-3 |
  | `z_lineshape` | 3/2 | 6.4e-5 | 6.4e-5 | 1.6e-4 |
  | `zprime_spectrum` | 3/2 | 2.0e-3 | 2.8e-3 | 2.8e-3 |
  | `kinematic_edge`, 64 points | 5/2 | 1.0e-3 | 1.1e-3 | 8.3e-4 |

  A larger factor declines alignments that help: on `cmb_peaks` every leaf
  declines with a factor 2. It would avoid the one case in which an alignment
  that passed the test did harm, `kinematic_edge` on 64 points with Matérn
  5/2 leaves: on the test curves the NRMSE is 6.6e-4 with the factor 1.5 and
  3.4e-4 with 3, which is that of `'shared'`. The warp is fitted to all
  curves of a leaf, the one left out included, so the test is optimistic
  where the grid hardly resolves the feature.

### 10.7 Global models with curves

Streams of 800 curves as in section 10.1, one seed, with the additive GP
(`global_mean='additive_gp'`, "GP") or the network (`'net'`) as the global
model and the default leaves. NRMSE of the last 400 predictions:

| family | shared | shared + GP | shared + net | aligned | aligned + GP | aligned + net |
|---|---:|---:|---:|---:|---:|---:|
| `matter_power` | 5.0e-5 | 1.4e-5 | 8.6e-5 | 5.0e-5 | 1.4e-5 | 8.6e-5 |
| `eft_distribution` | 1.5e-4 | 2.1e-6 | 2.4e-5 | 1.5e-4 | 2.1e-6 | 2.4e-5 |
| `jet_spectrum` | 5.6e-5 | 9.0e-6 | 9.2e-5 | 5.6e-5 | 9.0e-6 | 9.2e-5 |
| `dust_sed` | 1.3e-5 | 2.6e-6 | 2.8e-5 | 1.3e-5 | 2.6e-6 | 2.8e-5 |
| `z_lineshape` | 2.6e-4 | 7.4e-5 | 7.1e-5 | 6.4e-5 | 6.4e-5 | 6.4e-5 |
| `zprime_spectrum` | 8.2e-3 | 9.3e-3 | 9.4e-3 | 2.0e-3 | 2.0e-3 | 2.0e-3 |
| `kinematic_edge` | 1.3e-3 | 9.7e-4 | 2.0e-3 | 7.8e-4 | 7.8e-4 | 7.8e-4 |
| `kinematic_edge`, 128 points | 1.4e-3 | 1.1e-3 | 2.2e-3 | 1.0e-4 | 1.0e-4 | 1.0e-4 |
| `oscillation_probability` | 1.1e-3 | 4.3e-4 | 4.7e-4 | 1.1e-3 | 4.3e-4 | 4.7e-4 |
| `cmb_peaks` | 1.9e-4 | 1.1e-4 | 6.0e-5 | 1.2e-4 | 1.2e-4 | 1.2e-4 |
| `form_factor` | 4.7e-3 | 4.0e-3 | 6.7e-3 | 4.7e-3 | 4.0e-3 | 6.7e-3 |
| `ringdown` | 2.8e-3 | 1.8e-3 | 9.4e-4 | 8.8e-4 | 8.8e-4 | 8.8e-4 |
| `peak_counts_mc` | 4.1e-3 | 3.3e-3 | 4.3e-3 | 4.1e-3 | 3.3e-3 | 4.3e-3 |

* **The additive GP lowers the error of `'shared'`** by a factor 3.6 to 70 on
  the four smooth families, 3.5 on `z_lineshape`, 2.6 on
  `oscillation_probability` and 1.2 to 1.7 on five others, and raises it by
  13% on `zprime_spectrum`: 2.9 (0.89 to 69) in the
  geometric mean over the twelve families. With 800 curves its sample holds
  500 of them, and on the smooth families the GP alone has the error of the
  tree with it (`matter_power`: 9.7e-6 on the test curves against 8.3e-6).
* **The network lowers it by a factor 2.3 to 6 on five families and raises
  it by up to 2.2 on the other seven**: 1.3 (0.45 to 6.1)
  in the geometric mean.
* **A leaf of `'aligned'` with an alignment models the curves themselves**,
  and the others the residual of the global model. The error is that of
  `'aligned'` alone where the leaves hold alignments and that of `'shared'`
  with the global model where they decline: with the GP
  3.6 (1.2 to 69) times lower than `'shared'` alone, and
  3.6 (1.1 to 103 by family) over three seeds. When the leaves still aligned
  the residual curves they declined, and `'aligned'` with the network had
  9.7e-3 on `zprime_spectrum`.
* **Coverage**: 0.63 to 0.72 with the GP and 0.53 to 0.72 with the network on
  the exact curves, and 0.95 and 0.93 on `peak_counts_mc`, where it is 0.99
  without a global model. It dips after each new version of the global
  model: on a stream of 400 curves of `matter_power`, with the GP fitted at
  200 and at 400, it was 0.48 over the last 200.
* **Time for the stream**: 4 to 16 s with `'shared'`, 18 to 68 s with the GP,
  which is fitted three times, and 17 to 41 s with the network.

3200 curves, NRMSE of the last 1600 predictions:

| family | shared | shared + GP | shared + net | aligned | aligned + GP |
|---|---:|---:|---:|---:|---:|
| `matter_power` | 3.2e-5 | 5.8e-6 | 3.9e-5 | 3.2e-5 | 5.8e-6 |
| `z_lineshape` | 9.5e-5 | 3.0e-5 | 2.1e-5 | 2.7e-5 | 2.7e-5 |
| `zprime_spectrum` | 3.3e-3 | 4.3e-3 | 4.5e-3 | 7.4e-4 | 7.4e-4 |
| `kinematic_edge`, 128 points | 3.7e-4 | 2.8e-4 | 4.5e-4 | 4.9e-5 | 4.9e-5 |
| `cmb_peaks` | 9.7e-5 | 2.9e-5 | 1.7e-5 | 5.5e-5 | 5.5e-5 |
| `peak_counts_mc` | 3.1e-3 | 2.6e-3 | 2.2e-3 | 3.1e-3 | 2.6e-3 |

* The gains over `'shared'` are 2 (0.78 to 5.5) for the GP,
  1.6 (0.74 to 5.8) for the network,
  2.5 (1 to 7.6) for `'aligned'` and
  3.4 (1.2 to 7.6) for `'aligned'` with the GP.
* **A leaf that could use either takes the alignment.** On `cmb_peaks` the
  alignment gains a factor 1.8 and the global models 3.3 and 5.7, and
  `'aligned'` with a global model has the gain of the alignment.

Six parameters (`zprime_spectrum_6d`, `eft_distribution_6d`), with the
default and with Matérn 5/2 leaves ("5/2"), 800 curves:

| family | shared | shared, 5/2 | shared + GP | aligned | aligned, 5/2 | aligned + GP, 5/2 | aligned + GP, 5/2 in both |
|---|---:|---:|---:|---:|---:|---:|---:|
| `zprime_spectrum_6d` | 6.5e-3 | 7.1e-3 | 5.8e-3 | 3.7e-3 | 3.6e-3 | 2.7e-3 | 2.7e-3 |
| `eft_distribution_6d` | 1.7e-3 | 7.8e-5 | 6.2e-7 | 1.7e-3 | 7.8e-5 | 6.2e-7 | 2.0e-6 |

3200 curves:

| family | shared | aligned | aligned, 5/2 | aligned + GP | aligned + GP, 5/2 |
|---|---:|---:|---:|---:|---:|
| `zprime_spectrum_6d` | 3.3e-3 | 1.2e-3 | 8.6e-4 | 1.0e-3 | 1.0e-3 |
| `eft_distribution_6d` | 1.3e-3 | 1.3e-3 | 4.7e-5 | 4.3e-7 | 4.3e-7 |

"GP, 5/2" has Matérn 5/2 leaves and the global GP with its default kernel, and
"5/2 in both" also the global GP with `nu=2.5` (section 10.8).

* **`eft_distribution_6d` is quadratic in its six coefficients**, which the
  additive kernel of order 2 of the global GP describes: with 800 curves its
  error is 850 to 2700 times lower than that of the default leaves and 40 to
  130 times lower than that of Matérn 5/2 leaves.
* **The alignment also gains with six parameters**: by a factor 1.8 with 800
  curves, where a leaf of 100 curves spans more of each parameter than with
  three, and 2.8 with 3200 (3.8 with Matérn 5/2 leaves).

Global networks that take `x` into account, each alone, fitted once to 800
curves, NRMSE on the test curves:

| family | one output per grid point | with a warp of its grid | grid coordinate as an input |
|---|---:|---:|---:|
| `matter_power` | 1.0e-4 | 3.6e-4 | 2.3e-3 |
| `z_lineshape` | 5.6e-4 | 4.2e-4 | 8.1e-4 |
| `zprime_spectrum` | 1.1e-2 | 4.4e-3 | 4.1e-3 |
| `kinematic_edge`, 128 points | 5.6e-3 | 2.6e-3 | 2.7e-3 |
| `cmb_peaks` | 2.2e-4 | 4.8e-4 | 1.4e-2 |
| `ringdown` | 2.4e-3 | 1.2e-2 | 7.1e-2 |
| `peak_counts_mc` | 4.0e-3 | 2.2e-3 | 5.5e-3 |

* **The grid coordinate as one more input**, `(theta, u) -> f(u; theta)`,
  is 2 to 2.7 times more accurate than one output per grid point on the two
  families with a sharp feature that moves and 20 to 65 times less accurate
  on `matter_power`, `cmb_peaks` and `ringdown`. Its fit takes 15 to 20 times
  as long.
* **A warp of the output grid**: a second head gives the knots of a monotone
  map of the grid, and the network's curve is read at the mapped positions.
  It is 2.2 to 2.5 times more accurate on the sharp features and 1.8 on the
  noisy histograms, and 2 to 5 times less accurate on `matter_power`,
  `cmb_peaks` and `ringdown`. As the global model of `'shared'` leaves it
  lowered the error on the stream by a factor 1.4 on `zprime_spectrum` and
  1.6 on `kinematic_edge` with 128 points, where `'aligned'` gives 4 and 14.

The curve as a single-output target with `x` as one more input: 200 curves
as `200 p` points of a stream in random order, with and without the network.
NRMSE on the test curves, and the time of the stream:

| family | tree on `(x, theta)` | with the network | `'shared'`, 5/2 | `'aligned'`, 5/2 |
|---|---:|---:|---:|---:|
| `z_lineshape` | 7.8e-4, 68 s | 6.8e-5, 147 s | 7.1e-5, 1.3 s | 2.4e-5, 2.8 s |
| `zprime_spectrum` | 1.6e-2, 149 s | 7.4e-3, 191 s | 1.4e-2, 1.7 s | 5.4e-3, 4.8 s |

With the network the single-output tree has the error of `'shared'` on
`z_lineshape` and half of it on `zprime_spectrum`, at 100 times the time. On a
stream it is 8 times better than with `fit` (section 10.5), which does not use
the fitted length scales for its splits.

A grid relative to a known position. 800 curves, `'shared'` and `'aligned'`
with the default leaves, RMSE of the last 400 predictions in the unit of the
curve:

| family and grid | shared | aligned |
|---|---:|---:|
| `zprime_spectrum`, 128 points in `m` [dex] | 0.102 | 0.025 |
| `zprime_spectrum`, 128 points in `m / M` [dex] | 0.015 | 0.0091 |
| `kinematic_edge`, 128 points in `m` [events] | 3.9e-3 | 2.9e-4 |
| `kinematic_edge`, 128 points in `m / endpoint` [events] | 3.0e-4 | 3.0e-4 |

On the grid in `m / endpoint` the edge does not move, every leaf declines the
alignment, and `'shared'` has the error that `'aligned'` has on the grid in
`m`. On the grid in `m / M` the peak does not move and the dip below it does.

### 10.8 Leaf kernels for curves

Streams of 800 curves as in section 10.1, one seed, with the default leaf
kernel (Matérn 3/2) and with Matérn 5/2 (`Default_GPR(nu=2.5)`, "5/2"), and
with the additive GP as the global model, with its default kernel ("GP, 5/2")
or with Matérn 5/2 in it too (`global_mean_kwargs=dict(nu=2.5)`, "5/2 in
both"). NRMSE of the last 400 predictions:

| family | shared | shared, 5/2 | shared + GP, 5/2 | shared + GP, 5/2 in both | aligned | aligned, 5/2 | aligned + GP, 5/2 | aligned + GP, 5/2 in both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `matter_power` | 5.0e-5 | 1.2e-5 | 1.4e-5 | 1.0e-5 | 5.0e-5 | 1.2e-5 | 1.4e-5 | 1.0e-5 |
| `eft_distribution` | 1.5e-4 | 2.5e-6 | 2.1e-6 | 1.3e-6 | 1.5e-4 | 2.5e-6 | 2.1e-6 | 1.3e-6 |
| `jet_spectrum` | 5.6e-5 | 1.7e-5 | 9.0e-6 | 7.2e-6 | 5.6e-5 | 1.7e-5 | 9.0e-6 | 7.2e-6 |
| `dust_sed` | 1.3e-5 | 3.2e-6 | 2.6e-6 | 2.6e-6 | 1.3e-5 | 3.2e-6 | 2.6e-6 | 2.6e-6 |
| `z_lineshape` | 2.6e-4 | 4.2e-5 | 7.4e-5 | 2.5e-5 | 6.4e-5 | 3.2e-5 | 2.3e-5 | 1.8e-5 |
| `zprime_spectrum` | 8.2e-3 | 8.9e-3 | 9.7e-3 | 9.8e-3 | 2.0e-3 | 2.1e-3 | 2.1e-3 | 2.1e-3 |
| `kinematic_edge` | 1.3e-3 | 8.2e-4 | 1.1e-3 | 9.3e-4 | 7.8e-4 | 1.0e-3 | 1.0e-3 | 1.0e-3 |
| `kinematic_edge`, 128 points | 1.4e-3 | 8.8e-4 | 1.1e-3 | 9.7e-4 | 1.0e-4 | 4.4e-5 | 4.4e-5 | 4.4e-5 |
| `oscillation_probability` | 1.1e-3 | 4.7e-4 | 4.3e-4 | 2.7e-4 | 1.1e-3 | 4.7e-4 | 4.3e-4 | 2.7e-4 |
| `cmb_peaks` | 1.9e-4 | 3.3e-5 | 1.1e-4 | 2.4e-5 | 1.2e-4 | 2.2e-5 | 2.4e-5 | 2.2e-5 |
| `form_factor` | 4.7e-3 | 4.9e-3 | 4.2e-3 | 4.2e-3 | 4.7e-3 | 4.9e-3 | 4.2e-3 | 4.2e-3 |
| `ringdown` | 2.8e-3 | 1.4e-3 | 1.8e-3 | 7.7e-4 | 8.8e-4 | 4.7e-4 | 4.7e-4 | 4.7e-4 |
| `peak_counts_mc` | 4.1e-3 | 3.9e-3 | 3.3e-3 | 3.0e-3 | 4.1e-3 | 3.9e-3 | 3.3e-3 | 3.0e-3 |

* **Matérn 5/2 leaves lower the error of `'shared'` by a factor
  3.1 (0.93 to 58)** in the geometric mean over the twelve
  families, at the same time per update. The largest gains are on the smooth
  families, and on `zprime_spectrum` and `form_factor` the error is 4 to 9%
  higher.
* **With `'aligned'` the gain over the default `'shared'` is
  4 (0.95 to 58)**, against
  1.5 (1 to 4.1) for `'aligned'` with the default leaves.
  On `kinematic_edge` with 64 points, where the grid hardly resolves the
  edge, half of the leaves keep an alignment and the error is 1.2 times
  that of `'shared'` with Matérn 5/2 leaves, and twice that on the test
  curves (section 10.6).
* **The global GP needs the smoother kernel too.** With its default kernel it
  raises the error of Matérn 5/2 `'shared'` leaves on six families, by up
  to a factor 3.3 (`cmb_peaks`). With Matérn 5/2 in both it lowers it on the
  ten families without a sharp feature, by a factor 1.2 to 2.4, and raises
  it by 10 to 13% on `zprime_spectrum` and `kinematic_edge`:
  4.3 (0.84 to 110) over the default `'shared'`.
* **`'aligned'` with Matérn 5/2 in the leaves and in the global GP** has
  5.3 (1.1 to 110) over the default `'shared'`: the
  lowest error of the eight columns on ten families, 5% above it on
  `zprime_spectrum` and 30% above it on `kinematic_edge` with 64 points. The
  global GP adds a factor 1.3 to `'aligned'` with Matérn 5/2 leaves in the
  geometric mean, between 1.0 and 2.3. Over three seeds the gain over the
  default `'shared'` is 5.3, 4.8 and 5.3, between 0.97 and 128 for a single
  family, and 4.0, 3.9 and 4.6 without the global GP.
* **Coverage** is 0.59 to 0.72 for these columns on the exact curves.

RBF leaves (`Default_GPR(nu=float('inf'))`), six families:

| family | shared | shared, 5/2 | shared, RBF | aligned | aligned, 5/2 | aligned, RBF |
|---|---:|---:|---:|---:|---:|---:|
| `matter_power` | 5.0e-5 | 1.2e-5 | 1.5e-5 | 5.0e-5 | 1.2e-5 | 1.5e-5 |
| `z_lineshape` | 2.6e-4 | 4.2e-5 | 1.9e-5 | 6.4e-5 | 3.2e-5 | 9.6e-6 |
| `cmb_peaks` | 1.9e-4 | 3.3e-5 | 8.3e-6 | 1.2e-4 | 2.2e-5 | 8.3e-6 |
| `kinematic_edge`, 128 points | 1.4e-3 | 8.8e-4 | 8.8e-4 | 1.0e-4 | 4.4e-5 | 3.9e-5 |
| `zprime_spectrum` | 8.2e-3 | 8.9e-3 | 6.1e-2 | 2.0e-3 | 2.1e-3 | 4.2e-3 |
| `peak_counts_mc` | 4.1e-3 | 3.9e-3 | 3.9e-3 | 4.1e-3 | 3.9e-3 | 3.9e-3 |

RBF leaves are 2 to 4 times better than Matérn 5/2 leaves on `z_lineshape`
and `cmb_peaks`, and 7 times worse on `zprime_spectrum` with `'shared'` and 2
times worse with `'aligned'`.

3200 curves, NRMSE of the last 1600 predictions:

| family | shared | shared, 5/2 | aligned | aligned, 5/2 | aligned + GP, 5/2 | aligned + GP, 5/2 in both |
|---|---:|---:|---:|---:|---:|---:|
| `matter_power` | 3.2e-5 | 2.1e-5 | 3.2e-5 | 2.1e-5 | 5.8e-6 | 5.8e-6 |
| `z_lineshape` | 9.5e-5 | 1.3e-5 | 2.7e-5 | 1.2e-5 | 1.1e-5 | 8.3e-6 |
| `zprime_spectrum` | 3.3e-3 | 3.4e-3 | 7.4e-4 | 8.4e-4 | 8.4e-4 | 8.4e-4 |
| `kinematic_edge`, 128 points | 3.7e-4 | 1.9e-4 | 4.9e-5 | 3.0e-5 | 3.0e-5 | 3.0e-5 |
| `cmb_peaks` | 9.7e-5 | 1.3e-5 | 5.5e-5 | 9.2e-6 | 1.7e-5 | 7.7e-6 |
| `peak_counts_mc` | 3.1e-3 | 3.0e-3 | 3.1e-3 | 3.0e-3 | 2.6e-3 | 2.4e-3 |

The gains over the default `'shared'` are 2.4 (0.97 to 7.6)
for Matérn 5/2 leaves, 4.3 (1.1 to 12) with `'aligned'`,
and 6.1 (1.3 to 13) with Matérn 5/2 in the global GP
too. On `matter_power` the error of Matérn 5/2 leaves is higher with 3200
curves than with 800 (2.1e-5 against 1.2e-5), and the global GP lowers it to
5.8e-6.

The eleven single-output targets of `examples/target_functions_physics.py`,
streams of 4000 uniform points, NRMSE of the last 2000 predictions: with
Matérn 5/2 leaves the error is 8 to 19% higher on eight targets and lower on
three (`eft_signal_strength` by a factor 2.1, `oscillation_probability` 1.3,
`signal_yield_mc` 1.1), 1.02 times lower in the geometric mean. Their
thresholds, cusps and resonances lie in the input space. With a curve as the
target such a feature lies along the grid, and the kernel only sees how the
curve changes with the parameters: `jet_spectrum` gains a factor 3.3 from
Matérn 5/2 leaves as a curve in `p_T`, and loses 18% with `p_T` as an input.

### 10.9 Curves from outside physics, and targets that are not a curve

Sixteen further targets with two to four inputs and 64 to 144 outputs, streams
of 800 as in section 10.1, two seeds, default leaves. NRMSE of the last 400
predictions (geometric mean of the seeds), the ratio for each seed, the share
of the leaves that hold an alignment, and the ratio with Matérn 5/2 leaves:

| target (inputs) | what moves | shared | aligned | shared / aligned, two seeds | leaves aligned | with Matérn 5/2 leaves |
|---|---|---:|---:|---:|---:|---:|
| infected fraction of an SIR epidemic against time (3) | a peak | 2.9e-3 | 9.2e-4 | 3.69, 2.72 | 1.00 | 3.50, 2.85 |
| travelling viscous shock (3) | a steep front | 1.2e-2 | 1.0e-3 | 11, 12 | 1.00 | 8.58, 9.57 |
| density of a mixture of two Gaussians (4) | two peaks, independently | 7.3e-3 | 1.2e-3 | 6.25, 5.83 | 1.00 | 9.27, 7.43 |
| gain of a second-order low-pass filter against frequency (3) | a resonance | 3.4e-4 | 4.3e-5 | 9.90, 6.62 | 1.00 | 6.92, 6.07 |
| elastic, then hardening (3) | a kink | 6.8e-4 | 5.2e-4 | 1.25, 1.35 | 0.60 | 1.24, 1.39 |
| logistic growth against time (3) | an inflection | 8.3e-4 | 8.5e-4 | 1.07, 0.90 | 0.88 | 1.04, 1.16 |
| Black-Scholes gamma against the strike (3) | a bell, a little | 1.1e-3 | 1.1e-3 | 1.03, 1.00 | 0.29 | 0.91, 1.00 |
| drug concentration after an oral dose (3) | a broad peak | 8.1e-4 | 8.1e-4 | 1.00, 1.00 | 0.04 | 1.00, 1.00 |
| Black-Scholes call price against the strike (3) | nothing | 2.4e-4 | 2.3e-4 | 1.00, 1.06 | 0.08 | 1.00, 1.00 |
| five fixed shapes with varying amplitudes (3) | nothing | 1.6e-3 | 1.6e-3 | 1.00, 1.00 | 0.00 | 1.00, 1.00 |
| a jump between two grid points (2) | a jump | 2.4e-2 | 2.4e-2 | 1.00, 1.00 | 0.00 | 1.00, 1.48 |
| a peak on a grid of 12 x 12 points, row by row (3) | a peak, in two directions | 2.6e-3 | 1.5e-3 | 1.83, 1.64 | 0.96 | 1.15, 0.98 |
| 64 unrelated functions of the inputs (3) | not a curve | 3.2e-3 | 3.2e-3 | 1.00, 1.00 | 0.00 | 1.00, 1.00 |
| `sir_epidemic` with its grid shuffled | not a curve | 2.9e-3 | 2.9e-3 | 1.00, 1.00 | 0.00 | 1.00, 1.00 |
| `viscous_shock` with its grid shuffled | not a curve | 1.2e-2 | 1.2e-2 | 1.00, 1.00 | 0.00 | 1.00, 1.00 |
| `z_lineshape` with its grid shuffled | not a curve | 2.4e-4 | 2.4e-4 | 1.00, 1.00 | 0.00 | 1.00, 1.00 |

* **What `'aligned'` needs is a feature that moves along an ordered grid**,
  whatever the origin of the curve: a factor 3 to 12 for the epidemic peak,
  the shock front, the two peaks of the mixture and the resonance of the
  filter.
* **Where nothing moves, or where the outputs are in no order, every leaf
  declines**, and the predictions are those of `'shared'`: the three
  shuffled grids and the unrelated outputs have a ratio of exactly 1.
* **In between the gain is small and can be a loss.** Over the 64 runs of
  this table (two kernels, two seeds) `'aligned'` was more than 5% worse
  than `'shared'` in two, by 10% at most, and on the test curves after the
  stream in one, by 21% (`stress_strain`).
* **A stream takes 1.3 to 2.8 times as long** as with `'shared'`.
* **A surface is not a curve.** Row by row, the peak on the grid of two
  dimensions gains a factor 1.7 with the default leaves and nothing with
  Matérn 5/2 leaves: a warp of one ordered grid follows a shift along the
  rows only. Section 10.12 has the warp of both axes.
* **Matérn 5/2 leaves and a jump.** They lower the error of `'shared'` on 13
  of these targets, by a factor 1.3 to 6.2, change nothing on the shock, and
  raise it by a factor 4 on the jump (0.10 against 0.024), where each output
  is a step in the parameters.

### 10.10 Other representations of the curve as the outputs (not in the code)

The outputs of the tree are the weights of a representation of the curve, in
place of its values on the grid, and a prediction is turned back into a curve:

* **B-spline coefficients**: the least-squares coefficients of the curve in a
  cubic B-spline basis of 48, 64 or 96 functions on uniform knots.
* **Arc-length knots**: for each curve one position per grid point, at equal
  steps of the curve's arc length (grid and values each scaled to a length of
  one), and the curve's values at these positions. The predicted curve is the
  interpolant through the predicted positions and values.

800 curves added with `batch_update_tree`, default leaves, one seed. NRMSE on
300 test curves; the first two columns are the curve on its grid:

| family | points in x | shared | aligned | 96 coefficients, shared | 64 coefficients, independent | arc-length knots, shared | 48 coefficients, aligned | 96 coefficients, aligned | arc-length knots, aligned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `zprime_spectrum` | 128 | 6.9e-3 | 1.8e-3 | 7.0e-3 | 7.7e-3 | 3.2e-3 | 3.7e-3 | 1.6e-3 | 3.2e-3 |
| `kinematic_edge` | 128 | 5.3e-4 | 8.0e-5 | 5.2e-4 | 1.1e-3 | 3.7e-4 | 2.4e-3 | 1.8e-4 | 1.9e-4 |
| `viscous_shock` | 128 | 8.0e-3 | 9.5e-4 | 8.4e-3 | | 2.4e-3 | 9.2e-3 | 2.3e-3 | 2.4e-3 |
| `cmb_peaks` | 128 | 1.9e-4 | 1.0e-4 | 1.9e-4 | | 4.5e-3 | 1.7e-4 | 9.8e-5 | 2.0e-3 |
| `sir_epidemic` | 97 | 1.5e-3 | 3.7e-4 | 1.5e-3 | 1.4e-3 | 1.3e-3 | 3.9e-4 | 4.9e-4 | 1.2e-3 |
| `mixture_density` | 96 | 5.3e-3 | 8.7e-4 | 5.3e-3 | 2.3e-3 | 8.3e-3 | 8.4e-4 | 8.5e-4 | 2.8e-3 |
| `z_lineshape` | 64 | 1.5e-4 | 4.0e-5 | 1.5e-4 | | 3.5e-4 | 3.8e-5 | 1.5e-4 | 3.5e-4 |
| `matter_power` | 64 | 5.2e-5 | 5.2e-5 | 5.0e-5 | | 3.0e-4 | 4.4e-4 | 5.0e-5 | 3.0e-4 |
| `peak_counts_mc` | 60 | 3.5e-3 | 3.5e-3 | 3.8e-3 | | 5.8e-3 | 9.5e-3 | 3.8e-3 | 5.8e-3 |

* **A fixed basis with a shared kernel is `'shared'`**: with 96 coefficients
  the error is within 5% of that on the grid (9% on the noisy histograms).
  The prediction is the same weighted sum of the leaf's curves, in other
  coordinates. A peak that moves along the grid passes from one basis function
  to the next, so each coefficient is as narrow a function of the parameters
  as the peak is of `x`.
* **A GP per coefficient** lowers the error on `mixture_density` by a factor
  2.3 and on none of the other three, at about 15 times the time. With 64
  coefficients the basis itself has an error of 2e-3 on `zprime_spectrum` and
  8e-4 on `kinematic_edge`.
* **Arc-length knots follow a feature without a fit**: against `'shared'` the
  error is lower by a factor 2.1 on `zprime_spectrum`, 3.3 on `viscous_shock`,
  1.4 on `kinematic_edge` and 1.2 on `sir_epidemic`. It is higher by 1.6 on
  `mixture_density`, 1.7 on the noisy histograms and 23 on `cmb_peaks`. With
  two positions per grid point the error on `z_lineshape` is 1.2e-4 and on
  `matter_power` 9.6e-5, and the others change by less than 5%. `'aligned'`
  on the grid is lower on all nine. In a cube of side 0.15 of the parameter
  range, the part of the knots' values that is not linear in the parameters
  is 10 to 17 times smaller than that of the grid values for
  `kinematic_edge` and `viscous_shock`, 2.7 and 1.6 times smaller for
  `zprime_spectrum` and `mixture_density`, and 4 times larger for the eight
  peaks of `cmb_peaks`.
* **`'aligned'` leaves on the coefficients**, which are in the order of the
  grid. With 48 coefficients the error is within 5% of that of `'aligned'`
  on the grid for `sir_epidemic`, `mixture_density` and `z_lineshape`, in 9
  to 11 s against 13 to 18 s. It is 2 to 30 times higher where 48 functions
  do not represent the curve: the basis alone has an error of 3e-3 on
  `zprime_spectrum`, 2e-3 on `kinematic_edge`, 7e-3 on `viscous_shock` and
  4e-4 on `matter_power`. With 96 coefficients for 128 grid points the
  error is that of the grid on `zprime_spectrum` and `cmb_peaks`, and 2.2
  and 2.4 times higher on `kinematic_edge` and `viscous_shock`. On the
  arc-length knots it is never below `'aligned'` on the grid.
* **Time of the 800 curves**: 4 to 8 s for `'shared'` and 7 to 23 s for
  `'aligned'` on the grid, 86 to 128 s with a GP per coefficient, 6 to 13 s
  for the arc-length knots, 6 to 11 s and 10 to 18 s for `'aligned'` on 48
  and 96 coefficients.

### 10.11 Parameters with mixed effects, and oblique leaves

A family is evaluated in a rotated cube of its parameter space, so that each
input of the tree is a combination of the family's parameters:
`theta_family = 0.5 + (theta - 0.5) R / s` with a random rotation `R` and `s`
the largest column sum of `|R|`, for which the cube stays inside the unit
cube. The reference is the cube of the same size along the family's own axes
(`R = 1`). 800 curves added with `batch_update_tree`, default leaves and
`Default_GPR(oblique=True)`, two rotations. NRMSE on 300 test curves,
geometric mean of the two rotations:

| family | leaves | own axes | own axes, oblique | rotated | rotated, oblique | rotated / own | rotated / rotated oblique | rotated oblique / own |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `zprime_spectrum` | shared | 3.5e-3 | 2.8e-3 | 1.3e-2 | 4.7e-3 | 3.9 | 2.9 | 1.3 |
| | aligned | 3.1e-4 | 3.2e-4 | 1.2e-3 | 6.3e-4 | 3.8 | 1.9 | 2.0 |
| `sir_epidemic` | shared | 3.4e-4 | 1.8e-4 | 6.9e-4 | 3.8e-4 | 2.0 | 1.8 | 1.1 |
| | aligned | 7.5e-5 | 7.5e-5 | 1.9e-4 | 1.1e-4 | 2.6 | 1.7 | 1.5 |
| `cmb_peaks` | shared | 5.5e-5 | 5.5e-5 | 1.2e-4 | 7.4e-5 | 2.2 | 1.7 | 1.3 |
| | aligned | 3.5e-5 | 3.5e-5 | 5.4e-5 | 4.7e-5 | 1.5 | 1.2 | 1.3 |
| `matter_power` | shared | 2.6e-5 | 2.6e-5 | 5.8e-5 | 3.0e-5 | 2.2 | 1.9 | 1.1 |
| | aligned | 2.6e-5 | 2.6e-5 | 5.8e-5 | 3.0e-5 | 2.2 | 1.9 | 1.1 |
| `mixture_density` | shared | 1.0e-3 | 1.0e-3 | 2.2e-3 | 1.4e-3 | 2.1 | 1.6 | 1.3 |
| | aligned | 1.7e-4 | 1.7e-4 | 2.0e-4 | 2.0e-4 | 1.2 | 1.0 | 1.2 |
| `kinematic_edge`, 128 points | shared | 1.6e-4 | 1.6e-4 | 5.1e-3 | 4.2e-4 | 32 | 12 | 2.7 |
| | aligned | 4.4e-5 | 4.4e-5 | 7.6e-5 | 5.1e-5 | 1.7 | 1.5 | 1.2 |
| `z_lineshape` | shared | 6.0e-5 | 6.0e-5 | 2.5e-4 | 6.2e-5 | 4.2 | 4.1 | 1.0 |
| | aligned | 1.7e-5 | 1.7e-5 | 3.0e-5 | 2.1e-5 | 1.8 | 1.5 | 1.2 |

* **The rotation costs `'shared'` a factor 2.0 to 4.2**, and 32 on
  `kinematic_edge`, whose curves depend on one of the three parameters far
  more than on the others. `'aligned'` loses a factor 1.2 to 3.8: its warp is
  linear in the offsets in any direction.
* **Oblique leaves recover a factor 1.6 to 4.1 for `'shared'`** (12 on
  `kinematic_edge`) and 1.0 to 1.9 for `'aligned'`, and leave the error
  within a factor 1.0 to 1.3 (2.7) and 1.1 to 2.0 of that on the family's own
  axes.
* **On the family's own axes** the leaves of five families declined the
  shear, with unchanged predictions. Those of `zprime_spectrum` and
  `sir_epidemic` kept it with `'shared'`, for an error 1.2 and 1.9 times
  lower. With `'aligned'` the error changed by 2% at most.
* **The kernel alone, in one leaf.** 100 curves of five of the families in a
  cube of side 0.5 or 0.25 of the rotated space, 300 test curves, one
  shared-kernel GP: the oblique kernel has 1.4 to 5.8 times the accuracy of
  the default kernel, and 1.0 to 1.9 times that of the default kernel on the
  parameters turned back onto the family's own axes.
* **The kernel without the cuts**, in the tree: a factor 1.4 to 3.1 for
  `'shared'` on the rotated families, and an error 1.3 to 3.1 times that on
  the own axes (10 on `kinematic_edge`).
* **Time of the 800 curves**: 1.1 to 2.8 times that of the default leaves.

The eleven single-output targets of `examples/target_functions_physics.py`,
2000 points added with `batch_update_tree`, one rotation. NRMSE on 1000 test
points, and the time of the 2000 points on the own axes:

| target | inputs | own axes | own axes, oblique | rotated | rotated, oblique | rotated / own | rotated / rotated oblique | s, default | s, oblique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `resonance_xsec` | 4 | 2.4e-2 | 1.8e-2 | 2.9e-2 | 8.6e-3 | 1.2 | 3.4 | 15 | 34 |
| `dilepton_spectrum` | 4 | 2.3e-2 | 7.2e-3 | 4.2e-2 | 8.0e-3 | 1.9 | 5.3 | 10 | 35 |
| `jet_spectrum` | 4 | 1.1e-4 | 1.1e-4 | 7.5e-4 | 2.2e-4 | 6.7 | 3.5 | 18 | 41 |
| `oscillation_probability` | 5 | 1.8e-3 | 1.4e-3 | 4.8e-2 | 9.9e-3 | 27 | 4.9 | 9 | 39 |
| `branching_ratio` | 5 | 3.5e-2 | 2.6e-2 | 1.5e-1 | 7.1e-2 | 4.2 | 2.1 | 11 | 40 |
| `loop_decay_width` | 4 | 1.9e-3 | 1.8e-3 | 1.1e-2 | 2.6e-3 | 5.8 | 4.3 | 10 | 29 |
| `relic_density` | 4 | 1.2e-2 | 3.1e-3 | 4.3e-2 | 3.7e-3 | 3.7 | 12 | 16 | 42 |
| `neutralino_mass` | 4 | 9.2e-4 | 1.1e-3 | 3.8e-3 | 1.3e-3 | 4.2 | 2.9 | 10 | 31 |
| `eft_signal_strength` | 6 | 1.6e-3 | 1.8e-3 | 1.5e-3 | 1.7e-3 | 1.0 | 0.9 | 10 | 52 |
| `signal_yield_mc` | 4 | 2.1e-3 | 2.1e-3 | 2.2e-3 | 2.3e-3 | 1.0 | 1.0 | 11 | 17 |
| `global_fit_loglike` | 10 | 5.5e-3 | 5.9e-3 | 6.6e-2 | 4.2e-2 | 12 | 1.6 | 18 | 137 |

* **On the rotated inputs** oblique leaves lower the error by a factor 2.1
  to 12 on eight targets and 1.6 on the target with ten inputs. On the two
  that the rotation does not make harder they change it by 5 to 10%.
* **On the targets' own parameters** the error is lower on six targets, by a
  factor 1.3 to 3.8 on five, and higher on three, by 7 to 20%. With one
  output the leave-one-out error of 100 points decides on up to 45 entries
  of the shear.
* **The 2000 points take 1.5 to 5 times as long**, and 8 times with ten
  inputs.
* **The coverage** is 0.61 to 0.77 with the default and with the oblique
  leaves (0.96 to 0.98 for the noisy target).

A rotation of the whole parameter space, in place of a kernel per leaf (not
in the code): the eigenvectors of the mean of `J J^T`, with `J` the slopes of
the curves in the parameters from linear fits to the 18 to 24 nearest
curves, as the inputs of a tree with default leaves. Same curves and
rotations:

| family | rotated / rotated with the learned rotation, shared | aligned | own axes / own axes with the learned rotation, shared | aligned |
|---|---:|---:|---:|---:|
| `zprime_spectrum` | 2.7 | 2.1 | 1.2 | 1.1 |
| `sir_epidemic` | 1.5 | 1.3 | 1.3 | 0.7 |
| `cmb_peaks` | 1.7 | 1.4 | 0.8 | 1.3 |
| `matter_power` | 1.2 | 1.2 | 0.4 | 0.4 |
| `mixture_density` | 1.3 | 0.5 | 0.9 | 0.9 |
| `kinematic_edge`, 128 points | 12 | 1.9 | 0.8 | 1.4 |
| `z_lineshape` | 4.2 | 1.7 | 1.0 | 1.1 |

The directions in which the curves change most are not always those of their
length scales: on the own axes of `matter_power` this rotation makes the
error 2.4 times higher.

### 10.12 Functions on a grid of two dimensions

Five families of functions on a grid of 32 x 32 points (surfaces) with three
parameters each, four of them in `examples/multioutput_surfaces.py`: a peak
whose centre moves along both axes and whose width changes (`peak`), two
peaks side by side that move in opposite directions along the second axis
(`two_peaks`), a ring about the centre of the grid with a radius that
changes (`ring`), a resonance along the first axis at a position that moves,
on a spectrum that falls along the second axis at a rate that changes
(`ridge`), and three fixed shapes with varying amplitudes (`smooth`). 800
surfaces added with `batch_update_tree`, default leaves, one seed. NRMSE on
300 test surfaces, for `'shared'`, for `'aligned'` with the surface taken as
a curve of 1024 points, and for `'aligned'` with `output_shape=(32, 32)`:

| family | shared | aligned, as a curve | aligned, as a surface | shared / as a curve | shared / as a surface | leaves aligned, as a surface |
|---|---:|---:|---:|---:|---:|---:|
| `peak` | 2.4e-3 | 6.3e-4 | 3.0e-4 | 3.9 | 8.2 | 1.00 |
| `two_peaks` | 6.6e-4 | 2.2e-4 | 1.1e-4 | 3.0 | 5.8 | 1.00 |
| `ring` | 5.1e-4 | 5.1e-4 | 2.6e-4 | 1.0 | 2.0 | 1.00 |
| `ridge` | 4.0e-4 | 4.0e-4 | 4.0e-4 | 1.0 | 1.0 | 0.00 |
| `smooth` | 1.6e-6 | 1.6e-6 | 1.6e-6 | 1.0 | 1.0 | 0.00 |

* **The warp of both axes lowers the error by a factor 2.0 to 8.2** on the
  three families with a feature that moves across the grid, and its leaves
  decline on the other two, with the predictions of `'shared'`.
* **As a curve** the surface gains where the feature moves along the rows,
  which is the order of the outputs: half as much as with the warp of both
  axes for the two families with peaks, and nothing for the ring.
* **The coverage** is 0.61 to 0.69 for the three.

Single leaves: 100 surfaces in a cube of side 0.5 or 0.3 of the parameter
range, 200 test surfaces in the same cube, three positions of the cube, one
shared-kernel GP on the surfaces or on the aligned surfaces. NRMSE on the test
surfaces, and the ratio of the leave-one-out errors that decides on the
alignment:

| family | side | shared | aligned | shared / aligned | the same from leave-one-out |
|---|---:|---:|---:|---:|---:|
| `peak` | 0.5 | 2.8e-3 | 2.3e-4 | 12 | 14 |
| | 0.3 | 8.7e-4 | 6.9e-5 | 13 | 14 |
| `two_peaks` | 0.5 | 7.1e-4 | 1.2e-4 | 6.0 | 6.3 |
| | 0.3 | 2.2e-4 | 4.2e-5 | 5.2 | 5.5 |
| `ring` | 0.5 | 1.2e-3 | 4.7e-4 | 2.6 | 2.7 |
| | 0.3 | 3.6e-4 | 2.1e-4 | 1.7 | 1.9 |
| `ridge` | 0.5 | 2.1e-4 | 2.0e-4 | 1.1 | 1.1 |
| | 0.3 | 5.9e-5 | 6.3e-5 | 0.9 | 0.9 |
| `smooth` | 0.5 | 4.3e-6 | 3.7e-6 | 1.2 | 1.2 |
| | 0.3 | 2.2e-6 | 2.2e-6 | 1.0 | 1.0 |

* **The bound and the ridge of the warp.** With the hard bound of the curves
  on the exponent of the warp and their ridge of 0.001, the leaves of `peak`
  with side 0.5 had the test error of `'shared'` (2.8e-3) at a leave-one-out
  error of 1.0e-3: the median error of a test surface was that of the
  leave-one-out predictions, and the largest 300 times the median, at the
  corners of the cube. With the `tanh` bound and ridges of 0.001, 0.01, 0.03,
  0.1 and 0.3 the test error was 7.5e-4, 2.9e-4, 2.3e-4, 2.6e-4 and 3.6e-4,
  and for `two_peaks` 2.0e-4, 1.4e-4, 1.2e-4, 1.1e-4 and 1.5e-4. The code has
  0.03.
* **The aligned surface read back, against each surface warped into the frame of
  the query** (with the hard bound, side 0.25): with the aligned surfaces on
  the grid itself the error was 6.6e-5 against 3.9e-5 for `peak` and 1.0e-4
  against 3.2e-5 for `two_peaks`. On a grid with twice the points along each
  axis it was 4.0e-5 against 4.0e-5 and 3.9e-5 against 3.3e-5, at 7 ms for
  a prediction against 80 ms.
* **A warp of each axis that is the same along the other one** cannot move
  the two peaks of `two_peaks` in opposite directions. The network below has
  such a warp, and no gain on that family.
* **Time on one core**, 800 surfaces of `peak` and `two_peaks`: 46 to 48 s for
  `'shared'`, 106 to 116 s for `'aligned'` as a curve and 178 to 204 s as an
  surface. A prediction takes 0.1 ms, 10 ms and 5 to 6 ms.
* **The same ridge for curves** changes little: with 0.03 in place of 0.001
  in the warp of a curve, the error of `'aligned'` on 800 curves was 1.3
  times lower for `zprime_spectrum`, 1.1 times higher for `kinematic_edge`
  on 64 points, and within 3% for five other families (two seeds). The warp
  of a curve is unchanged.

A global warp from a network, in place of a warp per leaf (not in the code).
A network with two hidden layers of 64 units maps the parameters to the knots
of the warp, and is trained so that curves at neighbouring parameters agree
after the warp: the squared difference of each curve and its 8 nearest
neighbours, each read at its own warp and integrated over the grid of either
curve, over the squared distance of their parameters. A tree with `'shared'`
leaves learns the aligned curves on a grid three times as fine (twice for
surfaces), and its prediction is read back through the warp of the query. 800
curves or surfaces, one seed, NRMSE on 300 test curves:

| family | shared | aligned | network warp, shared leaves |
|---|---:|---:|---:|
| `zprime_spectrum` | 6.9e-3 | 1.8e-3 | 1.5e-3 |
| `viscous_shock` | 8.0e-3 | 9.5e-4 | 1.0e-3 |
| `kinematic_edge`, 128 points | 5.3e-4 | 8.0e-5 | 2.4e-4 |
| `mixture_density` | 5.3e-3 | 8.7e-4 | 2.0e-3 |
| `sir_epidemic` | 1.5e-3 | 3.7e-4 | 5.9e-4 |
| `z_lineshape` | 1.5e-4 | 4.0e-5 | 1.8e-4 |
| `cmb_peaks` | 1.9e-4 | 1.0e-4 | 2.1e-4 |
| `matter_power` | 5.2e-5 | 5.2e-5 | 1.7e-3 |
| `peak` (surface) | 2.4e-3 | 3.0e-4 | 4.4e-4 |
| `two_peaks` (surface) | 6.6e-4 | 1.1e-4 | 7.3e-4 |
| `ring` (surface) | 5.1e-4 | 2.6e-4 | 4.3e-4 |
| `ridge` (surface) | 4.0e-4 | 4.0e-4 | 9.4e-4 |

For the surfaces the network gave one warp per axis, the same along the other
axis. The network took 45 to 134 s to train for the curves, where the whole
`'aligned'` tree takes 13 to 23 s, and 22 minutes for the surfaces. It has no
test that declines it: where nothing moves the error is up to 33 times that
of `'shared'`.

## 11. Saving and loading

Streams of uniform points in the unit cube, plain tree with
`retrain_every_n_points=25`, and the kernel hyperparameters fixed
(`Default_GPR(optimizer=None)`), which leaves the size of the tree as it is
and lets a million points be streamed in 11 minutes. "joblib" is the tree
written with `joblib.dump` as it is in memory and read with `joblib.load`.
"`save`" is `save` and `GPTree.load`. The times are with the file in the page
cache; read from the disk of the container, a load took up to 2 s longer with
joblib and up to 0.3 s longer with `save`. The last column is the size of the
loaded tree by its own estimate (`memory_mb`).

| points | d | `Nbar` | | leaves | | file MB | save s | load s | loaded tree MB |
|---|---|---|---|---|---|---|---|---|---|
| 300 000 | 8 | 100 | | 4340 | joblib | 560 | 9.1 | 6.1 | 586 |
| | | | | | `save` | 103 | 0.92 | 0.43 | 156 |
| 1 000 000 | 8 | 100 | | 14 613 | joblib | 1878 | 35.3 | 23.9 | 1967 |
| | | | | | `save` | 346 | 3.6 | 2.0 | 525 |
| 300 000 | 12 | 50 | | 8753 | joblib | 492 | 18.3 | 13.6 | 532 |
| | | | | | `save` | 168 | 1.8 | 1.0 | 278 |
| 300 000 | 8 | 200 | | 2098 | joblib | 851 | 4.3 | 2.9 | 863 |
| | | | | | `save` | 87 | 0.47 | 0.24 | 107 |
| 300 000 | 8 | 100 | `max_memory_mb=200` | 2990 | joblib | 182 | 6.4 | 4.0 | 200 |
| | | | | | `save` | 71 | 0.57 | 0.42 | 107 |
| 300 000 | 8 | 100 | `predict_from='neighbours'` | 4340 | joblib | 170 | 25.9 | 18.1 | 157 |
| | | | | | `save` | 110 | 1.4 | 0.63 | 156 |

The first save of a tree, which puts its leaves to sleep, took as long as the
later ones. A loaded tree made its first 1000 predictions, at points close to
1000 of the stream's, in 0.35, 0.5 and 1.0 s at `Nbar=50`, 100 and 200, and
the same predictions again in 0.16 s: the first ones woke 800 to 950 leaves.
They differed from those of the tree before it was saved by at most 1.3e-16
of the target's range in the mean, and by a factor within 2.4e-15 of one in
the sigma.

What each part of `save` gives, for the tree of the first row. The last two
rows are prototypes of the file alone, in which the state of each leaf is a
pickle of its own that a loaded tree would unpickle when it first uses the
leaf. Their time to save is without the 0.8 s of pickling the leaves, as for
a tree that keeps the pickles of the leaves it has not changed.

| | file MB | save s | load s |
|---|---|---|---|
| joblib, leaves awake | 560 | 9.1 | 5.8 |
| joblib, leaves asleep | 130 | 8.5 | 5.3 |
| pickle, leaves awake | 548 | 2.2 | 1.8 |
| pickle, leaves asleep | 118 | 1.7 | 1.6 |
| and one random state for all GPs, where 2850 of the 4340 leaves had a copy | 103 | 1.3 | 1.2 |
| and without the garbage collector (`save`) | 103 | 0.88 | 0.34 |
| each leaf a pickle of its own (not in the code) | 134 | 0.41 | 0.21 |
| and the file mapped into memory (not in the code) | 134 | 0.46 | 0.14 |

With predictions from neighbours a node also keeps the errors of its last 50
points (`_scores`). On a stream of 60 000 points (885 leaves), saved as the
two arrays per point that they are: 25.2 MB, 0.49 s to save and 0.29 s to
load. Saved as one array per node: 22.4 MB, 0.25 s and 0.10 s.

## 12. Choosing the training points (`Trainer`)

A `Trainer` with its defaults (2000 candidates, one random check in ten
points, the first 200 points random) on `GPTree(Nbar=100)` with its defaults,
which suggests the points in batches of 50, uniform in the unit cube. It is
compared with a tree of the same settings that is given as many uniform random
points, in the same batches. 4000 evaluations (2000 in two dimensions, 6000
for `global_fit_loglike`), exact observations with a declared noise of
`1e-3 |y|`, and a tolerance `atol` of 2% of the standard deviation of the
target over the cube. The errors are those of the finished trees at 5000
uniform test points, or at 2000 test points in the range. Three seeds; a ratio
is the geometric mean of the ratios of the seeds, with the same test points
for both trees of a seed.

### 12.1 A tree that is to be accurate over the whole box

| target | d | RMSE, uniform points | RMSE, `Trainer` | ratio | over the seeds | ratio of the 95% error | coverage, uniform points | coverage, `Trainer` |
|---|---|---|---|---|---|---|---|---|
| `Chirp` | 3 | 0.502 | 0.436 | 0.87 | 0.82 to 0.94 | 0.84 | 0.69 | 0.71 |
| `eft_signal_strength` | 6 | 0.00877 | 0.00732 | 0.83 | 0.78 to 0.89 | 0.80 | 0.65 | 0.67 |
| `GaussianPeaks` | 2 | 0.00555 | 0.00189 | 0.34 | 0.23 to 0.65 | 0.41 | 0.65 | 0.72 |
| `GaussianPeaks` | 4 | 0.19 | 0.14 | 0.74 | 0.66 to 0.78 | 0.74 | 0.67 | 0.69 |
| `GaussianPeaks` | 6 | 0.511 | 0.482 | 0.94 | 0.91 to 0.98 | 0.90 | 0.69 | 0.67 |
| `Himmelblau` | 4 | 0.0623 | 0.0596 | 0.96 | 0.92 to 0.98 | 0.96 | 0.65 | 0.64 |
| `Michalewicz` | 4 | 0.266 | 0.277 | 1.04 | 0.95 to 1.10 | 1.05 | 0.68 | 0.65 |
| `relic_density` | 4 | 0.177 | 0.0945 | 0.54 | 0.51 to 0.59 | 0.48 | 0.65 | 0.64 |
| `resonance_xsec` | 4 | 0.248 | 0.166 | 0.67 | 0.58 to 0.81 | 0.52 | 0.67 | 0.66 |
| `RotatedRosenbrock` | 4 | 1.21e+04 | 7.07e+03 | 0.59 | 0.50 to 0.67 | 0.74 | 0.67 | 0.67 |
| `StepRidge` | 4 | 0.25 | 0.231 | 0.92 | 0.88 to 0.99 | 0.88 | 0.67 | 0.65 |
| geometric mean | | | | 0.73 | | 0.73 | | |

The 95% error is the error that 95% of the predictions stay below.

### 12.2 A tree that is to be accurate where the target lies in a range

The errors and the shares within the tolerance are those at test points in the
range. "Points to the final RMSE" is the first of the checks, made every 500
points, at which the tree of the `Trainer` has the RMSE that the tree with
uniform points has at the end. The last two columns are over the whole box,
with the coverage of the tree with uniform points in brackets.

| target | d | range | share of the box | RMSE, uniform points | ratio | over the seeds | ratio of the 95% error | within the tolerance, uniform points | within the tolerance, `Trainer` | points to the final RMSE of uniform points | RMSE over the box, ratio | coverage over the box |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `GaussianPeaks` | 4 | below 3 | 10.3% | 0.114 | 0.35 | 0.25 to 0.51 | 0.18 | 0.87 | 0.99 | 1000 of 4000 | 3.61 | 0.41 (0.67) |
| `GaussianPeaks` | 6 | below 2.02 | 1.0% | 0.297 | 0.36 | 0.23 to 0.81 | 0.24 | 0.56 | 0.94 | 1000 of 4000 | 2.06 | 0.43 (0.67) |
| `GaussianPeaks` | 6 | below 3.53 | 5.0% | 0.299 | 0.51 | 0.37 to 0.78 | 0.32 | 0.54 | 0.87 | 1500 of 4000 | 2.26 | 0.46 (0.68) |
| `global_fit_loglike` | 10 | above -5 | 0.6% | 1.61 | 0.36 | 0.26 to 0.47 | 0.25 | 0.35 | 0.60 | 4000 of 6000 | 1.29 | 0.70 (0.70) |
| `global_fit_loglike` | 10 | above -8 | 4.0% | 1.47 | 0.58 | 0.43 to 0.81 | 0.40 | 0.34 | 0.50 | 3000 of 6000 | 1.33 | 0.55 (0.69) |
| geometric mean | | | | | 0.42 | | 0.27 | | | | | |

In three of the fifteen runs the ratio of the RMSE is about 0.8 (0.78 to
0.81), where the other seeds have 0.23 to 0.57. Their 95% errors are in line
with the other seeds (0.28 to 0.34). In the two of them that were looked at
(`GaussianPeaks` in six dimensions, the third seed of both ranges), 41 and 23
of the 2000 test points have an error above five times the tolerance and
carry 87% and 83% of the squared error. The tree predicts them outside the
range, with a probability of the range below 0.1 for 88% and 57% of them, and
with a median error of 9.8 and 8.4 sigma. No such point is found in the first
seed.

### 12.3 The report against the finished tree

`Trainer.report()` at the end of the runs of sections 12.1 and 12.2, against
the same quantities of the finished tree at the test points in the range. The
report has 172 to 200 checks without a range and 76 to 110 checks in the
range with one. The true share of a range is that of two million uniform
points.

| | without a range (33 runs) | with a range (15 runs) |
|---|---|---|
| `share_in_range`, over the true share | | 0.76 to 1.21, median 1.06 |
| `share_above_tolerance`, less the share of the test points with a sigma above the tolerance | -0.01 to 0.01 | -0.06 to 0.04, and -0.25 in one run |
| `share_within_tolerance`, less that of the test points | -0.09 to 0.02, median -0.02 | -0.20 to 0.05, and -0.36 in one run; median -0.02 |
| `coverage`, less that of the test points | -0.09 to 0.09 | -0.10 to 0.12, and -0.31 in one run |
| `rms_error`, over that of the test points | 0.57 to 11, median 1.15 | 0.26 to 2.0, median 0.89 |

The checks are those of the last 2000 points, so the report lags behind a
tree whose error still falls: the factor 11 is `GaussianPeaks` in two
dimensions.

### 12.4 Settings of the `Trainer`

Four of the ranges of section 12.2 (not `global_fit_loglike` above -5), four
seeds. The RMSE and the 95% error within the range, relative to all checks
uniform, as the geometric mean over the four ranges; the share of the test
points within the tolerance, as their mean; and the checks in the range that
the report has at the end.

| | RMSE | 95% error | within the tolerance | checks in the range |
|---|---|---|---|---|
| all random checks uniform | 1.00 | 1.00 | 0.82 | 2 to 21 |
| half of them by the probability of the range (`CHECKS_BY_RANGE`) | 1.05 | 0.94 | 0.83 | 88 to 107 |
| all uniform, twice the sigma in the probability of the range (not in the code) | 1.01 | 1.04 | 0.81 | 2 to 21 |
| half by the range, twice the sigma (not in the code) | 1.13 | 1.02 | 0.82 | 83 to 106 |

A probability proportional to the square of the need (not in the code), three
seeds: a ratio to uniform points of 0.69 in place of 0.73 over the eleven
targets of section 12.1, and of 0.39 in place of 0.42 over the five ranges of
section 12.2. Per target the square has 0.60 to 1.19 times the error of the
need itself.

### 12.5 Other ways to draw the points: the prototype (not in the code)

A prototype outside the package, before the `Trainer`: 200 points of a Sobol
sequence first, then batches of 50 of which 5 are uniform random points and
45 are chosen from 2000 uniform candidates. Two seeds, the targets and the
numbers of points of sections 12.1 and 12.2, and the ratio of the RMSE to
that with uniform random points.

Over the whole box, by the sigma alone:

| the 45 points | geometric mean of eleven targets | lowest | highest |
|---|---|---|---|
| a Sobol sequence, for all 50 | 0.85 | 0.43 | 1.14 |
| the highest sigma | 0.72 | 0.19 | 1.22 |
| the highest sigma, with the variance of the other candidates of a leaf reduced for each point picked in it | 0.73 | 0.26 | 1.13 |
| drawn with a probability proportional to the variance | 0.74 | 0.41 | 1.06 |
| drawn with a probability proportional to its square | 0.68 | 0.25 | 1.07 |

The reduction of the variance is that of the leaf's GP with the point added,
which does not depend on the target value there. Sobol points and uniform
points differ by up to 14% on targets where neither should gain, which is the
spread of two seeds.

Within a range, by the variance times the probability of the range, and by
the variance alone:

| target | range | highest | highest, variance reduced | drawn | highest variance alone |
|---|---|---|---|---|---|
| `GaussianPeaks`, 4 | below 3 | 0.27 | 0.23 | 0.29 | 0.93 |
| `GaussianPeaks`, 6 | below 2.02 | 0.25 | 0.26 | 0.28 | 1.38 |
| `GaussianPeaks`, 6 | below 3.53 | 0.33 | 0.31 | 0.38 | 1.24 |
| `global_fit_loglike` | above -5 | 0.39 | 0.41 | 0.39 | 0.89 |
| `global_fit_loglike` | above -8 | 0.48 | 0.47 | 0.47 | 0.79 |

With 20 000 candidates in place of 2000, the highest need on
`global_fit_loglike` above -5 gave an RMSE in the range of 0.63 and 0.49 on
the two seeds, against 0.42 and 0.73.

Where the points with the highest sigma go, in one run of 3000 points each:

| target | d | largest number of the 45 points in one leaf, median over the batches | at the edge of the box | RMS error of the picked points over that of the random points |
|---|---|---|---|---|
| `relic_density` | 4 | 23 | 0.21 (0.15) | 7.1 |
| `resonance_xsec` | 4 | 19 | 0.27 (0.15) | 2.0 |
| `GaussianPeaks` | 6 | 18 | 0.47 (0.22) | 1.7 |
| `Michalewicz` | 4 | 26 | 0.24 (0.15) | 1.15 |

"At the edge" is the share of the picked points with an input within 2% of
its bounds, with that of uniform points in brackets. The errors are those of
the predictions made before the points were added, over the second half of
the run. The sigma covered 0.69 to 0.73 of the errors of the picked points
and 0.55 to 0.66 of those of the random points.

### 12.6 With predictions from neighbours

The runs of sections 12.1 and 12.2 on three targets and two ranges, with
`predict_from='neighbours'` and with predictions from the leaves, two seeds.
The RMSE over the box or within the range; the last column is the tree that
predicts from neighbours with a `Trainer` against the tree that predicts from
its leaves with uniform points.

| target | d | range | leaves, uniform points | leaves, `Trainer` | neighbours, uniform points | neighbours, `Trainer` | ratio with neighbours | last over first |
|---|---|---|---|---|---|---|---|---|
| `GaussianPeaks` | 4 | | 0.197 | 0.141 | 0.133 | 0.083 | 0.62 | 0.42 |
| `GaussianPeaks` | 6 | | 0.512 | 0.473 | 0.352 | 0.317 | 0.90 | 0.62 |
| `relic_density` | 4 | | 0.187 | 0.103 | 0.164 | 0.0675 | 0.41 | 0.36 |
| `GaussianPeaks` | 4 | below 3 | 0.113 | 0.048 | 0.0558 | 0.0113 | 0.20 | 0.10 |
| `GaussianPeaks` | 6 | below 3.53 | 0.311 | 0.128 | 0.169 | 0.0511 | 0.30 | 0.16 |

With predictions from neighbours the sigma covers 0.66 to 0.75 of the errors
of the tree of the `Trainer`, over the box or within the range.

### 12.7 Cost

One process, `GaussianPeaks` in four and six dimensions, between 2000 and
4000 points (about 60 leaves), batches of 50.

| | predictions from the leaves | from neighbours |
|---|---|---|
| `suggest(50)`, 2000 candidates | 25 ms | 1.4 s |
| `suggest(500)`, 10 000 candidates | 0.14 s | 7.2 s |
| `report()`, 2000 candidates | 35 to 48 ms | 1.3 to 1.5 s |
| `add` of 50 points | 69 to 79 ms | 147 to 161 ms |

With predictions from the leaves the candidates are predicted leaf by leaf
(`predict(mode='loop')`), and with predictions from neighbours one by one, at
0.7 ms each.
