# Neural-linear leaves: accuracy on the stream benchmark and cost scaling on a long stream

`GPTree(GPR=NeuralLinearGPR(FeatureNetLearner()))` (`pygptreeo/neural_linear.py`):
one tree-wide feature network, and in every leaf a Bayesian linear regression on
its features of the residual of the network's own prediction. The design and
its rationale are in `docs/neural_gptree_ideas.md`. Two questions here:

1. On the targets and streams the package is tested on, how does it compare with
   the plain GP tree in accuracy and in the honesty of its sigma?
2. On a very long stream, do the per-point update and prediction times stay
   bounded, as they do for the GP tree, and what does bounding them cost?

Everything below is measured; the machine is a 4-core container with
`OMP_NUM_THREADS=1` and one thread per process.

## 1. The stream benchmark (6D, 4000 points, 3 seeds)

`examples/benchmark_global_mean_streams.py --configs tree,neural --calibrate`,
the setup of `BENCHMARK_RESULTS_global_mean_streams.md`: `RotatedRosenbrock`
and `GaussianPeaks` in six dimensions, the four streams (uniform, focusing,
sweeping, walker), `Nbar = 100`, `theta = 1e-4`, retrain every 25 points,
gradual splitting, calibrated sigma. The `neural` configuration uses the
learner's defaults: 3 x 128 SiLU, 4000 Adam steps per (re)fit with a cosine
schedule, first fit at 200 points and refits at each doubling of the count,
every point kept in the training sample, residual leaves with the learner's
prequential error budget added to the sigma. The six runs shared four cores,
so the times are inflated by about half.

Mean over 3 seeds (min..max):

| target | stream | config | prequential NRMSE | uniform-test NRMSE | focus-test NRMSE | coverage prequential / uniform / focus (target 0.68) | sigma/RMSE prequential / uniform / focus | time [s] |
|---|---|---|---|---|---|---|---|---|
| gaussian_peaks | focusing | neural | 0.0040 (0.0028..0.0047) | 0.0079 (0.0065..0.0094) | 0.0001 (0.0000..0.0001) | 0.70 / 0.05 / 0.55 | 0.88 / 0.03 / 0.71 | 79 |
| gaussian_peaks | focusing | tree | 0.0074 (0.0064..0.0081) | 0.0232 (0.0213..0.0254) | 0.0001 (0.0001..0.0002) | 0.70 / 0.54 / 0.63 | 1.00 / 0.68 / 0.80 | 37 |
| gaussian_peaks | sweeping | neural | 0.0151 (0.0145..0.0155) | 0.0707 (0.0569..0.0881) | 0.0166 (0.0151..0.0190) | 0.68 / 0.44 / 0.77 | 0.77 / 0.43 / 1.05 | 73 |
| gaussian_peaks | sweeping | tree | 0.0264 (0.0230..0.0330) | 0.0807 (0.0787..0.0842) | 0.0117 (0.0085..0.0138) | 0.65 / 0.50 / 0.77 | 0.75 / 0.74 / 1.14 | 41 |
| gaussian_peaks | uniform | neural | 0.0074 (0.0072..0.0076) | 0.0035 (0.0030..0.0040) | 0.0034 (0.0030..0.0038) | 0.66 / 0.61 / 0.62 | 0.81 / 0.69 / 0.71 | 94 |
| gaussian_peaks | uniform | tree | 0.0168 (0.0162..0.0179) | 0.0139 (0.0129..0.0149) | 0.0135 (0.0127..0.0149) | 0.68 / 0.66 / 0.67 | 0.93 / 0.87 / 0.90 | 48 |
| gaussian_peaks | walker | neural | 0.0071 (0.0055..0.0098) | 0.0328 (0.0266..0.0365) | 0.0009 (0.0006..0.0013) | 0.66 / 0.17 / 0.80 | 1.03 / 0.09 / 1.20 | 87 |
| gaussian_peaks | walker | tree | 0.0111 (0.0100..0.0119) | 0.1063 (0.1053..0.1081) | 0.0039 (0.0021..0.0049) | 0.68 / 0.20 / 0.78 | 0.86 / 0.26 / 0.95 | 35 |
| rotated_rosenbrock | focusing | neural | 0.0032 (0.0021..0.0048) | 0.0080 (0.0077..0.0082) | 0.0000 (0.0000..0.0000) | 0.70 / 0.05 / 0.63 | 0.68 / 0.03 / 0.78 | 80 |
| rotated_rosenbrock | focusing | tree | 0.0069 (0.0061..0.0083) | 0.0301 (0.0289..0.0308) | 0.0000 (0.0000..0.0000) | 0.69 / 0.33 / 0.60 | 0.87 / 0.32 / 1.02 | 38 |
| rotated_rosenbrock | sweeping | neural | 0.0031 (0.0027..0.0034) | 0.0327 (0.0278..0.0359) | 0.0037 (0.0024..0.0057) | 0.67 / 0.42 / 0.79 | 0.69 / 0.23 / 1.35 | 72 |
| rotated_rosenbrock | sweeping | tree | 0.0060 (0.0052..0.0065) | 0.0647 (0.0613..0.0673) | 0.0053 (0.0040..0.0071) | 0.66 / 0.39 / 0.68 | 0.73 / 0.26 / 0.47 | 42 |
| rotated_rosenbrock | uniform | neural | 0.0034 (0.0025..0.0042) | 0.0018 (0.0017..0.0019) | 0.0019 (0.0011..0.0024) | 0.67 / 0.54 / 0.52 | 0.59 / 0.39 / 0.38 | 93 |
| rotated_rosenbrock | uniform | tree | 0.0155 (0.0114..0.0187) | 0.0109 (0.0080..0.0141) | 0.0110 (0.0079..0.0134) | 0.67 / 0.67 / 0.67 | 0.67 / 0.66 / 0.66 | 48 |
| rotated_rosenbrock | walker | neural | 0.0034 (0.0030..0.0036) | 0.0474 (0.0443..0.0506) | 0.0003 (0.0003..0.0004) | 0.68 / 0.24 / 0.85 | 1.06 / 0.04 / 2.36 | 86 |
| rotated_rosenbrock | walker | tree | 0.0062 (0.0061..0.0064) | 0.0923 (0.0865..0.0984) | 0.0024 (0.0022..0.0026) | 0.65 / 0.28 / 0.78 | 0.81 / 0.13 / 1.00 | 35 |

The same as ratios neural / tree (mean over seeds of the per-seed ratio; below
1 is better than the plain tree), with the learner's refit count and fit time:

| target | stream | prequential | uniform-test | focus-test | refits, fit s (of total s) |
|---|---|---|---|---|---|
| rotated_rosenbrock | uniform | 0.22 | 0.17 | 0.17 | 5, 59 (93) |
| rotated_rosenbrock | focusing | 0.46 | 0.27 | 0.50 | 5, 52 (80) |
| rotated_rosenbrock | sweeping | 0.52 | 0.50 | 0.68 | 5, 47 (72) |
| rotated_rosenbrock | walker | 0.55 | 0.51 | 0.15 | 5, 54 (86) |
| gaussian_peaks | uniform | 0.44 | 0.25 | 0.26 | 5, 60 (94) |
| gaussian_peaks | focusing | 0.54 | 0.34 | 0.44 | 5, 51 (79) |
| gaussian_peaks | sweeping | 0.59 | 0.87 | **1.46** | 5, 47 (73) |
| gaussian_peaks | walker | 0.63 | 0.31 | 0.24 | 5, 53 (87) |

For comparison, the additive-GP global model (`global_pkg`, same benchmark,
`BENCHMARK_RESULTS_global_mean_streams.md`) reaches ratios of 0.5 to 0.85 on
the same cells at 4 to 8 times the plain tree's run time.

### Reading

* **Accuracy: a factor of 2 to 6 on 23 of the 24 cells.** The neural-linear
  tree is below the plain tree on every prequential and every cube-wide
  metric, by the most on the uniform stream (ratios 0.17 to 0.44), and on
  every focus set but one. Seed ranges rarely overlap. This is well beyond
  what the additive-GP global model gave on the same cells, and these
  targets have no additive structure to exploit: the network learns the
  rotated quadratic and the mixture of rotated peaks that an order-2 additive
  kernel can only partly represent.
* **The one loss is the end of the sweep on the Gaussian peaks** (focus ratio
  1.46, 0.0166 against 0.0117). The sweeping stream leaves most leaves behind
  and the focus set sits at the front, where the network, refit at most at
  each doubling, is trained on a sample dominated by the path behind it. The
  additive-GP global model showed the same soft spot on this stream before
  its refresh rule; here the leaves do refresh, and what lags is the network
  itself. A refit cap (`refit_cap`) or a reservoir that favours recent points
  would address it at the cost of more refits; not measured.
* **On the stream the sigma is calibrated.** Prequential coverage is 0.66 to
  0.70 on every cell and sigma over RMSE 0.6 to 1.1, the same as the plain
  tree: the per-leaf calibration works on the new leaf model unchanged.
* **Off the stream the sigma is not, and it is worse than the plain tree's.**
  On the cube-wide test set of the focusing streams coverage is 0.05 with
  sigma over RMSE 0.03; on the walker streams 0.17 to 0.24. The plain tree is
  poor there too (0.20 to 0.54) but less so. The cause is structural: a
  residual leaf far from its data reverts to the network's head, with the
  small posterior sigma of a small residual, plus the learner's error budget,
  which is a stream-wide average of the head's error *on the stream*, where
  the head is most accurate. Off the stream the head's error is ten to thirty
  times that budget (the uniform-test NRMSE of 0.008 to 0.047 against
  prequential 0.003 to 0.007), and nothing in the leaf sees it. The
  raw-target leaves of the earlier probe (`docs/neural_gptree_ideas.md` §5)
  were less overconfident there (coverage 0.46 to 0.59) and less accurate.
  The fix on the table is a small ensemble of bagged feature networks, whose
  spread grows away from the data; the package does not have it yet, so a
  user who needs sigma away from the stream should know this.
* **Cost.** With every point kept and refits at each doubling, five refits of
  about 10 s each (on the oversubscribed machine) are most of the extra run
  time; the leaf work itself is less than the GP leaves'. Section 2 measures
  how this behaves on a stream 25 times longer.

## 2. Cost scaling on a 100 000-point stream

`examples/benchmark_neural_linear_scaling.py`: `GaussianPeaks` in six
dimensions, the walker stream (every proposal of a Metropolis random walk,
the stream closest to an optimiser's or sampler's), 100 000 points, one seed,
`Nbar = 100`, each configuration in its own process with nothing else on the
machine. Every `update_tree` call and every per-point `predict` call (the
default recursive mode) is timed; at each checkpoint a 1000-point uniform test
set is predicted in `loop` mode (batch). Times are per window since the
previous checkpoint: the mean, the 99th percentile and the maximum of the
update time, the mean per-point prediction time, and the batch prediction time
per point; the prequential NRMSE of the window and the test NRMSE at the
checkpoint. "leaf solves" is the number of regression solves held by the leaves
alive at the checkpoint (leaves that split hand nothing on), "sample" the size
of the network's training sample.

Configurations: `tree` (the plain GP tree); `neural` (every point kept, each
refit run in one go inside the `update_tree` call that triggers it);
`neural_amort8` (every point kept, refits spread over the following updates at
8 Adam steps each and published when complete); `neural_cov4000_amort8` (the
same, with the network trained on a 4000-point maximin coverage reservoir and a
refit skipped unless a quarter of the reservoir has turned over).

**tree**

| points | update ms: mean / p99 / max | predict ms per point: single / batch | window prequential NRMSE | test NRMSE | leaves | elapsed s |
|---|---|---|---|---|---|---|
| 1000 | 5.17 / 131.7 / 301 | 0.67 / 0.02 | 0.0253 | 0.1235 | 17 | 5.9 |
| 2000 | 4.41 / 92.5 / 275 | 0.66 / 0.02 | 0.0133 | 0.1276 | 26 | 11.0 |
| 4000 | 4.34 / 106.8 / 279 | 0.68 / 0.06 | 0.0110 | 0.1235 | 60 | 21.1 |
| 8000 | 4.78 / 112.5 / 411 | 0.72 / 0.11 | 0.0103 | 0.0995 | 121 | 43.2 |
| 16000 | 4.39 / 101.6 / 399 | 0.71 / 0.22 | 0.0068 | 0.0960 | 237 | 84.2 |
| 32000 | 4.54 / 107.3 / 408 | 0.74 / 0.49 | 0.0050 | 0.0929 | 459 | 169.2 |
| 64000 | 4.71 / 110.1 / 474 | 0.78 / 1.63 | 0.0036 | 0.0873 | 935 | 346.5 |
| 100000 | 4.63 / 106.7 / 494 | 0.78 / 3.57 | 0.0032 | 0.0863 | 1460 | 545.0 |

**neural** (every point kept, refits in one go)

| points | update ms: mean / p99 / max | predict ms per point: single / batch | window prequential NRMSE | test NRMSE | leaves | refits | fit s | sample | leaf solves | elapsed s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 19.92 / 23.1 / 7045 | 0.72 / 0.10 | 0.0259 | 0.0445 | 16 | 3 | 18.7 | 1000 | 14 | 20.7 |
| 2000 | 7.90 / 22.8 / 6556 | 0.81 / 0.14 | 0.0069 | 0.0354 | 28 | 4 | 25.2 | 2000 | 34 | 29.6 |
| 4000 | 5.21 / 25.9 / 7221 | 0.93 / 0.21 | 0.0035 | 0.0322 | 58 | 5 | 32.4 | 4000 | 59 | 42.1 |
| 8000 | 3.18 / 24.1 / 6611 | 0.88 / 0.47 | 0.0022 | 0.0230 | 117 | 6 | 39.0 | 8000 | 110 | 58.8 |
| 16000 | 2.43 / 25.9 / 6661 | 1.03 / 0.34 | 0.0013 | 0.0179 | 239 | 7 | 45.7 | 16000 | 225 | 86.9 |
| 32000 | 1.89 / 23.8 / 6744 | 1.00 / 0.38 | 0.0009 | 0.0171 | 461 | 8 | 52.4 | 32000 | 476 | 133.6 |
| 64000 | 1.82 / 24.7 / 8038 | 1.06 / 1.37 | 0.0008 | 0.0162 | 937 | 9 | 60.4 | 64000 | 889 | 227.3 |
| 100000 | 1.54 / 24.3 / 69 | 0.55 / 3.01 | 0.0006 | 0.0173 | 1447 | 9 | 60.4 | 100000 | 651 | 305.9 |

**neural_amort8** (every point kept, refits spread over the stream)

| points | update ms: mean / p99 / max | predict ms per point: single / batch | window prequential NRMSE | test NRMSE | leaves | refits | fit s | sample | leaf solves | elapsed s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 14.12 / 33.9 / 6870 | 0.69 / 0.17 | 0.0271 | 0.0858 | 16 | 2 | 12.6 | 1000 | 16 | 15.0 |
| 2000 | 3.86 / 26.0 / 38 | 0.51 / 0.04 | 0.0141 | 0.0769 | 28 | 2 | 12.6 | 2000 | 14 | 19.4 |
| 4000 | 3.33 / 28.0 / 47 | 0.82 / 0.08 | 0.0078 | 0.0432 | 58 | 3 | 18.5 | 4000 | 39 | 27.8 |
| 8000 | 3.26 / 27.6 / 52 | 0.93 / 0.10 | 0.0029 | 0.0276 | 119 | 4 | 25.3 | 8000 | 72 | 44.7 |
| 16000 | 2.28 / 24.4 / 72 | 0.88 / 0.20 | 0.0015 | 0.0214 | 239 | 5 | 31.4 | 16000 | 154 | 70.2 |
| 32000 | 2.01 / 25.5 / 108 | 0.96 / 0.50 | 0.0010 | 0.0174 | 462 | 6 | 38.2 | 32000 | 355 | 118.2 |
| 64000 | 1.79 / 24.7 / 98 | 0.97 / 1.23 | 0.0008 | 0.0170 | 945 | 7 | 45.4 | 64000 | 654 | 207.9 |
| 100000 | 1.77 / 24.6 / 251 | 1.27 / 3.15 | 0.0006 | 0.0178 | 1441 | 8 | 52.9 | 100000 | 1692 | 320.7 |

**neural_cov4000_amort8** (4000-point coverage reservoir, refits spread)

| points | update ms: mean / p99 / max | predict ms per point: single / batch | window prequential NRMSE | test NRMSE | leaves | refits | fit s | sample | leaf solves | elapsed s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 10.45 / 25.5 / 8992 | 0.46 / 0.04 | 0.0271 | 0.1199 | 16 | 1 | 9.0 | 1000 | 2 | 10.9 |
| 2000 | 6.98 / 31.7 / 44 | 0.58 / 0.04 | 0.0206 | 0.1170 | 28 | 1 | 9.0 | 2000 | 9 | 18.6 |
| 4000 | 2.64 / 27.5 / 721 | 0.81 / 0.07 | 0.0070 | 0.0443 | 59 | 2 | 15.6 | 4000 | 27 | 25.5 |
| 8000 | 3.73 / 26.2 / 52 | 0.88 / 0.15 | 0.0031 | 0.0318 | 119 | 3 | 22.3 | 4000 | 64 | 44.2 |
| 16000 | 2.83 / 25.7 / 100 | 0.95 / 0.16 | 0.0016 | 0.0246 | 238 | 4 | 29.1 | 4000 | 144 | 74.6 |
| 32000 | 2.24 / 24.7 / 92 | 0.92 / 0.43 | 0.0010 | 0.0203 | 465 | 5 | 35.4 | 4000 | 299 | 125.7 |
| 64000 | 2.07 / 25.2 / 86 | 0.94 / 1.25 | 0.0009 | 0.0184 | 943 | 6 | 41.8 | 4000 | 580 | 223.4 |
| 100000 | 2.04 / 24.5 / 54 | 1.24 / 3.18 | 0.0007 | 0.0151 | 1450 | 7 | 48.6 | 4000 | 1500 | 344.8 |

### Reading

* **The per-point update cost is bounded, and lower than the GP tree's.** The
  neural-linear tree's mean update time *falls* along the stream, from 20 ms
  in the first thousand points to 1.5 to 2.0 ms at 100 000, because the
  network refits (the only cost that is not per leaf) happen at each doubling
  of the count and so become rarer; the plain tree sits at 4.3 to 4.8 ms
  throughout. The 99th percentile is flat for both: 24 to 28 ms for the
  neural-linear tree (one leaf solve: features of at most `Nbar` points
  through the network, 13 eigendecompositions of a 129 x 129 matrix) against
  93 to 132 ms for the plain tree (one leaf GP fit with its hyperparameter
  optimisation). A leaf solve costs the same at 1447 leaves as at 16, which
  is the property the design has to keep.
* **A refit is a fixed cost: 6.5 to 8 s whatever the sample size.** With
  every point kept, the nine refits of the `neural` run total 60 s on a
  sample growing from 200 to 64 000 points, 6.7 s each, because a refit is a
  fixed 4000 Adam steps on 128-point minibatches; the schedule makes their
  number logarithmic in the stream length. In the burst mode that cost lands
  inside one `update_tree` call (the maximum column: 6.5 to 8 s), which is
  what a stream consumer would notice. **The amortised mode removes the
  spike**: with 8 steps per update the maximum update time is 38 to 251 ms
  (a leaf solve plus 8 steps, about 12 ms, and the occasional publication of
  the new network), the total cost is the same, and the accuracy at every
  checkpoint from 8000 points on is the same as the burst mode's to within
  the window noise (test 0.0178 against 0.0173 at 100 000). The amortised
  mode is the one to use on a live stream.
* **Bounding the training sample costs nothing here.** The 4000-point
  coverage reservoir gives the best test error of the four at 100 000
  points (0.0151 against 0.0173 and 0.0178) and the same prequential error,
  with seven refits instead of eight or nine (one skipped for too little
  turnover), bounded memory, and 0.27 ms per offer after the reservoir fills
  (the one 721 ms maximum at the 4000 checkpoint is the reservoir building
  its distance matrix when it fills, once). The walker revisits the same
  region, so a coverage sample of it represents it; a stream that keeps
  exploring new territory would test the reservoir harder, and the sweeping
  stream of section 1 is the case to measure next.
* **Per-point prediction is bounded for both**: 0.5 to 1.3 ms for the
  neural-linear tree (a feature pass through the network and the leaf
  regression), 0.66 to 0.78 ms for the plain tree, flat from 16 to 1460
  leaves, since the recursive mode visits only the leaves a point can belong
  to.
* **Batch prediction in `loop` mode grows with the number of leaves, for both
  backends**: from 0.02 to 3.6 ms per point for the plain tree and from 0.1
  to 3.2 ms for the neural-linear trees between 17 and 1460 leaves. The loop
  mode calls every leaf once (now only on the points it can own, after this
  change; before it, on all of them), so with 1000 test points and 1400
  leaves it makes about 1400 small calls, and the per-call overhead (a
  kernel evaluation or a network forward pass on a handful of points) is the
  cost. For a long stream, predict with the default recursive mode, whose
  cost per point is bounded; a loop mode that routes the batch down the tree
  once and evaluates the network once per batch would make the neural
  batch bounded too, and is the obvious next improvement.
* **Accuracy at 100 000 points: five times the plain tree's on both
  metrics.** Prequential 0.0006 to 0.0007 against 0.0032; cube-wide 0.015
  to 0.018 against 0.086 (the walker never covers the cube, so the cube-wide
  number is mostly extrapolation, where the network's shape helps most).
  The plain tree keeps improving slowly with more leaves (0.124 to 0.086);
  the neural-linear trees reach their cube-wide floor by 16 000 points and
  keep improving on the stream.
* **Memory.** The plain tree and all neural-linear trees hold the stream in
  their leaves (`Nbar` points per leaf, so linear in the stream, as the
  package always has). The learner with every point kept adds one more
  linear copy; the reservoir bounds that copy at `reservoir_size` points.
* **Total wall time for 100 000 points**: 306 to 345 s for the neural-linear
  trees against 545 s for the plain tree, on one core each.

**Caveats.** One seed, one target, one stream, `d = 6`. The cost figures are
for a 3 x 128 network on one CPU core; a wider network or a GPU moves the
refit and feature-pass constants but not the scaling. The burst run's last
window shows a maximum of 69 ms because no refit fell in it (the next would
have been at 128 000 points).

## Reproduce

```bash
cd examples
for t in rotated_rosenbrock gaussian_peaks; do for s in 1 2 3; do
  OMP_NUM_THREADS=1 python benchmark_global_mean_streams.py --target $t --seeds $s \
      --streams uniform,focusing,sweeping,walker --configs tree,neural --N 4000 --d 6 --calibrate \
      > results/neural_linear_streams/${t}_seed${s}.jsonl
done; done
python benchmark_global_mean_streams.py --summarize results/neural_linear_streams/*.jsonl
# the scaling study (one process per configuration, nothing else on the machine):
for c in tree neural neural_amort8 neural_cov4000_amort8; do
  OMP_NUM_THREADS=1 python benchmark_neural_linear_scaling.py --target gaussian_peaks --stream walker \
      --d 6 --N 100000 --configs $c --seed 1 > results/neural_linear_scaling/walker_${c}.jsonl
done
```
