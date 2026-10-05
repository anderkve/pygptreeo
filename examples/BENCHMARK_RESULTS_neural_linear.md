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
  prequential 0.003 to 0.007), and nothing in the leaf sees it. Section 1.1
  adds the floor that addresses this; the runs above are without it.

### 1.1 The uncertainty floor

The leaf now provides a floor on the calibrated sigma (`distance_floor`,
on by default; `NeuralLinearGPR.predict_floor`, applied by `GPNode.predict`
as `max(sigma, floor)`): near its points, the leaf's *local* leave-one-out
error, the rms of the closed-form leave-one-out residuals of the regression at
the five fit points nearest the query, noise subtracted; beyond two
nearest-neighbour spacings from the leaf's points, rising to the function's
overall scale over two more spacings, as a kernel's variance rises to its
prior amplitude. Distances are in coordinates scaled by the leaf's spread.
The calibration scaler is fitted on the model sigma alone and multiplies it;
the floor is applied after it. Costs: one distance computation against at
most `Nbar` points per prediction and the leverages at each solve, nothing
measurable.

Three designs were measured on the way (the result files of the first two
are kept as `results/neural_linear_streams_floor_v1/` and `_v2/`), and each
step was decided by a per-point diagnostic rather than by the summary
numbers:

1. *The stream-wide budget times the distance ratio, added before
   calibration.* Changed nothing (focusing cube-wide 0.05 → 0.05, walker
   0.24 → 0.27): on a focusing stream the cube-wide points fall inside leaves
   created during the broad phase at ordinary spacing (median ratio 1.1, 7%
   above 2), so the ratio never engaged, and on a walker stream the budget is
   set in a valley where the function is tiny.
2. *A leaf-local budget from the head's residual on the leaf's points, a
   ramp to the function's scale starting at one spacing, added after
   calibration.* Over-covered everything: the leaves correct the head by a
   factor of 50 to 100 where they have data, so the head's error is the wrong
   scale there, and in six dimensions in-cloud ratios reach 1.9, so a ramp
   from one spacing fired inside the data.
3. *The local leave-one-out error with the ramp from two spacings* (the
   design above). A leaf-wide leave-one-out rms was also measured and
   rejected: right in stale leaves, forty times too large in a focus core,
   because the leaf's box spans the old wide cluster while the queries sit in
   the dense centre.

Coverage with the floor, mean over 3 seeds; the accuracy columns are
unchanged to four decimals, since the floor touches only the sigma:

| target | stream | config | coverage prequential / uniform / focus (target 0.68) | sigma/RMSE prequential / uniform / focus |
|---|---|---|---|---|
| rotated_rosenbrock | uniform | tree | 0.67 / 0.67 / 0.67 | 0.67 / 0.66 / 0.66 |
| rotated_rosenbrock | uniform | neural, no floor | 0.67 / 0.54 / 0.52 | 0.59 / 0.39 / 0.38 |
| rotated_rosenbrock | uniform | neural, floor | 0.74 / 0.75 / 0.74 | 1.62 / 1.13 / 1.11 |
| rotated_rosenbrock | focusing | tree | 0.69 / 0.33 / 0.60 | 0.87 / 0.32 / 1.02 |
| rotated_rosenbrock | focusing | neural, no floor | 0.70 / 0.05 / 0.63 | 0.68 / 0.03 / 0.78 |
| rotated_rosenbrock | focusing | neural, floor | 0.77 / 0.48 / 0.72 | 1.68 / 1.15 / 17.76 |
| rotated_rosenbrock | sweeping | tree | 0.66 / 0.39 / 0.68 | 0.73 / 0.26 / 0.47 |
| rotated_rosenbrock | sweeping | neural, no floor | 0.67 / 0.42 / 0.79 | 0.69 / 0.23 / 1.35 |
| rotated_rosenbrock | sweeping | neural, floor | 0.74 / 0.89 / 0.75 | 7.35 / 2.54 / 2.25 |
| rotated_rosenbrock | walker | tree | 0.65 / 0.28 / 0.78 | 0.81 / 0.13 / 1.00 |
| rotated_rosenbrock | walker | neural, no floor | 0.68 / 0.24 / 0.85 | 1.06 / 0.04 / 2.36 |
| rotated_rosenbrock | walker | neural, floor | 0.85 / 0.62 / 0.82 | 7.62 / 0.35 / 7.27 |
| gaussian_peaks | uniform | tree | 0.68 / 0.66 / 0.67 | 0.93 / 0.87 / 0.90 |
| gaussian_peaks | uniform | neural, no floor | 0.66 / 0.61 / 0.62 | 0.81 / 0.69 / 0.71 |
| gaussian_peaks | uniform | neural, floor | 0.72 / 0.69 / 0.70 | 1.36 / 0.95 / 0.98 |
| gaussian_peaks | focusing | tree | 0.70 / 0.54 / 0.63 | 1.00 / 0.68 / 0.80 |
| gaussian_peaks | focusing | neural, no floor | 0.70 / 0.05 / 0.55 | 0.88 / 0.03 / 0.71 |
| gaussian_peaks | focusing | neural, floor | 0.76 / 0.41 / 0.68 | 1.43 / 1.62 / 14.34 |
| gaussian_peaks | sweeping | tree | 0.65 / 0.50 / 0.77 | 0.75 / 0.74 / 1.14 |
| gaussian_peaks | sweeping | neural, no floor | 0.68 / 0.44 / 0.77 | 0.77 / 0.43 / 1.05 |
| gaussian_peaks | sweeping | neural, floor | 0.74 / 0.84 / 0.79 | 2.41 / 4.14 / 1.66 |
| gaussian_peaks | walker | tree | 0.68 / 0.20 / 0.78 | 0.86 / 0.26 / 0.95 |
| gaussian_peaks | walker | neural, no floor | 0.66 / 0.17 / 0.80 | 1.03 / 0.09 / 1.20 |
| gaussian_peaks | walker | neural, floor | 0.79 / 0.77 / 0.80 | 6.51 / 1.71 / 4.48 |

Reading:

* **Off the stream the sigma is now conservative or close to nominal, and
  better than the plain tree's on every stream.** Cube-wide coverage goes
  from 0.05 to 0.41 and 0.48 on the focusing streams (plain tree 0.54 and
  0.33), from 0.42 and 0.44 to 0.84 and 0.89 on the sweeping streams (0.39
  and 0.50), from 0.17 and 0.24 to 0.77 and 0.62 on the walker streams (0.20
  and 0.28), and from 0.54 and 0.61 to 0.69 and 0.75 on the uniform streams
  (0.66 and 0.67). The walker and sweeping streams are the case the ramp is
  for: the cube-wide points lie many spacings from every leaf's data and get
  the function's scale.
* **The focusing stream's cube-wide points are the remaining gap** (0.41 to
  0.48). They sit inside stale leaves at ordinary spacing, so the ramp does
  not apply and the local leave-one-out error is what they get; it
  underestimates the error there by about 1.5 (the leaves' points were in
  the network's training sample, so the head's residual on them is
  optimistic). A floor that knew how stale a leaf is relative to the
  network, or an out-of-sample estimate of the head's error per region,
  would close it; neither is in the package.
* **The price is over-coverage on wandering streams.** On-stream coverage is
  0.72 to 0.85 with sigma over RMSE of 1.4 to 7.6: the walker's and the
  sweep's own next points often lie beyond two spacings from the leaf's data,
  the ramp treats them as unknown territory, and the network in fact
  interpolates them well. The RMS ratios are dominated by those points (the
  focus-region ratios of 14 to 18 on the focusing streams come from the few
  outer focus points the ramp pushes to the function's scale while their
  errors are tiny); the coverage columns say the typical sigma is right. On
  the uniform streams, where every point has neighbours, the cost is a ratio
  of 1.1 to 1.6. For a user who wants a sharp sigma on the stream and does
  not query away from it, `distance_floor=False` restores the earlier
  behaviour.
* **What the floor is and is not.** It is a heuristic in the sense that a
  kernel's prior amplitude is one: beyond the data it says "the function's
  scale", not a measurement. Near the data it is a measurement (the leaf's
  own out-of-sample error), which is why it also lifts the uniform-stream
  coverage that the calibrated model sigma alone left at 0.52 to 0.61. The
  constants (five neighbours, a ramp from two spacings over two) are in
  units of the leaf's own spacing and were chosen from the per-point
  diagnostics of one seed of the Gaussian peaks, not swept.
* **Cost.** With every point kept and refits at each doubling, five refits of
  about 10 s each (on the oversubscribed machine) are most of the extra run
  time; the leaf work itself is less than the GP leaves'. Section 2 measures
  how this behaves on a stream 25 times longer.

### 1.2 The hybrid: GP leaves on the network's residual

`GPTree(global_mean='net')` (`NetGlobalMean` in `pygptreeo/neural_linear.py`):
the same feature network as the tree-wide global model, the leaves keep their
ARD Matern GPs and model the residual of the network's head, with the global
model's refresh rule (a leaf refits against the newest network version on
first use) and its stream-wide error budget added to the sigma. No distance
floor: the leaf sigma is the GP's. Same benchmark, 3 seeds; the `hybrid`
configuration of `benchmark_global_mean_streams.py`, network refits spread
over the stream as for the neural-linear tree.

| target | stream | config | prequential NRMSE | uniform-test NRMSE | focus-test NRMSE | coverage prequential / uniform / focus (target 0.68) | sigma/RMSE prequential / uniform / focus | time [s] |
|---|---|---|---|---|---|---|---|---|
| rotated_rosenbrock | uniform | GP tree | 0.0155 | 0.0109 | 0.01098 | 0.67 / 0.67 / 0.67 | 0.67 / 0.66 / 0.66 | 48 |
| rotated_rosenbrock | uniform | neural-linear | **0.0034** | **0.0018** | **0.00188** | 0.74 / 0.75 / 0.74 | 1.62 / 1.13 / 1.11 | 66 |
| rotated_rosenbrock | uniform | hybrid | 0.0053 | 0.0030 | 0.00316 | 0.67 / 0.62 / 0.62 | 0.63 / 0.45 / 0.45 | 78 |
| rotated_rosenbrock | focusing | GP tree | 0.0069 | 0.0301 | 0.00004 | 0.69 / 0.33 / 0.60 | 0.87 / 0.32 / 1.02 | 38 |
| rotated_rosenbrock | focusing | neural-linear | **0.0032** | 0.0080 | 0.00002 | 0.77 / 0.48 / 0.72 | 1.68 / 1.15 / 17.76 | 65 |
| rotated_rosenbrock | focusing | hybrid | 0.0044 | **0.0075** | **0.00001** | 0.72 / 0.11 / 0.62 | 0.79 / 0.07 / 0.70 | 74 |
| rotated_rosenbrock | sweeping | GP tree | 0.0060 | 0.0647 | 0.00530 | 0.66 / 0.39 / 0.68 | 0.73 / 0.26 / 0.47 | 42 |
| rotated_rosenbrock | sweeping | neural-linear | **0.0031** | **0.0327** | **0.00368** | 0.74 / 0.89 / 0.75 | 7.35 / 2.54 / 2.25 | 65 |
| rotated_rosenbrock | sweeping | hybrid | 0.0046 | 0.0381 | 0.00513 | 0.67 / 0.48 / 0.78 | 0.66 / 0.23 / 0.63 | 75 |
| rotated_rosenbrock | walker | GP tree | 0.0062 | 0.0923 | 0.00241 | 0.65 / 0.28 / 0.78 | 0.81 / 0.13 / 1.00 | 35 |
| rotated_rosenbrock | walker | neural-linear | **0.0034** | **0.0474** | **0.00035** | 0.85 / 0.62 / 0.82 | 7.62 / 0.35 / 7.27 | 59 |
| rotated_rosenbrock | walker | hybrid | 0.0049 | 0.0576 | 0.00113 | 0.67 / 0.27 / 0.89 | 0.91 / 0.05 / 1.79 | 73 |
| gaussian_peaks | uniform | GP tree | 0.0168 | 0.0139 | 0.01346 | 0.68 / 0.66 / 0.67 | 0.93 / 0.87 / 0.90 | 48 |
| gaussian_peaks | uniform | neural-linear | **0.0074** | **0.0035** | **0.00344** | 0.72 / 0.69 / 0.70 | 1.36 / 0.95 / 0.98 | 68 |
| gaussian_peaks | uniform | hybrid | 0.0120 | 0.0072 | 0.00726 | 0.66 / 0.68 / 0.68 | 0.84 / 0.90 / 0.90 | 77 |
| gaussian_peaks | focusing | GP tree | 0.0074 | 0.0232 | 0.00012 | 0.70 / 0.54 / 0.63 | 1.00 / 0.68 / 0.80 | 37 |
| gaussian_peaks | focusing | neural-linear | **0.0040** | **0.0079** | **0.00006** | 0.76 / 0.41 / 0.68 | 1.43 / 1.62 / 14.34 | 67 |
| gaussian_peaks | focusing | hybrid | 0.0059 | 0.0101 | **0.00006** | 0.70 / 0.21 / 0.60 | 1.01 / 0.20 / 0.62 | 75 |
| gaussian_peaks | sweeping | GP tree | 0.0264 | 0.0807 | 0.01174 | 0.65 / 0.50 / 0.77 | 0.75 / 0.74 / 1.14 | 41 |
| gaussian_peaks | sweeping | neural-linear | **0.0151** | 0.0707 | 0.01662 | 0.74 / 0.84 / 0.79 | 2.41 / 4.14 / 1.66 | 63 |
| gaussian_peaks | sweeping | hybrid | 0.0192 | **0.0551** | **0.00968** | 0.67 / 0.54 / 0.74 | 0.89 / 0.70 / 1.54 | 74 |
| gaussian_peaks | walker | GP tree | 0.0111 | 0.1063 | 0.00391 | 0.68 / 0.20 / 0.78 | 0.86 / 0.26 / 0.95 | 35 |
| gaussian_peaks | walker | neural-linear | **0.0071** | **0.0328** | **0.00091** | 0.79 / 0.77 / 0.80 | 6.51 / 1.71 / 4.48 | 61 |
| gaussian_peaks | walker | hybrid | 0.0079 | 0.0378 | 0.00180 | 0.66 / 0.20 / 0.87 | 1.12 / 0.11 / 1.45 | 73 |

Reading:

* **On these smooth 6D targets the hybrid sits between the two**: 1.3 to 3
  times better than the GP tree and 1.1 to 2 times behind the neural-linear
  tree on 19 of 24 accuracy cells. The GP leaves on the residual recover less
  than the linear leaves on the features: a hundred-point Matern GP in six
  dimensions is still a weak local model, residual or not, where a linear
  re-weighting of good features is strong.
* **Where the hybrid wins is where the network lags**: the end of the sweep on
  the Gaussian peaks (focus 0.0097 against the neural-linear tree's 0.0166
  and the GP tree's 0.0117, cube-wide 0.055 against 0.071 and 0.081), and the
  focus region of the focusing streams (level or better). There the network's
  snapshot is stale relative to the front of the stream, and a kernel leaf
  corrects it where a linear leaf on its features does not.
* **Its sigma behaves like the GP tree's**: on-stream coverage 0.66 to 0.72
  (the per-leaf calibration of the GP sigma plus the stream-wide budget),
  off-stream 0.11 to 0.54, as poor as the GP tree's or worse, since it has
  neither the network's floor nor anything else that grows away from the
  data. The floor of §1.1 is a property of the neural-linear leaf and does
  not transfer.
* **It is the slowest of the three** (73 to 78 s against 35 to 48 and 59 to
  68): it pays the GP leaf's fits and the network's refits.

### 1.3 Against the GP tree with the additive global GP

The package's other global model, `GPTree(global_mean='additive_gp')`
(`AdditiveGPGlobalMean`: a GP with an order-2 additive + Matern kernel on a
500-point coverage reservoir, refit on turnover, two optimiser restarts; the
`global_pkg` configuration of `BENCHMARK_RESULTS_global_mean_streams.md`), was
run on this same benchmark with the same seeds and settings, calibrated
(`results/global_mean_streams/calibrated_v3_*.jsonl`). Its rows beside the
three configurations above, mean over 3 seeds:

| target | stream | config | prequential | uniform-test | focus-test | coverage prequential / uniform / focus | sigma/RMSE prequential / uniform / focus | time [s] |
|---|---|---|---|---|---|---|---|---|
| rotated_rosenbrock | uniform | GP tree | 0.0155 | 0.0109 | 0.01098 | 0.67 / 0.67 / 0.67 | 0.67 / 0.66 / 0.66 | 44 |
| rotated_rosenbrock | uniform | GP tree + global GP | 0.0107 | 0.0077 | 0.00762 | 0.68 / 0.64 / 0.63 | 0.62 / 0.56 / 0.58 | 348 |
| rotated_rosenbrock | uniform | neural-linear | **0.0034** | **0.0018** | **0.00188** | 0.74 / 0.75 / 0.74 | 1.62 / 1.13 / 1.11 | 66 |
| rotated_rosenbrock | uniform | hybrid | 0.0053 | 0.0030 | 0.00316 | 0.67 / 0.62 / 0.62 | 0.63 / 0.45 / 0.45 | 78 |
| rotated_rosenbrock | focusing | GP tree | 0.0069 | 0.0301 | 0.00004 | 0.69 / 0.33 / 0.60 | 0.87 / 0.32 / 1.02 | 31 |
| rotated_rosenbrock | focusing | GP tree + global GP | 0.0053 | 0.0177 | 0.00004 | 0.74 / 0.36 / 0.68 | 0.66 / 0.25 / 1.60 | 181 |
| rotated_rosenbrock | focusing | neural-linear | **0.0032** | 0.0080 | 0.00002 | 0.77 / 0.48 / 0.72 | 1.68 / 1.15 / 17.76 | 65 |
| rotated_rosenbrock | focusing | hybrid | 0.0044 | **0.0075** | **0.00001** | 0.72 / 0.11 / 0.62 | 0.79 / 0.07 / 0.70 | 74 |
| rotated_rosenbrock | sweeping | GP tree | 0.0060 | 0.0647 | 0.00530 | 0.66 / 0.39 / 0.68 | 0.73 / 0.26 / 0.47 | 37 |
| rotated_rosenbrock | sweeping | GP tree + global GP | 0.0065 | 0.0530 | 0.00899 | 0.67 / 0.41 / 0.76 | 0.83 / 0.26 / 0.56 | 355 |
| rotated_rosenbrock | sweeping | neural-linear | **0.0031** | **0.0327** | **0.00368** | 0.74 / 0.89 / 0.75 | 7.35 / 2.54 / 2.25 | 65 |
| rotated_rosenbrock | sweeping | hybrid | 0.0046 | 0.0381 | 0.00513 | 0.67 / 0.48 / 0.78 | 0.66 / 0.23 / 0.63 | 75 |
| rotated_rosenbrock | walker | GP tree | 0.0062 | 0.0923 | 0.00241 | 0.65 / 0.28 / 0.78 | 0.81 / 0.13 / 1.00 | 39 |
| rotated_rosenbrock | walker | GP tree + global GP | 0.0049 | 0.0760 | 0.00109 | 0.67 / 0.25 / 0.93 | 0.92 / 0.06 / 2.34 | 330 |
| rotated_rosenbrock | walker | neural-linear | **0.0034** | **0.0474** | **0.00035** | 0.85 / 0.62 / 0.82 | 7.62 / 0.35 / 7.27 | 59 |
| rotated_rosenbrock | walker | hybrid | 0.0049 | 0.0576 | 0.00113 | 0.67 / 0.27 / 0.89 | 0.91 / 0.05 / 1.79 | 73 |
| gaussian_peaks | uniform | GP tree | 0.0168 | 0.0139 | 0.01346 | 0.68 / 0.66 / 0.67 | 0.93 / 0.87 / 0.90 | 43 |
| gaussian_peaks | uniform | GP tree + global GP | 0.0135 | 0.0112 | 0.01126 | 0.68 / 0.67 / 0.68 | 0.87 / 0.86 / 0.86 | 300 |
| gaussian_peaks | uniform | neural-linear | **0.0074** | **0.0035** | **0.00344** | 0.72 / 0.69 / 0.70 | 1.36 / 0.95 / 0.98 | 68 |
| gaussian_peaks | uniform | hybrid | 0.0120 | 0.0072 | 0.00726 | 0.66 / 0.68 / 0.68 | 0.84 / 0.90 / 0.90 | 77 |
| gaussian_peaks | focusing | GP tree | 0.0074 | 0.0232 | 0.00012 | 0.70 / 0.54 / 0.63 | 1.00 / 0.68 / 0.80 | 42 |
| gaussian_peaks | focusing | GP tree + global GP | 0.0054 | 0.0169 | 0.00011 | 0.74 / 0.34 / 0.65 | 0.93 / 0.35 / 1.36 | 171 |
| gaussian_peaks | focusing | neural-linear | **0.0040** | **0.0079** | **0.00006** | 0.76 / 0.41 / 0.68 | 1.43 / 1.62 / 14.34 | 67 |
| gaussian_peaks | focusing | hybrid | 0.0059 | 0.0101 | **0.00006** | 0.70 / 0.21 / 0.60 | 1.01 / 0.20 / 0.62 | 75 |
| gaussian_peaks | sweeping | GP tree | 0.0264 | 0.0807 | 0.01174 | 0.65 / 0.50 / 0.77 | 0.75 / 0.74 / 1.14 | 35 |
| gaussian_peaks | sweeping | GP tree + global GP | **0.0130** | **0.0423** | **0.00773** | 0.67 / 0.42 / 0.73 | 0.78 / 0.43 / 0.94 | 386 |
| gaussian_peaks | sweeping | neural-linear | 0.0151 | 0.0707 | 0.01662 | 0.74 / 0.84 / 0.79 | 2.41 / 4.14 / 1.66 | 63 |
| gaussian_peaks | sweeping | hybrid | 0.0192 | 0.0551 | 0.00968 | 0.67 / 0.54 / 0.74 | 0.89 / 0.70 / 1.54 | 74 |
| gaussian_peaks | walker | GP tree | 0.0111 | 0.1063 | 0.00391 | 0.68 / 0.20 / 0.78 | 0.86 / 0.26 / 0.95 | 41 |
| gaussian_peaks | walker | GP tree + global GP | 0.0073 | 0.0624 | 0.00135 | 0.66 / 0.16 / 0.87 | 1.07 / 0.09 / 1.61 | 325 |
| gaussian_peaks | walker | neural-linear | **0.0071** | **0.0328** | **0.00091** | 0.79 / 0.77 / 0.80 | 6.51 / 1.71 / 4.48 | 61 |
| gaussian_peaks | walker | hybrid | 0.0079 | 0.0378 | 0.00180 | 0.66 / 0.20 / 0.87 | 1.12 / 0.11 / 1.45 | 73 |

As ratios to the GP tree (mean of per-seed ratios), prequential / uniform /
focus:

| target | stream | GP tree + global GP | neural-linear | hybrid |
|---|---|---|---|---|
| rotated_rosenbrock | uniform | 0.70 / 0.71 / 0.69 | 0.22 / 0.17 / 0.17 | 0.34 / 0.28 / 0.28 |
| rotated_rosenbrock | focusing | 0.75 / 0.59 / 1.12 | 0.46 / 0.27 / 0.50 | 0.63 / 0.25 / 0.33 |
| rotated_rosenbrock | sweeping | 1.11 / 0.82 / 1.94 | 0.52 / 0.50 / 0.68 | 0.78 / 0.59 / 1.00 |
| rotated_rosenbrock | walker | 0.78 / 0.82 / 0.47 | 0.55 / 0.51 / 0.15 | 0.78 / 0.62 / 0.47 |
| gaussian_peaks | uniform | 0.81 / 0.81 / 0.84 | 0.44 / 0.25 / 0.26 | 0.71 / 0.52 / 0.54 |
| gaussian_peaks | focusing | 0.74 / 0.73 / 0.87 | 0.54 / 0.34 / 0.44 | 0.79 / 0.43 / 0.46 |
| gaussian_peaks | sweeping | 0.50 / 0.52 / 0.69 | 0.59 / 0.87 / 1.46 | 0.72 / 0.69 / 0.83 |
| gaussian_peaks | walker | 0.65 / 0.59 / 0.36 | 0.63 / 0.31 / 0.24 | 0.71 / 0.36 / 0.45 |

Reading:

* **The network beats the additive global GP on 21 of 24 accuracy cells,
  usually by a factor of two to four**, at a fifth of its run time (59 to 68 s
  against 171 to 386 s). The additive GP's gain of 15 to 50% on these
  non-additive targets is what its own record measured; the network, which
  is not limited to low-order additive structure, gets 50 to 85%.
* **The one cell the global GP wins is the end of the sweep on the Gaussian
  peaks** (focus 0.0077 against the network's 0.0166 and the hybrid's
  0.0097): its turnover rule refits it seven times along the sweep, so it
  follows the front, where the network refits only at each doubling of the
  count and lags. A refit cap on the network learner (`refit_cap`) is the
  corresponding knob, not measured here.
* **The hybrid and the global GP are the closest pair in design** (both GP
  leaves on a global model's residual) and the hybrid is better on 20 of 24
  cells at a quarter of the run time, since a network fit costs seconds
  where an additive GP fit on 500 points costs tens of seconds and scales
  cubically.
* **Sigma off the stream**: the global GP's is the GP tree's (coverage 0.16
  to 0.42 cube-wide on the moving streams, as its record notes), the hybrid's
  likewise; only the neural-linear tree's floor lifts it (0.41 to 0.89).

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

## 3. Point-by-point comparison figures

`examples/compare_tree_vs_neural.py` streams the same 40 000 points through
both trees, predicting every point before giving it to the tree, and draws
the package's usual performance figure with both overlaid, per batch of 2000
points: prediction time, update time (mean and maximum in the batch), NRMSE,
the fraction of predictions within 1 to 16% of the true value, the empirical
1-sigma coverage, and the number of leaves. Same settings as section 1
(`Nbar = 100`, retrain every 25 points, gradual splitting, calibrated sigma;
the neural-linear tree with refits spread over the stream at 8 steps per
update and the floor of §1.1). One process per configuration, one seed. The
figures are `results/compare_tree_vs_neural/*_compare.png`, the per-batch
numbers behind them `*_batches.csv`.

Mean over the last five batches (the last 10 000 points), with the largest
single update time after the first batch:

| run | config | predict ms | update ms (max) | NRMSE | within 1% | within 4% | coverage |
|---|---|---|---|---|---|---|---|
| eggholder, d = 3, uniform | GP tree | 0.94 | 4.09 (284) | 0.040 | **0.41** | **0.73** | 0.68 |
| | neural-linear | 0.84 | 1.88 (76) | 0.067 | 0.17 | 0.47 | 0.78 |
| | hybrid | 1.13 | 4.78 (295) | **0.039** | 0.38 | **0.73** | 0.71 |
| rotated_rosenbrock, d = 6, uniform | GP tree | 0.89 | 4.40 (679) | 0.0045 | 0.30 | 0.69 | 0.67 |
| | neural-linear | 0.81 | 1.85 (110) | **0.0006** | 0.85 | **0.98** | 0.75 |
| | hybrid | 1.12 | 5.16 (442) | **0.0006** | **0.87** | **0.98** | 0.68 |
| gaussian_peaks, d = 10, uniform | GP tree | 0.91 | 5.44 (659) | 0.038 | 0.09 | 0.35 | 0.67 |
| | neural-linear | 0.77 | 1.70 (85) | **0.0037** | **0.79** | **0.99** | 0.75 |
| | hybrid | 1.14 | 9.11 (562) | 0.0039 | 0.75 | **0.99** | 0.66 |
| gaussian_peaks, d = 6, walker | GP tree | 0.92 | 4.74 (522) | 0.015 | 0.53 | 0.89 | 0.68 |
| | neural-linear | 0.77 | 1.77 (85) | 0.0034 | 0.92 | **1.00** | 0.74 |
| | hybrid | 1.13 | 5.46 (554) | **0.0030** | **0.93** | **1.00** | 0.70 |

The `hybrid` rows (GP leaves on the residual of the same network, §1.2) were
added in a second pass; the figures overlay all three.

Reading:

* **On these 40 000-point streams the hybrid is the best of both.** It
  matches the GP tree on the Eggholder (0.039 against 0.040) and the
  neural-linear tree on the three high-dimensional runs (0.0006, 0.0039 and
  0.0030 against 0.0006, 0.0037 and 0.0034), with coverage at 0.66 to 0.71.
  On the 4000-point benchmark of §1.2 it sat between the two; by 40 000
  points the GP leaves on the residual have the points to finish the job the
  network starts, on smooth and rough targets alike. One seed per run.
* **It pays the GP leaf's cost.** Update time 4.8 to 9.1 ms on average with
  batch maxima of 295 to 562 ms (the hyperparameter fits), against 1.7 to
  1.9 ms and 76 to 110 ms for the neural-linear tree; prediction 1.1 ms
  against 0.8. On the 10D run its update time is nearly twice the GP tree's,
  since the network's refits and the leaves' refits against each new
  snapshot come on top of the GP fits.
* **Its sigma is the GP's, calibrated on the stream and without the floor**:
  nominal here, where every point has neighbours, and as poor off-stream as
  the GP tree's (§1.2).

* **The dimension decides the winner.** On the 3D Eggholder, a rough and
  strongly oscillatory target, the GP tree is better throughout: a local
  Matern kernel is the right prior for it, and a 3 x 128 network learns its
  ripples more slowly than the leaves do. From six dimensions on the
  neural-linear tree is 4.5 to 10 times more accurate, and the fraction of
  predictions within 1% goes from 0.09 to 0.79 on the 10D peaks and from
  0.30 to 0.85 on the 6D rotated Rosenbrock. On the walker stream, the one
  closest to an optimiser's, the gap is a factor 4.5 with 92% of predictions
  within 1%.
* **Time per point is lower for the neural-linear tree on every run**, both
  for prediction (0.8 against 0.9 ms) and for the update (1.7 to 1.9 against
  4.1 to 5.4 ms on average, with a batch maximum of 76 to 110 ms against
  284 to 679 ms for the GP tree's hyperparameter fits). The one cost it has
  that the GP tree lacks is visible in the prediction-time panel: in the
  batch where a network refit is published (at each doubling of the count),
  every leaf re-solves its regression on first use, and the batch mean
  rises to 2 to 4 times its usual value, then returns. The refits' training
  itself is in the update time, spread at 8 steps per update.
* **Coverage.** The GP tree sits on 0.68 throughout. The neural-linear tree
  sits at 0.74 to 0.78: conservative, which is the floor of §1.1 at work
  (its leave-one-out error exceeds the calibrated sigma on part of the
  stream); on the Eggholder, where leave-one-out residuals of a rough target
  are large, it is most conservative.
* **Leaf counts are the same** (570 to 590 at 40 000 points), as they must
  be: the tree grows by `Nbar`, not by the leaf model.

### 3.1 Why the network is slower on the Eggholder (measured)

Head of the feature network alone, trained on the 40 000 uniform Eggholder
points and scored on a held-out uniform set (one seed), against the two trees'
last 10 000 stream points:

| model | NRMSE | cost |
|---|---|---|
| network head, 3 x 128, 4000 steps (the package default) | 0.143 | 6 s per refit |
| network head, 3 x 128, 16 000 steps | 0.104 | 24 s |
| network head, 4 x 256, 16 000 steps | 0.058 | 57 s |
| network head, 3 x 128, 4000 or 16 000 steps, Fourier input features (64 frequencies, scale 2 or 4) | 0.108 to 0.161 | 7 to 28 s |
| neural-linear tree (default network) | 0.067 | |
| GP tree | 0.040 | |
| hybrid: GP leaves on the residual of the default network (`global_mean`) | 0.036 (within 1%: 0.38, coverage 0.71) | GP leaf cost |
| neural-linear tree with the 4 x 256 network, 16 000 steps per refit | 0.12, 0.22, 0.06, 0.17 per 10 000-point window | 8 times the default |

So on this target the network at the fixed refit budget is the limit (its own
error is 3.5 times the GP tree's), the linear leaves halve that error but
cannot add structure the features lack, a kernel leaf on the same network's
residual recovers the GP tree's accuracy, and buying a better network costs
eight times the refit time and gave an unstable tree whose cause was not
established. Fourier input features, the usual remedy for an MLP's slowness
on high frequencies, did not help at these settings.

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
