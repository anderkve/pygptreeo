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

SCALING_PLACEHOLDER

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
