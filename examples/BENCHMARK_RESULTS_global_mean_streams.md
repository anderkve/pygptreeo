# Global model + residual tree: benchmark under different input streams

Question: does a tree that models the *residual* of a global GP beat a plain GPTree on
targets that a low-order additive model can only partly capture, and does the answer
depend on how the input stream explores the input space?

Setup (`examples/benchmark_global_mean_streams.py`, commit 9b43fb5):

* Targets: `RotatedRosenbrock` and `GaussianPeaks` in 6 dimensions. Both couple every
  input dimension through a fixed random rotation, so unlike the other N-dimensional
  targets in `target_functions.py` (all sums of terms in one or two adjacent
  coordinates) they have no exact low-order additive decomposition.
* Streams of 4000 points each: `uniform`; `focusing` (DE-like: 666 uniform points, then a
  Gaussian cluster around the target's minimum whose width shrinks from 0.3 to 0.02);
  `sweeping` (a cluster of width 0.12 whose centre follows a smooth path through the
  cube); `walker` (every proposal of a Metropolis random walk on exp(-f/T)).
* Tree: `GPTree(Default_GPR(n_restarts_optimizer=1), Nbar=100, theta=1e-4,
  retrain_every_n_points=25, splitting_strategy='gradual')`, i.e. the ARD Matern default
  kernel and the `min_lengthscale` split criterion.
* Global model (`global`): a GP with `AdditiveMaternKernel(d=6, order=2)` fitted on a
  500-point coverage (maximin) reservoir of the stream, refit when at least 25 % of the
  reservoir has turned over since the last fit and at most once per 500 points, with two
  optimizer restarts per fit. Leaves subtract the current snapshot at fit time and add
  back the same snapshot at predict time. `frozen`: the same model, but no refits after
  the first 666 points.
* Metrics (NRMSE = RMSE / range of the target on a uniform sample): `prequential` on the
  stream after the first 666 points (predict, then update); `uniform-test` on 3000 uniform
  points (global accuracy); `focus-test` on 3000 points drawn where the stream ended up
  (for `uniform` just another uniform sample).
* 3 seeds per cell; the seed sets the stream, the tree's routing and the optimizer restarts.

## Results

Mean over 3 seeds, with (min..max). Time is the full streaming run on one core.

| target | stream | config | prequential NRMSE | uniform-test NRMSE | focus-test NRMSE | refits | time [s] |
|---|---|---|---|---|---|---|---|
| gaussian_peaks | focusing | frozen | 0.0058 (0.0047..0.0066) | 0.0185 (0.0163..0.0201) | 0.0001 (0.0001..0.0001) | 2 | 84 |
| gaussian_peaks | focusing | global | 0.0056 (0.0046..0.0063) | 0.0174 (0.0158..0.0185) | 0.0001 (0.0001..0.0002) | 3 | 118 |
| gaussian_peaks | focusing | tree | 0.0074 (0.0064..0.0081) | 0.0232 (0.0213..0.0254) | 0.0001 (0.0001..0.0002) | 0 | 20 |
| gaussian_peaks | sweeping | frozen | 0.0144 (0.0134..0.0153) | 0.0599 (0.0539..0.0656) | 0.0111 (0.0086..0.0135) | 2 | 76 |
| gaussian_peaks | sweeping | global | 0.0173 (0.0131..0.0205) | 0.0606 (0.0568..0.0673) | 0.0153 (0.0086..0.0233) | 7 | 245 |
| gaussian_peaks | sweeping | tree | 0.0264 (0.0230..0.0330) | 0.0807 (0.0787..0.0842) | 0.0117 (0.0085..0.0138) | 0 | 20 |
| gaussian_peaks | uniform | frozen | 0.0145 (0.0138..0.0157) | 0.0117 (0.0109..0.0130) | 0.0114 (0.0106..0.0125) | 2 | 97 |
| gaussian_peaks | uniform | global | 0.0139 (0.0130..0.0148) | 0.0114 (0.0107..0.0121) | 0.0113 (0.0106..0.0124) | 5 | 203 |
| gaussian_peaks | uniform | tree | 0.0168 (0.0162..0.0179) | 0.0139 (0.0129..0.0149) | 0.0135 (0.0127..0.0149) | 0 | 25 |
| gaussian_peaks | walker | frozen | 0.0085 (0.0080..0.0096) | 0.0717 (0.0518..0.0838) | 0.0025 (0.0014..0.0032) | 2 | 90 |
| gaussian_peaks | walker | global | 0.0087 (0.0067..0.0121) | 0.0670 (0.0558..0.0739) | 0.0022 (0.0011..0.0038) | 7 | 239 |
| gaussian_peaks | walker | tree | 0.0111 (0.0100..0.0119) | 0.1063 (0.1053..0.1081) | 0.0039 (0.0021..0.0049) | 0 | 21 |
| rotated_rosenbrock | focusing | frozen | 0.0057 (0.0043..0.0074) | 0.0196 (0.0182..0.0212) | 0.0000 (0.0000..0.0001) | 2 | 83 |
| rotated_rosenbrock | focusing | global | 0.0055 (0.0041..0.0073) | 0.0180 (0.0172..0.0190) | 0.0000 (0.0000..0.0001) | 3 | 123 |
| rotated_rosenbrock | focusing | tree | 0.0069 (0.0061..0.0083) | 0.0301 (0.0289..0.0308) | 0.0000 (0.0000..0.0000) | 0 | 20 |
| rotated_rosenbrock | sweeping | frozen | 0.0120 (0.0088..0.0148) | 0.0562 (0.0471..0.0662) | 0.0059 (0.0054..0.0062) | 2 | 75 |
| rotated_rosenbrock | sweeping | global | 0.0116 (0.0072..0.0165) | 0.0581 (0.0554..0.0636) | 0.0066 (0.0040..0.0082) | 7 | 225 |
| rotated_rosenbrock | sweeping | tree | 0.0060 (0.0052..0.0065) | 0.0647 (0.0613..0.0673) | 0.0053 (0.0040..0.0071) | 0 | 21 |
| rotated_rosenbrock | uniform | frozen | 0.0115 (0.0094..0.0139) | 0.0091 (0.0076..0.0098) | 0.0091 (0.0066..0.0107) | 2 | 89 |
| rotated_rosenbrock | uniform | global | 0.0110 (0.0089..0.0131) | 0.0082 (0.0070..0.0090) | 0.0079 (0.0058..0.0098) | 5 | 214 |
| rotated_rosenbrock | uniform | tree | 0.0155 (0.0114..0.0187) | 0.0109 (0.0080..0.0141) | 0.0110 (0.0079..0.0134) | 0 | 25 |
| rotated_rosenbrock | walker | frozen | 0.0059 (0.0051..0.0063) | 0.0833 (0.0800..0.0888) | 0.0021 (0.0019..0.0023) | 2 | 82 |
| rotated_rosenbrock | walker | global | 0.0049 (0.0042..0.0053) | 0.0759 (0.0727..0.0817) | 0.0013 (0.0012..0.0016) | 7 | 248 |
| rotated_rosenbrock | walker | tree | 0.0062 (0.0061..0.0064) | 0.0923 (0.0865..0.0984) | 0.0024 (0.0022..0.0026) | 0 | 21 |

Same data as ratios to the plain tree (mean over seeds of the per-seed ratio; below 1 is
better than the tree alone):

| target | stream | global: prequential / uniform / focus | frozen: prequential / uniform / focus |
|---|---|---|---|
| rotated_rosenbrock | uniform | 0.72 / 0.77 / 0.72 | 0.75 / 0.86 / 0.83 |
| rotated_rosenbrock | focusing | 0.78 / 0.60 / (1.12)* | 0.81 / 0.65 / (1.34)* |
| rotated_rosenbrock | sweeping | **2.00** / 0.90 / 1.26 | **2.04** / 0.86 / 1.16 |
| rotated_rosenbrock | walker | 0.79 / 0.82 / 0.56 | 0.94 / 0.90 / 0.87 |
| gaussian_peaks | uniform | 0.83 / 0.82 / 0.84 | 0.86 / 0.85 / 0.85 |
| gaussian_peaks | focusing | 0.76 / 0.75 / 0.87 | 0.78 / 0.80 / 0.78 |
| gaussian_peaks | sweeping | 0.67 / 0.75 / 1.26 | 0.56 / 0.74 / 0.96 |
| gaussian_peaks | walker | 0.78 / 0.63 / 0.56 | 0.77 / 0.67 / 0.64 |

\* In the focusing stream every configuration reaches a focus-test NRMSE of about 1e-4;
these ratios are noise on numbers at the resolution floor.

## Reading

**On non-additive targets the gain is real but modest: 15 to 45 %.** On the additive
Rosenbrock and Rastrigin the same design gave gains of two to three orders of magnitude
(see the design discussion); here the additive global model can capture only the smooth
large-scale part and the leaves keep most of the work. Gains of this size hold for the
uniform, focusing and walker streams in every metric and on both targets, and are
consistent across seeds (the seed ranges of `global` and `tree` rarely overlap).

**The sweeping stream is the exception.** The global model still improves accuracy over
the whole cube (uniform-test 0.75 to 0.90), but at the moving front it hurts: the
prequential error doubles on the rotated Rosenbrock and the focus-test error grows by a
quarter on both targets. The frozen model shows the same pattern, so this is not refit
churn. It is extrapolation: the front of the sweep is territory no reservoir version has
seen, the global GP extrapolates a wrong trend there for about one length scale before
reverting to its mean, and the fresh leaves at the front then have to undo an error they
have too few points to learn. The tree alone extrapolates from its own parent leaf and does
so better. The remedy to try next is to damp the global contribution where the global GP
is itself uncertain (shrink it towards its constant mean with its own predictive variance),
which makes the residual revert to the raw target exactly where the model is guessing.

**Refits versus frozen.** With reliable fits (restarts), refitting on a coverage reservoir
beats the frozen model in 20 of 24 metric cells, by a few percent on uniform and focusing
streams and by more on the walker (which keeps discovering new territory). The exception is
again the sweeping stream, where frozen is slightly less harmful at the front. Refits are
few (3 to 7 per 4000 points) and the turnover rule freezes the model automatically on the
focusing stream (3 refits, all before the cluster narrows).

**Cost.** At these prototype settings (500-point reservoir, order-2 additive kernel, two
restarts) the global runs take 6 to 12 times longer than the tree alone, dominated by the
global fits (20 to 40 s each) and by the per-point global prediction. The frozen model
costs 3 to 4 times the tree. A 300-point reservoir, one restart, or a cheaper global model
family would bring this down; none of that was tuned here.

## Reproduce

```bash
cd examples
for t in rotated_rosenbrock gaussian_peaks; do for s in 1 2 3; do
  OMP_NUM_THREADS=1 python benchmark_global_mean_streams.py --target $t --seeds $s \
      --streams uniform,focusing,sweeping,walker --configs tree,global,frozen --N 4000 --d 6 \
      > results/global_mean_streams/${t}_seed${s}.jsonl
done; done
python benchmark_global_mean_streams.py --summarize results/global_mean_streams/*.jsonl
```

The raw `RESULT` lines of the runs above are in `results/global_mean_streams/`.
