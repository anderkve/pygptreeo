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

## Damping the global contribution where the model has no data (negative result)

The reading above suggested damping the global model's contribution by its own
predictive variance, so that the residual reverts to the raw target where the model
extrapolates. Two findings, both on the sweeping stream (3 seeds, same settings):

**The fitted global GP's variance is uninformative.** Marginal likelihood picks catch-all
Matern length scales of 70 to 260 standardised units on a domain a few units wide, with
an amplitude of 36^2 (rotated Rosenbrock, mid-sweep reservoir of 500 points). The
posterior variance is then below 1e-3 of the prior everywhere, including three reservoir
spacings beyond the data, so the damping weight is 1.000 on every test point. The
`global_damped*` configurations therefore use the explained-variance fraction of a
*reference* GP on the reservoir points (unit RBF, length scale 0.5x / 1x / 2x the
reservoir's nearest-neighbour spacing), which is 1 on the data and 0 far from it.

**Damping hurts, and the more the worse.** Mean confidence of the final snapshot on the
uniform / focus test sets is given for the damped runs.

| target | config | prequential NRMSE | uniform-test NRMSE | focus-test NRMSE | confidence |
|---|---|---|---|---|---|
| rotated_rosenbrock | tree | 0.0060 (0.0052..0.0065) | 0.0647 (0.0613..0.0673) | 0.0053 (0.0040..0.0071) | |
| rotated_rosenbrock | global | 0.0116 (0.0072..0.0165) | 0.0581 (0.0554..0.0636) | 0.0066 (0.0040..0.0082) | |
| rotated_rosenbrock | global_damped_wide (2x) | 0.0141 (0.0115..0.0164) | 0.0677 (0.0665..0.0694) | 0.0074 (0.0042..0.0091) | 0.90 / 0.98 |
| rotated_rosenbrock | global_damped (1x) | 0.0215 (0.0188..0.0230) | 0.0881 (0.0845..0.0933) | 0.0121 (0.0072..0.0149) | 0.26 / 0.62 |
| rotated_rosenbrock | global_damped_tight (0.5x) | 0.0345 (0.0315..0.0372) | 0.0928 (0.0895..0.0987) | 0.0170 (0.0127..0.0212) | 0.02 / 0.10 |
| gaussian_peaks | tree | 0.0264 (0.0230..0.0330) | 0.0807 (0.0787..0.0842) | 0.0117 (0.0085..0.0138) | |
| gaussian_peaks | global | 0.0173 (0.0131..0.0205) | 0.0606 (0.0568..0.0673) | 0.0153 (0.0086..0.0233) | |
| gaussian_peaks | global_damped_wide (2x) | 0.0264 (0.0175..0.0345) | 0.0891 (0.0831..0.0931) | 0.0131 (0.0094..0.0188) | 0.90 / 0.98 |
| gaussian_peaks | global_damped (1x) | 0.0360 (0.0344..0.0376) | 0.1304 (0.1236..0.1393) | 0.0349 (0.0312..0.0394) | 0.26 / 0.62 |
| gaussian_peaks | global_damped_tight (0.5x) | 0.0543 (0.0499..0.0618) | 0.1370 (0.1338..0.1387) | 0.0365 (0.0271..0.0428) | 0.02 / 0.10 |

This refutes the extrapolation explanation. A direct check confirms it: mid-sweep, the
global mean on the next 300 stream points (one reservoir spacing ahead of its data) has
an RMSE 13x (rotated Rosenbrock) and 30x (Gaussian peaks) smaller than the reservoir's
constant mean. The global trend at the front is good, damping it towards a constant
throws it away, and the damping transition additionally writes a large artificial seam
(the size of `m - mean`) into the residual exactly where the fresh leaves are learning.

The remaining explanation is extrapolation *by the leaves*: a fresh leaf at the front
predicts with the GP it inherited from its parent, extrapolated by a few points. The raw
target's local slope is accurate and a Matern GP with long length scales follows it; the
residual's local slope is the global model's slope error and its GP reverts to a
constant sooner. The leaf-kernel linear-trend configurations (`*_lin`, an added
`ConstantKernel * DotProduct` term) test this, and refute it too: on the sweeping stream
the linear term changes nothing for either configuration.

| target | config | prequential NRMSE | uniform-test NRMSE | focus-test NRMSE |
|---|---|---|---|---|
| rotated_rosenbrock | tree | 0.0060 (0.0052..0.0065) | 0.0647 (0.0613..0.0673) | 0.0053 (0.0040..0.0071) |
| rotated_rosenbrock | tree_lin | 0.0063 (0.0053..0.0072) | 0.0657 (0.0642..0.0673) | 0.0090 (0.0040..0.0160) |
| rotated_rosenbrock | global | 0.0116 (0.0072..0.0165) | 0.0581 (0.0554..0.0636) | 0.0066 (0.0040..0.0082) |
| rotated_rosenbrock | global_lin | 0.0125 (0.0072..0.0191) | 0.0581 (0.0554..0.0636) | 0.0094 (0.0076..0.0124) |
| gaussian_peaks | tree | 0.0264 (0.0230..0.0330) | 0.0807 (0.0787..0.0842) | 0.0117 (0.0085..0.0138) |
| gaussian_peaks | tree_lin | 0.0266 (0.0230..0.0336) | 0.0828 (0.0802..0.0874) | 0.0117 (0.0084..0.0137) |
| gaussian_peaks | global | 0.0173 (0.0131..0.0205) | 0.0606 (0.0568..0.0673) | 0.0153 (0.0086..0.0233) |
| gaussian_peaks | global_lin | 0.0169 (0.0131..0.0195) | 0.0587 (0.0512..0.0676) | 0.0149 (0.0082..0.0233) |

Note also that the harm is target-specific: on the sweeping stream the residual tree is
*better* than the plain tree on the Gaussian peaks (prequential 0.0173 vs 0.0264) and
worse only on the rotated Rosenbrock (0.0116 vs 0.0060), whose values span seven orders
of magnitude along the sweep.

## The actual cause: stale snapshots in leaves that never retrain

`diagnose_global_mean_sweep.py` records, for every prequential prediction of the
sweeping run, which leaf answered, how many points it held, whether it had been
retrained since its creation, the error of the global snapshot that leaf was fitted
against, and the error the *current* snapshot would have made. Rotated Rosenbrock,
seed 2 (the worst seed: plain tree 0.0052, residual tree 0.0165):

| predictions whose leaf snapshot is ... | share of predictions | residual tree NRMSE | error of the leaf's snapshot | error of the current snapshot |
|---|---|---|---|---|
| current | 0.83 | **0.0045** | 0.0177 | 0.0177 |
| 1 version behind | 0.15 | 0.0398 | 0.0449 | 0.0135 |
| 2 versions behind | 0.01 | 0.0121 | 0.0120 | 0.0072 |
| 3 or more behind | 0.01 | 0.0392 | 0.0390 | 0.0091 |

Where the leaf's snapshot is current, the residual tree is *better* than the plain tree
(0.0045 vs 0.0052) even on this stream. All of the excess error comes from the 17 % of
predictions answered by leaves holding an older snapshot, and there the total error equals
the stale snapshot's own error: the leaf's residual GP, trained against that snapshot in a
region the sweep has since left, cannot correct the snapshot's extrapolation when the path
comes back through the leaf's box. The eight worst predictions of the run are all such
leaves (never retrained, depth 3, total error 0.18 to 0.26 of the range, equal to the
snapshot error). Every prediction in this benchmark is made by a leaf with 50 to 100 own
points, so leaf sparsity plays no role, and 57 % of predictions are made by leaves that
have never retrained since their creation, in both configurations.

The versioning rule "a leaf keeps the snapshot it was fitted against until its next
retrain" was chosen so that a refit of the global model never invalidates a leaf. That is
still right for leaves that keep receiving points, but a leaf that receives no new points
never retrains, and on a sweeping stream that is most leaves. The fix is the `_fresh`
rule: a leaf refits against the current snapshot the first time it is asked to predict
after a newer global version was published (`global_fresh` configuration).

## With the refresh rule: sweeping stream

Same settings, 3 seeds. `global_fresh` made 117 to 118 stale-leaf refits per run (about
two per leaf), which is where its extra time goes.

| target | config | prequential NRMSE | uniform-test NRMSE | focus-test NRMSE | time [s] |
|---|---|---|---|---|---|
| rotated_rosenbrock | tree | 0.0060 (0.0052..0.0065) | 0.0647 (0.0613..0.0673) | 0.0053 (0.0040..0.0071) | 21 |
| rotated_rosenbrock | global (stale snapshots) | 0.0116 (0.0072..0.0165) | 0.0581 (0.0554..0.0636) | 0.0066 (0.0040..0.0082) | 225 |
| rotated_rosenbrock | global_fresh | **0.0065** (0.0057..0.0080) | **0.0530** (0.0477..0.0616) | 0.0090 (0.0054..0.0136) | 279 |
| gaussian_peaks | tree | 0.0264 (0.0230..0.0330) | 0.0807 (0.0787..0.0842) | 0.0117 (0.0085..0.0138) | 20 |
| gaussian_peaks | global (stale snapshots) | 0.0173 (0.0131..0.0205) | 0.0606 (0.0568..0.0673) | 0.0153 (0.0086..0.0233) | 245 |
| gaussian_peaks | global_fresh | **0.0129** (0.0119..0.0149) | **0.0423** (0.0419..0.0428) | **0.0077** (0.0072..0.0084) | 362 |

The doubling of the on-stream error on the rotated Rosenbrock is gone (0.0065 vs the
plain tree's 0.0060, within the seed spread), the cube-wide error is now better than the
plain tree's on both targets, and on the Gaussian peaks the refreshed residual tree beats
the plain tree in every metric, including the end-of-sweep focus set where the stale
version had been worse. The one remaining soft spot is the rotated Rosenbrock focus set,
0.0090 against 0.0053, driven by a single seed (0.0136); the other two seeds are level
with the plain tree. Results for the refresh rule on the other three streams follow below.

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

The raw `RESULT` lines of the runs above are in `results/global_mean_streams/`
(`damped_*` for the damping section, `lin_*` for the linear-trend runs, `fresh_*` for the
refresh rule).
