# Design notes

Why the tree has a global model, how the mechanism works, how the tree stays
within a fixed size, how a point reaches a leaf's GP between two fits, how a
fit can run in another process, how a batch of points is added, how a leaf
models a curve, what was tried and dropped, and what is still open. The
measurements are in [`benchmarks.md`](benchmarks.md).

## What is global and what is local

A leaf GP learns the shape of the function from its own `Nbar` points. A
stationary kernel with one length scale per input needs a number of points that
grows like `(box side / length scale)^d` to pin a function down, and splitting
only shrinks the box side. In ten dimensions a leaf of 100 points with eleven
kernel hyperparameters is underdetermined, and away from its points its
prediction reverts to the leaf's constant mean.

So the split that matters is between what is global and what is local. One
model trained on everything the tree has seen carries the large-scale shape of
the function. The leaves keep what a partition is good at: the local
correction, an uncertainty tied to the local density of the data, and a
calibration of that uncertainty from local residuals.

## What the tree provides, whatever the leaf model

* **A partition that follows the data.** A leaf only has to be right in its
  box. The `min_lengthscale` criterion cuts along the dimension in which the
  leaf's GP says the function varies fastest, or across the direction of its
  shortest length scale if its kernel is oblique (below).
* **A bounded cost per point.** A point is placed once, a retrain touches at
  most `Nbar` points, and inner nodes drop their data (Lederer et al. 2020).
* **Per-leaf calibration.** Each leaf keeps its last 25 normalised prediction
  errors and scales its sigma to their 68% quantile. This holds the on-stream
  coverage at the nominal value for every leaf model and global model measured.
* **A sigma that grows beyond a leaf's points** (`widen_sigma_with_distance`,
  on by default). The distance of a query from a leaf's points is its
  distance from the mean of the points of the leaf's last fit in units of
  their standard deviation, the largest over the inputs. Beyond a distance of
  2 the calibrated sigma is multiplied by 1 + slope * (distance - 2). The
  slope is one number for the tree: over the last 200 predictions registered
  beyond 2, the 68% quantile of the errors in units of sigma, minus 1,
  divided by their mean distance beyond 2. The predictions do not change, and
  on a stream that stays in a region hardly any query is beyond 2.
* **Blending near a cut.** A prediction in the overlap region of a split is a
  mixture over the leaves the point could belong to.

## Length scales along combinations of the inputs

A kernel with one length scale per input, in a tree that cuts along one input,
fits a target whose effects lie along the inputs. If the target changes fastest
along a combination of them (parameters with partly degenerate effects), every
input involved gets a short length scale and no cut separates the fast
direction: the leaf resolves more dimensions than the target has.

* **The kernel.** `Default_GPR(oblique=True)` gives the leaves an
  `ObliqueMatern` kernel: a Matern kernel of the distance
  `r^2 = |(x - x') L / length_scale|^2` with `L` unit lower triangular. Its
  entries below the diagonal (the shear) are hyperparameters, any positive
  definite metric is of this form, and without shear it is the default
  kernel. The likelihood has a slope in the shear where all length scales are
  equal, which it does not have in the angles of a rotation.
* **A fit keeps the shear if it helps.** A leaf is fitted with the shear at
  zero and then with it free, from that fit and from the shear its kernel
  started with. The free fit is kept if its leave-one-out error is lower by a
  factor 1.1. A leaf that declines predicts as with the default kernel.
* **A bounded cost.** The free fit has at most 50 steps of the optimiser,
  and after a rejection it is left out of the next 1, then 3, then 7 fits.
* **The cut.** A leaf whose kernel has a shear is cut across the eigenvector
  of its metric with the largest eigenvalue, with the inputs in units of
  their spread, at the median of its points along that direction.
* **Both are needed.** In one leaf of curves in a rotated parameter space the
  kernel alone gave the error of the default kernel on the unrotated
  parameters or less. In a tree with `'shared'` leaves and cuts along the
  inputs it left the error at 1.3 to 3.1 times that on the unrotated
  parameters (10 times for a kinematic edge), and with the cuts across the
  leaves' directions at 1.0 to 1.3 times (2.7 times; `docs/benchmarks.md`,
  section 9.11).
* **Many outputs determine the shear better than one.** The likelihood of a
  shared-kernel GP sees every output. On curves no leaf that kept a shear
  made the error higher by more than 2%. On three of eleven single-output
  targets the error was 7 to 20% higher than with the default leaves, where
  the leave-one-out error of 100 points decides on 6 to 45 entries.

## The global model

`GPTree(global_mean=...)` gives the tree one shared model whose residual the
leaf GPs fit (`pygptreeo/global_mean.py`). A leaf with few points then reverts
to the global model instead of to a constant.

* **Snapshots with versions.** A fitted global model is immutable. A leaf
  subtracts the current snapshot when it fits and adds the same one back when
  it predicts.
* **Refresh on first use.** A leaf asked to predict after a newer snapshot
  exists refits against it first. Without this rule a leaf that has stopped
  receiving points keeps an old snapshot, whose error its residual GP cannot
  correct once the stream comes back through its box; on a sweeping stream that
  doubled the on-stream error.
* **A coverage sample.** The global model trains on a maximin reservoir of the
  stream (`CoverageReservoir`), so the sample represents the explored region
  however the stream moves through it.
* **Few refits.** A refit is due when the point count has doubled. The
  additive GP also requires a quarter of its reservoir to have turned over, so
  it stops refitting when the stream stops exploring.
* **An error budget in the sigma.** The learner tracks the mean squared
  prequential error of its current snapshot and subtracts the observation
  noise of the same points (`error_scale`). A leaf adds that variance to its
  own before calibration. The residual GP knows how smooth the residual is on
  the leaf's points; this term accounts for how wrong the global model is at
  new ones. The sigma stays an uncertainty about the underlying function.

Two global models ship:

* **`'additive_gp'`** (`AdditiveGPGlobalMean`): a GP with an order-2 additive
  plus Matérn kernel on a 500-point reservoir. Strong on targets with low-order
  additive structure. Its fit is cubic in the reservoir size and its kernel has
  `d (d - 1) / 2` pair terms, so it becomes slow beyond about six dimensions.
* **`'net'`** (`FeatureNetLearner`, `NetGlobalMean`): an MLP trained at a fixed
  cost per refit, not limited to additive structure. Each refit runs L-BFGS on
  the reservoir and then Adam steps over the stored points (every point of the
  stream up to `store_size`, a uniform sample of it after that), and publishes
  the weights with the lowest loss on a held-out fifth of the points. A leaf
  takes up a new version by refitting on the new residual with its kernel
  hyperparameters kept.

## A memory limit

`GPTree(max_memory_mb=...)` keeps the tree within a fixed size on an endless
stream. One setting in MB replaces a count of leaves because a leaf has no
fixed size: a fitted leaf takes 40 kB at `Nbar=50` in three dimensions and
370 kB at `Nbar=200` in six.

* **Where the memory is.** A fitted leaf at `Nbar=100` in six dimensions takes
  about 113 kB: 80 kB for the Cholesky factor of its GP, 7 kB for the GP's
  copy of the training data and its weights, 6 kB for the leaf's own and shared
  points, and about 20 kB for Python objects (the node, an inner node, scalers,
  kernels). The factor follows from the training data and the kernel.
* **A sleeping leaf** has released the factors and weights of its GPs and
  keeps everything else. Restoring them takes one kernel matrix and one
  factorisation, and the GP then predicts as before: exactly, or to the
  numerical precision of the factorisation if points were appended since its
  fit (about 1e-8 of the target's range in the mean). The GP's own
  training data are kept for this: rebuilt from the leaf's points, the GP would
  also include the points that arrived after its last fit, and a prediction
  would depend on whether the leaf had slept.
* **A leaf is woken before a point is stored in it**, so that the prediction
  registered for the calibration is still one made without the point.
* **Two orders.** Leaves go to sleep in the order in which their GPs were last
  used, by a point or by a prediction. Leaves are removed in the order in
  which a point last reached them. A prediction can therefore keep a GP awake,
  but it does not protect a leaf from removal, and it never causes one: a
  prediction that wakes a leaf sends another to sleep.
* **Sleep first, then remove.** Below the limit nothing sleeps. At the limit
  the least recently used leaf is put to sleep while the GPs awake take more
  than half of the memory left for the leaves (`FITTED_SHARE`), and the leaf
  that has gone longest without a point is removed otherwise. With an even
  split the tree never has fewer than half as many GPs awake, or half as many
  leaves, as another split would give it.
* **The tree adds up its own size.** The arrays of the leaves and their GPs
  are counted exactly. Their other objects are counted as constants, 10 kB per
  leaf and 10 kB per GP of a leaf (`NODE_OVERHEAD_BYTES`,
  `GPR_OVERHEAD_BYTES`), so that a run does not depend on the Python version.
  The estimate was within 5% of the memory Python had allocated for the tree
  in six configurations. The global model and the output basis count at the
  size they have once their samples are full, and a limit below that is an
  error.
* **A removal changes the structure only.** The leaf and its parent are
  removed, and the leaf's sibling, a leaf or a subtree, takes the parent's
  place. The leaves along the removed cut then extend over the removed leaf's
  region. No remaining leaf changes its points or its GP, so a removal costs
  no fit.
* **The removed leaf's points are discarded.** With gradual splitting much of
  them is still in the tree: the sibling is a leaf in 47 to 91% of the
  removals, and it then usually holds copies of about half of the removed
  leaf's points as shared points. Beyond that, the memory of the regions the
  stream has left is the global model, whose reservoir is a coverage sample of
  the whole stream: a leaf that extends over a forgotten region adds its
  extrapolated residual to the global model there.
* **The network's point store is a sample.** Beyond `store_size` points
  (100 000) the store is a uniform sample of the stream, with the stream's own
  distribution of points. The Adam phase of a refit draws about 400 000 points
  (3000 steps of 128), a few passes over the store.

## Points between two fits of a leaf

* **Appended points** (`append_points_between_fits`, on by default). Between
  two fits of a leaf a point is added to its fitted GPs as they are: the
  Cholesky factor gets one more row, at a cost quadratic in the number of
  points. The hyperparameters, the scaling and the global-model snapshot of
  the last fit are kept, so the result is the GP that a fit with fixed
  hyperparameters to the points of that fit and the new one would give. The
  next fit starts from the leaf's points as usual, so nothing accumulates.
  The prediction registered for the calibration is made before the point is
  appended.
* **With every point appended the retrain interval matters little**: the
  hyperparameters change slowly, and what a leaf lacked between two fits was
  the points themselves.

## Fits in other processes

`update_tree(defer_fits=True)` returns the fit that a leaf is due as a task,
`solve_fit_task` carries it out in any process, and `apply_fit_result` gives
its result to the tree. The tree has no part in sending them.

* **A fitted leaf is its points and its hyperparameters.** Optimising the
  hyperparameters takes 60 ms for 100 points in six dimensions, and three
  quarters of the time of an update. The factorisation that follows from the
  points and the hyperparameters takes 1.3 ms. So the points stay with the
  tree and only the optimisation is handed out.
* **The result is the hyperparameters alone.** The leaf is refitted with them
  to the points it holds when the result arrives, which include those that
  arrived in the meantime. A GP fitted by the worker would lack them. Applied
  at once, the result gives the same leaf as a fit in place.
* **The result is applied in the frame of its task.** The hyperparameters
  belong to the scaling of the points they were found for, so the tree keeps
  the scalers of a task until its result arrives and the leaf takes them with
  the hyperparameters. The leaf is then as fitted in place when the task was
  made, with the points since then appended. Scalers fitted to the leaf's
  points at that moment would differ once it has split: a result applied to
  six leaves two splits below its task gave twice the error. The residual of
  a task is that of one snapshot of the global model, so a result is dropped
  if there is a newer snapshot.
* **A task is self-contained.** It holds an unfitted copy of each GP of the
  leaf with the scaled inputs, targets and noise, about 7 kB for 100 points in
  six dimensions, so a worker needs no tree and no settings. The result is the
  task's number and one array per GP.
* **A leaf waits for its result.** It is handed no other task before the
  result is applied, so no more tasks are out than the tree has leaves, and
  results cannot arrive in the wrong order. Under load a leaf is refitted
  less often and nothing queues up. A worker has to answer every task, with
  `hyperparameters=None` for a fit that failed.
* **A full leaf splits without its fit.** It asks for the fit and splits with
  the hyperparameters it has. Its children inherit the wait, and the result
  refits every leaf under the node the task was made for.
* **Tasks that are out are not saved with the tree.** A loaded tree has none,
  and a leaf of it asks again with its next full retrain buffer.

## A batch of points

* **`batch_update_tree` adds the points one by one**, whatever the tree
  holds. Point by point is the most accurate way into a tree with points in
  every case measured (`docs/benchmarks.md`, section 8.2), and it keeps the
  calibration, which needs the prediction of each point before it is added.
* **It does not use `fit` for a tree without points.** `fit` is about 2.5
  times as fast there at about the same error, but its sigma is not
  calibrated, it shuffles the points and it hands no fits out, so the same
  call would behave differently with the state of the tree. A tree of ten
  points that takes 5000 is also in the situation of a tree without points:
  what matters is the size of the batch against that of the tree. The choice
  is the caller's.
* **`fit` is for a tree without points.** It sets the root up anew, so it
  raises an error for a tree that holds points.

## Curves as outputs

A multi-output target is often a function `f(x; theta)` of a physics quantity
`x`, given on a grid in `x`, with the parameters `theta` as the inputs of the
tree. `examples/target_functions_curves.py` has twelve families of such
curves, and `benchmarks.md`, section 9, what was measured on them.

* **A leaf's prediction is a combination of its curves.** The weights come
  from the kernel: one set per grid point with `'independent'`, one per
  component with `'pca'`, one for the whole curve with `'shared'`. On curves
  without noise the three have the same error within a factor of 2, so
  `'shared'` is the reference: one GP per leaf.
* **The fit of a shared-kernel GP is computed from `Y Y^T`.** Its likelihood
  depends on the targets `Y`, of N points and p outputs, only through that
  N x N matrix, so the adapter optimises the hyperparameters at a cost that
  does not grow with p. The optimum is that of scikit-learn's own fit, which
  takes 9 times as long for 128 outputs and 53 times for 1000.
* **A sigma per output from leave-one-out errors.** A shared-kernel GP has
  one variance for all outputs, whether an output varies across the leaf or
  not. Its predictions of each of the leaf's curves from the others follow
  from the inverse of the kernel matrix, without a refit, and each output's
  sigma gets the factor with which it covers 68% of that output's
  leave-one-out errors.
* **A smoother kernel for curves from a smooth model.** The default leaf
  kernel, Matérn 3/2, was chosen on rough targets and on the residual of the
  network (`benchmarks.md`, section 4.4). The curves here are smooth in their
  three or four parameters, and with Matérn 5/2 leaves
  (`Default_GPR(nu=2.5)`) the error of `'shared'` is 3.1 times lower in the
  geometric mean over the twelve families, between 0.9 and 58 times. RBF
  leaves are better still on `z_lineshape` and `cmb_peaks`, by a factor 2 to
  4, and 7 times worse on `zprime_spectrum`. The default is unchanged: on
  the eleven single-output physics targets, whose thresholds and cusps lie in
  the input space, Matérn 5/2 leaves have the same error in the geometric
  mean and a higher one on eight. A feature along the grid of a curve costs
  the kernel nothing as long as it does not move (section 9.8). Where a jump
  moves along the grid, each output is a step in the parameters, and Matérn
  5/2 leaves have 4 times the error of the default (section 9.9).
* **What limits the accuracy is a feature that moves.** Where a peak, an edge
  or the phase of an oscillation has a position in `x` that depends on
  `theta`, a combination of curves with the feature at different positions
  has several weaker features, and none in between. The error then falls
  slowly with the number of curves.
* **`'aligned'`: each leaf brings the features of its curves to common
  positions.** Take a spectrum with a resonance whose mass is one of the
  parameters. The curves of a leaf have their peaks at different masses. The
  leaf stretches the grid of each curve in some places and compresses it in
  others, by an amount that depends on the curve's parameters, until the
  peaks lie on top of each other (the registration of functional data). The
  aligned curves differ in the height and the width of the peak, which change
  slowly with the parameters, and its position is carried by the stretch.
* **The stretch is a warp of the grid.**

      phi(u; delta) = (p - 1) I(u) / I(p - 1),    I(u) = int_0^u exp(delta . w(s)) ds

  `u` is the grid index, `delta` the offset of `theta` from the mean of the
  leaf's points in units of their standard deviation, and each `w_i` is
  piecewise linear on 8 knots. The aligned curve is `g(u) = f(phi(u; delta))`.
  The warp is monotone, maps the grid onto itself, is smooth in `delta` and is
  the identity at the leaf's centre. It is 8 numbers per input.
* **The warp makes the aligned curves linear in `theta`.** It minimises what a
  fit linear in `delta` leaves of the aligned curves, by least squares. A
  feature that only moves is then carried by the warp, and the GP is left
  with curves that change slowly. A leaf's first fit is also run from a warp
  fitted to blurred curves, since narrow features that lie apart do not pull
  on each other, and the better of the two results is kept. A later fit
  starts from the leaf's previous warp.
* **A prediction warps each curve into the frame of the query** and combines
  the warped curves with the weights of the GP mean (transformed snapshot
  interpolation, Welper 2017). Each curve is interpolated once, at
  `phi_n(phi^-1(x))`, and a curve of the leaf is returned exactly at its own
  `theta`. Predicting the aligned curve and warping it back interpolates
  twice, and leaves an error that does not fall with the number of curves.
* **The leaf decides by leave-one-out.** It fits one GP to the curves and one
  to the aligned curves, and keeps the alignment if it predicts each curve
  from the others with an error lower by a factor 1.5. On the families without
  a feature that moves every leaf declined, and the predictions are those of
  `'shared'`. After a rejection the next attempt waits for 1, then 3, then 7
  fits.
* **An alignment learns the grid one would choose by hand.** Where the
  position of the feature is known, the curve can be given on a grid relative
  to it, and nothing moves. For a kinematic endpoint the leaves of `'shared'`
  on a grid in `m / endpoint` have the error of `'aligned'` on the grid in
  `m` (an RMSE of 3.0e-4 and 2.9e-4 events, against 3.9e-3 for `'shared'` on
  the grid in `m`). For a Z' peak the grid in `m / M` is the better of the two
  (0.015 against 0.025 dex, and 0.10 dex for `'shared'` on the grid in `m`),
  and an alignment on top of it lowers the error to 0.009 dex, since the dip
  below the peak still moves.
* **The grid has to resolve the feature.** An alignment interpolates the
  curves between the grid points, by cubic splines. The gain on a kinematic
  edge was a factor 9 where the grid has three points across the edge or
  more, 1.4 to 3 with half as many, and none with a quarter, where every leaf
  declined.
* **Images: a warp of each axis that varies along the other.** With
  `output_shape=(p1, p2)` the outputs are an image, flattened row by row, and
  its grid is warped as

      (u, v) -> (phi_1(u; v, delta), phi_2(v; u, delta))

  `phi_1` is the warp above along the first axis, with a `w` that varies
  linearly between two knots along the second axis, and `phi_2` the same with
  the axes exchanged: 32 numbers per input. Rows can so move by different
  amounts, and so can columns. The warp is fitted like that of a curve, by
  damped Gauss-Newton steps on at most 32 points along each axis.
* **The exponent of that warp is bounded by a `tanh` and has a ridge.** Where
  an image is flat nothing determines the warp. With the hard bound and the
  weak ridge of the curves it grew there until it met the bound, and at the
  corners of a leaf, where several offsets are large at once, it misplaced
  the feature: a few queries had 300 times the median error, which the
  leave-one-out errors did not show. With the smooth bound and a ridge of
  0.03 the leave-one-out error of a leaf is that of its test points
  (`docs/benchmarks.md`, section 9.12).
* **A prediction of an image is the aligned image, read back.** The GP is
  fitted to the aligned images on a grid with twice the points along each
  axis, and its mean is read where each grid point of the query comes from,
  found by Newton steps on the warp. Warping each of the leaf's images into
  the frame of the query, as for curves, is the same prediction where the
  finer grid resolves the aligned images, and took 12 times as long.
* **Both global models take the curve as their target.** The network has one
  output per grid point and the additive GP one kernel for all outputs, fitted
  from `Y Y^T` like the GP of a leaf, and a leaf models the residual curves.
  With 800 curves and `'shared'` leaves the additive GP lowered the error by a
  factor 3.6 to 70 on the four smooth families, by 3.5 on the Z line shape and
  2.6 on the oscillation probability, and by 1.2 to 1.7 on five others. The
  network lowered it by a factor 2.3 to 6 on five families and raised it by up
  to 2.2 on the other seven, where the tree alone is as accurate as the
  network (`benchmarks.md`, section 9.7).
* **Much of that gain is the gain of a smoother model.** With Matérn 5/2
  leaves the additive GP with its default kernel raises the error on four of
  eight families (`cmb_peaks`: 1.1e-4 against 3.3e-5): its Matérn 3/2 part
  leaves a residual that is rougher than the curves. With Matérn 5/2 in the
  global kernel too (`global_mean_kwargs=dict(nu=2.5)`) it lowers the error
  of `'shared'` leaves on the ten families without a sharp feature, by a
  factor 1.2 to 2.4, and raises it by 10% on the other two.
* **The additive GP gains most where its kernel fits the target.**
  `eft_distribution_6d` is quadratic in six coefficients, which is what an
  additive kernel of order 2 describes: the error is 40 to 130 times lower
  than with Matérn 5/2 leaves alone, and 850 to 2700 times lower than with
  the default leaves.
* **No global model follows a sharp feature that moves.** On
  `zprime_spectrum` and `kinematic_edge` the additive GP changed the error by
  less than a factor 1.3 and the network raised it by up to 1.6.
* **A leaf with an alignment does without the global model.** What a global
  model leaves of a peak that it places slightly wrong does not move with the
  parameters, so the residual curves cannot be aligned: with a global model
  every leaf of `'aligned'` declined, and the gain of the alignment was lost.
  A leaf therefore tries an alignment of the curves themselves. If it keeps
  it, it models the curves. If it declines, it models the residual of the
  global model, as a leaf of `'shared'` does. With the additive GP the error
  on each of the thirteen streams was then within 10% of the lower of
  `'aligned'` alone and `'shared'` with the global GP, and 3.6 times lower
  than that of `'shared'` alone in the geometric mean. With Matérn 5/2 in
  the leaves and in the global GP it was 4.8 to 5.3 times lower than that of
  the default `'shared'` leaves over three seeds, and 3.9 to 4.6 times
  without the global model.

## Tried and dropped

* **Neural-linear leaves**: a Bayesian linear regression per leaf on the
  network's last-layer features, in place of the GP. On smooth targets they
  matched the hybrid at half its update cost. On rough targets they were 5 to
  55% behind the plain tree, and on a target with a jump a few predictions were
  wrong by thousands of times the target's range: a leaf that is linear in the
  features extrapolates without bound where a point's features lie outside the
  leaf's cloud, while a GP leaf reverts to the network there.
* **Damping the global model by its own uncertainty**, so that the residual
  reverts to the raw target where the model extrapolates. The fitted global
  GP's predictive variance is uninformative (below 0.1% of the prior even far
  from the data), and damping by distance to the reservoir made every metric
  worse: the global trend ahead of the data is better than a constant.
* **A linear-trend term in the leaf kernel.** No effect.
* **A floor on the predicted sigma at the observation-noise level.** It bounds
  the calibration scaler, but turns sigma into the uncertainty of a noisy
  observation. The error budget above addresses the term that was missing.
* **Appending only the points far from their leaf's other points**, in place
  of every point: half of the gain on a sweeping stream, and none on the
  streams that stay in a region, where every point appended lowers the error
  by 7 to 23%.
* **Points beyond a leaf's cuts**, with every query predicted by one leaf. An
  overlap of 0.05 for the stored points alone lowered the error by 3% over
  18 runs (4% and 6% on the uniform and sweeping streams, none on the
  walker), and overlaps of 0.2 and 0.5 did worse. Leaves that keep 25 of
  their sibling's points nearest the cut gained 6%, most of which was the
  larger GP. Neither changed the largest errors (`docs/benchmarks.md`,
  section 7.6).
* **Other forms of the sigma beyond a leaf's points.** A factor per band of
  the distance, each learned from the errors in its band: the same coverage
  as the one slope, with a list per band. The slope as the 68% quantile of
  the slopes that each error would need: learned from queries just beyond 2,
  it reached 40 on uniform streams.
* **A leaf of its own for new territory**: a point far from its leaf's other
  points splits the leaf between those points and itself. The error rose by 73% on a sweeping stream and
  90% on a walker: the new leaves have too few points to predict from.
* **Another leaf for a query far from its leaf's points** (`far_threshold`).
  The leaves at the edge of the explored region own everything beyond it, and
  a stream that moves on can enter the region of a leaf whose points lie far
  away. A query more than 7 standard deviations from the mean of its leaf's
  fitted points, along some input, was given to the leaf of its family with
  the nearest points; the family has, for each ancestor, the leaf the query
  reaches beyond the ancestor's cut. On the seed the threshold was chosen on,
  the error on moving streams fell by 10% with no run more than 3% worse. On
  two further seeds single runs changed by -47% to +35%, with a mean of -4%
  on a sweeping stream and +2% on a walker (`docs/benchmarks.md`, section
  7.4). Where every candidate leaf is far from the query, no distance tells
  which one is right. No better: lower thresholds (up to +53% in single runs
  at 4), the recently updated leaves as the candidates, nearness by distance
  from the leaf's mean, distances in units of the kernel's length scales or
  of the GP's variance, a margin on the nearness, a sigma that grows with the
  distance, and the nearest leaf for every query.
* **A decision function of its own for predictions.** The cuts decide where a
  point is stored, and the same cuts decide which leaf predicts a query. A
  rule for the queries alone can use the distance from the query to each
  candidate leaf's points (`docs/benchmarks.md`, sections 7.3 and 7.4):
  * The distance has to be in units of the leaf's own spread. In units of
    the inputs' range the leaf with the nearest points is the worse one in
    most runs.
  * A blend of the own leaf and its neighbour beyond the nearest cut,
    weighted by that distance, lowers the error in 117 of 140 runs, by 6% on
    average, on every stream type, and the share of predictions off by more
    than 10 sigma from 0.8% to 0.6%. It predicts with 1.7 leaves per query.
  * A leaf can be wrong by many times the target's range just beyond its
    cut, with a small sigma, and no distance shows it. Two such queries made
    one run 4.9 times worse. Using the blend only where the two predictions
    agree within 5 sigma left no run more than 2% worse and a gain of 5%,
    but it keeps the own leaf's largest errors.
  * An overlap of 0.2 for the queries alone gave about the error of that
    overlap for both, 10 to 18% below the default on `GaussianPeaks`, with a
    sigma that is too large and up to four times the prediction time.
  * Leaves chosen by a query's nearest stored points (a vote of the 25
    nearest, or the nearest leaf centroids) predict 2 to 4 times worse than
    the own leaf when every input is in units of its spread over the stream.
    In units of the spread of the own leaf's points, and with the other
    leaves used only where they agree with the own leaf within 5 sigma, the
    vote lowered the error by 8% on two seeds with no run worse, at 3.4
    leaves per query. A GP on the 100 nearest points with the own leaf's
    hyperparameters gained 13% under the same test and was worse in 5 to 9
    of the runs of a seed.

  Not pursued: the gains do not pay for several leaves per query.
* **Merging a removed leaf and its sibling back into their parent**, with the
  parent's points and hyperparameters as they were at the split. On four
  40 000-point runs the error was 11 to 49% higher than with plain removal and
  the tree split 31 to 140% more often: the restored parent is full and splits
  again at its next point, and it has lost the points that arrived after the
  split. With gradual splitting the sibling already holds about 80 of the
  parent's 100 points. A variant that kept the pair's newest points was within
  the seed-to-seed spread of plain removal.

* **Other warps for `'aligned'`** (`docs/benchmarks.md`, section 9.6):
  * Free ends, i.e. a shift and a stretch, with a curve continued beyond the
    grid by its value at the edge: worse than the warp that maps the grid
    onto itself in 11 of 12 single-leaf runs.
  * Sixteen knots, or a warp that is also linear in the squares of the
    offsets: no family gained more than 10%, most lost, at 2 to 7 times the
    time.
  * Every fit started from blurred curves: the gain on a narrow peak rose
    from 2.0 to 3.2, and that on a damped oscillation fell from 2.6 to 1.3,
    since the blur removes the oscillation. Hence both starts.
* **Weights of a representation of the curve as the outputs**
  (`docs/benchmarks.md`, section 9.10):
  * Coefficients of a fixed B-spline basis. With a shared kernel the
    predictions are those of `'shared'`, and a GP per coefficient helped on
    one of four families at 15 times the time: a feature that moves makes
    every coefficient as rough in the parameters as the curve is in `x`.
  * Knots at equal steps of each curve's own arc length, and the values
    there. 1.2 to 3.3 times more accurate than `'shared'` on a single sharp
    feature, 1.6 to 23 times less accurate on two peaks, eight peaks and noisy
    curves, and behind `'aligned'` on every family. For one sharp feature the
    knots are far closer to linear in the parameters than the grid values,
    and for eight peaks further from it.
  * `'aligned'` leaves on the B-spline coefficients: the error of `'aligned'`
    on the grid at half the time where half as many coefficients as grid
    points represent the curve, and far worse where they do not.
* **Global networks that take `x` into account** (`docs/benchmarks.md`,
  section 9.7), each alone on 800 curves against the network with one output
  per grid point:
  * The grid coordinate as one more input, `(theta, u) -> f(u; theta)`:
    2 to 2.7 times more accurate on the two families with a sharp feature
    that moves, 20 to 65 times less accurate on smooth and oscillating
    curves, at 15 times the time of a fit.
  * A warp of the output grid, from a second head that gives the knots of a
    monotone map at which the network's curve is read: 2.2 to 2.5 times more
    accurate on the sharp features and 1.8 on the noisy histograms, 2 to 5
    times less accurate on `matter_power`, `cmb_peaks` and `ringdown`. As
    the global model of `'shared'` leaves it lowered the error on
    `zprime_spectrum` by a factor 1.4 and on `kinematic_edge` by 1.6, where
    the alignment in the leaves gives 4 and 14.
* **A shared-kernel GP on outputs in units of their noise**, for noise that
  differs between the outputs: within 2% of the common scale with 800 and
  3200 noisy curves, and 5 to 16% worse with 200.
* **`x` as one more input of a single-output tree**, in place of the curve as
  a multi-output target: with `fit` 2 to 80 times the error with 200 curves,
  and a fit that takes 12 to 65 seconds where `'shared'` takes less than one.
  On a stream, and with the network as its global model, the single-output
  tree has the error of `'shared'` with Matérn 5/2 leaves on `z_lineshape`
  and half of it on `zprime_spectrum`, at 100 times the time
  (`docs/benchmarks.md`, section 9.7).

## Known limitations

* **Sigma far from the data.** The calibration is fitted on the stream. In
  regions the stream has not visited the coverage of every configuration falls
  to 0.1 to 0.5, and a tree with a global model is more accurate there without
  its sigma growing accordingly.
* **A leaf predicts beyond its own points.** The leaves at the edge of the
  explored region own everything beyond it. On a sweeping stream 27% of the
  queries lie more than 2 standard deviations from the mean of their leaf's
  fitted points, along some input, and carry 81% of the squared error; the
  sigma covers 0.60 of them, and 0.44 of those beyond 7 (`docs/benchmarks.md`,
  section 7.5). Appended points shorten the time a leaf extrapolates, and
  the sigma that grows with the distance then covers 0.70 to 0.78 of them.
  No rule was found for another leaf to predict in its place.
* **The errors have heavier tails than the sigma says.** Over 140 runs 2.2%
  of the predictions are off by more than 5 sigma and 0.8% by more than 10,
  and the five largest of 5000 errors carry 23% of the squared error of the
  median run. Near a leaf's points the sigma that grows with the distance
  changes nothing: 2.5% of the predictions within a distance of 2 are off by
  more than 5 sigma.
* **GPyTorch leaves take no appended points.** The adapter cannot extend a
  fit, so a point enters the GPs with the leaf's next fit.
* **The network lags at the front of a moving stream**, since it refits only
  when the point count doubles. The leaves correct most of this.
* **Regions the tree has forgotten.** Under a memory limit, a prediction in a
  region whose leaves were removed is the extrapolation of a neighbouring
  leaf, on top of the global model if there is one. Its sigma is too small in
  the same way as far from the data.
* **GPyTorch leaves do not sleep.** The adapter cannot release and restore a
  fit, so a memory limit is met by removing leaves only.
* **Not every fit can be handed out.** GPyTorch leaves and the leaves of the
  `'shared'` and `'pca'` output models are fitted in place also with
  `defer_fits`. So is a leaf that meets a new version of the additive global
  GP: it refits before its next prediction.
* **Calibration after a new network version.** Coverage can dip for a batch
  until the per-leaf calibration catches up with the shifted residual.

* **`'aligned'` is for one curve on one ordered grid, or one image.** It
  gains wherever a feature moves along such a grid, in curves from physics
  or not (an epidemic peak, a shock front, a filter resonance: a factor 3 to
  12), and its leaves decline where nothing moves or the outputs are in no
  order, at 1.3 to 1.8 times the time of `'shared'`. Over 64 runs on sixteen
  such targets it was more than 5% worse than `'shared'` in two, by 10% at
  most (`docs/benchmarks.md`, section 9.9). An image needs its
  `output_shape`: as a curve, row by row, it is aligned along the rows only.
  The warp of an image moves its rows and its columns. A feature that turns,
  or grids of more than two dimensions, are beyond it.
* **`'aligned'` follows what stays on the grid.** The warp maps the grid onto
  itself, so a change of scale that moves features across its end, like the
  frequency of an oscillation, is followed only in part. A leaf either
  aligns its curves or uses the global model: where both would help
  (`cmb_peaks` with the network) it has the gain of the alignment only.
* **The calibrated sigma of noisy observations is that of an observation.**
  The calibration compares a prediction with the observed value. With noise
  the sigma grows to the noise level, and covers the underlying function too
  often: 0.98 of the predictions on a stream of Monte Carlo histograms,
  and 0.99 for a single output with a noise of 5% of its range. The GP's own
  sigma (`use_calibrated_sigma=False`) is that of the function.
* **`'pca'` keeps nearly all components of curves without noise**, since few
  are below the declared noise: 125 of 128 for `zprime_spectrum`, where 24
  give the same error within 3% in a fifth of the time.
* **`fit` splits by the spread of the points.** It builds the tree before a
  leaf is fitted, so there are no length scales for `min_lengthscale`.

## Open ideas

* **Thinning sleeping leaves before removing them.** A leaf reduced to its 50
  most informative points (chosen one by one for the largest remaining GP
  variance, hyperparameters kept) had 1.2 times the error of the full leaf in
  a one-step test, and 1.6 times with 25 points. Keeping the hyperparameters
  was better than fitting them again to the remaining points, and the
  calibration has to be redone, for which the discarded points are a held-out
  set.
* **A calibration that takes the declared noise out.** The scale `s` for
  which `sqrt(s^2 sigma^2 + noise^2)` covers 68% of the last residuals, in
  place of the one for which `s sigma` does. Where the noise dominates, 25
  residuals say little about `s`, so it needs a longer window or a prior.
* **The number of `'pca'` components from the prediction error.** Components
  whose variance is below the error of the leaf GPs add fits and no accuracy.
* **A covariance between the outputs.** The leave-one-out errors of a
  shared-kernel leaf also give the correlation of the errors along the grid,
  which a chi-square fit to a predicted curve needs.
* **A basis per leaf.** Across a leaf the curves vary in a few directions,
  far fewer than across the stream, so a leaf's own principal components
  would do with fewer GPs than the global ones.
* **Warps for other grids.** A free end for changes of scale, a grid in two
  dimensions, several curves in one target, or a warp the user supplies
  where the position of a feature is known.
* **Bagged networks for the off-stream sigma.** The spread of a few networks
  trained on resampled points is an uncertainty about the global model that
  grows away from the data (Lakshminarayanan et al. 2017; Osband et al. 2023
  for a cheaper variant).
* **Leaf kernels on a local active subspace.** With the network's gradients at
  a leaf's points, take the leading eigenvectors of their average outer
  product and put the leaf's kernel on those few directions, so a leaf fits
  three or four hyperparameters instead of `d + 1` (Constantine et al. 2014;
  the same bet as LWPR, Vijayakumar et al. 2005).
* **In-context regression leaves.** A leaf with no fit at all: a pretrained
  transformer takes the leaf's points as context and returns a predictive
  distribution (Müller et al. 2023; Hollmann et al. 2025). The tree would
  supply the locality that keeps the context small.
* **Linearised-network kernels.** A leaf GP whose kernel is built from the
  network's parameter gradients (Immer et al. 2021).

## References

* Constantine, Dow, Wang 2014, Active subspace methods in theory and practice,
  SIAM J. Sci. Comput., arXiv:1304.2070.
* Hollmann et al. 2025, Accurate predictions on small data with a tabular
  foundation model (TabPFN v2), Nature.
* Immer, Korzepa, Bauer 2021, Improving predictions of Bayesian neural nets
  via local linearization, AISTATS, arXiv:2008.08400.
* Lakshminarayanan, Pritzel, Blundell 2017, Simple and scalable predictive
  uncertainty estimation using deep ensembles, NIPS, arXiv:1612.01474.
* Lederer et al. 2020, Real-time regression with dividing local Gaussian
  processes, arXiv:2006.09446.
* Müller, Feurer, Hollmann, Hutter 2023, PFNs4BO: in-context learning for
  Bayesian optimization, ICML, arXiv:2305.17535.
* Osband et al. 2023, Epistemic neural networks, NeurIPS, arXiv:2107.08924.
* Riquelme, Tucker, Snoek 2018, Deep Bayesian bandits showdown (the
  neural-linear model), arXiv:1802.09127.
* Vijayakumar, D'Souza, Schaal 2005, Incremental online learning in high
  dimensions (LWPR), Neural Computation.
* Welper 2017, Interpolation of functions with parameter dependent jumps by
  transformed snapshots, SIAM J. Sci. Comput., arXiv:1505.01227.
