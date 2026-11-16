# Design notes

Why the tree has a global model, how the mechanism works, how the tree stays
within a fixed size, how a point reaches a leaf's GP between two fits, how a
prediction can use the points of other leaves, how the leaf of a point is
found, what was tried and dropped, and what is still open. The measurements
are in [`benchmarks.md`](benchmarks.md).

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
  leaf's GP says the function varies fastest.
* **A bounded cost per point.** A point is placed once, a retrain touches at
  most `Nbar` points, and inner nodes drop their data (Lederer et al. 2020).
  The leaf of a point is found through a balanced index of the leaves, since
  the depth of the tree of cuts depends on the order of the points (see "An
  index of the leaves").
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

## Predictions from a query's neighbours

`GPTree(predict_from='neighbours')` conditions a prediction on the stored
points nearest to the query, in place of the points of the query's leaf
(`pygptreeo/neighbours.py`). The measurements are in `docs/benchmarks.md`,
section 8.

* **Most queries lie at an edge of their leaf.** The points of a leaf lie on
  one side of each of its cuts, and in several dimensions most of a cell is
  close to a cut. In the benchmark runs the leaf of a query holds 37% of the
  query's 100 nearest points, and 4% of the queries lie within 0.75 standard
  deviations of the mean of their leaf's points along every input. The gain
  from the neighbours grows with that distance: none for those 4%, 23%
  between 1.25 and 1.75 standard deviations (45% of the queries) and 32%
  beyond 2.5.
* **The leaf supplies the kernel.** The hyperparameters, the scaling of the
  inputs and targets, the calibration of sigma and the global-model snapshot
  are those of the query's leaf. Leaves are fitted, split, put to sleep and
  removed as before. A sleeping leaf is not woken: its kernel is kept.
* **The distance** is Euclidean with each input in units of the spread of the
  leaf's fitted points. The cuts make a leaf narrow along the inputs in which
  the target varies fastest, so the shape of the leaf carries what its kernel
  has learned about the inputs.
* **A kernel describes the points it was fitted to.** A leaf beside a jump or
  a kink of the target has a smooth kernel, since its own points do not
  contain the jump. Conditioned on neighbours from across the jump, that
  kernel overshoots by several times the target's range. A leaf's own GP is
  far less exposed to this, since its kernel is fitted to the points it is
  conditioned on. The mean squared target of the neighbours after
  whitening with the kernel matrix, `y^T K^-1 y / k`, is about 1 where the
  kernel describes them and in the hundreds or thousands in these cases.
  Above `neighbour_consistency` (10) the leaf's own GP predicts, as in the
  tree without neighbours. Limits from 5 to 100 give the same error, and the
  number comes with the factorisation that the prediction needs anyway.
* **A leaf keeps to its own GP while that predicts its points better.**
  Where the declared noise is large against the change of the target across
  a query's neighbours, they say less about its slope than the points of the
  leaf, which lie further apart. On a stream that had contracted into a
  dense cluster of such a target the neighbours were 26% worse than the
  leaves, with a consistency of 0.2: no limit on the consistency tells these
  queries apart, since most queries with a low one are predicted best. So
  each leaf compares the two predictions at the last 50 points it received,
  and predicts from its own points while their RMS error is below 0.9 times
  that of the neighbours (`prefers_own`). That run is then 4% worse than
  the leaves, and the mean error over the 75 runs of a seed, with the rule
  and the search of 16 leaves, 0.6% above that without the two.
* **A point is not its own neighbour.** The prediction registered for the
  calibration is made before the point is stored.
* **The search.** The 16 leaves whose points' boxes are nearest to the query
  are taken from the index of the leaves (`neighbour_leaves`), and the
  neighbours are the nearest points among theirs. With `None` the search is
  exact: the query's leaf bounds the distance of the neighbours, and every
  leaf whose points can lie within it is looked at, 7 to 46 leaves in three
  to five dimensions and 35 to 79 in six to ten at 20 000 points. Sixteen
  leaves give the error of the exact search, eight 0.6% more, four 5%.
* **Cost.** A prediction factorises the kernel matrix of its neighbours, at a
  cost cubic in their number, where a leaf's own GP needs one triangular
  solve: about 0.5 ms against 0.14 ms with 100 neighbours. The leaf's own GP
  is still fitted and extended with each point, for the queries it predicts
  itself, and each new point is predicted both ways before it is stored, so
  an update takes 20 to 40% longer.

## An index of the leaves

* **The tree of cuts is not balanced.** A leaf is cut at the median of its
  own points, so on a stream that moves on, the child the stream has left
  stays as it is and the other is cut again: every split adds a level. On a
  cluster that drifts along one input the deepest leaf is 131 levels down
  after 12 000 points, where a balanced tree has 8. The time to reach a leaf
  through the cuts grows with the depth, at about a thousand levels the
  recursive descent that predictions used failed, and saving a tree failed
  from about 150 levels.
* **`LeafIndex`** (`pygptreeo/leaf_index.py`) holds the same leaves in a
  hierarchy of bounding boxes: groups of up to 32 leaves or groups, as the
  nodes of an R-tree (Guttman 1984). Its depth grows with the logarithm of
  the number of leaves, whatever the order of the points.
* **A group that overflows is divided where the cells of its two halves
  overlap least**, along any input and with 30 to 70% of the children in
  each half (the overlap criterion of the R*-tree, Beckmann et al. 1990).
  Cells that were made by cuts can nearly always be separated by one of
  those cuts, so the boxes of the groups do not overlap, and a search opens
  few groups however many there are: 5 at 20 000 points and 9 at a million
  for the 16 nearest leaves in six dimensions (`docs/benchmarks.md`,
  section 8.9).
* **Two boxes per leaf.** The cell that the cuts give the leaf, which finds
  the leaf of a point, and the bounding box of its stored points, which
  bounds the distance from a query to the leaf's points for the neighbour
  search. The cells are those of the tree of cuts, so the predictions do not
  change.
* **Within an overlap the cuts are followed.** A cell stops half an overlap
  short of a cut. A point in no cell lies in the overlap of two children,
  and is given to a leaf, or predicted by several, through `prob_func` as
  before.
* **A removed leaf.** The leaves along the removed cut extend their cells
  over the cell of the removed leaf, as they do in the tree of cuts.
* **Saving.** The nodes are saved as a list with their links as positions in
  it, so that saving does not recurse through the levels.

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

  Not pursued: the gains do not pay for several leaves per query. The GP on
  the nearest points became `predict_from='neighbours'` with another test:
  agreement with the own leaf rejects the neighbours where they differ from
  it, which is where they gain most, while the test of the neighbours
  against the kernel rejects them only where the kernel cannot describe them.
* **Cuts that are not along the inputs** (`docs/benchmarks.md`, section 8.1).
  A cut is a hyperplane with a free normal: the direction of largest spread
  of the leaf's points, the line between the centroids of two clusters of
  them, a random direction, the direction in which the mean of the leaf's GP
  changes most, or the one in which its gradient changes most. With one GP
  per leaf the first three were 30 to 35% worse than the cut along the input
  with the shortest length scale, and the two from the GP 8% and 4% worse.
  The last one finds a ridge or a kink that runs across the inputs, and
  lowers the error by 21 to 26% on a target with such a ridge, but raises it
  by 8 to 70% on smooth ones: an axis-aligned cell is a box, which the
  scaling of each input maps to a cube for the leaf's kernel, and an oblique
  cell is not.
* **Decorrelated inputs for the leaf GPs**, so that the kernel of an oblique
  cell has its length scales along the cell: the symmetric whitening of the
  leaf's points in place of a scale per input. With the oblique cuts along
  the change of the gradient the error fell by 50 to 80% on a target with an
  oblique ridge and rose by 7 to 93% on `GaussianPeaks` (14% lower on
  average, 15 of 33 runs worse). With the cuts along the inputs it changed
  nothing.
* **A cut where the gradient of the leaf's GP changes most**, in place of
  the median of the points. 1% lower error with cuts along the inputs and
  10% with the oblique cuts along the change of the gradient, which are then
  6% below the default, with a third of the runs worse.
* **A model in every node** (a multi-resolution tree). A node that splits
  leaves its GP behind, and the GPs of its descendants model what is left of
  the target. 41% worse: the residual of an interpolating GP is zero at the
  node's points and oscillates between them, so it is rougher than the
  target and the children fit it badly, where a GP on the target itself
  generalises with long length scales.
* **Far points among the neighbours** of `predict_from='neighbours'`: 30 of
  the 100 at geometrically growing ranks of distance, or four from a sample
  that every node keeps of its subtree, for each level above the leaf. No
  gain over the 100 nearest (the same error, and 2% higher).
* **Neighbours in units of the fitted length scales.** Without the test of
  the neighbours against the kernel the error was 5.8 times that of the tree
  on a target with a cliff, against 1.5 times in units of the leaf's spread:
  a long length scale along an input does not mean that the target does not
  depend on it. With the test the two metrics give the same error (0.73
  and 0.74 of the tree's).
* **Another leaf's kernel for the neighbours.** The leaf that holds most of
  the neighbours is the query's own for 76% of the queries. Its kernel in
  place of the own leaf's gave 1% lower error, and the one of the two with
  the higher likelihood of the neighbours 2%, at twice the cost.
* **Fewer neighbours where the kernel does not describe them.** With the
  neighbours ordered by distance, one factorisation gives the prediction
  from the nearest `k` for every `k`. Taking the largest `k` that passes
  the test, or stopping at the first neighbour that the nearer ones predict
  badly, gave the error of the plain fallback to the leaf's GP.
* **A blend of the neighbours' prediction and the leaf's own**, with equal
  weights or weights from the two sigmas: 1 to 13% worse than the
  neighbours' prediction alone.
* **Dividing a group of the index at the median of its children's centres**,
  along the input in which they are spread most. The boxes of the groups
  overlapped, a point lay in those of 5 bottom groups in four dimensions and
  of 9 in eight, and the search for the 16 nearest leaves opened 9 groups at
  20 000 points and 55 at a million in six dimensions.
* **Merging a removed leaf and its sibling back into their parent**, with the
  parent's points and hyperparameters as they were at the split. On four
  40 000-point runs the error was 11 to 49% higher than with plain removal and
  the tree split 31 to 140% more often: the restored parent is full and splits
  again at its next point, and it has lost the points that arrived after the
  split. With gradual splitting the sibling already holds about 80 of the
  parent's 100 points. A variant that kept the pair's newest points was within
  the seed-to-seed spread of plain removal.

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
  No rule was found for another leaf to predict in its place. Predictions
  from a query's neighbours lower the error of the queries beyond 2.5 by a
  third.
* **The errors have heavier tails than the sigma says.** Over 140 runs 2.2%
  of the predictions are off by more than 5 sigma and 0.8% by more than 10,
  and the five largest of 5000 errors carry 23% of the squared error of the
  median run. Near a leaf's points the sigma that grows with the distance
  changes nothing: 2.5% of the predictions within a distance of 2 are off by
  more than 5 sigma.
* **GPyTorch leaves take no appended points.** The adapter cannot extend a
  fit, so a point enters the GPs with the leaf's next fit.
* **GPyTorch leaves predict from their own points.** The adapter cannot
  condition a fitted kernel on other points, so `predict_from='neighbours'`
  needs the scikit-learn adapter.
* **A global model is evaluated at the neighbours of every query.** With
  `predict_from='neighbours'` the residuals of the neighbours are computed
  for each prediction, which is slow with the additive global GP.
* **The network lags at the front of a moving stream**, since it refits only
  when the point count doubles. The leaves correct most of this.
* **Regions the tree has forgotten.** Under a memory limit, a prediction in a
  region whose leaves were removed is the extrapolation of a neighbouring
  leaf, on top of the global model if there is one. Its sigma is too small in
  the same way as far from the data.
* **GPyTorch leaves do not sleep.** The adapter cannot release and restore a
  fit, so a memory limit is met by removing leaves only.
* **Calibration after a new network version.** Coverage can dip for a batch
  until the per-leaf calibration catches up with the shifted residual.

## Open ideas

* **Oblique cuts where the target has a ridge across the inputs.** Cuts along
  the change of the GP's gradient with decorrelated leaf inputs halve the
  error on such targets and raise it on smooth ones (`docs/benchmarks.md`,
  section 8.1). A leaf could try both kinds of cut on its own points when it
  splits and keep the better, as `enable_split_evaluation` does for the
  input of an axis-aligned cut.
* **A kernel for the neighbours.** The leaf's kernel was fitted at the scale
  of the leaf, and the neighbours of a query in a dense region lie within a
  small part of it. A few steps of the hyperparameter optimisation on the
  neighbours themselves, from the leaf's values, would fit the kernel to the
  points it is conditioned on, at several times the cost of a prediction.
* **Thinning sleeping leaves before removing them.** A leaf reduced to its 50
  most informative points (chosen one by one for the largest remaining GP
  variance, hyperparameters kept) had 1.2 times the error of the full leaf in
  a one-step test, and 1.6 times with 25 points. Keeping the hyperparameters
  was better than fitting them again to the remaining points, and the
  calibration has to be redone, for which the discarded points are a held-out
  set.
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

* Beckmann, Kriegel, Schneider, Seeger 1990, The R*-tree: an efficient and
  robust access method for points and rectangles, SIGMOD.
* Constantine, Dow, Wang 2014, Active subspace methods in theory and practice,
  SIAM J. Sci. Comput., arXiv:1304.2070.
* Gramacy, Apley 2015, Local Gaussian process approximation for large
  computer experiments, J. Comput. Graph. Stat., arXiv:1303.0383 (a GP on a
  neighbourhood chosen for each prediction).
* Guttman 1984, R-trees: a dynamic index structure for spatial searching,
  SIGMOD.
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
