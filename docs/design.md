# Design notes

Why the tree has a global model, how the mechanism works, how the tree stays
within a fixed size, how a point reaches a leaf's GP between two fits, how a
prediction can use the points of other leaves, how the leaf of a point is
found, how a leaf's fit can run in another process, how a batch of points is
added, how a leaf models a curve, what was tried and dropped, and what is
still open. The measurements are in [`benchmarks.md`](benchmarks.md).

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
  section 10.11).
* **A tree with such cuts has no index of its leaves.** Its cells are no
  boxes, so the leaf of a point is found through the cuts from the root, and
  with `predict_from='neighbours'` every leaf is searched for the points
  nearest to a query. On a stream that drifts the time to reach a leaf then
  grows with the depth of the tree (see "An index of the leaves").
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

## Saving and loading

`save` writes the tree to one file and `GPTree.load` reads it. They are made
for a tree that one process trains and saves again and again while other
processes load it to predict, so both have to be fast for a tree of many
leaves (`docs/benchmarks.md`, section 11).

* **The leaves are saved asleep.** `save` puts every leaf to sleep before it
  writes, so the file holds no Cholesky factor. The factors are 80% of a
  tree at `Nbar=100` in eight dimensions, 60% at `Nbar=50` in twelve and 90%
  at `Nbar=200` in eight. A leaf wakes when it is next used, in the tree
  that was saved as in a loaded one, and then predicts as before it slept
  (see "A memory limit"). Waking takes about 0.2, 0.4 and 1.0 ms at
  `Nbar=50`, 100 and 200. A process that loads a tree to predict holds the
  factors of the leaves it uses and of no others.
* **They sleep in the tree itself, not in a copy of it.** The tree's
  accounts of its memory then hold for the file as they are, nothing is
  computed for a save, and the next save finds asleep all leaves but those
  used since. A tree that is saved often is a tree whose leaves are asleep,
  as with `predict_from='neighbours'`.
* **A pickle, not a joblib file.** joblib writes each array apart from the
  rest, with a header of its own, and uses the pickler written in Python.
  That suits a few large arrays. A sleeping tree has about 40 arrays and a
  dozen other objects per leaf, most of them small, which the `pickle`
  module writes and reads three to five times as fast. With joblib a tree
  saved asleep took nearly as long to write and to read as one saved awake.
* **Without the garbage collector.** Pickling and unpickling a tree create
  millions of objects, and the collector goes through all of them again and
  again without finding one to free. It is off during a save and a load,
  which takes a third off the time of a save and two thirds off that of a
  load.
* **One random state.** A scikit-learn GP keeps the random state of its last
  fit: 5 kB, and 0.1 ms to unpickle. A clone shares that of the GP it was
  cloned from, since each fit replaces it before using it. A copy for every
  leaf that has not been fitted since its split was 12% of the file.
* **The scores of a node are saved as one array.** With predictions from
  neighbours a node keeps two small arrays for each of its last 50 points
  (`_scores`), and an array takes 1 microsecond to read and 3 to write
  however small it is. As they are, they doubled the time of a save and
  tripled that of a load.
* **The file is replaced in one step.** `save` writes to `path + '.tmp'` and
  renames that to `path`. A process that opens `path` gets the previous file
  or the new one, whole, and one that is reading the previous file keeps it
  until it closes it. This is what a file system does that follows POSIX;
  over NFS a read from another machine can fail when the file is replaced.
* **A save writes the whole tree**, also when one leaf has changed. Writing
  takes 0.2 to 0.3 ms per leaf and reading 0.1 to 0.15 ms, at 20 to 40 kB
  per leaf.
* **A tree that is replaced is released.** The nodes of a tree refer to each
  other in loops: a node and its children, a group of the leaf index and its
  parent, the tree and its neighbour search. Python frees objects in a loop
  only when its garbage collector finds them, not when the tree is dropped.
  A load leaves the objects of the new tree with the collector as new ones,
  so a collection follows each load. It goes through the new tree alone,
  except about every twelfth (with CPython's default thresholds), which
  also goes through the trees loaded since the last such collection and
  frees those that were dropped: ten trees in one pause, of about ten times
  the duration of a load. Until then the process holds all of them
  (`docs/benchmarks.md`, section 11). `release` breaks the loops, so that
  the tree is freed when it is dropped, and the collections after a load
  find no tree to free. The process that replaces a tree calls it; a tree
  does not release itself when it is dropped, since something else may
  still hold its nodes. The same links are cut for a leaf that a memory
  limit removes.

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
section 9.

* **Most queries lie at an edge of their leaf.** The points of a leaf lie on
  one side of each of its cuts, and in several dimensions most of a cell is
  close to a cut. In the benchmark runs the leaf of a query holds 37% of the
  query's 100 nearest points, and 4% of the queries lie within 0.75 standard
  deviations of the mean of their leaf's points along every input. The gain
  from the neighbours grows with that distance: none for those 4%, 23%
  between 1.25 and 1.75 standard deviations (45% of the queries) and 32%
  beyond 2.5.
* **No options of its own.** A prediction is conditioned on `Nbar`
  neighbours, as many points as a leaf holds at most, so that one setting
  says how many points a prediction can rest on. With 100 points in a leaf,
  150 neighbours gave an error 6% lower at more than three times the
  factorisation, and 100 neighbours the same error with 200 points in a
  leaf as with 100 (`docs/benchmarks.md`, sections 9.2 and 9.20). The other
  settings are constants of `pygptreeo/neighbours.py`, named
  `NEIGHBOUR_...` below, with the values of the benchmarks. A tree keeps
  what it was built with on its `NeighbourSearch`, so a saved tree predicts
  as it did whatever the constants are later.
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
  Above `NEIGHBOUR_CONSISTENCY` (10) the leaf's own GP predicts, as in the
  tree without neighbours. Limits from 5 to 100 give the same error, and the
  number comes with the factorisation that the prediction needs anyway.
* **The change to the leaf's own GP is gradual** (`NEIGHBOUR_TRANSITION`,
  2). The prediction is the mixture of the two, with a weight for the
  neighbours that falls from 1 at half the limit to 0 at twice the limit,
  linearly in the logarithm of the consistency. In the zones at the cuts
  the other prediction is the mixture of the leaves' predictions. With a
  switch at the limit the prediction steps where the consistency crosses
  it: along lines across the whole input space the steps above one sigma
  are fewer by a third to two thirds with the mixture, and the largest
  about half as high (`docs/benchmarks.md`, section 9.20).
* **A lower limit for a query outside its neighbours**
  (`NEIGHBOUR_CONSISTENCY_OUTSIDE`, 3). Away from the data the neighbours
  of a query come from several leaves and span many times the range of
  targets of the query's leaf. A kernel that fits them badly then carries
  their trend on to the query, where the leaf's own GP returns to the mean
  of its points: with the additive kernel this gave patches of twice the
  target's largest value, or below its smallest, in corners without data.
  A query is outside by the distance of the neighbours' centre from it over
  that of the furthest neighbour, near 0 with neighbours on all sides and
  towards 1 with all of them on one side. The limit is that among the
  neighbours up to 0.5 and the lower one from 0.8, with a geometric
  passage between them; 95% of a stream's own points stay below 0.6, in two
  dimensions and in six. A neighbour that fades out counts in the centre by
  what is left of it, so the measure is continuous in the query. It is
  worked out only where the weight depends on it.
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
* **Neighbours fade out** (`NEIGHBOUR_FADE`, 0.9). With neighbours that
  all count alike the prediction steps wherever the set changes: 33 to 39
  times by more than 0.1 sigma on a path through two or three leaves, where
  the leaves' own predictions step 4 to 5 times. So the noise variance of a
  neighbour grows with its distance `r` from the query, measured against
  that of the first stored point beyond the set: by `t^2 / (1 - t)^2` times
  the sum of its own noise variance and the variance of a prediction, with
  `t = (r - 0.9) / 0.1` beyond 0.9. At the edge the noise is infinite, a
  point enters and leaves the set with no weight, and the prediction is
  continuous in the query while its leaf and the leaves searched stay the
  same. It then steps as often as the leaves' own predictions and by less,
  for 1% more error than without fading in the median run and 3% in the
  mean of 150 runs (`docs/benchmarks.md`, sections 9.14 and 9.15).
* **The unit of that noise** is what decides whether a point still counts:
  its variance given the other neighbours, which the kernel sets. The
  median variance that the leaf's GPs gave their last predictions is of that
  size whether or not the kernel is calibrated, and the point's own noise
  is added for targets where the noise is the larger of the two. The squared
  error of the last predictions, the unit of the first version, is an eighth
  to a half of that variance on noise-free targets, and far less under a
  kernel that claims more uncertainty than there is: the fading then takes
  place within the last hundredths of the radius, and is a step again.
* **The kernels of neighbouring leaves are mixed across a cut**
  (`NEIGHBOUR_BLEND`, 0.1). The kernel, the scaling of the inputs and
  targets, the metric of the distances and the calibration are those of a
  leaf, so a prediction would step where the query passes into the next
  leaf. Each leaf therefore has a zone at each of its faces, half of it on
  either side of the cut, and a weight that is 1 within its faces less half
  a zone, a half at a face and 0 half a zone beyond it; with several inputs
  the weight is the product over them, and the weights of the leaves at a
  query are scaled to sum to 1. Within a zone the prediction is conditioned
  once, on the query's neighbours, with the weighted sum of the leaves'
  kernels in the units of the inputs and targets, and with their metrics,
  target means, snapshots of the global model, units of the fading and sigma
  scalers averaged with the same weights.
* **A sum of kernels, not an average of hyperparameters.** Each leaf scales
  its inputs and targets itself, so its hyperparameters mean something only
  with its scalers, and an average over leaves has no meaning for a kernel
  that is not of the default form. A weighted sum of kernels is a kernel
  whatever the leaves' kernels and scalers are, and it changes continuously
  with the weights.
* **The width of a zone is a share of the leaf's own width** along the
  input that is cut: the distance between its faces, or to the edge of its
  points where no cut bounds it. So a zone stays local however deep the tree
  grows. Measured against the node that was cut, as in the first version,
  the zone of an old cut covers leaves that have since become far smaller
  than it: the deeper the tree, the more zones a query lies in, and its
  prediction mixes in the kernels of leaves that are not beside it. The
  width needs no change with the number of inputs. At 20 000 points 1.3 to
  1.4 leaves take part in a prediction in three to ten dimensions; the
  number grows slowly with the tree, to at most `1.1^d` once every leaf has
  two faces along every input.
* **Where the kernels cannot be mixed** the predictions of the leaves are,
  with the same weights: where a leaf predicts from its own points or the
  neighbours do not fit the mixed kernel, and for the output models with a
  shared kernel or a basis, whose kernels are not those of single outputs.
* **Sigma is calibrated on the prediction that is made.** A point within the
  zones of several leaves is registered, for the sigma scaler of the leaf it
  is stored in, with the mixed prediction. A leaf's choice between its own
  GP and the neighbours still compares the two with its own kernel.
* **Most leaves sleep** (`NEIGHBOUR_AWAKE`, 8). A prediction from neighbours
  uses a leaf's kernel, scalers and points and never the factor of its
  kernel matrix, which is three quarters of a leaf's memory. The factor is
  needed where a point arrives, for the leaf's own prediction of it and to
  append it, and where the leaf's own GP predicts. So only the leaves whose
  GPs were fitted or woken last stay awake, and a leaf that predicts from
  its own points (`prefers_own`), which would otherwise be woken for every
  query. The predictions are the same to a part in a million; the tree
  takes 27 to 63% of the memory, the more the more leaves predict from
  their own points; an update takes 3 to 11% longer, for the factor of the
  leaf that is woken, and a prediction up to 6%.
* **A memory limit removes leaves later.** With the leaves asleep already,
  `max_memory_mb` has only leaves to remove once it is reached. With every
  leaf awake it removes leaves while half of their memory is left to the
  GPs awake (`FITTED_SHARE`), which suits predictions from the leaves and
  not those from neighbours: at 6 MB the tree kept a third of 20 000 points
  that way, and keeps half of them now.
* **A point is not its own neighbour.** The prediction registered for the
  calibration is made before the point is stored.
* **The search.** The 16 leaves whose points' boxes are nearest to the query
  are taken from the index of the leaves (`NEIGHBOUR_LEAVES`), and the
  neighbours are the nearest points among theirs. With `None` the search is
  exact: the query's leaf bounds the distance of the neighbours, and every
  leaf whose points can lie within it is looked at, 7 to 46 leaves in three
  to five dimensions and 35 to 79 in six to ten at 20 000 points. Sixteen
  leaves give the error of the exact search, eight 0.6% more, four 5%.
* **Cost.** A prediction factorises the kernel matrix of its neighbours, at a
  cost cubic in their number, where a leaf's own GP needs one triangular
  solve, and within a zone it evaluates the kernel of each leaf that takes
  part: 0.7 to 0.9 ms against about 0.2 ms with 100 neighbours. The leaf's
  own GP is still fitted and extended with each point, for the queries it
  predicts itself, and each new point is predicted both ways before it is
  stored, and a third time within a zone, and its leaf is woken for it, so
  an update takes 40 to 70% longer. Both times grow slowly with the stream,
  by about a quarter from
  5000 to 200 000 points, as leaves get faces along more inputs
  (`docs/benchmarks.md`, section 9.17). The gradual change to the leaf's
  own GP asks for that GP wherever the weight of the neighbours is below
  1, and wakes its leaf for it: at most 0.5% on an update, 4% on a
  prediction along the stream and 11% on one at a uniform test point
  (section 9.21).

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
  section 9.9).
* **Three boxes per leaf.** The cell that the cuts give the leaf, which
  finds the leaf of a point; the bounding box of its stored points, which
  bounds the distance from a query to the leaf's points for the neighbour
  search; and its reach, the box of its faces with the outer halves of its
  zones, which finds the leaves that take part in a prediction. The cells
  are those of the tree of cuts, so the predictions from the leaves do not
  change.
* **Within an overlap the cuts are followed.** A cell stops half an overlap
  short of a cut. A point in no cell lies in the overlap of two children, or
  on a cut that has none, and is given to a leaf, or predicted by several,
  through `prob_func` as before.
* **A removed leaf.** The leaves along the removed cut extend their cells
  over the cell of the removed leaf, as they do in the tree of cuts, and
  their faces and widths with them.
* **Saving.** The nodes are saved as a list with their links as positions in
  it, so that saving does not recurse through the levels. A node that is
  pickled or copied on its own, without its tree, comes back without its
  links.

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
curves, and `benchmarks.md`, section 10, what was measured on them.

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
  the kernel nothing as long as it does not move (section 10.8). Where a jump
  moves along the grid, each output is a step in the parameters, and Matérn
  5/2 leaves have 4 times the error of the default (section 10.9).
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
* **With `predict_from='neighbours'`** a leaf that holds an alignment
  predicts from its own aligned curves, and a leaf without one from the
  points nearest to the query, as a `'shared'` leaf does.
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
* **A function on a grid of two dimensions: a warp of each axis that varies
  along the other.** With `output_shape=(p1, p2)` the outputs are such a
  function, a surface, flattened row by row, and its grid is warped as

      (u, v) -> (phi_1(u; v, delta), phi_2(v; u, delta))

  `phi_1` is the warp above along the first axis, with a `w` that varies
  linearly between two knots along the second axis, and `phi_2` the same with
  the axes exchanged: 32 numbers per input. Rows can so move by different
  amounts, and so can columns. The warp is fitted like that of a curve, by
  damped Gauss-Newton steps on at most 32 points along each axis.
* **The exponent of that warp is bounded by a `tanh` and has a ridge.** Where
  a surface is flat nothing determines the warp. With the hard bound and the
  weak ridge of the curves it grew there until it met the bound, and at the
  corners of a leaf, where several offsets are large at once, it misplaced
  the feature: a few queries had 300 times the median error, which the
  leave-one-out errors did not show. With the smooth bound and a ridge of
  0.03 the leave-one-out error of a leaf is that of its test points
  (`docs/benchmarks.md`, section 10.12).
* **A prediction of a surface is the aligned surface, read back.** The GP is
  fitted to the aligned surfaces on a grid with twice the points along each
  axis, and its mean is read where each grid point of the query comes from,
  found by Newton steps on the warp. Warping each of the leaf's surfaces into
  the frame of the query, as for curves, is the same prediction where the
  finer grid resolves the aligned surfaces, and took 12 times as long.
* **Both global models take the curve as their target.** The network has one
  output per grid point and the additive GP one kernel for all outputs, fitted
  from `Y Y^T` like the GP of a leaf, and a leaf models the residual curves.
  With 800 curves and `'shared'` leaves the additive GP lowered the error by a
  factor 3.6 to 70 on the four smooth families, by 3.5 on the Z line shape and
  2.6 on the oscillation probability, and by 1.2 to 1.7 on five others. The
  network lowered it by a factor 2.3 to 6 on five families and raised it by up
  to 2.2 on the other seven, where the tree alone is as accurate as the
  network (`benchmarks.md`, section 10.7).
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

  Not pursued: the gains do not pay for several leaves per query. The GP on
  the nearest points became `predict_from='neighbours'` with another test:
  agreement with the own leaf rejects the neighbours where they differ from
  it, which is where they gain most, while the test of the neighbours
  against the kernel rejects them only where the kernel cannot describe them.
* **Cuts that are not along the inputs** (`docs/benchmarks.md`, section 9.1).
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
  cell is not. The cut of a leaf that kept an oblique kernel
  (`Default_GPR(oblique=True)`, "Length scales along combinations of the
  inputs") is none of these: its direction and the kernel's are fitted
  together.
* **Decorrelated inputs for the leaf GPs**, so that the kernel of an oblique
  cell has its length scales along the cell: the symmetric whitening of the
  leaf's points in place of a scale per input. With the oblique cuts along
  the change of the gradient the error fell by 50 to 80% on a target with an
  oblique ridge and rose by 7 to 93% on `GaussianPeaks` (14% lower on
  average, 15 of 33 runs worse; on 20 000-point streams 16% lower, 11 of
  33 runs worse, by up to 2.7 times). With the cuts along the inputs it
  changed nothing.
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
* **A switch at a lower limit for queries outside their neighbours**, or
  at a limit of 3 for every query (`docs/benchmarks.md`, section 9.20). Both
  remove the overshoots outside the data. On 5000-point streams the first
  costs 2.0% in error along the stream and the second 3.6%, where the
  gradual change costs 1.0%, and neither removes a step.
* **The leaf's remaining points with the neighbours**, so that a
  prediction is conditioned on the 100 nearest points and on every other
  point of its leaf (`docs/benchmarks.md`, section 9.20). The overshoots
  outside the data stay, since the points of other leaves near the query
  are still there, and without the test against the kernel they grow
  several times. The error along the stream is 6% lower, as with 150
  nearest neighbours, which need no rule for points that enter at a cut.
* **A split into two clusters of the leaf's points**, each point going to
  the child with the nearer centroid (`docs/benchmarks.md`, section 9.13).
  This is a cut by the hyperplane half-way between the centroids. With
  clusters of the inputs it was 26 to 30% worse than the default cuts, and
  with distances in the fitted length scales of the leaf's kernel level with
  them. The shape of a cell does not change how many of its queries lie at
  its edge.
* **Neighbours shared by the queries of a small cell**, to keep the
  prediction smooth within the cell: 1 to 6% lower error than the leaves,
  against 25% with the neighbours of each query. In several dimensions a
  cell with few points is still a large part of its leaf across.
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

* **Other warps for `'aligned'`** (`docs/benchmarks.md`, section 10.6):
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
  (`docs/benchmarks.md`, section 10.10):
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
  section 10.7), each alone on 800 curves against the network with one output
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
  (`docs/benchmarks.md`, section 10.7).

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
* **Outside the data both kinds of prediction can overshoot.** With the
  additive kernel and a declared noise of a millionth of the value, a leaf's
  own GP predicted up to three times the target's largest value in corners
  without data, and the neighbours with it (`docs/benchmarks.md`, section
  9.20). The lower limit for queries outside their neighbours removes the
  overshoots that the neighbours added to these. On the benchmark targets,
  where the neighbours are the more accurate of the two outside the data
  as well, it costs 1% in error on 5000-point streams and nothing on
  20 000-point ones.
* **GPyTorch leaves take no appended points.** The adapter cannot extend a
  fit, so a point enters the GPs with the leaf's next fit.
* **GPyTorch leaves predict from their own points.** The adapter cannot
  condition a fitted kernel on other points, so `predict_from='neighbours'`
  needs the scikit-learn adapter.
* **Predictions from neighbours still step, by a tenth of a sigma or
  less,** where the 16 leaves that are searched change and where a leaf
  that predicts from its own points takes part. Away from the data they
  step more often than those of the leaves.
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
* **Not every fit can be handed out.** GPyTorch leaves and the leaves of the
  `'shared'` and `'pca'` output models are fitted in place also with
  `defer_fits`. So is a leaf that meets a new version of the additive global
  GP: it refits before its next prediction.
* **Calibration after a new network version.** Coverage can dip for a batch
  until the per-leaf calibration catches up with the shifted residual.

* **`'aligned'` is for one curve on one ordered grid, or one surface on a
  grid of two dimensions.** It
  gains wherever a feature moves along such a grid, in curves from physics
  or not (an epidemic peak, a shock front, a filter resonance: a factor 3 to
  12), and its leaves decline where nothing moves or the outputs are in no
  order, at 1.3 to 1.8 times the time of `'shared'`. Over 64 runs on sixteen
  such targets it was more than 5% worse than `'shared'` in two, by 10% at
  most (`docs/benchmarks.md`, section 10.9). A surface needs its
  `output_shape`: as a curve, row by row, it is aligned along the rows only.
  The warp of a surface moves its rows and its columns. A feature that turns,
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

* **Oblique cuts where the target has a ridge across the inputs.** Cuts along
  the change of the GP's gradient with decorrelated leaf inputs halve the
  error on such targets and raise it on smooth ones (`docs/benchmarks.md`,
  section 9.1). A leaf could try both kinds of cut on its own points when it
  splits and keep the better, as `enable_split_evaluation` does for the
  input of an axis-aligned cut. With `Default_GPR(oblique=True)` a leaf
  decides by its kernel instead: it is cut across the direction of its
  shortest length scale if it kept an oblique kernel.
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
* **Leaves that are unpickled when they are first used.** With the state of
  each leaf as a pickle of its own inside the file, a process that loads a
  tree unpickles the inner nodes and the index of the leaves, and a leaf
  when it first needs it. In a prototype of the file format this took a
  third off the time of a load, and more than half with the file mapped
  into memory, which also reads from disk only the leaves that are used and
  lets the processes of one machine share them (`docs/benchmarks.md`,
  section 11). A tree that keeps the pickle of each leaf it has not changed
  could save in half the time. Both need a leaf that can be without its
  state, and the second a record of which leaves have changed.
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
* Welper 2017, Interpolation of functions with parameter dependent jumps by
  transformed snapshots, SIAM J. Sci. Comput., arXiv:1505.01227.
