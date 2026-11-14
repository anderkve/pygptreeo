# Design notes

Why the tree has a global model, how the mechanism works, what was tried and
dropped, and what is still open. The measurements are in
[`benchmarks.md`](benchmarks.md).

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
* **Per-leaf calibration.** Each leaf keeps its last 25 normalised prediction
  errors and scales its sigma to their 68% quantile. This holds the on-stream
  coverage at the nominal value for every leaf model and global model measured.
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
  the reservoir and then Adam steps over every point seen, and publishes the
  weights with the lowest loss on a held-out fifth of the points. A leaf takes
  up a new version by refitting on the new residual with its kernel
  hyperparameters kept.

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

## Known limitations

* **Sigma far from the data.** The calibration is fitted on the stream. In
  regions the stream has not visited the coverage of every configuration falls
  to 0.1 to 0.5, and a tree with a global model is more accurate there without
  its sigma growing accordingly.
* **The network lags at the front of a moving stream**, since it refits only
  when the point count doubles. The leaves correct most of this.
* **Calibration after a new network version.** Coverage can dip for a batch
  until the per-leaf calibration catches up with the shifted residual.

## Open ideas

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
