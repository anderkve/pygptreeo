# Neural networks in a GPTreeO-style tree: ideas for many-input online regression

A design note (October 2026) on how pyGPTreeO's machinery could be kept while
the leaf Gaussian processes are replaced, or supplemented, by neural networks,
with the aim of scaling online regression to functions of many inputs. It
reads the current `pygptreeo` code, the neural-network emulator prototype on
noisyprof's `claude/friendly-wright-tznhd6` branch (`comparisons/nle/simemu.py`
and `docs/records/simulator_emulator_proposal.md` there), and the literature,
and it includes one small measurement (section 5) of the idea it recommends
first. Epistemic labels: **measured** (run in this session), **read** (from
the code or a record in these repositories), **verified** (abstract or full
text of the paper checked), **reasoning** (an argument, not a measurement).

## 1. The short answer

The thing that fails in many dimensions is not the tree, it is asking each
leaf to learn the *shape* of the function from its own hundred points. A
stationary GP with one length scale per input needs a number of points that
grows like `(box side / length scale)^d` to pin a function down, and the tree
only shrinks the box side (reasoning; the standard covering argument). In ten
dimensions a leaf with `Nbar = 100` points and eleven kernel hyperparameters is
underdetermined, and its posterior reverts to the leaf constant a short way
from its data.

So the useful split is not "GP versus neural network" but "what is global
versus what is local". Share across the whole tree what is global, the
representation of the function (its relevant directions, its large-scale
shape), by training one network on everything the tree has seen; keep local
what the tree is good at, the correction in each box, the uncertainty tied to
the local data density, and the per-leaf calibration of that uncertainty. The
simplest model with this division is a **neural-linear tree**: one shared
feature network, and in every leaf a Bayesian linear regression on its
features, best on the residual of the network's own prediction. The leaf fit
is a closed form, its update is rank-one, its sigma is exact within the
feature span, and the tree code does not change (section 3, idea 1). A first
measurement on two 10D targets with 8000 streamed points (section 5) puts it
at a third to a sixth of the plain tree's error on every metric, with
on-stream coverage at the nominal 0.68 and a comparable run time. The same
measurement shows where the credit goes: the shared network alone is nearly
as accurate cube-wide, and what the tree adds is the calibrated sigma and a
local correction worth a factor of five where the stream is dense. What it
does not yet give is an honest sigma far from the data, which needs the
network's own error budget or a few bagged networks, exactly as the
global-model benchmark found for the GP. Ideas 2 to 6 are the alternatives
and extensions, ranked.

## 2. What is worth keeping, and what the GP costs

### 2.1 The tree's strengths that have nothing to do with the GP (read)

* **Locality by data-driven partitioning.** A leaf model only has to be right
  in its box, and the partition follows the data, not a grid
  (`GPNode.compute_split_position_and_overlap`, `prob_func`). The
  `min_lengthscale` split criterion puts the cut where the function varies
  fastest (`examples/BENCHMARK_RESULTS_split_direction.md`).
* **Bounded update cost.** Every point is placed once, retrains touch at most
  `Nbar` points, and non-leaf nodes drop their data, so the total cost is
  sublinear in the stream (Lederer et al. 2020, the DLGP paper; verified).
* **Per-leaf prequential calibration.** `register_pred_perf` and
  `update_sigma_scaler` keep a window of the last 25 normalised residuals per
  leaf and rescale sigma to the 68% quantile. This is a *local* calibration,
  and in the stream benchmarks it holds coverage at 0.65 to 0.74 for every
  configuration (`examples/BENCHMARK_RESULTS_global_mean_streams.md`).
* **Mixture-of-experts aggregation with overlap.** Predictions near a cut are
  blended over the leaves the point could belong to.
* **A tree-wide model the leaves correct, already wired in.** `global_mean`
  gives the tree one shared model, versioned snapshots, a coverage reservoir
  for its training sample, a turnover rule for refits, a refresh rule for
  leaves holding a stale snapshot, and an epistemic error budget added to the
  leaf sigma (`pygptreeo/global_mean.py`). Everything below that needs a
  shared network can hang on this hook; the mechanism was debugged on the
  sweeping stream so that stale snapshots never answer a prediction.

### 2.2 What the GP leaf costs in many dimensions (read, reasoning)

* **Hyperparameters from too few points.** Each retrain maximises the
  marginal likelihood of `d + 1` kernel parameters on at most `Nbar` points,
  with restarts, at `O(Nbar^3)` per evaluation. At `d = 10` that is eleven
  parameters from a hundred points, and the fitted length scales are noisy,
  which also makes the split criterion noisy.
* **No sharing between leaves.** A leaf cannot use what the leaf next door
  learned about the function's directions, except through the parent's GP it
  inherits until its first retrain.
* **Reversion to a constant.** Away from its points a leaf GP returns to its
  prior mean. The global-model benchmark shows what a shared trend buys even
  when it is a GP: 15 to 50% lower error on the rotated 6D targets, and orders
  of magnitude on additive ones.
* **The global GP does not scale either.** The additive global GP costs 20 to
  40 s per fit on a 500-point reservoir at `d = 6`, cubic beyond, and its
  order-2 additive kernel has `d(d-1)/2` pair terms.

### 2.3 What noisyprof's network prototype showed, and what it still lacks (read)

The prototype on `claude/friendly-wright-tznhd6` trains an ensemble of five
MLPs (3 x 128, SiLU) on single simulator runs and reads profile likelihoods
off it by batched gradient ascent. Against the per-cell DE with local GPs it
is decisive in six and ten dimensions (record §4.2, §4.9: 0.01 to 0.2 nats
from 40k to 70k simulations where noisyprof is 0.2 to 10 nats off after 2
million). The flat directions that defeat a local search are easy for one
regression trained on everything. That is the "global" half of the argument
above, measured.

What it pays for, also measured in that record, is the list a tree is built
to fix:

* **Every round is a full retrain of five networks** (44 to 89 s per fit
  at 8 to 32 outputs, §4.14; 541 of about 750 s of wall time in fits in the
  all-settings example, §7). There is no incremental update.
* **Refit-to-refit instability.** Early stopping left each member at an
  arbitrary point and the certified count oscillated (§4.15); a fixed cosine
  schedule and bagging were needed to make consecutive refits agree, and warm
  refits on a design whose newest points are concentrated tilt the bulk fit
  (§4.11, the posterior stage now refits cold).
* **The ensemble spread misses shared errors**, so an error map from
  out-of-bag residuals and a calibration factor solved from fresh
  simulations had to be added (§4.17); the factor is one global number, and
  the map needs 128 neighbours in whitened coordinates to resolve a few
  tenths of a nat.
* **Uncertainty is not tied to local data density** in any closed form; it
  is whatever five initialisations happen to disagree on.

A tree whose leaves are closed-form models on shared features gives
incremental updates, local uncertainty from local data, and local
calibration from local residuals, which is the list above.

## 3. The ideas, ranked

### Idea 1. The neural-linear tree (recommended first)

**What.** One shared feature network `phi(x) in R^m` (an MLP whose last hidden
layer is the feature map), trained on everything the tree has seen on a
doubling schedule, as `AdditiveGPGlobalMean` schedules its refits. Each leaf
holds a Bayesian linear regression on `[phi(x), 1]` of the residual
`y - h(x)` of the network's own head `h`: a Gaussian prior `N(0, tau^2 I)` on
the weights, per-point observation noise from the `sigma` the user passes
plus a learned extra noise `s^2` for what the features cannot represent, both
`tau^2` and `s^2` chosen by the evidence on a small grid. The leaf's
posterior is `Sigma = (Phi^T Lambda^-1 Phi + I / tau^2)^-1`,
`mu = Sigma Phi^T Lambda^-1 (y - h)`; the prediction is `h(x) + phi(x)^T mu`,
the latent sigma `sqrt(phi(x)^T Sigma phi(x))`. Regressing the residual
rather than the target is what lets a leaf with few points default to the
network instead of to a constant; the probe measures both forms. The tree
stays as it is: `NeuralLinearGPR` implements `GPRegressorInterface`, the
feature learner sits beside the tree like a `GlobalMeanLearner`, and a leaf
that is asked to predict after the network was refit re-solves its
regression on the current features first (the refresh rule of
`global_mean`, reused).

**Why it scales.** The leaf estimates `m + 1` weights, however many inputs
there are; the network, which sees all `N` points, is what has to learn the
function's directions, and a 3 x 128 MLP on ten thousand points in ten
dimensions is a routine fit (the noisyprof prototype's bread and butter). The
tree then does what it is good at: a local re-weighting of the global
features where the global fit is off, and a sigma that grows where the leaf
has no data in feature space.

**What it gives that the prototype lacks.**

* *Rank-one online updates.* Adding a point to a leaf is a rank-one update
  of `Sigma` (or of its Cholesky factor), `O(m^2)`; no gradient steps, no
  retrain schedule between network refits. The probe below re-solves every
  25 points, which costs about a hundred small Cholesky factorisations per
  leaf fit and is already cheap; the incremental form is the next step.
* *A closed-form, local sigma.* The leaf's `Sigma` is the inverse of what the
  leaf's own points say about each feature direction, so sigma is small where
  the leaf has data along `phi` and large where it does not. The per-leaf
  calibration scaler then absorbs what the features cannot see.
* *Differentiability.* `phi` is a network, so `d(prediction)/dx` comes from
  autograd, and a profiler of the noisyprof kind can run batched gradient
  ascent on the tree's mean exactly as it does on the ensemble. The
  mixture-of-experts gate is piecewise linear in `x` with the overlap
  `theta`, so the mixture is continuous and almost everywhere differentiable.
* *A split criterion without ARD length scales.* The fitted leaf mean is
  `w . phi(x)`, so its gradient per input dimension is one backward pass; the
  rms gradient along dimension `i` times the leaf's spread along `i` is the
  analogue of "spread over length scale", and the existing `min_lengthscale`
  criterion uses it unchanged through `get_length_scales` (implemented in the
  probe).
* *Multi-output for free.* Several outputs sharing one `Sigma` cost one
  factorisation and `p` right-hand sides, which is what noisyprof's
  "emulate the outputs, not the log-likelihood" design needs; the plug-in
  likelihood and its Fisher weighting apply to the leaf means unchanged.

**Literature (verified).** Snoek et al. 2015 (DNGO, "Scalable Bayesian
optimization using deep neural networks") is exactly adaptive basis
regression: a network learns the basis, Bayesian linear regression on top
gives the posterior, linear in the data instead of cubic. Riquelme, Tucker and
Snoek 2018 ("Deep Bayesian bandits showdown") name it neural-linear and find
it among the most reliable uncertainty methods for Thompson sampling. Titsias
et al. 2020 (ICLR, "Functional regularisation for continual learning with
Gaussian processes") use the same last-layer-Gaussian construction to turn a
network into a GP for continual learning, with a fixed-size summary per task,
which is the continual-learning precedent for "a shared network plus a
per-region closed-form posterior". Gijsberts and Metta 2013 (incremental
sparse spectrum GP regression) is the fixed-feature version of the same leaf,
random Fourier features plus recursive least squares, with constant update
cost and better accuracy than LWPR on robot data; idea 1 replaces the random
features by learned ones and adds the tree.

**Limitations and their mitigations.**

* *The sigma lives in the feature span.* Where `phi` cannot represent the
  local function, the regression is confidently wrong, and in the residual
  form a leaf far from its data inherits the head's error with no term for
  it. Four layers of defence: the evidence-chosen extra noise `s^2` (which
  inflates `Sigma` where the residual is large), the leaf's prequential
  calibration, the learner's epistemic error budget added in quadrature (as
  `global_mean.error_scale` does for the GP global model), and an ensemble
  of `K` bagged feature networks giving `K` regressions per leaf, whose
  spread is the uncertainty *about the features* (the prototype's ensemble,
  moved one level down). The probe uses the first two only, and its
  off-stream coverage (section 5) is the measurement that the third or
  fourth is needed.
* *Network refits invalidate every leaf.* Each leaf keeps its points, so the
  refresh is one `m x m` solve per leaf; at a few hundred leaves that is a
  second. Refits happen on a doubling schedule, so their number is
  logarithmic in the stream length.
* *Catastrophic forgetting in the feature network.* Training on everything
  seen (the probe) is safe but grows; training on a coverage reservoir (the
  `CoverageReservoir` the global model already uses) bounds it and keeps the
  features honest over the explored region whatever the stream does. The
  leaves hold the fine detail in any case, which is the point of the
  division of labour.
* *Deep kernel learning's pathology does not apply.* Ober, Rasmussen and van
  der Wilk 2021 show that training a feature extractor *by the GP marginal
  likelihood* overfits badly. Here the network is trained by least squares
  on all the data and then frozen for the leaves; the evidence only chooses
  two scalars per leaf.

**For pygptreeo.** A `NeuralLinearGPR` backend plus a `FeatureNetLearner`
(torch optional, as GPyTorch is now); `GPTree(GPR=NeuralLinearGPR(learner))`
with `use_standard_scaling=False` (the learner standardises internally). The
probe script `examples/probe_neural_linear_tree.py` is a working draft of both.

**For noisyprof.** The prototype's `MomentEmulator` would become "one feature
network plus a tree of output regressions": the acquisition keeps the
ensemble spread if `K` feature networks are used, gains a local sigma from
the leaf posteriors, and gets its between-round updates for free (new runs
enter their leaves; the network refits on the doubling schedule, not every
round). Whether the local sigma can replace the out-of-bag error map is a
measurement to make (section 6).

### Idea 2. The network as `global_mean`, leaves as GPs on its residual in a local active subspace

**What.** The smallest change: `GPTree(global_mean=NetGlobalMean(...))`, a
`GlobalMeanLearner` whose snapshot is an MLP (or a small ensemble) fitted on
the coverage reservoir. Everything else in the tree, including the error
budget `error_scale` and the refresh rule, applies as it does to the additive
GP, and the leaf GPs model a residual that is smaller and smoother than the
raw target. To address the leaf kernel's dimension problem directly, give
each leaf a kernel on a **local active subspace**: with the network's
gradients at the leaf's points (one backward pass), form the average outer
product `C = mean(grad f grad f^T)`, take its top `r` eigenvectors `U` (r of
2 to 4), and use a Matern kernel on `U^T x` plus a weak isotropic catch-all on
the rest. The leaf then fits `r + 2` hyperparameters, not `d + 1`.

**Why.** Constantine's active subspaces (verified) are precisely this
construction for response surfaces of expensive simulators, and LWPR
(Vijayakumar, D'Souza and Schaal 2005, verified) is the classical online
local-model method that survives 50 input dimensions by giving each local
model a few projection directions. TuRBO (Eriksson et al. 2019, verified)
makes the same bet in Bayesian optimisation: local GPs in trust regions
scale where one global GP does not. SAASBO (Eriksson and Jankowiak 2021,
verified) reaches hundreds of dimensions with sparsity priors on the length
scales, the axis-aligned cousin of a subspace.

**Trade-offs.** Keeps the GP's sigma semantics outside the feature span, which
idea 1 gives up, and keeps the leaf cost cubic in `Nbar`. The global network's
own error is tracked only as a stream-wide scalar (`error_scale`), which the
calibrated benchmark already shows is conservative in dense regions and
optimistic far from the data. Expect a smaller accuracy gain than idea 1 and
a larger run time; worth measuring as the comparison point, since it changes
the least.

### Idea 3. In-context regression leaves (prior-fitted networks)

**What.** A leaf with no fit at all: its model is a transformer pretrained to
do Bayesian regression in context (Müller et al. 2023, PFNs4BO, verified;
Hollmann et al. 2025, TabPFN v2, verified for regression on up to ten
thousand samples). The leaf's `Nbar` points *are* the context; an update
appends a point; a prediction is one forward pass that returns a full
predictive distribution; `Nbar` becomes the context length the model was
trained for. The tree supplies what such models lack: locality, so the
context stays small and relevant, and the per-leaf calibration.

**Why it fits the brief.** "Fast update, fast prediction, reasonable
uncertainty, no training" is the definition of in-context regression. PFNs
can be pretrained on a prior matched to the target class (GP samples with
anisotropic Matern kernels, additive structure, the length-scale ranges the
tree expects), which is the PFNs4BO recipe, and the prior can encode
"ignore irrelevant dimensions", which is the many-inputs case. Conditional
and attentive neural processes (Garnelo et al. 2018; Kim et al. 2019,
verified) are the same idea with a lighter architecture.

**Risks.** Prior mismatch (a PFN cannot represent what its pretraining prior
never produced); inference cost on CPU with contexts of a thousand points,
likely tens of milliseconds per query, which is fine for noisyprof's batched
profiler and slow for a per-point stream; and gradients with respect to `x`
are available by autograd but cost a full forward pass each. This is the most
speculative idea and the one with the largest upside for the user-facing
simplicity; a one-day probe with TabPFN v2 as the leaf would settle whether
it is worth a dedicated pretraining.

### Idea 4. The tree as the memory and calibrator of a global ensemble

**What.** Keep the prototype's ensemble as the model, and make the tree the
data structure around it: each leaf keeps `Nbar` points as a coverage
sample of its box, the ensemble refits on the union of the leaves (a bounded,
coverage-balanced rehearsal set), and each leaf keeps its own prequential
residual window and calibration scaler. The leaf scalers replace the one
global calibration factor with a local one, and the leaf residual pools
replace the error map's 128-neighbour whitened k-d tree with a partition
that already follows the data.

**Why.** It needs no new regressor and could be tried in `simemu.py` in a
day: the lesson of record §4.11 (warm refits on a design concentrated in
one region tilt the bulk) is a rehearsal-set problem, and a coverage-balanced
union of leaf samples is the standard remedy in continual learning; the
lesson of §4.17 (one scalar factor set by the worst champions) is a locality
problem, which a partition solves by construction. Mondrian forests
(Lakshminarayanan, Roy and Teh 2016, verified) are the precedent for "an
online partition that carries uncertainty in regression".

**Trade-offs.** It does not reduce the cost of the refits or make updates
incremental; it makes them stable and the uncertainty local. It composes
with idea 1 (the leaves' regressions and the global ensemble can coexist).

### Idea 5. Leaf-specific heads on a shared trunk (hard mixture of experts)

**What.** Each leaf owns the last layer, or a FiLM scale-and-shift, of a
shared trunk; a leaf update is a handful of Adam steps on its `Nbar` points
(milliseconds), the trunk is refit on the doubling schedule. Uncertainty from
an epinet (Osband et al. 2023, verified: an additive uncertainty network
that matches large ensembles at a fraction of the cost) or from `K` heads.

**Why not first.** It is idea 1 with gradient steps in place of the closed
form, which loses the exact sigma, the exact evidence for the noise, and the
rank-one update, and gains only the ability of the head to be nonlinear in
the features. The ranking would change if the leaves turn out to need more
than a linear re-weighting of the shared features; the probe's residual
diagnostics (the fitted `s^2` per leaf) will say.

### Idea 6. Linearised-network kernels (the generalised linear model view)

**What.** A leaf GP whose kernel is `J(x) J(x')^T` with `J = d f / d theta`,
the network's parameter gradients at `x`: Immer, Korzepa and Bauer 2021
(verified) show that the Laplace approximation with the generalised
Gauss-Newton Hessian *is* this GP, and that predicting with the linearised
model fixes the usual underfitting. Idea 1 is the special case where `J` is
restricted to the last layer; using more layers' gradients, or a random
projection of all of them, gives a richer prior from the same network.

**Why last.** Cost (the full Jacobian per point) and no evidence yet that
the last layer is not enough; a research direction, not a first step.

## 4. Cross-cutting points

* **Uncertainty has to be measured, not assumed.** The tree's per-leaf
  calibration is the one component that has delivered nominal coverage on
  every configuration so far, and it does so on whatever the leaf model is.
  Any of the ideas should be run through `--calibrate` and judged on the
  three coverage columns, not only on NRMSE; off-stream coverage (the uniform
  test set on a focusing stream) is where every model so far is poor, and a
  sigma that grows outside the feature span (bagged feature networks, idea
  1) is the candidate fix.
* **Refit scheduling and forgetting.** Reuse the global model's rules:
  doubling schedule, a coverage reservoir as the training sample, snapshot
  versions, lazy refresh of leaves. They were debugged on the sweeping
  stream, where the naive rule doubled the error.
* **The split criterion.** With features in place of length scales, the
  gradient-sensitivity criterion (idea 1) is the natural replacement and
  costs one backward pass per split; `enable_split_evaluation` remains as
  the fallback that tests candidates by held-out error.
* **For noisyprof, differentiability and batching matter more than per-point
  update speed.** Both idea 1 and idea 3 keep autograd through the mean;
  the mixture gate is continuous with `theta > 0`. Prediction over a batch of
  profiler starts groups by leaf, so a tree of a few hundred leaves costs a
  few hundred small matrix products per profiler step.

## 5. A first measurement: the neural-linear tree in 10D

**Setup (measured).** `examples/probe_neural_linear_tree.py`: the
`NeuralLinearGPR` backend and `FeatureNet` learner of idea 1 inside the
unchanged `GPTree`. Targets `RotatedRosenbrock` and `GaussianPeaks` in
`d = 10` (no low-order additive structure); streams `uniform` and `focusing`
from `benchmark_global_mean_streams.py`; `N = 8000` points, one seed; the
benchmark's nominal noise (`sigma = 1e-3 |y|`, realised as Gaussian noise of
that size); `use_calibrated_sigma=True` everywhere. Configurations:

* `tree`: `Default_GPR(n_restarts_optimizer=1)`, `Nbar = 100`, `theta = 1e-4`,
  `retrain_every_n_points = 25`, gradual splitting (the benchmark's plain
  tree).
* `nltree100`, `nltree400`: the neural-linear tree with `Nbar` of 100 and
  400; feature network 3 x 128 SiLU, 4000 Adam steps with a cosine schedule
  per refit, first fit at 200 points then at each doubling (warm), `m = 128`
  features; per leaf the evidence grid is 9 extra-noise by 13 prior-variance
  values; the leaf re-solves every 25 points and on first use after a
  network refit.
* `nlres100`, `nlres400`: the same, but each leaf regresses the *residual*
  of the network's own head, so a leaf with little data reverts to the
  network rather than to a constant (the `global_mean` idea applied inside
  the leaf; a second run after the first results were read).
* `network head alone`: the feature network's own linear head, reported
  alongside (no sigma), to separate what the network does from what the
  tree adds.

Metrics as in the stream benchmark: NRMSE (RMSE over the target's range on a
uniform sample) prequentially on the stream after warm-up, on 3000 uniform
test points and on 3000 points where the stream ended; 1-sigma coverage
(target 0.68) and RMS sigma over RMSE on the same three sets. A second set
of runs repeats `tree`, `nltree400` and `nlres400` on the rotated Rosenbrock
with Gaussian noise of 5% of `|y|` added to the targets (and passed as the
per-point `sigma`). Raw lines in `examples/results/neural_linear_probe/`.
The `nlres` and noisy runs ran six processes on four cores, so their wall
times are inflated by about half relative to the first set.

**Results, nominal noise** (`sigma = 1e-3 |y|`):

| target | stream | config | prequential | uniform-test | focus-test | coverage preq / uni / focus | sigma/RMSE preq / uni / focus | leaves | time [s] (of which net fits) |
|---|---|---|---|---|---|---|---|---|---|
| gaussian_peaks | focusing | tree | 0.0210 | 0.0694 | 0.0007 | 0.69 / 0.57 / 0.60 | 1.00 / 0.88 / 0.86 | 114 | 68.5 |
| gaussian_peaks | focusing | nltree100 | 0.0052 | 0.0204 | 0.00004 | 0.68 / 0.59 / 0.72 | 1.19 / 0.67 / 1.10 | 118 | 74.9 (39.8) |
| gaussian_peaks | focusing | nltree400 | 0.0043 | 0.0193 | 0.00005 | 0.67 / 0.48 / 0.69 | 0.90 / 0.46 / 1.07 | 27 | 61.1 (37.3) |
| gaussian_peaks | focusing | nlres100 | 0.0048 | 0.0188 | 0.00005 | 0.71 / 0.37 / 0.67 | 1.28 / 0.29 / 1.25 | 115 | 118.9 (60.7) |
| gaussian_peaks | focusing | nlres400 | 0.0049 | 0.0188 | 0.00004 | 0.67 / 0.32 / 0.65 | 0.91 / 0.20 / 1.10 | 29 | 96.1 (56.6) |
| gaussian_peaks | focusing | network head alone | 0.0052 | 0.0187 | 0.00024 | (no sigma) | | | |
| gaussian_peaks | uniform | tree | 0.0613 | 0.0572 | 0.0566 | 0.67 / 0.66 / 0.68 | 0.98 / 0.98 / 0.99 | 116 | 67.1 |
| gaussian_peaks | uniform | nltree100 | 0.0168 | 0.0105 | 0.0108 | 0.68 / 0.71 / 0.67 | 0.90 / 0.99 / 1.01 | 117 | 81.8 (39.5) |
| gaussian_peaks | uniform | nltree400 | 0.0152 | 0.0088 | 0.0096 | 0.68 / 0.65 / 0.64 | 0.76 / 0.80 / 0.72 | 32 | 62.7 (37.6) |
| gaussian_peaks | uniform | nlres100 | 0.0156 | 0.0085 | 0.0089 | 0.68 / 0.66 / 0.65 | 0.92 / 0.83 / 0.82 | 116 | 130.5 (59.4) |
| gaussian_peaks | uniform | nlres400 | 0.0157 | 0.0085 | 0.0087 | 0.67 / 0.67 / 0.66 | 0.72 / 0.73 / 0.73 | 32 | 94.0 (56.4) |
| gaussian_peaks | uniform | network head alone | 0.0159 | 0.0086 | 0.0089 | (no sigma) | | | |
| rotated_rosenbrock | focusing | tree | 0.0158 | 0.0639 | 0.00019 | 0.69 / 0.36 / 0.62 | 0.97 / 0.42 / 1.15 | 116 | 87.8 |
| rotated_rosenbrock | focusing | nltree100 | 0.0059 | 0.0153 | 0.00005 | 0.68 / 0.56 / 0.67 | 1.08 / 0.71 / 1.23 | 120 | 75.8 (39.9) |
| rotated_rosenbrock | focusing | nltree400 | 0.0053 | 0.0132 | 0.00007 | 0.67 / 0.50 / 0.66 | 0.89 / 0.52 / 1.03 | 28 | 60.9 (36.5) |
| rotated_rosenbrock | focusing | nlres100 | 0.0054 | 0.0131 | 0.00005 | 0.70 / 0.38 / 0.63 | 1.13 / 0.41 / 1.31 | 116 | 114.0 (59.8) |
| rotated_rosenbrock | focusing | nlres400 | 0.0054 | 0.0134 | 0.00008 | 0.68 / 0.33 / 0.62 | 0.87 / 0.33 / 1.09 | 31 | 99.4 (57.6) |
| rotated_rosenbrock | focusing | network head alone | 0.0055 | 0.0131 | 0.00026 | (no sigma) | | | |
| rotated_rosenbrock | uniform | tree | 0.0454 | 0.0389 | 0.0386 | 0.67 / 0.66 / 0.66 | 0.81 / 0.83 / 0.84 | 119 | 71.3 |
| rotated_rosenbrock | uniform | nltree100 | 0.0137 | 0.0086 | 0.0089 | 0.68 / 0.66 / 0.64 | 1.00 / 0.96 / 0.92 | 115 | 80.4 (39.5) |
| rotated_rosenbrock | uniform | nltree400 | 0.0117 | 0.0058 | 0.0059 | 0.67 / 0.67 / 0.66 | 0.87 / 0.83 / 0.82 | 32 | 62.7 (38.1) |
| rotated_rosenbrock | uniform | nlres100 | 0.0118 | 0.0057 | 0.0060 | 0.68 / 0.67 / 0.66 | 1.01 / 0.84 / 0.82 | 118 | 126.5 (59.5) |
| rotated_rosenbrock | uniform | nlres400 | 0.0118 | 0.0057 | 0.0060 | 0.66 / 0.69 / 0.68 | 0.83 / 0.83 / 0.80 | 32 | 96.4 (58.0) |
| rotated_rosenbrock | uniform | network head alone | 0.0119 | 0.0058 | 0.0060 | (no sigma) | | | |

**Results, 5% noise** (rotated Rosenbrock, `sigma = 0.05 |y|` added to the
targets):

| stream | config | prequential | uniform-test | focus-test | coverage preq / uni / focus | sigma/RMSE preq / uni / focus | leaves | time [s] (of which net fits) |
|---|---|---|---|---|---|---|---|---|
| focusing | tree | 0.0154 | 0.0628 | 0.00011 | 0.69 / 0.42 / 0.59 | 1.00 / 0.48 / 1.04 | 121 | 137.4 |
| focusing | nltree400 | 0.0057 | 0.0180 | 0.00005 | 0.68 / 0.46 / 0.63 | 0.87 / 0.53 / 1.08 | 28 | 88.4 (55.9) |
| focusing | nlres400 | 0.0058 | 0.0177 | 0.00006 | 0.68 / 0.40 / 0.60 | 0.92 / 0.47 / 1.07 | 29 | 62.0 (36.2) |
| focusing | network head alone | 0.0059 | 0.0173 | 0.00033 | (no sigma) | | | |
| uniform | tree | 0.0460 | 0.0407 | 0.0402 | 0.68 / 0.68 / 0.68 | 0.83 / 0.83 / 0.85 | 117 | 116.2 |
| uniform | nltree400 | 0.0142 | 0.0098 | 0.0104 | 0.76 / 0.80 / 0.78 | 1.06 / 1.24 / 1.17 | 32 | 94.2 (56.0) |
| uniform | nlres400 | 0.0138 | 0.0091 | 0.0094 | 0.75 / 0.79 / 0.78 | 1.04 / 1.16 / 1.13 | 32 | 66.4 (39.8) |
| uniform | network head alone | 0.0140 | 0.0093 | 0.0096 | (no sigma) | | | |

**Reading (measured, one seed).**

* **Three to six times lower error than the plain tree, everywhere.** On
  the uniform stream the cube-wide error falls from 0.057 to 0.009 (peaks)
  and from 0.039 to 0.006 (rotated Rosenbrock); on the focusing stream the
  cube-wide error falls from 0.069 to 0.019 and 0.064 to 0.013, and the
  error in the focus region from 0.0007 to 0.00004 and from 0.00019 to
  0.00005. With 5% noise the ratios are the same. The plain tree's error
  barely moves between the two noise levels, which says it is limited by
  what a hundred-point leaf GP can represent in ten dimensions, not by
  the data.
* **The gain comes from the shared network, and the tree keeps it.** The
  network's own head is within a few percent of the neural-linear tree on
  the prequential and cube-wide metrics. With `Nbar = 100` and 128 features
  the leaves that regress the raw target lose 20 to 50% against the head
  cube-wide (0.0105 against 0.0086; 0.0086 against 0.0058), a leaf of a
  hundred points re-weighting 129 features; `Nbar = 400`, or the residual
  form at any `Nbar`, closes that to the head's number. The residual form is
  the one to adopt: it makes the leaf's default the network, not a constant.
* **What the tree adds to the network.** Two things the head does not have.
  A sigma: on-stream coverage is 0.66 to 0.71 for every neural
  configuration at nominal noise, with sigma over RMSE between 0.7 and 1.3,
  so the per-leaf prequential calibration works on the new leaf model
  unchanged; with 5% noise on the uniform stream it is conservative (0.75
  to 0.80), on the focusing stream nominal. And a local correction: where
  the stream is dense (the focus test set) the leaves are five times more
  accurate than the head, 0.00004 to 0.00008 against 0.00024 to 0.00033,
  at both noise levels. This is the division of labour of section 1 made
  visible: the network carries the shape, the leaves the detail where the
  data are.
* **Off the stream the sigma is too small, and the residual form makes it
  smaller.** On the focusing streams the cube-wide coverage is 0.46 to 0.59
  for the raw-target leaves (the plain tree: 0.36 to 0.57) and 0.32 to 0.40
  for the residual leaves, with sigma over RMSE down to 0.2. A residual
  leaf far from its data reverts to the head with the small posterior sigma
  of a small residual and no term for the head's own error there. This is
  the finding of the global-model benchmark again, where the fix was the
  learner's epistemic error budget added in quadrature; here the same term,
  or the spread of bagged feature networks, is the next step (the plan,
  step 2). On the uniform stream, where every leaf keeps receiving points,
  coverage is nominal on all three sets.
* **Cost is comparable, and the leaf work is cheaper than the GP's.** The
  first set ran one process per core: 61 to 82 s for the neural-linear
  trees against 67 to 88 s for the plain tree, of which 37 to 40 s are the
  six network refits of 4000 Adam steps. With `Nbar = 400` the tree has
  about 30 leaves instead of 115, and the whole 8000-point stream plus the
  6000 test predictions take 20 to 25 s of leaf work: the leaf solve is a
  129 x 129 system, 117 times for the evidence grid, every 25 points, and a
  rank-one update would remove most of that. The plain tree's leaves cannot
  grow that large without their cubic cost showing.

**Caveats.** One seed, two targets, ten dimensions, 8000 points. The
network trains on every point seen rather than on a reservoir, which is
fine at this size and not a design. The additive-GP global model
(`global_pkg`) was not run at `d = 10`, where its 45 pair terms make each
refit slow; the 6D benchmark has it at 15 to 50% below the plain tree,
which is far from the factor of three to six here. The rotated Rosenbrock
and the Gaussian peaks are smooth; a rough or thresholded target would test
the leaves' local correction harder and the network's features less well.

## 6. A plan

*Status.* Step 1 is done: `pygptreeo/neural_linear.py` is the package
version (residual leaves, evidence by eigendecomposition, the error budget,
optional bounded reservoirs, refits spread over the stream), with the 6D
four-stream benchmark and a 100 000-point scaling study in
`examples/BENCHMARK_RESULTS_neural_linear.md`. The rank-one update was not
needed: a leaf solve on at most `Nbar` points costs 25 ms at the 99th
percentile and does not grow with the stream. Step 2 was taken without
bagging: the leaf's sigma now has a floor, its local leave-one-out error
near its points rising to the function's scale beyond two spacings
(`BENCHMARK_RESULTS_neural_linear.md` §1.1), which makes the off-stream
sigma conservative on every stream at no cost; the focusing stream's stale
leaves remain the one place it is still under (coverage 0.41 to 0.48).

1. **Promote the probe into the package** (done): `NeuralLinearGPR` in its
   residual form, `FeatureNetLearner` with the doubling schedule and
   optional reservoir, `get_length_scales` from the Jacobian; torch as an
   optional dependency like GPyTorch. Still to run: the four-stream benchmark
   at `d = 10, 20` with real noise against `global_pkg`.
2. **An honest sigma off the stream** (a day): first the cheap version, the
   learner's prequential error budget added in quadrature as
   `global_mean.error_scale` already does; then `K = 3` to `5` bagged
   feature networks with `K` regressions per leaf and the between-network
   spread added. Judge on the cube-wide coverage of the focusing and
   sweeping streams, 0.32 to 0.59 in the probe, against the 0.68 target.
3. **Idea 4 in `simemu.py`** (a day): leaf-local calibration and a
   coverage-balanced rehearsal set for the ensemble refits; measure the
   certified-count stability on the 10D wave model against §4.15.
4. **Idea 1 as the emulator in the noisyprof prototype** (a few days): the
   tree of output regressions on the shared features, the profiler's
   gradient ascent through it, the leaf sigma in place of (or beside) the
   error map; measure accuracy per simulation, fit time per round and
   certification stability on the 6D and 10D models of the record.
5. **Idea 3 probe** (a day): TabPFN v2 as the leaf model of a tree with
   `Nbar` of 500 to 2000 on the same benchmark, to see whether an in-context
   leaf is competitive before investing in a pretrained prior of our own.

## 7. Sources

Verified in this session (abstract or full text):

* Lederer et al. 2020, Real-time regression with dividing local Gaussian
  processes, arXiv:2006.09446.
* Snoek et al. 2015, Scalable Bayesian optimization using deep neural
  networks, ICML (PMLR 37).
* Riquelme, Tucker, Snoek 2018, Deep Bayesian bandits showdown,
  arXiv:1802.09127.
* Titsias et al. 2020, Functional regularisation for continual learning with
  Gaussian processes, ICLR, arXiv:1901.11356.
* Gijsberts, Metta 2013, Real-time model learning using incremental sparse
  spectrum Gaussian process regression, Neural Networks.
* Vijayakumar, D'Souza, Schaal 2005, LWPR: incremental online learning in
  high dimensions, Neural Computation.
* Nguyen-Tuong, Seeger, Peters 2008/2009, Local Gaussian process regression
  for real-time online model learning, NIPS 21 / Advanced Robotics.
* Bui, Nguyen, Turner 2017, Streaming sparse Gaussian process
  approximations, NIPS 30, arXiv:1705.07131.
* Wilson, Hu, Salakhutdinov, Xing 2016, Deep kernel learning, AISTATS
  (PMLR 51); Ober, Rasmussen, van der Wilk 2021, The promises and pitfalls
  of deep kernel learning, UAI, arXiv:2102.12108.
* Immer, Korzepa, Bauer 2021, Improving predictions of Bayesian neural nets
  via local linearization, AISTATS (PMLR 130), arXiv:2008.08400.
* Eriksson et al. 2019, Scalable global optimization via local Bayesian
  optimization (TuRBO), NeurIPS, arXiv:1910.01739; Eriksson, Jankowiak
  2021, High-dimensional Bayesian optimization with sparse axis-aligned
  subspaces, UAI, arXiv:2103.00349.
* Constantine, Dow, Wang 2014, Active subspace methods in theory and
  practice, SIAM J. Sci. Comput., arXiv:1304.2070.
* Lakshminarayanan, Roy, Teh 2016, Mondrian forests for large-scale
  regression when uncertainty matters, AISTATS; Lakshminarayanan, Pritzel,
  Blundell 2017, Deep ensembles, NIPS 30, arXiv:1612.01474.
* Osband et al. 2023, Epistemic neural networks, NeurIPS 36,
  arXiv:2107.08924.
* Müller, Feurer, Hollmann, Hutter 2023, PFNs4BO: in-context learning for
  Bayesian optimization, ICML (PMLR 202), arXiv:2305.17535; Hollmann et al.
  2025, TabPFN v2 (Nature), with the ten-thousand-sample regime from the
  follow-up analyses arXiv:2502.17361 and arXiv:2505.20003.
* Garnelo et al. 2018, Conditional neural processes, ICML (PMLR 80),
  arXiv:1807.01613; Kim et al. 2019, Attentive neural processes,
  arXiv:1901.05761.
* Gramacy, Lee 2008, Bayesian treed Gaussian process models, JASA,
  arXiv:0710.4536 (the Bayesian ancestor of a treed GP).

Read in these repositories: `pygptreeo/gptree.py`, `gpnode.py`,
`global_mean.py`, `gp_interface.py`, the two benchmark results documents;
noisyprof branch `claude/friendly-wright-tznhd6`: `comparisons/nle/simemu.py`,
`simemu_api.py`, `docs/emulator_prototype_stages.md`,
`docs/records/simulator_emulator_proposal.md`, the two literature reviews.
