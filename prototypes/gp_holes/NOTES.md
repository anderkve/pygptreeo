# GP holes: limitations, issues and ideas

Notes for the further development of the prototype. The README has what the
prototype does and the results.

## Where the prototype stands

Closing notes of the session of 27 November 2026, for the one that takes over.

**Added in this session.** Each has a section in the README, from "A faster
search" on.

- `truth.py` has the right formula of each target as a model that the rules
  can write, and scores it on a data set. For I.30.5 and Planck's law it is a
  model with a hole, and SRSD iii.15.12 has none. `run_baseline.py` runs the
  search on the five sets with that score beside each result, and with
  `--check` on 41 of the runs.
- Speed: the gradient of the evidence without a second kernel matrix, outer
  functions that trail are left alone, and the short fit starts with scaled
  variances and has 20 iterations. A search takes less than half the time it
  took: 31 s against 71 s for `toy`.
- Rules: "add hole", for a model that has no hole left, and "look inside",
  which proposes a sum, a power of a sum or the logarithm of a sum from the
  values of a hole.

**The check runs.** `python run_baseline.py NAME --check` takes half an hour
on 4 cores: 19 runs of the benchmark, 11 and 7 of the two SRSD samples and 4
empirical laws. `results/baseline_check.txt` has them for the last commit:
33 of the 37 runs with a rule are right, against 24 at the start of the
session, and the 4 laws are found. No result scores more than 0.3 below the
right formula.
Not right are II.11.28 in two of three seeds (another function fits within
the noise), I.30.5 (no arcsin in the library) and SRSD bonus.20 (see
"Inputs over many decades").

These runs are not a test: every change was tried on them, and some were
made for one of them (SRSD i.6.20b, iii.10.19 and i.39.11).

**Not done: the full pass.** `python run_baseline.py NAME` runs the 150
searches of the five sets. No version of this session has run it; it should
take about two hours on 4 cores. The tables of the README before "A faster
search" are from the search as it was before. The 109 runs outside the check
runs are the only ones on which the changes have not been tried. The docs
set needs PySR, for its selection of inputs.

What to do next is in "Where to take it" below.

**Added in the session after it** (the same day). The question was whether
the rules can come from one principle in place of a growing list. Each of
these has a section in the README.

- The rule "symmetry" (`CFG['symmetry']`, off by default): candidates from
  linear relations between the values of a hole and its derivatives at the
  data. With `'first'` the listed merges, factors and functions that a hole is
  linear in are kept as a fallback only. `run_baseline.py` has `--symmetry`.
- `run_ode.py`: a differential equation for a hole of one input. It is not
  part of the search.
- A third SRSD sample (`srsd_sample3.json`, the set `srsd3` of
  `run_baseline.py`), drawn after the rule was written and before any run on
  it. It has been used once, for the search without the rule and with
  `'add'` and `'first'`, and nothing was changed afterwards. It is no longer
  held out.

In short: the rule can stand in for most of the list. With `'first'` the
check runs need 29% of the fits and the third sample 36%, and 30 of 37 and 19
of 30 results are right, against 33 and 20 without the rule. It does not
find more than the list does, and where a result changes it is more often
for the worse than for the better. The default is unchanged.

See "The rule symmetry" and "Representations that were weighed" below for
what did not work.

**Added on 28 November**, for data sets of more than a few hundred points.
The README has the measurements, in "Larger data sets".

- `local.py`: `LocalModel`, a model whose holes are GPs within blocks of at
  most 100 points, with the formula one for all points. The evidence is exact
  for this model and its cost grows in proportion to the number of points.
  `search(X, y, model=LocalModel)`, and `run_benchmark.py --local`.
- `bigdata.py`: `distil`, the predictions of a GPTree of all the points as the
  data of a search on 100 of them, and `validate`, the best models of a search
  fitted to all the points, with a GPTree for each hole, and compared on
  held-out points. `run_scaling.py` runs both and the search with local holes.
  They need pygptreeo and scikit-learn, which the prototype did not need
  before.
- `Model.holes` gives what a fit says of each hole, for the rules "look
  inside" and "symmetry"; `search` has `keep`, the number of models it
  returns, and `model`, the class of its models. The search without the rule
  "symmetry" gives the same results as before. With the rule its path can
  differ in the last digits: `'first'` on I.37.4 takes 867 fits in place of
  865, with the same score.

**Practical.**

- The prototype needs numpy, scipy and threadpoolctl; the docs set also pysr
  and scikit-learn. The first import of pysr sets up Julia, which takes some
  minutes.
- A search gives the same result every time for the same data and seed. A
  change that does not alter the path of a search leaves its score the same
  to two decimals, so two versions can be compared run by run
  (`results/baseline_check.json`).
- Run times vary by 10% between runs on a shared machine. The number of fits
  is the steadier measure.
- A search on 100 points takes about 20 s on 4 cores, 80 s of processor time,
  and between 2 s and 2 minutes. By the number of inputs, over the 41 check
  runs and the 30 runs of the third SRSD sample: 8 to 9 s and 1,000 to 1,300
  fits for 2 inputs, 16 to 24 s and 2,000 to 3,000 fits for 3, 35 to 51 s and
  4,300 to 5,600 fits for 4, and 39 to 69 s and 4,600 to 6,900 fits for 5.
  With the rule "symmetry" as `'first'` the mean is 8 to 12 s.
- What is under "Speed" and "The short fit cannot be much shorter" was
  measured with a copy of the search that logged every candidate of every
  step and its short fit. It is not in the repository.

## Where to take it

An assessment of 28 November 2026, from a discussion of what the prototype
is for: a global model of pygptreeo, or a method of its own.

**The order.** First the method on its own: the fixes and the tests below, a
version that is then held fixed, and a fresh held-out sample. A global model
for pygptreeo is measured on the side, with a learner outside the package,
and built only after that. Whether a paper is worth writing is decided after
the full pass and the held-out sample.

**Next steps, in this order.**

1. The two defects that change results (Issues): parentheses in the text of
   merged inputs, by which the search tells structures apart, and `snap` with
   the best rounding in place of the first. Then the 41 check runs, and
   `results/baseline_check.*` anew.
2. The full pass of 150 searches with that code, which no version has run:
   1.5 to 2 hours on 4 cores, with its results beside those of the README.
3. A fourth SRSD sample (`make_srsd_sample.py --sample 4`; 52 equations are
   left), drawn before anything further is changed and run once, with PySR
   beside it. The first three samples are spent.
4. A learner for `GPTree(global_mean=...)` in this folder that runs the
   search on the coverage sample of the tree each time the number of points
   has doubled, compared with `'additive_gp'` and `'net'` on streams of up to
   5 inputs. It is a measurement of which weaknesses matter there.
5. What the two uses both need: a check against formulas that fit within the
   noise and are wrong outside the data; inputs screened before the search,
   for more than 5; and the noise of each point as given, where it is known.

Later, from the list this replaces: one function of holes with symbolic
terms around it, `y = a(x)·g(H) + b(x)`, which is exact, and a product of
two holes (Ideas).

**Why not a global model in pygptreeo yet.**

- The number of inputs does not fit. The global model of pygptreeo is for
  targets of about ten inputs, where a leaf of 100 points cannot learn the
  shape. The search takes up to 5 and is out of reach at 30.
- A wrong formula is used with confidence. The search prefers a formula that
  fits within the noise to a hole, and a global model is what the tree
  falls back on ahead of the stream.
- The prototype is not settled: the full pass has not been run, the rules
  and the costs were set on the benchmark, and it fits a noise level of its
  own where pygptreeo has the noise of each point.

A learner outside the package is possible as it is: `global_mean` takes an
object of a subclass of `GlobalMeanLearner`, whose `observe` sees every point
and publishes fitted models with a `predict`.

**The method on its own.** The search needs numpy, scipy and threadpoolctl
only, so it can be a package of its own. Its parts are known:

- AI Feynman (Udrescu & Tegmark 2020, arXiv:1905.11481 and 2006.10782) fits
  a neural network and tests it for symmetry and separability, which it uses
  to divide the problem.
- A Kolmogorov–Arnold network (Liu et al. 2024, arXiv:2404.19756) has spline
  functions of one input that are replaced by symbols one at a time.
- The search over kernels of Duvenaud et al. (2013, arXiv:1302.4922) is
  greedy and scored by the evidence of a GP, but its result is a kernel.
- A formula with a GP for what it leaves out is the model of Kennedy &
  O'Hagan (2001) for a simulator. Zhu et al. (2022, "Robust equation
  discovery considering model discrepancy") have it with a sparse Bayesian
  choice of terms, and universal differential equations (Rackauckas et al.
  2020, arXiv:2001.04385) with a neural network that is turned into symbols
  afterwards.

No paper was found that puts a GP prior on the part of an expression that is
not known and scores expressions with such parts by their exact evidence.
The search for one was short (two reviews: arXiv:2507.19540 and 2606.06567).
What looks new:

- the exact evidence of every partial expression, so that each step of the
  search is a comparison of models;
- a hole in the result, where the data or the rules cannot decide;
- the rules as linear constraints on a hole (README, "The symmetries of a
  hole"), which is a principle in place of a list;
- about 100 points with noise, where most methods use thousands.

What a paper needs and is not there:

- a method that is held fixed and a held-out set that is fresh;
- baselines that a referee accepts: PySR with the time its paper gives it and
  not 60 s, AI Feynman, and a Bayesian method;
- the search with a rule or a cost left out, one at a time, and fewer
  numbers set by hand (the costs of 3 and 10, the thresholds);
- a count of the results that pass as right and are approximations.

The objection to expect is "AI Feynman with a GP". The answer is in the
first three of the points that look new, and has to be shown on data.

**Larger data sets, in the order to try** (see also "Larger data sets" under
Limitations).

1. Propose on few points and decide on all. A formula without a hole costs
   in proportion to the points (3 ms for an evaluation at 10,000), so every
   candidate without a hole can be fitted to all the points at each step,
   while the short fits stay on about 100.
2. Compress in place of taking a part, as `distil` does, which gave 13 of 26
   against 9. Further: the error of each compressed point given to the
   search, the residual compressed again once a formula is found, and
   several sets of compressed points.
3. A hole of one or two inputs as 20 to 50 basis functions. The cost is then
   in proportion to the points, without blocks and what they do to the order
   of models. For a hole of many inputs, each point conditioned on its
   nearest neighbours in place of blocks. Neither has been tried.
4. The relations of the rule "symmetry" from all the points, which costs in
   proportion to them.
5. A score for the size: the costs of a function and of an exponent were set
   for 100 points. With many points the likelihood of held-out points can
   decide.

Not blocks as they are, and not a GPTree for each candidate.

**Continual learning with pygptreeo.** Ideas, none of them built.

- The search runs on the coverage sample of 500 points that the tree keeps
  for its global models, in another process, each time the number of points
  has doubled.
- A formula is a third kind of global model, whose residual the leaves fit.
  A result with a hole helps as well: if the hole has `x1/x2` as its input,
  the leaves learn in that one input in place of two.
- The stream scores the candidates. Each point is predicted before it is
  added, and the sum of the log densities of these predictions is the
  evidence. A few formulas can be kept, updated at each point, and dropped
  when they fall behind.
- A new search when the points have doubled, or when the error of the
  formula at new points is above its noise. The residuals of the leaves show
  where it fails.
- A stream that moves covers a small region at first, where every function
  looks like a polynomial: in half of the box of I.37.4 the search found the
  formula in 0 of 8 runs. Early formulas are provisional. The few best can be
  kept, with their disagreement as the uncertainty ahead of the data.
- Where the formula is right to within the noise, the leaves can go, which
  helps under a memory limit.

## Limitations

### A hole cannot sit inside a function

A hole may be multiplied by symbolic factors, `x3·H(…)`, and take symbolic
inputs, `H(x1/x2)`. In both cases it enters the model linearly, so the model
is a GP and its evidence is exact. A hole inside a function, `log(H(x))`, is
not a GP.

**Around the whole expression** the prototype has it (`CFG['outer']`): for
`y = g(m(x))` the inner expression `m` is fitted to `g⁻¹(y)`, with the
Jacobian of the transform in the score, as in a warped GP (Snelson et al.
2003). This is exact for noise inside the function. Noise outside it is
carried through the transform to first order, as a second noise term with a
variance for each point. The same would work for one hole in one function
when the rest of the expression is symbolic, `y = a(x)·g(H) + b(x)`.

**From the values of a hole** the rule "look inside" proposes a sum of
functions of the inputs, a power of such a sum or its logarithm, with least
squares for the sum. Such a formula has no hole, so its evidence is exact.
The sum has one function for each input of the hole, so a product of
functions of two inputs inside it is out of reach. So is the logarithm of a
sine with its zeros at the ends of the data, as in SRSD bonus.1: with a
frequency that is a little off, the sine is negative there.

**In the interior**, `x1·log(H(x2, x3)) + H(x4)`, it is missing. The evidence
is then an integral over the values of the inner hole at the data points,
without a closed form. A way that keeps the structure of the search, as a
hierarchy of holes:

- Linearise the function around the current estimate ĥ of the inner hole,
  `g(H) ≈ g(ĥ) + g'(ĥ)·(H − ĥ)`. The hole then enters linearly, with the
  numerically evaluated factor `g'(ĥ(x))`: a term that the prototype already
  scores exactly.
- Iterate, with the posterior mean of the hole as the next ĥ (Gauss–Newton).
  At convergence ĥ is the posterior mode, and the evidence of the linearised
  model is the Laplace approximation of the evidence.
- Nested functions, `log(H1 + exp(H2))`, are linearised by the chain rule, so
  the holes of all levels end up in one GP, with factors that are products of
  derivatives at the current estimates. There is one evidence for the whole
  hierarchy: the data enter only at the top, so a level has none of its own.
- The approximation is good where `g` is close to linear over the spread of
  the inner hole, and poor for `H²` or `sin(H)` (several modes) and for
  `log(H)` with `H` near 0. Its error differs between models, so it should be
  checked against a numerical integral (e.g. importance sampling around the
  mode) on small problems before linearised and exact models are compared.
- Related: the extended GP (Steinberg & Bonilla 2014), the same linearisation
  in a variational framework; deep GPs (Damianou & Lawrence 2013) and chained
  GPs (Saul et al. 2016), with variational bounds on the evidence; Laplace
  evidence with a Gauss–Newton Hessian for model selection (Immer et al. 2021).

### The rule "symmetry"

The rule reads the structure of a hole off its fit, so it is as good as the
fit. Three ways in which it fell short of the listed rules on the check
runs:

- **A hole that the GP cannot fit has no relations.** Under the logarithm,
  the hole of SRSD i.6.20b is `−(θ − θ1)²/(2σ²) − log σ`, with `σ` down to
  0.1: it scores 268 where the formula scores 920, and no relation has a
  residual below the threshold. The listed merge to `x1 − x2` is tried
  without looking, and the evidence takes it. With `'first'` the listed rules
  come in once the rule has nothing new, and the formula is found.
- **The evidence can prefer a wrong merge.** SRSD bonus.7 is
  `√(a·x1 − b·x2/x3²)`, with inputs over two decades and `x2` of both signs.
  In seed 1 the hole `H(log(x1), x2/x3^0.5)` scores 313 and
  `H(log(x1), x2/x3²)` 257: the right input is spread over four decades, and
  a hole that is not linear in it is fitted worse. The search follows the
  wrong merge and does not come back. Under the square the order is the other
  way, 348 against 290, and the listed rules reach the formula there.
- **A relation tells less apart than the evidence does.** The hole
  `H(x1·x2)` of II.11.28 is `3/(1 − u/3) − 2`. The relation of `exp(c·u)`
  holds within the accuracy of the second derivatives, and the fit of
  `(1 + c·u)^p` from the values that they give ends at `(1 − 0.5·u)^−0.19`, 7
  below the exponential. From the listed starting values, `c = ±0.5` and
  `p = −1`, it ends at the right formula, 4.5 above the exponential.

Also:

- The residual of a relation does not use the uncertainty of the
  derivatives. A Bayes factor of the relation from the posterior of the hole
  would, and would not need the thresholds.
- Relations of more columns have smaller residuals by chance. They are
  limited to one coordinate for all inputs for this reason.
- "No interaction" is still listed. Its relation is a single column, the
  mixed second derivative, which needs a scale to be judged by.

### Larger data sets

- **The search with local holes found less than the search on distilled
  points** in its first test, 3 of 7 results against 6 of 7 for the same
  cases, and it lost two results that the search on 100 points has. With the
  same 300 points in blocks of 100 and in one block, the blocks cost one
  result of four (I.6.2 at 10% noise), so a part of this is the blocks
  themselves (next point). Not tested: the costs of a symbolic function and
  of a free exponent are fixed numbers that were set for 100 points, while
  differences in the evidence grow with the number of points, so with 1,000
  points the costs hold back less.
- **A search with local holes costs in proportion to the points.** With 1,000
  points a fit takes about 10 times as long as with 100, so a search of 10,000
  points would take hours. More points than about 1,000 have to come in
  another way: through a tree before the search (`distil`), or by fitting
  most candidates to a part of the points and only the best to all of them,
  which has not been tried.
- **Blocks lower the evidence of a hole.** A hole in blocks has no
  covariance between points of different blocks. With the same parameters,
  the evidence of a hole of three inputs on 240 points is 48 lower in 4
  blocks than in one. A formula does not change, so with blocks a formula
  that fits within the noise is preferred to a hole more than before. The
  loss also depends on how the inputs of a hole are cut, so it differs
  between models with different holes and changes their order.
- **One set of blocks for a model.** The blocks are the leaves of cuts
  through the inputs of all the holes of a model together, so in
  `x1·H(x2) + H(x3)` the hole `H(x2)` is also cut along `x3`, and learns
  less from the points than it could. Blocks of its own for each hole would
  make the covariance a sum of block matrices in different orders, whose
  evidence has no cheap exact form.
- **The blocks are made anew for each fit**, from the inputs of the holes as
  they are at its start, so two fits of the same structure can have different
  blocks, and their scores differ a little.
- **A hole is predicted from one block**, the one whose points are nearest
  in the mean, without the blending of pygptreeo in the overlap.
- **The predictions of a tree are not independent.** In `distil` their
  errors are smooth within a leaf, and the search takes them for noise. The
  choice between the best models is therefore made on the points themselves
  (`validate`).
- **`validate` drops a model that it cannot evaluate at every point.** With
  strong noise some targets of a positive function are not positive, and a
  model with a logarithm, a root or an inverse around the expression is
  dropped. A model without a hole could be fitted by least squares to the
  targets as they are.
- **`validate` does not refit the symbolic parameters of a model with a
  hole**, and its score is a likelihood of held-out points with the costs of
  the search subtracted, not an evidence. The numbers of models it compares,
  10 and at most 2 with a hole, were set once.
- **The noise is not given to a GPTree here.** Each leaf learns its own
  level, through a white-noise term in its kernel. From a start at 1% of the
  variance of the leaf's targets the fit often stays there: for points with
  30% noise the predictions are then off by 19% of the spread, against 9%
  from a start at 50%.

### Other limitations

- **A hole times a hole** is not a GP either. The rules get around a product
  only when one factor is in the library. A logarithm around the whole
  expression (above) turns a product of positive functions into a sum.
- **Inputs that change sign.** A fitted exponent is undefined for negative
  numbers, so such inputs get the integer powers 1, 2, 3, −1 and −2, `log|x|`,
  and the product and ratio with another input, but no fitted power `|x|^p`.
- **Relative noise.** The variance of the relative noise at a point is in
  proportion to the square of the observed value, not of the model's
  prediction, which is close for small noise only. A constant can therefore
  explain any data as relative noise of about 100%: on SRSD sample 2, i.43.16,
  it scored 309 against 297 for a single hole, and the search stalled. The
  relative noise is bounded at 30% for this reason; a variance from the
  model's own prediction would not need the bound. Without the term, a hole
  stayed beside the right formula for targets that change sign and have
  relative noise (SRSD sample 1: i.12.4, ii.11.20, ii.34.29b, bonus.18); with
  it, the formula alone scores higher on i.12.4 (608 against 603).
- **Inputs over many decades.** A hole takes its inputs as they are
  (standardised), or the logarithm of a positive input. Inputs that change
  sign and span decades remain a poor fit for the GP.
- **Targets over many decades.** The noise of the same size at every point
  cannot go below 1.6e-4 of the spread of the targets (its bound, and a
  jitter of 1e-8 on the variances). The targets of SRSD bonus.20 span 7.6
  decades, and 63% of them lie below that. Its formula is a sum of four
  terms, which the rules can write, but it scores 670 where a model under a
  logarithm scores 1102; there the formula is the logarithm of a sum with
  products of functions of two inputs.
- **A constant is always added** to the expression, also inside an outer
  function. It comes out small but not zero, e.g. `1/(1/d1 + n/d2 + 0.048)`.
- **Few points.** Below about 15 points the score (BIC for the parameters)
  lets spurious terms in, or the search stays at a hole.
- **A formula that fits within the noise is preferred to a hole**, also when
  it is wrong outside the data. The costs per function and per free exponent
  reduce this. They were chosen on the benchmark targets.
- **Greedy search.** A step that does not raise the score by itself is rarely
  followed up, so structure that pays off after two steps is missed, e.g. a
  series built one term at a time. The result depends on the seed for the
  harder targets, and on small changes to the search: for Newton's law, with
  30 points, one version went on from the law to `√(H(log(r/√(m1·m2))))`,
  which scores 1.7 higher, and the versions before and after it return the
  law. The search can also miss a model that scores higher, though in the
  check runs no result is now more than 0.3 below the right formula.
- **Number of inputs.** The candidates per step grow with the square of the
  number of inputs of a hole. 5 inputs take one to four minutes, 30 are out
  of reach.
- **The score decides alone.** The search returns its 5 best models, but no
  Pareto front of accuracy against complexity as PySR does; PySR's paper
  counts a law as found if it is anywhere on the front.
- **Speed.** A run fits thousands of candidates, and these short fits take
  90% of its time. All but 3% of the candidates are dropped after their fit.
  Half of them are further starting values of a structure that is in the
  step already, and a quarter come from the refinement of outer functions. An
  evaluation of the evidence takes about 1 ms for 100 points.
- **The short fit cannot be much shorter.** The score after fewer iterations
  is a poor guide to the score at the end of the short fit. Of the candidates
  that went on to a full fit and were refined later, one in ten would not
  have gone on had all short fits stopped after 10 iterations, and one in
  forty after 20 (of 25, without the scaling of the variances). With the
  scaling and 10 iterations, one search of 28 stopped at the first hole.

## Issues

- The benchmark has not been run with the polynomial in the library.
- A numerical criterion of success (an error below the noise outside the
  range of the data) also counts close approximations as right, for both
  codes, e.g. `exp(log(x1) + log(x2) + 8.3e-26·x2³)` for `x1·x2/√(1 − x2²/c²)`.
  The results need a look at the formulas as well.
- The rules, the library and the costs were developed on the benchmark
  targets, so the benchmark overstates how well the search does.
- The text of a merged input inside another one can be read in two ways,
  e.g. `x1/(x1*x2^0.5)/x3^0.5` for `x1/((x1·x2^0.5)/x3^0.5)`. The structures
  of the search are told apart by this text.
- A sine inside a sum keeps a phase of −π/2, as in
  `(1 − sin(−1.57 + 2·u))` for `1 + cos(2·u)`: `snap` rounds a phase to
  π/2 with the sign of a weight changed, and here there is no weight.
- `snap` rounds the first parameter whose rounding raises the score, so its
  result depends on the order of the parameters. For `(1 + c·u)^p` of
  II.11.28 it can round `c` and keep a free exponent, 12 below the score with
  the exponent rounded and `c` free. Trying every parameter and taking the
  best would not depend on the order.

## Ideas

- Models without the constant, for the evidence to decide.
- Functions of holes in the interior by linearisation (see above).
- A score for few points that integrates over the parameters in place of BIC.
- A hole of one input that the library cannot fill could be handed to another
  symbolic regression code, with the GP posterior as data.
- More from the values of a hole: products of functions of two inputs in the
  sums of the look inside, and a product of two holes where the values of a
  hole of both inputs, on a grid, have rank one. As a relation, a hole is a
  product of a function of `uj` and one of `uk` where
  `H·∂²H/∂uj∂uk = ∂H/∂uj·∂H/∂uk`.
- For the rule "symmetry": the relations of a hole whose inputs span decades
  from a fit of the hole in their logarithms; the listed starting values
  beside those of a relation; a continuous coordinate `u^μ` in place of the
  four.
- The differential equation of a hole of one input as a rule of the search
  (`run_ode.py`). For noisy data, relations between integrals of the hole
  in place of its derivatives. Equations with other coefficients than
  polynomials, e.g. `sin(u)` and `cos(u)` for the pendulum. For a hole of
  several inputs, the same columns for one input at a time give the function
  of that input that the hole has as a factor.
- A hole inside a function as its values at a few knots, with a GP prior,
  fitted as an exponent is. The rest of the model stays a GP.
- New points where the candidate formulas disagree most (active learning).
- For many points: the quick fits of a search on a part of the points and the
  full fits on all of them, with local holes; the relations of the rule
  "symmetry" from the derivatives of a tree of all the points; and in
  pygptreeo itself, a formula as a third kind of global model, whose
  residual the leaves fit.
- On the pygptreeo tree: one symbolic skeleton shared by all leaves, the holes
  local to each leaf, and the summed evidence of the leaves as score. The
  global model becomes a mean, a factor and an input map.
- A kernel from the grammar: the second moment of the prior over expressions
  as the kernel of a hole, so that refining a hole does not change the prior.
- A GP over expressions as a surrogate for an expensive score, with a distance
  between the functions two expressions can represent, not between their
  syntax. Only worth it when the score is expensive.

## Representations that were weighed

- **Kolmogorov–Arnold.** The theorem writes any continuous function as
  `Σ Φq(Σ φqp(xp))`, but with inner functions that are not smooth, so it says
  nothing about formulas. What is of use is its shape, a function of a sum of
  functions of one input each: the merges, the factors, the look inside and
  the function around the expression are all of this shape. The rule
  "symmetry" finds such nodes where a hole stays a hole (an input
  `sk + c·sj`, a factor `exp(γ·s)`), with the coordinates `u`, `log(u)`,
  `u²`, `1/u` as the functions of one input. A node with a hole below a
  function in the interior is where the evidence stops being exact, as
  before.
- **EML**, `eml(x, y) = exp(x) − ln(y)` with the constant 1 (Odrzywołek,
  arXiv:2603.21852), builds every elementary function from one operator. A
  hole in either argument is inside `exp` or `log`, so no partial expression
  is a GP. Its measure of simplicity is also far from that of a formula of
  physics: multiplication is a tree of depth 8, and its own search by
  gradient recovers formulas in 100% of runs at depth 2, 25% at depths 3 to 4
  and under 1% at depth 5.
- **A continuous function around the expression.** The score of the first
  hole was computed for `z = (y^λ − 1)/λ`, with `log(y)` for `λ = 0`, for 8
  values of `λ` from −2 to 3 and the targets of the check runs of one sign.
  The best `λ` is that of a right formula where the function matters most
  (`λ = 2` for I.8.14, by 122; the logarithm for SRSD i.6.20b), but not for
  SRSD bonus.7 (`λ = −1`, with the right `λ = 2` lower by 78) or i.27.6 (flat,
  with the right `λ = −1` lower by 21), and for products it prefers a root or
  the logarithm, in which the hole is simpler. So the function cannot be
  chosen from the first hole, and following several of them for some steps,
  as the search does, is needed. This was a scratch script and is not in the
  repository.

## Questions to discuss

- **The search.** It is greedy: a beam of the best models, refined step by
  step, so it can stop at a form that is only locally best. Alternatives to
  weigh: moves that make a part of a model a hole again (so that a step can be
  undone), stochastic search over models (simulated annealing, MCMC),
  sequential Monte Carlo over models with the data added in batches, a wider
  beam with more diversity, and proposals from the posterior of a hole. One
  move of the first kind is in: a new hole beside a model that has none left.
  `run_baseline.py` has the score of the right formula beside that of each
  result, which tells failures of the search from failures of the score.
- **Against handing the data to an AI agent.** What does the algorithmic way
  give that asking an agent for the formula does not, and what must hold for
  it to stay useful? Points raised: it should stay light enough to run inside
  other computations, e.g. live during the calibration of a large model or a
  simulation, or in continual learning.

## Tests still to do

- The full pass with the present search (see "Where the prototype stands").
- The benchmark with the polynomial in the library.
- More inputs.
- Noise levels other than 1%, and noise that is not relative.
- PySR with tuned settings and longer runs.
- A fresh held-out sample: the three SRSD samples have been used (see "Where
  to take it").
