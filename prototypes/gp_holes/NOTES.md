# GP holes: limitations, issues and ideas

Notes for the further development of the prototype. The README has what the
prototype does and the results.

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
  harder targets. The search can also miss a model that scores higher: on
  SRSD i.39.11 the right formula scores 567 and the returned model, which
  has a hole, 557.
- **Number of inputs.** The candidates per step grow with the square of the
  number of inputs of a hole. 5 inputs take minutes, 30 are out of reach.
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
- New points where the candidate formulas disagree most (active learning).
- On the pygptreeo tree: one symbolic skeleton shared by all leaves, the holes
  local to each leaf, and the summed evidence of the leaves as score. The
  global model becomes a mean, a factor and an input map.
- A kernel from the grammar: the second moment of the prior over expressions
  as the kernel of a hole, so that refining a hole does not change the prior.
- A GP over expressions as a surrogate for an expensive score, with a distance
  between the functions two expressions can represent, not between their
  syntax. Only worth it when the score is expensive.

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

- The benchmark with the polynomial in the library.
- More inputs.
- Noise levels other than 1%, and noise that is not relative.
- PySR with tuned settings and longer runs.
- A fresh held-out sample once the SRSD sample has been used to develop the
  search.
