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
  numbers, so such inputs only get the integer powers 1, 2, 3, −1 and −2, and
  the product and ratio with another input. They have no logarithm, so inside
  an outer logarithm `log|x|` is missing: for `y = k·x²/2` with `x` of both
  signs one version of the search returned a contrived formula in place of
  `2·log|x|`.
- **Relative noise.** The noise is the same at every point, apart from the
  noise outside an outer function. With relative noise (1% in the SRSD
  sample) a target of one sign does well with an outer logarithm. For a target
  that changes sign a hole stays beside the right formula to take up the
  difference (SRSD i.12.4, ii.11.20, ii.34.29b, bonus.18). A noise term in
  proportion to `y²` fixes this in a test: on i.12.4 the formula alone then
  scores 608, against 566 for formula and hole with the present noise.
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
  harder targets. The search can also miss a model that scores far higher: on
  SRSD i.6.20b the right formula scores 922 and the returned one 488, and on
  II.11.28 277 against 273.
- **Number of inputs.** The candidates per step grow with the square of the
  number of inputs of a hole. 5 inputs take minutes, 30 are out of reach.
- **The score decides alone.** The search returns its 5 best models, but no
  Pareto front of accuracy against complexity as PySR does; PySR's paper
  counts a law as found if it is anywhere on the front.

## Issues

- The benchmark has not been run with the polynomial in the library.
- A numerical criterion of success (an error below the noise outside the
  range of the data) also counts close approximations as right, for both
  codes, e.g. `exp(log(x1) + log(x2) + 8.3e-26·x2³)` for `x1·x2/√(1 − x2²/c²)`.
  The results need a look at the formulas as well.
- The rules, the library and the costs were developed on the benchmark
  targets, so the benchmark overstates how well the search does.

## Ideas

- A noise term in proportion to `y²` (see "Relative noise"), and `log|x|`
  for inputs that change sign.
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

## Tests still to do

- The benchmark with the polynomial in the library.
- More inputs.
- Noise levels other than 1%, and noise that is not relative.
- PySR with tuned settings and longer runs.
- A fresh held-out sample once the SRSD sample has been used to develop the
  search.
