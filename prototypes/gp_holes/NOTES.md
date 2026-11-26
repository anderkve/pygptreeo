# GP holes: limitations, issues and ideas

Notes for the further development of the prototype. The README has what the
prototype does and the results.

## Limitations

### A hole cannot sit inside a function

A hole may be multiplied by symbolic factors, `x3·H(…)`, and take symbolic
inputs, `H(x1/x2)`. In both cases it enters the model linearly, so the model
is a GP and its evidence is exact. A hole inside a function, `log(H(x))`, is
not a GP, and the prototype has no rule that makes one.

So a function can only be applied to what the rules put into it: an input, a
product of powers of inputs, or a linear combination of two inputs.
`log(x1 + c·x2)` can be written, `log(0.4 + 0.3·e^x)` cannot. Three laws of
EmpiricalBench fail for this reason (Bode, Rydberg, Planck).

Two ways to add it:

- **A function around the whole expression.** If `y = g(H)`, then
  `g⁻¹(y) = H`: transform the target and search as before, with the Jacobian
  of the transform in the score so that scores of different transforms can be
  compared. Exact. The same works for one hole in one function when the rest
  of the expression is symbolic, `y = a(x)·g(H) + b(x)`.
- **A function of a hole in the interior**, `x1·log(H(x2, x3)) + H(x4)`. This
  needs approximate inference, e.g. a linearisation around the current
  estimate of the hole, and gives up the exact evidence.

### Other limitations

- **A hole times a hole** is not a GP either. The rules get around a product
  only when one factor is in the library. A logarithm around the whole
  expression (above) turns a product of positive functions into a sum.
- **Inputs that change sign.** A fitted exponent is undefined for negative
  numbers, so such inputs only get the powers 1 and 2, and no ratio with
  another input. `1/x` cannot be written for them.
- **Inputs over many decades.** A hole takes its inputs as they are
  (standardised). For inputs spread log-uniformly over decades the points lie
  bunched at one end, and the GP is a poor model.
- **Few points.** Below about 15 points the score (BIC for the parameters)
  lets spurious terms in, or the search stays at a hole.
- **A formula that fits within the noise is preferred to a hole**, also when
  it is wrong outside the data. The costs per function and per free exponent
  reduce this. They were chosen on the benchmark targets.
- **Greedy search.** A step that does not raise the score by itself is rarely
  followed up, so structure that pays off after two steps is missed, e.g. a
  series built one term at a time. The result depends on the seed for the
  harder targets.
- **Number of inputs.** The candidates per step grow with the square of the
  number of inputs of a hole. 5 inputs take minutes, 30 are out of reach.
- **One formula is returned.** PySR returns a Pareto front, and its paper
  counts a law as found if it is anywhere on the front.

## Issues

- The gradient of the evidence with respect to a parameter of a symbolic
  function is a finite difference with a step of 1e-6 for parameters below 1
  in size. For inputs of size 1e9 a coefficient in `exp(c·u)` is far smaller
  than the step, and such candidates fail.
- The benchmark has not been run with the polynomial in the library.
- The rules, the library and the costs were developed on the benchmark
  targets, so the benchmark overstates how well the search does.

## Ideas

- The pieces above that keep the evidence exact: integer powers for inputs
  that change sign, the logarithm of a positive input as an input of a hole,
  and a function around the whole expression.
- Return the few best models with their scores.
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
- Noise levels other than 1%.
- PySR with tuned settings and longer runs.
