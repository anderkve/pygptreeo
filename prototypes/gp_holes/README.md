# Symbolic regression with GP holes (prototype)

A model is an expression in which each part that is not yet known is a *hole*:
an unknown function with a Gaussian process prior. If every hole enters as
`a(x)·H(u(x))` with `a` and `u` symbolic, the whole model is one GP and its
evidence is closed-form. The search starts from a single hole `H(x1, …, xd)`,
i.e. plain GP regression, and applies rewrite rules that replace a hole with
more specific structure, keeping the models with the highest score. A hole that
remains at the end marks a part that the data or the rules could not resolve.

The prototype does not use the `pygptreeo` package. It needs numpy, scipy and
threadpoolctl.

## Files

- `gpholes.py`: the model, its evidence, the rewrite rules and the search.
- `targets.py`: target functions, mostly from the Feynman database.
- `run_benchmark.py`: the search on each target.
- `run_growth.py`: the result for one target as the number of points grows.

```bash
python run_benchmark.py --seeds 3            # all targets
python run_benchmark.py --only I.6.2 --verbose
python run_growth.py --target toy
```

## Rewrite rules

For a hole `H(u1, …, uq)` with symbolic inputs `u`, and `φ` from a library of
functions of one input:

| Rule | Replaces the hole with |
|---|---|
| no interaction | `H(all but uj) + H(all but uk)` |
| factor | `φ(uj)·H(all but uj)`; for `q = 1` also `φ(u)·u^p` |
| linear | `φ(uj)·H(all but uj) + H(all but uj)` |
| peel (`q = 1`) | `w·φ(u) + H(u)` |
| merge | `H(…, v, …)` with `v = uj·uk^p` or `v = uj + c·uk` in place of `uj, uk` |
| change input | the same with `v` in place of only one of the two |
| drop input, drop factor, drop term | the model without it |

The library is `u^p`, `log(u)`, `exp(c·v)`, `(1 + c·v)^p` and `sin(a + b·u)`,
with `v` one of `u`, `u²`, `1/u`. A sine must show at least half a period over
the data.

## Score

The log evidence at the fitted parameters, minus `½ log n` per parameter (BIC),
minus 3 per symbolic function, minus 10 per exponent that is not a simple
number (`NICE` in `gpholes.py`). Fitted exponents, coefficients of merged
inputs and phases are rounded to simple values where that raises the score.

Each step refines those of the 4 best models that have not been refined yet.
Every candidate gets a short fit, and the 12 best structures a full one.

## Results

100 points, uniform in the box of each target, noise 1% of the standard
deviation of the target, 3 seeds, Matérn-5/2 holes (`run_benchmark.py`). A run
counts as right if the result has no hole and its error outside the training
box is below the noise level. A run takes 20 to 210 s with 2 worker processes.

| Target | Right | Hole left | Other |
|---|---|---|---|
| `x2·sin(2·x1) + 0.5·x3²` | 3 | | |
| `x1·x2·x3` | 3 | | |
| `x1·x2/(4π·x3²)` | 3 | | |
| `x1·x2·x3/x4` | 3 | | |
| `0.5·x1·(x2² + x3² + x4²)` | 3 | | |
| `0.25·x1·(x2² + x3²)·x4²` | 3 | | |
| `exp(−(x1/x2)²/2)/(√(2π)·x2)` | 3 | | |
| `x1·(x2 + x3·x4·sin(x5))` | 3 | | |
| `x1 + x2 + 2·√(x1·x2)·cos(x3)` | 3 | | |
| `x1·exp(−x2·x3/x4)` | 3 | | |
| `x1·x2·log(x3/x4)` | 3 | | |
| `x1·(cos(x2·x3) + x4·cos²(x2·x3))` | 3 | | |
| `x1/√(1 − x2²/x3²)` | 3 | | |
| `(x1·x3 + x2·x4)/(x1 + x2)` | 1 | 1 | 1 |
| `1 + x1·x2/(1 − x1·x2/3)` | 1 | | 2 |
| `arcsin(x1/(x2·x3))` | | 1 | 2 |
| `√((x2 − x1)² + (x4 − x3)²)` | | 2 | 1 |

The last two targets cannot be written with the rules and the library, so the
intended result is a hole: `H(x1/(x2·x3))` and `H(x1 − x2, x3 − x4)`.

- The runs under "Other" return a formula that agrees with the target to
  within the noise inside the training box and is wrong outside it, e.g.
  `1.25·exp(0.78·x1·x2) − 0.25` for the second of them. For the last target
  it is a sum of products of exponentials and rational functions, with an
  error of twice the noise.
- For arcsin one run returns `0.997·u + 0.203·u³` with `u = x1/(x2·x3)`, close
  to the first two terms of its series, `u + u³/6`.
- With squared-exponential holes (`--kernel rbf`) the first 13 targets give
  38 of 39, and the last target never gives the plain hole.

For the first target with fewer points (`run_growth.py`): with 1% noise the
formula is right from 15 points on, and at 10 points the results are wrong
(spurious factors, or a hole with a wrong factor). With 20% noise the result is a hole or a partial expression at 15
points, the right formula in 2 of 3 runs at 25 to 60 points (the other with a
hole or a wrong factor of `x2`), and in all runs at 100 and 200 points.

## What does not work yet

- A formula that fits within the noise is preferred to a hole. The two costs
  in the score were chosen after seeing this on these targets, and they
  reduce it without removing it.
- A hole times a hole, and a function of a hole, are not GPs. The rules get
  around a product only when one factor is in the library.
- A step that does not raise the score by itself is rarely followed up, so
  structure that pays off after two steps can be missed.
- The search is greedy and its result depends on the seed for the harder
  targets.

## Tests still to do

- A target without a closed form but with a known series expansion (erf, a
  Bessel function): does the search find the terms of the series with the right
  coefficients, and does the order of the truncation follow the noise level?
- More inputs. The number of candidates per step grows with the square of the
  number of inputs of a hole.
- Noise levels other than 1%, and inputs that are not all positive.
- A comparison with an established symbolic regression code on the same data.
