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
- `run_series.py`: the search on targets with a series expansion and no closed
  form.
- `compare_pysr.py`: [PySR](https://github.com/MilesCranmer/PySR) on the same
  data (needs `pip install pysr`).
- `make_plots.py`: figures of the results, drawn into `plots/` from the logs
  of the runs in `results/` (needs matplotlib).

```bash
python run_benchmark.py --seeds 3            # all targets
python run_benchmark.py --only I.6.2 --verbose
python run_growth.py --target toy
python run_series.py
python compare_pysr.py --seconds 60
python make_plots.py
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
the data. A polynomial `1 + c1·v + … + ck·v^k` with `k` from 2 to 4 can be
added (`'poly'` in `CFG['library']`); it is not in the library by default.

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

The last column is PySR on the same data (`compare_pysr.py`): default search
settings, the operators `+ − * /`, `sin`, `cos`, `exp`, `log`, `sqrt` and
`square`, 60 s on 4 threads per run, and the equation it selects by default.

| Target | Right | Hole left | Other | PySR right |
|---|---|---|---|---|
| `x2·sin(2·x1) + 0.5·x3²` | 3 | | | 3 |
| `x1·x2·x3` | 3 | | | 3 |
| `x1·x2/(4π·x3²)` | 3 | | | 3 |
| `x1·x2·x3/x4` | 3 | | | 2 |
| `0.5·x1·(x2² + x3² + x4²)` | 3 | | | 0 |
| `0.25·x1·(x2² + x3²)·x4²` | 3 | | | 1 |
| `exp(−(x1/x2)²/2)/(√(2π)·x2)` | 3 | | | 0 |
| `x1·(x2 + x3·x4·sin(x5))` | 3 | | | 3 |
| `x1 + x2 + 2·√(x1·x2)·cos(x3)` | 3 | | | 0 |
| `x1·exp(−x2·x3/x4)` | 3 | | | 3 |
| `x1·x2·log(x3/x4)` | 3 | | | 3 |
| `x1·(cos(x2·x3) + x4·cos²(x2·x3))` | 3 | | | 3 |
| `x1/√(1 − x2²/x3²)` | 3 | | | 0 |
| `(x1·x3 + x2·x4)/(x1 + x2)` | 1 | 1 | 1 | 2 |
| `1 + x1·x2/(1 − x1·x2/3)` | 1 | | 2 | 0 |
| `arcsin(x1/(x2·x3))` | | 1 | 2 | 0 |
| `√((x2 − x1)² + (x4 − x3)²)` | | 2 | 1 | 1 |

The last two targets cannot be written with the rules and the library, so the
intended result is a hole: `H(x1/(x2·x3))` and `H(x1 − x2, x3 − x4)`. PySR can
write the last one, and neither code has arcsin.

- The runs under "Other" return a formula that agrees with the target to
  within the noise inside the training box and is wrong outside it, e.g.
  `1.25·exp(0.78·x1·x2) − 0.25` for the second of them. For the last target
  it is a sum of products of exponentials and rational functions, with an
  error of twice the noise.
- For arcsin one run returns `0.997·u + 0.203·u³` with `u = x1/(x2·x3)`, close
  to the first two terms of its series, `u + u³/6`.
- With squared-exponential holes (`--kernel rbf`) the first 13 targets give
  38 of 39, and the last target never gives the plain hole.
- Three of the PySR results counted as right contain a small spurious term.
- With 300 s per run on the four of the first 13 targets where PySR has 0, it
  has 1, 1, 1 and 0 of 3 right, in the order of the table, and 1, 2, 2 and 0
  with the best equation of its Pareto front in place of the one it selects.
  The one for `x1 + x2 + 2·√(x1·x2)·cos(x3)` is not that formula, but close
  enough to it to count as right.
- The rules and the library were developed on these targets, and PySR's
  settings were not adapted to them.

For the first target with fewer points (`run_growth.py`), right out of 3:

| Points | 15 | 25 | 40 |
|---|---|---|---|
| 1% noise | 3 | 3 | 3 |
| 1% noise, PySR | 0 | 0 | 0 |
| 20% noise | 0 | 2 | 3 |
| 20% noise, PySR | 0 | 1 | 1 |

At 10 points with 1% noise the results are wrong (spurious factors, or a hole
with a wrong factor). With 20% noise the three results at 15 points and one at
25 points have a hole, and one counted as right at 40 points has `x2^1.5` for
`x2`. PySR's results at 1% noise have the right main terms and small spurious
ones.

## Targets without a closed form

`erf(x1)`, the Bessel function `J0(x1)` and the period of a pendulum,
`4·√(x1/x2)·K(sin(x3/2))`, whose series is
`2π·√(x1/x2)·(1 + x3²/16 + 11·x3⁴/3072 + …)`. 100 points, 2 seeds, noise of
10%, 1% and 0.1%, integer powers up to 9 (`run_series.py`).

- **Default library.** No series. The result is a short formula that agrees
  with the target to within the noise, e.g.
  `6.28·(x1/x2)^0.5·(1 − 0.0618·x3²)^−1` for the pendulum at 1% and 0.1% noise
  (its expansion is `1 + 0.0618·x3² + 0.0038·x3⁴`, the target's
  `1 + 0.0625·x3² + 0.0036·x3⁴`), or a hole.
- **Powers only.** A hole in every run but one, with the right structure
  around it: `(x1/x2)^0.5·H(x3)` for the pendulum. One term of a series beside
  the hole does not raise the score, so the series is not built term by term.
- **Powers and the polynomial.** A polynomial in every run but one, of a
  degree that rises as the noise falls. At 0.1% noise it is the series in 5
  of 6 runs, with the number of terms a least-squares fit needs for that
  noise or one more:
  `1 − 0.25·x1² + 0.0152·x1⁴ − 0.00035·x1⁶` for `J0`
  (series `1 − 0.25·x1² + 0.0156·x1⁴ − 0.00043·x1⁶`) and
  `6.28·(x1/x2)^0.5·(1 + 0.063·x3² + 0.0031·x3⁴ + 0.0004·x3⁶)` for the
  pendulum. At 1% and 10% noise it is a polynomial of degree 2 to 4 with even
  and odd powers, which fits within the noise and is not the series. The
  coefficients are those of a fit, so the last ones differ from the series'.
- **All of them.** The polynomial for `J0`, either form for `erf`, and for the
  pendulum the rational form above at 1% and 0.1% noise.

PySR on the same data also returns short formulas that agree to within the
noise, e.g. `1.96·(cos(0.857·x1) + 0.471)` for `J0` at 1% noise (in units of
the standard deviation of the target), and no series.

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

- The benchmark with the polynomial in the library.
- More inputs. The number of candidates per step grows with the square of the
  number of inputs of a hole.
- Noise levels other than 1%, and inputs that are not all positive.
- Targets that were not used to develop the rules, and PySR with tuned
  settings.
