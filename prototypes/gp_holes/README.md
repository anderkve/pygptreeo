# Symbolic regression with GP holes (prototype)

A model is an expression in which each part that is not yet known is a *hole*:
an unknown function with a Gaussian process prior. If every hole enters as
`a(x)·H(u(x))` with `a` and `u` symbolic, the whole model is one GP and its
evidence is closed-form. The search starts from a single hole `H(x1, …, xd)`,
i.e. plain GP regression, and applies rewrite rules that replace a hole with
more specific structure, keeping the models with the highest score. A hole that
remains at the end marks a part that the data or the rules could not resolve.

The search itself does not use the `pygptreeo` package. It needs numpy, scipy
and threadpoolctl. `bigdata.py` and `run_scaling.py`, for larger data sets,
need pygptreeo and scikit-learn as well.

## Files

- `gpholes.py`: the model, its evidence, the rewrite rules and the search.
- `targets.py`: target functions, mostly from the Feynman database.
- `run_benchmark.py`: the search on each target.
- `run_growth.py`: the result for one target as the number of points grows.
- `run_series.py`: the search on targets with a series expansion and no closed
  form.
- `compare_pysr.py`: [PySR](https://github.com/MilesCranmer/PySR) on the same
  data (needs `pip install pysr`).
- `run_held_out.py`: the search or PySR on targets that PySR provides.
  `make_empirical_bench.py` writes the data of one set, `empirical_bench.json`.
- `run_baseline.py`: the search on every set of targets, with the score of the
  right formula beside each result. With `--check`, on 41 of the runs only.
- `truth.py`: the right formula of each target as a model that the search can
  write.
- `run_ode.py`: a differential equation for a hole of one input.
- `local.py`: a model whose holes are local GPs, for a search on more points.
- `bigdata.py`: a tree of local GPs before the search and trees after it, for
  many more points. `run_scaling.py` runs these on the benchmark.
- `make_plots.py`: figures of the results, drawn into `plots/` from the logs
  of the runs in `results/` (needs matplotlib).

```bash
python run_benchmark.py --seeds 3            # all targets
python run_benchmark.py --only I.6.2 --verbose
python run_growth.py --target toy
python run_series.py
python compare_pysr.py --seconds 60
python run_held_out.py --set empirical --code gp_holes
python run_baseline.py NAME                  # every set, into results/baseline_NAME.*
python run_baseline.py NAME --check          # the 41 check runs, half an hour
python run_baseline.py NAME --check --symmetry first    # ... with the rule "symmetry"
python truth.py                              # checks the formulas of truth.py
python run_ode.py                            # equations for erf, J0 and others
python run_benchmark.py --n 1000 --local     # holes of local GPs
python run_scaling.py NAME                   # 10,000 points per target
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
| change input | the same with `v` in place of only one of the two, or `log(uj)` in place of a positive `uj` |
| drop input, drop factor, drop term | the model without it |
| add hole | for a model with no hole left: the model and `H(xj)`, or a hole of all inputs; kept only if it scores above the model |
| look inside | for a hole: the sum of functions of its inputs that is closest to the values of the hole at the data, as terms or as one factor, and a power or the logarithm of such a sum |
| symmetry | for a hole: what follows from a linear relation between its values and its derivatives at the data (below); off by default |
| outer function | `g(m)` for a model `m` of holes only, with `g(m)` one of `exp(m)`, `log(m) + c`, `c − log(m)`, `1/m`, `√m` |

The library is `u^p`, `log(u)`, `exp(c·v)`, `(1 + c·v)^p` and `sin(a + b·u)`,
with `v` one of `u`, `u²`, `1/u`. For an input that changes sign, `u^p` and
`log(u)` are replaced by the powers 1, 2, 3, −1 and −2 and by `log|u|`. A sine must show at
least half a period over the data. A polynomial `1 + c1·v + … + ck·v^k` with
`k` from 2 to 4 can be added (`'poly'` in `CFG['library']`); it is not in the
library by default.

## Score

The log evidence at the fitted parameters, minus `½ log n` per parameter (BIC),
minus 3 per symbolic function, minus 10 per exponent that is not a simple
number (`NICE` in `gpholes.py`). Fitted exponents, coefficients of merged
inputs and phases are rounded to simple values where that raises the score.

Every model has noise of the same size at every point and noise in
proportion to the target, of at most 30%, whose variance at a point is in
proportion to the square of the observed value there. With a function `g` around the expression,
the inner expression is fitted to `g⁻¹(y)`, and the logarithm of the Jacobian
of this transform is added to the score, as in a warped GP. The noise of the
same size is then inside the function; outside it there is noise of the same
size and noise in proportion to the target, carried through the transform to
first order.

Each step refines those of the 4 best models that have not been refined yet,
and the best model of an outer function: once for each function, and in up to
6 steps for the 3 functions with the best scores. A function whose score is
more than 10 per point below the best is left out. Every candidate gets a
short fit of 20 iterations, from the parameters of the model it was made
from, with all variances scaled by the factor that maximises the evidence.
The 12 best structures, and the 3 best of each outer function, get a full
fit. The search returns its 5 best models.

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

## Targets that PySR provides

These were not used to develop the search, and it was run on them as it was
(`run_held_out.py`). The results are judged by the form of the formula.

**The examples of PySR's documentation.** Normal inputs, so also negative ones,
and 5 inputs of which 1 to 3 matter. 3 seeds, PySR with 60 s on 4 threads.

| Target | Points | Right | PySR right |
|---|---|---|---|
| `2·cos(x4) + x1² − 2` | 100 | 3 | 3 |
| `1/x1` | 100 | 0 | 3 |
| `x4² − x20² + 1.5`, 30 inputs | 300 | 3 | 3 |
| `exp(x1) + x2 + x3`, noise 0.1 | 100 | 3 | 0 |

- For an input that changes sign the library has no `1/u`, so `1/x1` cannot be
  written; the search returns products of holes and other factors.
- The third target has 30 inputs, which is out of reach of the search. As in
  the documentation, PySR's selection of inputs first keeps 5 of them, for
  both codes. A run then takes 16 minutes.
- For the last target PySR's three results have further terms, with 3 to 5
  times the error of the search inside the range of the data. Several of its
  other results have terms with coefficients below 1e-6.

**EmpiricalBench of the PySR paper** (arXiv:2305.01582): nine empirical laws,
with the original data for four of them and generated data with 1 to 20% noise
for the others (`make_empirical_bench.py`). Six targets are logarithms, and the
inputs are in their units, over up to eight decades. The data are fixed, so the
3 runs differ in the seed of the search only. PySR has the settings of the
paper, but 60 s on 4 threads in place of up to 1 hour on 8 cores. "Front" counts
the runs with the law somewhere on PySR's Pareto front, which is how the paper
counts; its own numbers are in the last column.

| Law | Points | Right | PySR right | PySR front | PySR in its paper |
|---|---|---|---|---|---|
| Hubble, `v = H0·D` | 32 | 3 | 0 | 3 | 5 of 5 |
| Kepler, `P = k·a^1.5` | 6 | 0 | 0 | 3 | 5 of 5 |
| Newton, `log(G·m1·m2/r²)` | 30 | 0 | 3 | 3 | 5 of 5 |
| Planck | 100 | 0 | 0 | 0 | 0 of 5 |
| Leavitt, `M = α·log10(P) + δ` | 26 | 0 | 0 | 0 | 5 of 5 |
| Schechter, `log(n)` | 27 | 3 | 1 | 1 | 5 of 5 |
| Bode, `log(0.4 + 0.3·2^n)` | 8 | 0 | 0 | 3 | 5 of 5 |
| Ideal gas, `log(n·R·T/V)` | 30 | 0 | 3 | 3 | 5 of 5 |
| Rydberg | 50 | 0 | 0 | 0 | 0 of 5 |

- The search returns `473·x1` for Hubble and
  `−1.21·log(x1) − 4.03e-9·x1 − 2.13` for Schechter. PySR has both laws on its
  front and mostly selects a longer formula.
- Kepler: `365·x1^1.5` with a further term that takes up the residual of the
  last of the 6 points.
- Newton and ideal gas: one part right, `−2.02·log(x1)` and `−0.97·log(x3)`,
  and a hole or a product of logarithms for the rest. The holes take the
  inputs as they are, which does not suit inputs spread over many decades.
- Leavitt: `3.89·exp(−0.043·x1) + 11.2`. The same form is on PySR's front.
- Bode: a hole. Bode, Planck and Rydberg need the logarithm of a sum, which
  the library does not have. For Planck the search finds the term
  `−4.8e-11·x1/x2`, which is `h·ν/(k·T)`, and for Rydberg `2·log(x1)` and the
  input `x1/x2`.

## After the outer function and the logarithms of inputs

The results above were obtained before the function around the expression,
the logarithms of inputs and the further powers were added. With them:

**A held-out sample of SRSD equations.** 15 Feynman equations of the SRSD
benchmark (Matsubara et al. 2022, arXiv:2206.10540), drawn at random from those
with at most 5 inputs that had not been used (`make_srsd_sample.py`), with the
benchmark's ranges of the inputs: often many decades, and of both signs. 100
points, 1% relative noise, 2 seeds. A result is right if it has no hole and
its median relative error is below 1% for inputs from ranges half a decade
wider on each side. The sample and this rule were fixed before the search was
changed (`run_held_out.py --set srsd`). PySR had 60 s on 4 threads; "front"
counts the runs with a right equation anywhere on its Pareto front.

| Equation | Before | After | PySR | PySR front |
|---|---|---|---|---|
| i.6.20b `exp(−((θ − θ1)/σ)²/2)/(√(2π)·σ)` | 0 | 0 | 2 | 2 |
| i.12.4 `q1/(4π·ε0·r²)` | 0 | 0 | 2 | 2 |
| i.14.4 `k·x²/2` | 0 | 1 | 1 | 2 |
| i.27.6 `1/(1/d1 + n/d2)` | 0 | 2 | 2 | 2 |
| i.34.27 `ħ·ω` | 1 | 2 | 2 | 2 |
| i.39.11 `p·V/(γ − 1)` | 0 | 0 | 1 | 1 |
| ii.11.20 `n·p²·E/(3·k·T)` | 0 | 0 | 0 | 0 |
| ii.13.34 `ρ·v/√(1 − v²/c²)` | 0 | 1 | 0 | 1 |
| ii.27.18 `ε0·E²` | 0 | 2 | 2 | 2 |
| ii.34.29b `g·μB·B·J/ħ` | 0 | 0 | 0 | 0 |
| ii.35.21 `n·μ·tanh(μ·B/(k·T))` | 0 | 2 | 0 | 0 |
| iii.10.19 `μ·√(Bx² + By² + Bz²)` | 0 | 0 | 0 | 0 |
| iii.15.12 `2·U·(1 − cos(k·d))` | 0 | 0 | 0 | 0 |
| bonus.8 `E/(1 + E·(1 − cos θ)/(m·c²))` | 0 | 2 | 0 | 2 |
| bonus.18 `3·(c²·k/r² + H²)/(8π·G)` | 0 | 0 | 0 | 1 |
| Total of 30 | 1 | 12 | 12 | 17 |

- Before the change the search stopped with an error on 8 of the equations:
  the bounds of the frequency of a sine crossed for inputs of small range.
- The targets of iii.15.12 are dominated by rounding errors (`1 − cos(k·d)`
  with `k·d` about 1e-8, 10% in the range of the data), so no result can be
  right.
- In ii.35.21 and bonus.8 a part of the formula has no visible effect in the
  ranges of the benchmark (`tanh(z)` with `z` below 0.01, and a correction of
  1e-10). The search returns the formula without it, `n·μ²·B/T` times a
  constant and `y = E`.
- The rule also counts approximations as right. By eye, of the 12 right
  results of the search 7 are the formula, at most with a constant added (in
  i.14.4 that constant is not small for the largest targets), 4 the formula
  without the invisible part, and 1 an approximation (ii.13.34). PySR's 12 are
  the formula with further terms of up to a few percent (9), or
  approximations (3). Its fronts were not logged for this sample.
- In 10 of the other 18 runs of the search the right formula, or its main
  terms, stands beside a hole. 8 of them are for targets that change sign,
  where the noise of the model does not grow with the target (NOTES.md,
  "Relative noise").
- The code before the last fix, a scale for `(1 + c·u)^p` (below), had 14 of
  30 right; the two results it had in addition were approximations.

**The original benchmark again**, seed 0 only: 15 of the 17 targets right,
against 14 before. I.8.14 is now right, as
`√((x3 − x4)² + 0.998·(x1 − x2)² + 0.0007)`, and so is I.18.4; II.11.28 is
missed, although the right formula scores higher (277 against 273). I.13.4
comes out as `√(0.25·(x1·x2² + x1·x3² + x1·x4²)² + 1.09)`: right within the
noise, but the constant does not belong there. A first version of the change
lost I.10.7 and II.11.28: the coefficient in `(1 + c·u)^p` was in units of the
median of `|u|`, so that `1 + c·u` could turn negative within the data. It is
in units of the largest `|u|` again.

**EmpiricalBench after the change.** This is no longer a held-out test: the
change was made while looking at it. 3 seeds, the same result each time.
Right: Hubble, Schechter, Bode, as `log(0.0154·exp(0.691·n) + 0.0208) + 2.96`
(that is `log(0.4 + 0.3·2^n)`), and Rydberg, as
`−16.2 − log(1.01·(1/n1² − 1/n2²))`. The ideal gas law has a small constant
inside the logarithm; Newton and Kepler have a further term at the level of
the noise; Leavitt is still an exponential and Planck still a hole. Before
the change: 2 of 9.

## Relative noise, log|x|, and a second held-out sample

The first SRSD sample showed two problems: for targets that change sign, a
hole stayed beside the right formula, since the noise did not grow with the
target, and `log|x|` was missing. The search got a noise term in proportion
to the target, of at most 30%, and `log|x|` for inputs that change sign. A
second sample of 15 SRSD equations, none used before, was drawn and fixed
before the change (`make_srsd_sample.py --sample 2`), with the same data and
rule (`run_held_out.py --set srsd2`).

| Equation | Before | After | PySR | PySR front |
|---|---|---|---|---|
| i.12.5 `q·E` | 0 | 2 | 2 | 2 |
| i.25.13 `q/C` | 0 | 2 | 2 | 2 |
| i.34.14 `(1 + v/c)·ω0/√(1 − v²/c²)`, see below | 0 | 2 | 2 | 2 |
| i.43.16 `μ·q·V/d` | 0 | 2 | 0 | 0 |
| i.47.23 `√(γ·p/ρ)` | 2 | 2 | 2 | 2 |
| ii.8.7 `3·q²/(20π·ε0·d)` | 1 | 2 | 0 | 2 |
| ii.8.31 `ε0·E²/2` | 2 | 2 | 0 | 2 |
| ii.11.3 `q·E/(m·(ω0² − ω²))` | 0 | 0 | 0 | 0 |
| ii.38.3 `Y·A·x/d` | 0 | 2 | 0 | 2 |
| iii.13.18 `2·E·d²·k/ħ` | 0 | 2 | 0 | 0 |
| bonus.1 `(Z1·Z2·α·ħ·c/(4·E·sin²(θ/2)))²` | 1 | 0 | 0 | 0 |
| bonus.7 `√(8π·G·ρ/3 − α·c²/d²)` | 0 | 2 | 0 | 0 |
| bonus.14 `E·cos θ·(−r + d³·(α − 1)/(r²·(α + 2)))` | 0 | 0 | 0 | 0 |
| bonus.19 `−(c⁴·k/r² + H²·c²·(1 − 2α))/(8π·G)` | 0 | 2 | 0 | 1 |
| bonus.20 `(ω0/ω)²·(ω0/ω + ω/ω0 − sin²β)` times a constant | 0 | 0 | 0 | 0 |
| Total of 30 | 6 | 22 | 8 | 17 |

- "Before" is the search of the previous section. Its misses were mostly
  the right formula with a hole beside it (i.12.5, i.25.13, i.43.16, ii.8.7).
- By eye, 20 of the 22 right results of the search are the formula, at most
  with a small constant added, e.g. `exp(2·log|q| − log(d) + 22.4)` for ii.8.7
  and `√(−9e16·α/d² + 5.6e-10·ρ)` for bonus.7. The other 2 are the leading
  terms of i.34.14 (next point). Of PySR's 8, 2 are the formula (i.47.23), 4
  the formula with further small terms, and 2 approximations.
- The benchmark's code computes i.34.14 as `(1 + v/ω0)·ω0/√(1 − v²/ω0²)`,
  with ω0 in place of c. Both codes saw that function; the search returned its
  leading terms, `ω0 + v`.
- On ii.11.3, which has a pole within the data (ω0 = ω), one run returned
  only a constant.

**The first sample again**, on the five equations where a hole had stood
beside the formula (i.12.4, i.14.4, ii.11.20, ii.34.29b, bonus.18): right in 9
of 10 runs, against 1 of 10 before. This sample was used to find the change.

**The original benchmark again**, seed 0: 14 of the 17 targets right, against
15 before. I.18.4 and II.11.28 are missed; both have turned out right in some
runs of nearly the same code and wrong in others.

## A faster search

Three changes. They were checked on 28 runs from the four sets above (11
targets of the benchmark, 12 equations of SRSD and 4 empirical laws, mostly
seed 0), and on 13 runs of harder targets with other seeds.

- **The gradient of the evidence** needs no second kernel matrix for a
  parameter of a symbolic function, and a fit computes once what does not
  change. An evaluation takes half the time, and the results are the same.
- **Outer functions.** Each used to have its best model refined in six steps.
  One that is not among the 3 best after the first of them is now left alone.
- **The start of the short fit.** A candidate used to start with the noise
  level of the model it was made from, far too low for it, and to spend 10 to
  20 iterations getting back from a score of minus several thousand. All its
  variances are now scaled first, by the factor that maximises the evidence,
  and the short fit has 20 iterations in place of 25.

| Search | Right of 24, first set | Seconds, first set | Right of 13, second set |
|---|---|---|---|
| With the faster gradient | 19 | 1025 | 5 |
| Outer functions limited | 19 | 823 | |
| ... and the scaled start, 25 iterations | 21 | 1051 | 7 |
| ... 20 iterations | 21 | 716 | 7 |
| ... 15 iterations | 21 | 613 | 7 |
| ... 10 iterations | 19 | 455 | |

- With the outer functions limited, all 28 results are the same as before.
- With the scaled start I.18.4 is found in all three seeds, as
  `x3 − (x3 − x4)/(1 + x1/x2)`, against none before, and II.11.28 in one of
  three. Of the 41 results, 8 have a higher score than before and one a score
  lower by 0.6, for another way of writing the same law.
- With 15 iterations one result of the second set is a less simple form with
  a lower score. With 10, one search of the first set stops at the first hole.
- The 4 empirical laws of the first set are found before and after.

The score of the right formula (`truth.py`) is beside each result. Of the 9
runs that are not right, 3 have a result with a score below that of the right
formula, so the search missed it (SRSD i.6.20b twice, by more than 400, and
i.39.11). In the other 6 the result scores as high as the right formula, or
as the model with a hole that the rules should arrive at, or higher.

## A new hole for a model that has none left

Once every hole of a model had been replaced, the model could only lose
terms, so a search that had filled its last hole with the wrong function
could not go on. The rule "add hole" proposes such a model with a new hole
beside its terms, in every search, and the search goes on from there if the
score is higher. On the two sets of the last section:

- SRSD i.6.20b is found in both seeds, as
  `exp(−0.5·(x1 − x2)²/x3² − log(x3) − 0.92)` with the score of the right
  formula, 920 and 1085, against 486 and 656 before. The search used to end
  at `0.05/x3²` in place of `−log(x3)`; a hole in `x3` beside that model
  scores 848, and the logarithm is the next step.
- The other 39 results are the same as before, with 2% more fits.
- With a new hole proposed for every model, also for those that still have
  one, the 41 results are the same again, with 10% more fits on the first set.

Right are now 23 of 24 on the first set and 7 of 13 on the second. Of the 7
runs that are not right, one has a result with a score below that of the
right formula (SRSD i.39.11, 557 against 567).

## A look inside a hole

The fit of a model gives the values of each of its holes at the data. The
rule "look inside" takes those of a hole of up to four inputs and finds, by
least squares, the sum of simple functions of the inputs that is closest to
them: one function for each input, a power or the logarithm, and a constant
if it is not small beside the sum. The hole is replaced by the terms of the
sum, or by the sum as one factor. For a hole of two to four inputs whose
values have one sign, the same is done with powers of the values, and the
hole is replaced by a power of the sum, `(sum)^p` or `(1 + c·sum)^p`.

The least squares only proposes: the proposals have no hole in place of the
one looked into, and are fitted and scored as every other candidate. A power
of a sum of several functions could not be reached before, since on the way
to it a hole would stand inside a function. On the two sets:

- SRSD iii.10.19 is found as `0.998·x1·(x2² + x3² + x4²)^0.5`, with a score
  of 428 against 268, in 2,200 fits in place of 5,100. The search comes to
  `x1·H(x2, x3, x4)` in its first step and looks into that hole in its second.
- SRSD i.39.11 is found as `1/(−0.997·(1 − x1)/(x2·x3))`, 567 against 557:
  the hole of `1/(H(x1)/(x2·x3))` is a line in `x1`.
- SRSD i.27.6 is now `1/(1/x1 + x2/x3)` without the constant of 0.1 that it
  had beside the two terms, and I.18.4 has `x4 − x3` as one factor.
- SRSD ii.13.34 counts as right with `x1·x2·(1 + 8.5e-26·x2³)`, in place of
  `x1·x2·H(x2)`. This is an approximation of the factor `1/√(1 − x2²/c²)`
  that fits within the noise, not the formula.
- The other results are the same, and the searches need 16% fewer fits.

Right are now 24 of 24 on the first set and 9 of 13 on the second, and no
result has a score more than 0.3 below that of the right formula. The four that are not
right are the two of II.11.28 with another function that fits within the
noise, I.30.5 (no arcsin in the library) and SRSD bonus.20.

**More functions and inputs.** The first version, above, had powers and the
logarithm in its sums and looked into holes of at most four inputs. Now the
sums can also have a sine or cosine, at one of seven frequencies, and an
exponential, at one of four rates; the fit of a candidate adjusts the
frequency or rate. A hole can have any number of inputs: all combinations of
functions are tried if they are at most 2,000, and with more the function of
one input at a time is chosen with the others kept, in turn until none
changes. A hole of one input gets a power of a sum too, where the sum is a
constant and a sine, exponential or logarithm.

- On four made-up targets with 1% noise, the search finds
  `x2/(1 + 0.5·cos(x1))`, `x3·√(x1² + 2·exp(−x2))` and, with six inputs,
  `x1·√(x2² + … + x6²)`, each in its second step. It does not find
  `√(x1² + x2² − 2·x1·x2·cos(x3))`, whose sum has a product of functions of
  three inputs.
- On the two sets, right are 24 of 24 and 9 of 13 as before, with 1% more
  fits on the first and 2% fewer on the second. I.50.26 is now
  `0.5·x1·x4·(1 + cos(2·x2·x3)) + x1·cos(x2·x3)`, 10 higher in score.
- For Newton's law of EmpiricalBench, with 30 points, this version went on
  from the law, which it finds in its first step with a score of 53.2, to
  `√(H(log(r/√(m1·m2))))` with 54.9. Of the nine laws it is the only one
  that changed. With the logarithm of a sum, below, the search returns the
  law again.

**The logarithm of a sum.** With `exp(h/v)` of the values `h` of a hole, for
six sizes of `v`, the same least squares gives the sum of `w·log(sum)`, and
the hole is replaced by that term.

- On made-up targets the search finds `x3·log(x1² + x2²)`, and
  `x2 + 2·log(1 + 0.6·cos(x1))` as `log(exp(x2)·(1 + 0.6·cos(x1))²)` and
  constants.
- SRSD bonus.1 is still not found. Its `log(sin(x4/2))` has the zeros of the
  sine at both ends of the range of `x4`, so a sine with a frequency that is
  a little off is negative there and has no logarithm, and the rule leaves
  such a sum out.

With all of the above, the 41 check runs (`python run_baseline.py NAME
--check`, in `results/baseline_check.txt`) have 33 right of the 37 with a
rule, and the 4 empirical laws, in 940 s of search. Not right are II.11.28
in two of its three seeds, I.30.5 and SRSD bonus.20, as before.

## The symmetries of a hole

Most of the rules of the table are entries of a list: a candidate for each
input, each pair of inputs and each function of the library, with a fit for
each. They have one thing in common. Each says that the hole obeys a linear
relation between its values and its derivatives:

| Rule | Relation |
|---|---|
| drop input | `∂H/∂uj = 0` |
| merge to `uj + c·uk` | `c·∂H/∂uj = ∂H/∂uk` |
| merge to `uj·uk^p` | `p·uj·∂H/∂uj = uk·∂H/∂uk` |
| factor `uj^p` | `uj·∂H/∂uj = p·H` |
| factor `exp(c·uj)` | `∂H/∂uj = c·H` |
| factor `sin(a + b·uj)` | `∂²H/∂uj² = −b²·H` |
| linear in `uj^p` | `uj·∂²H/∂uj² = (p − 1)·∂H/∂uj` |
| no interaction | `∂²H/∂uj∂uk = 0` |

The rule "symmetry" (`CFG['symmetry']`) finds such relations from the fit of
a model, in place of trying each entry of the list. The fit gives the
posterior mean of each hole at the data, and its first and second derivatives
with respect to the inputs of the hole in closed form. The rule looks for two
kinds of relation, in a coordinate `s` of each input: `u`, `log(u)`, `u²` or
`1/u`.

- **Of one to three inputs**, all in the same coordinate:
  `Σ cj·∂H/∂sj = γ·H`, `= δ` or `= 0`. The hole is then `exp(γ·sj)·G`,
  `δ·sj + G` or `G`, where `G` is a hole of one input less: of the invariants
  `sk − (ck/cj)·sj` and the inputs that are not in the relation. In `log(u)`
  this is the factor `uj^p` and the merge to `uk·uj^p`, in `u` the factor
  `exp(c·uj)` and the merge to `uk + c·uj`, in `u²` e.g. a hole of
  `x2² + c·x3²`. A relation of several inputs covers what took several steps:
  `H + x1·∂H/∂x1 + x2·∂H/∂x2 = 0` gives `H(x1/x2)/x2` for I.6.2 at once.
- **Of one input:** a factor `(1 + c·s)^p`, the factors `u^p·exp(c·s)`, a
  factor `sin(a + b·u)`, and a hole that is linear in `s`, in `exp(g·s)`, in
  `(1 + c·s)^p` or in `sin(a + b·u)`.

The first kind is the affine symmetry of NestyNet III (Ibata et al.,
arXiv:2608.21051, section 6), which finds it for a neural network fitted to
thousands of points. Here the derivatives are those of the GP posterior, and
a relation only proposes a candidate.

A relation is the combination of two to four columns (the values of the
hole, its derivatives in a coordinate, a constant) that is closest to zero;
its residual is the smallest singular value of the columns, scaled to unit
length. One of the first kind is kept if its residual is below 0.3 and it
does not hold as well, within a factor of 1.5, with fewer inputs or without
its right-hand side. The 6 best give a candidate for each of their inputs as
the one that is taken out, and there is one candidate with all the relations
that share no input. Of the second kind, each has a candidate for its best
coordinate of an input if the residual there is below 0.3. The coefficients
of a relation are the starting values of the candidate, which is fitted and
scored as every other candidate. The thresholds were set once and not tuned.

`CFG['symmetry']` is `'off'` (the default), `'add'` (the rule beside the
others), `'replace'` (the rule in place of the merges, of the factors but the
logarithm, and of the functions that a hole is linear in) or `'first'` (as
`'replace'`, but once the rule has nothing new for the best models, or the
search has stopped improving, the 4 best models get the rules that were left
out).

**The 41 check runs** (`run_baseline.py NAME --check --symmetry MODE`, in
`results/baseline_symmetry_MODE.txt`):

| `CFG['symmetry']` | Right of 37 | Fits |
|---|---|---|
| `'off'` | 33 | 117,505 |
| `'add'` | 33 | 102,846 |
| `'replace'` | 29 | 25,804 |
| `'first'` | 30 | 34,413 |

- With `'replace'` the searches need 22% of the fits and with `'first'` 29%.
  The 4 empirical laws come out the same in every mode.
- In all three modes I.8.14 is `((x2 − x1)² + (x4 − x3)²)^0.5`, without the
  constant under the root and with a score 6 and 7 higher, and SRSD i.27.6,
  seed 1, is `1/(1/(x3/x2) + 1/x1)`, 6 higher. With `'first'`, SRSD ii.13.34
  is `x1·x2·(1 − 1.22e-17·x2²)^−0.5`, the formula. Without the rule and with
  `'add'` it is an approximation that scores 1.6 higher, `x1·x2·(1 +
  8.5e-26·x2³)`, and with `'replace'` another that scores 7 lower.
- The toy target comes out as `0.5·x2·((x3/x2^0.5)² + 2·sin(2·x1))`, from the
  relation `x2·∂H/∂x2 + 0.5·x3·∂H/∂x3 = H`. This is the target, and it scores
  2 above the plain form, which has one weight more.
- Lost with `'replace'` and `'first'`: II.11.28 in seed 0, which becomes an
  exponential (the other two seeds are not right in any mode), and SRSD
  bonus.7 in both seeds. Lost with `'replace'` only: SRSD i.6.20b in seed 0,
  where the search stops at the first hole. NOTES.md has the reasons.
- I.18.4 is right in every mode, but with the rule its scores are lower: by
  3 and 6 in two seeds, and by 9 in the third with `'replace'` and `'first'`.
  Two of these results score below the right formula, by 3 and 6. Without
  the rule no result is more than 0.3 below it.

**A third sample of SRSD equations** (`srsd_sample3.json`,
`run_baseline.py NAME --sets srsd3 --symmetry MODE`, in
`results/baseline_srsd3_MODE.txt`): 15 equations, none used before, drawn
after the rule was written and before any run on them, with 2 seeds and the
data and the rule for a right result of the first two samples. Nothing was
changed after these runs.

| `CFG['symmetry']` | Right of 30 | Fits |
|---|---|---|
| `'off'` | 20 | 70,681 |
| `'add'` | 20 | 70,976 |
| `'first'` | 19 | 25,567 |

- With `'first'` the searches need 36% of the fits. 20 of the 30 results
  have the same score as without the rule, 3 a higher one and 7 a lower one.
- The result that is lost is i.16.6, `(u + v)/(1 + u·v/c²)`, in seed 1: a
  hole remains. Without the rule it is `0.998·(x1 + x2)` and a small term
  with a sine, an approximation that counts as right.
- i.15.3x, `(x − u·t)/√(1 − u²/c²)`, is right in every mode. Without the rule
  it is the formula, `(x1 − x2·x3)·(1 − 1.03e-17·x2²)^−0.5`; with `'first'`
  it is the first terms of its series, `x1 − x2·x3 − 5.6e-18·x2³·x3`, with a
  score 11 and 16 lower.
- bonus.4 in seed 0 is `1.42·x1^−0.5·(x2 − x3)^0.5` with `'first'`, 17 higher
  than the result without the rule, which has two further terms under the
  root. (The term of the formula in `L` and `r` is not visible in the data.)
- The 10 results that are not right without the rule are not right with it
  either. With `'first'` 4 of them score lower, by 12 to 132, and one higher.
- With `'add'` 28 of the 30 results are the same, with as many fits: on this
  sample, with inputs over decades, the candidates of the rule rarely win
  against the listed ones. On the check runs `'add'` saves 12% of the fits.
- The sample has no entries in `truth.py`, and the results were not judged
  by eye. As before the rule for a right result counts approximations: in
  every mode i.48.2 in seed 0 is `x1·(1 + 8.4e-26·x2³)` times a constant.

## A differential equation for a hole of one input

For a hole of one input the same can be done with the columns
`u^a·d^kH/du^k`, with `a` and `k` from 0 to 2, and a constant
(`run_ode.py`). A relation between two or three of them is a differential
equation with polynomial coefficients, which most functions of the library
obey, and so do functions without a closed form: `H'' + 2·u·H' = 0` for `erf`
and `u·H'' + H' + u·H = 0` for `J0`.

The solutions of an equation are a model without a hole: a term for each
basis solution, with a weight, and one more for an equation with a constant.
They are computed by numerical integration from the median of the data. The
evidence of the model is that of a formula, and it predicts outside the data.
The coefficients of the equation are fitted by least squares of its solutions
to the data, and each is held at a simple value where BIC allows. The cost
of a symbolic function is counted for each column of the equation. The 3
equations of two columns and of three with the smallest residuals are
scored.

100 points, 2 seeds, a squared-exponential hole. "Right" counts the runs
where the equation of the target has the best score and scores above the
hole.

| Target | Its equation | 10% noise | 1% | 0.1% |
|---|---|---|---|---|
| `erf(u)` | `H'' + 2·u·H' = 0` | 0 | 1 | 2 |
| `J0(u)` | `u·H'' + H' + u·H = 0` | 0 | 0 | 2 |
| `Ai(−u)` | `H'' + u·H = 0` | 2 | 2 | 2 |
| `exp(−u²/2)` | `H' + u·H = 0` | 2 | 2 | 2 |
| `sin(2·u)` | `H'' + 4·H = 0` | 2 | 2 | 2 |
| `1/(1 + u)` | `H' + u·H' + H = 0` | 0 | 2 | 2 |
| `arcsin(u)` | `H'' − u·H' − u²·H'' = 0` | 0 | 0 | 0 |

- Where the equation is found, it holds outside the data. For `erf` at 0.1%
  noise the rms error on a range 25% wider on each side is 8e-5 and 7e-5,
  against 2e-3 and 7e-3 for the hole; for the Airy function at 1% noise
  3e-3 and 2e-3 against 0.5 and 0.3.
- For `J0` at 10% and 1% noise no equation scores above the hole, so the hole
  stays. At 0.1% the result of one seed is `1.01·H' + u·H + u·H'' = 0`, with
  the coefficient not rounded.
- In 8 of the 42 runs another equation scores above the hole: for `erf` at
  10% noise and in one run at 1%, for `1/(1 + u)` in one run at 10%, and for
  `arcsin` at 1% and 0.1%. These fit within the noise and are as wrong
  outside the data as the hole. `arcsin` is never found: its equation is
  singular at `u = 1`, and the derivatives of the fit are poor near the end
  of the range.
- The factor `K(sin²(u/2))` of the period of a pendulum has an equation with
  trigonometric coefficients, which the columns cannot write. An equation
  scores above the hole in all 6 runs, e.g. `u·H'' − 7.62·H' + u·H = 0` at
  0.1% noise; its error outside the data is 0.03 and 0.001, against 0.05 and
  0.07 for the hole.

The script is not part of the search. A hole of one input that the library
cannot fill could be handed to it.

## Larger data sets

A fit of a model costs the cube of the number of points, and a search has
thousands of fits. Two ways to use more points were tried, both with trees of
local GPs as in pygptreeo.

**Holes of local GPs in the search** (`local.py`,
`search(X, y, model=LocalModel)`, `run_benchmark.py --local`). The points of
a model are divided into blocks of at most 100 points, the leaves of a tree
of cuts through the inputs of all its holes: a set of more points is cut at
the median of the input of largest range, and a point within 5% of that
range of the cut goes to either side (the overlap of pygptreeo). A hole has
no covariance between points of different blocks. The weights of the symbolic
terms and the constant stay those of the whole model, so the formula is one
for all points and only the holes are local. The covariance of the targets is
then a matrix of blocks plus one of low rank. The evidence and its gradient
are exact for this model, at the cost of a Cholesky factor per block, and
with one block for all points they are those of the search as it was.
Nothing else of the search changes. The blocks are in the evidence of the
prototype itself, and not GPTree objects: a GPTree fits the hyperparameters
of every leaf, which takes 4 to 9 s for 10,000 points, and a search has
1,000 to 7,000 fits. Here the blocks of a hole share its hyperparameters.

One evaluation of the evidence and its gradient, on one core:

| Points | One GP, hole of 3 inputs | Local holes | Local, no hole |
|---|---|---|---|
| 100 | 0.6 ms | 1.0 ms | 0.4 ms |
| 1,000 | 74 ms | 7 ms | 0.6 ms |
| 2,000 | 402 ms | | |
| 3,000 | | 27 ms | 1.0 ms |
| 10,000 | | 77 ms | 3 ms |

`run_scaling.py --local 1000` runs this search on 1,000 of the 10,000 points
of 7 of the cases below, and validates its best models as described there
(`results/scaling_n10000_local.txt`). A search takes 54 to 172 s, against
11 to 28 s on 100 points. 3 of the 7 results are right: the toy target at
10% and at 30% noise, and II.11.28 at 1%, which the search on 100 of the
points misses. The other 4 are not, and two of them, I.6.2 and I.37.4 at 10%
noise, are found by the search on 100 points. So the evidence scales as
intended, but in this first test the search does not gain from the points.

A second test has the same 300 points in blocks of 100 and in one block
(`--local 300 --nbar 100` and `--nbar 300`, in
`results/scaling_n300_blocks.txt` and `scaling_n300_one.txt`), for four
targets at 10% noise. With one block 3 results are right and with blocks 2:
I.6.2 ends at a hole, `H(log(x1)/x2, log(x2))`. The searches with blocks
take 30 to 54 s, those with one block 69 to 200 s. So the blocks themselves
can cost a result. Blocks lower the evidence of a hole, and by an amount
that depends on how the inputs of the hole are cut, so they also change the
order of models with different holes (NOTES.md). Another candidate, not
tested, is that the costs of a symbolic function and of a free exponent were
set for 100 points and weigh less against the evidence of 1,000.

**A tree before the search and trees after it** (`bigdata.py`). The search
stays on 100 points, and the other points come in at its two ends:

- *Distil.* A GPTree of pygptreeo is fitted to all the points, and the search
  runs on its predictions at 100 of them. The noise of the points is not
  given, so each leaf learns its own level. With leaves of 300 points the
  predictions are off by 5% of the spread of the target where the points have
  noise of 30%, and by 0.3% where they have 1%; with leaves of 100 points by
  9% and 0.4%. The tree takes 9 to 13 s for 8,000 points.
- *Validate.* The 10 best models of the search are fitted to all the points
  but a fifth that is held out, and the one with the best likelihood of the
  held-out points, less the costs of the search, is the result. A model
  without a hole is fitted by least squares. In a model with holes, each hole
  is a GPTree of the inputs of the hole (leaves of 100 points, overlap 5%),
  fitted to what the other terms leave of the target; at most 2 such models
  are fitted, and every model with a hole is also tried without it.

`run_scaling.py` compares three ways on 10,000 points of six targets of the
benchmark, with 2 seeds: the search on 100 of the points, with its best model
fitted to all ("subsample"); the same search with its 10 best models
validated ("validated"); and the search on the distilled points, validated
("distilled"). A result is right if it has no hole and its rms error outside
the box of the data is below 1% of the spread of the target, whatever the
noise. Right of 2, in `results/scaling_n10000.txt`:

| Target | Noise | Subsample | Validated | Distilled |
|---|---|---|---|---|
| toy | 10% | 2 | 2 | 2 |
| toy | 30% | 1 | 1 | 2 |
| I.6.2 | 10% | 2 | 2 | 2 |
| I.6.2 | 30% | 1 | 2 | 1 |
| I.37.4 | 10% | 2 | 2 | 2 |
| I.37.4 | 30% | 0 | 0 | 0 |
| I.10.7 | 10% | 0 | 0 | 1 |
| I.10.7 | 30% | 0 | 0 | 0 |
| I.18.4 | 10% | 1 | 1 | 1 |
| I.18.4 | 30% | 0 | 0 | 0 |
| II.11.28 | 1% | 0 | 0 | 2 |
| II.11.28 | 10% | 0 | 0 | 0 |
| II.11.28 | 30% | 0 | 0 | 0 |
| Right of 26 | | 9 | 10 | 13 |

- The four results that only the distilled search has right are formulas
  that the search on 100 of the points does not propose: II.11.28 at 1% noise
  in both seeds, where that search returns an exponential; I.10.7 at 10%,
  where it returns `x1·(1 − 0.5·x2/x3)^−0.5`; and the toy target at 30%.
- Two further results of the distilled search are the formula with an error
  just above the limit: `0.994·x1·(1 − (x2/x3)²)^−0.5 + 0.019` for I.10.7 at
  30% noise (1.2%), and
  `2·cos(x3)·x1^0.5·x2^0.5 + 1.01·(x1 + x2) − 0.06` for I.37.4 at 30%
  (1.1%). In the other seed the result for I.37.4 at 30% is off by 10%.
- Validation alone gains one result, I.6.2 at 30% noise, where the best model
  of the search has a hole and the formula is among the next.
- In one run, the distilled search for I.6.2 at 30% noise, no model could be
  validated. All 10 are fitted to the logarithm of the target, and 25 of the
  10,000 targets are not positive, while the predictions of the tree all
  are.
- A first version of the validation chose a model with a spurious hole
  beside the right formula in three of seven runs of the distilled search:
  the errors of the tree are smooth within a leaf, a hole fits them, and the
  10 best models were mostly such models. Each model with a hole is
  therefore also tried without it.
- The search takes 17 s on average, the validation 32 s more, and the
  distilled way 74 s in all.

## Limitations, issues and ideas

See [NOTES.md](NOTES.md).
