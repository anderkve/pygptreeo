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
- `run_held_out.py`: the search or PySR on targets that PySR provides.
  `make_empirical_bench.py` writes the data of one set, `empirical_bench.json`.
- `run_baseline.py`: the search on every set of targets, with the score of the
  right formula beside each result.
- `truth.py`: the right formula of each target as a model that the search can
  write.
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
python truth.py                              # checks the formulas of truth.py
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
paper, but 60 s on 4 threads in place of 50 minutes on 8 cores. "Front" counts
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

## Limitations, issues and ideas

See [NOTES.md](NOTES.md).
