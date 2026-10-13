# pyGPTreeO on 5D tree-level tt̄H amplitudes, and an uncertainty-gated integration

`examples/tth_amplitude_benchmark.py`, `examples/gated_integration_demo.py`,
`examples/plot_tth_results.py`; data as in `examples/tth_amplitude/README.md`; one
CPU core per run, up to four runs in parallel; seed 1; 12–13 October 2026.

## 1. The benchmark

"Interpolating amplitudes" (Bresó-Pla, Heinrich, Magerya, Olsson, SciPost Phys. 19,
123 (2025), arXiv:2412.09534) compares interpolation methods for 2 → 3 amplitudes on
test functions of five phase-space variables, two of them tree level: f1,
q q̄ → t t̄ H, and f3, g g → t t̄ H. Their error is the relative L1 distance
`ε = Σ|f̃ − f| / Σ f` of the weighted amplitude `f = w |M|²` over uniformly sampled
test points, with 1 % as the target, and they also evaluate on LO-unweighted points,
where it is the mean relative error. Their training points are Sobol points of the
unit cube. The amplitudes here are the same functions computed with MadGraph
(`tth_amplitude/tth_driver.cpp` builds their momenta from the five variables),
on the domain folded by the amplitude's symmetries as in their code. 131 072 Sobol
training points, 40 000 uniform and 5000 unweighted test points per channel.

The tree learns `log |M|²`: `GPTree(Nbar=100)` at the defaults; the hybrid adds the
feature network (`global_mean='net'`, refits amortised at 8 steps per update); a
larger network (`--net hidden=256,depth=4,steps=600,polish_steps=10000`) is run to
32 768 points. `acq` is the plain tree choosing which Sobol points to learn from,
500 at a time out of 20 000 random candidates, by the largest expected error
contribution `w |M|² σ`, against the Sobol order.

## 2. Exact amplitudes: ε against the number of training points

`eps_vs_points.png`; uniform-test ε, with the networks of the paper's Figure 13
(MLP and L-GATr, read off the plot) for reference.

| points | f1 tree | f1 acq | f1 hybrid | f1 hybrid, 4 × 256 | f1 paper's networks | f3 tree | f3 acq | f3 hybrid | f3 hybrid, 4 × 256 | f3 paper's networks |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 2.7e-2 | 2.7e-2 | 1.8e-2 | 1.5e-2 | 1e-3 | 0.19 | 0.19 | 0.18 | 0.20 | 1e-2 |
| 2000 | 1.9e-2 | 1.7e-2 | 9.6e-3 | 6.6e-3 | | 0.15 | 0.10 | 0.12 | 9.8e-2 | |
| 4000 | 1.4e-2 | 1.1e-2 | 6.5e-3 | 3.1e-3 | 3e-4 (3000) | 0.12 | 6.5e-2 | 6.8e-2 | 5.0e-2 | |
| 8000 | 9.6e-3 | 7.4e-3 | 4.2e-3 | 1.9e-3 | | 8.7e-2 | 4.6e-2 | 5.3e-2 | 3.2e-2 | |
| 16 000 | 6.6e-3 | 5.0e-3 | 2.4e-3 | 1.3e-3 | 1e-4 (10⁴) | 6.4e-2 | 3.5e-2 | 4.2e-2 | 2.0e-2 | 1e-3 (10⁴) |
| 32 768 | 4.3e-3 | 3.5e-3 | 1.4e-3 | 8.3e-4 | | 4.4e-2 | 2.8e-2 | 2.9e-2 | 1.3e-2 | |
| 65 536 | 3.0e-3 | | 8.2e-4 | | | 3.1e-2 | | 2.1e-2 | | |
| 131 072 | 2.2e-3 | | 5.1e-4 | 7e-5 (10⁵) | | 2.3e-2 | | 1.5e-2 | | 3e-4 (10⁵) |

Unweighted-test ε is within 10 % of the uniform one for f1 and 10–20 % above it for
f3 (the tails of f3 are where the error sits). Update cost: tree 2 ms per point,
hybrid 4–16 ms (falling as refits become rarer), the larger network 33–78 ms.
Coverage of the calibrated sigma at the 68 % level: tree 0.65–0.74, hybrid
0.65–0.86, larger network 0.56–0.73.

Reading:

* **The hybrid beats the tree by 2 to 4 times at equal points, and the gap grows
  with the stream**: f1 5.1e-4 against 2.2e-3 at 131 072 points, where the hybrid at
  16 000 points is already where the tree ends; f3 1.5e-2 against 2.3e-2. A larger,
  longer-trained network halves the hybrid's error again (f1 8.3e-4 against 1.4e-3
  at 32 768, f3 1.3e-2 against 2.9e-2), at four to six times the update cost: the
  network's budget, not the leaves, sets the hybrid's accuracy here, and the larger
  network at 32 768 points is beyond the default hybrid at 131 072.
* **Choosing the points by the expected error contribution saves about a third of
  them**: the tree with acquisition reaches at 8000 points what the Sobol order
  reaches at 12 000 (f1) and at 16 000 what Sobol reaches at 30 000 (f3).
* **The paper's networks are far ahead of these default settings**: 30 to 50
  times lower ε at 10⁴ points than the default hybrid, 10 to 20 times than the larger
  network. With the defaults the tree family sits with the paper's classical
  interpolants, which reach 1 % at 10⁴–10⁵ points in 5D. Section 6 shows what
  closes the gap: the inputs, the leaf kernel and size, and the sampling. Every
  method here keeps improving with points, with no plateau in sight, which is the
  behaviour the paper reports for its classical methods and not for its networks.
* **Where the tree's errors sit.** On f3 the median relative error of the tree is
  1.5e-3 at 131 072 points with 87 % of test points within 1 %, while ε is 2.3e-2:
  the weighted error is dominated by the peaks of `w |M|²` (collinear tops, the
  Coulomb region), which a 100-point leaf in five dimensions resolves last.

## 3. Noisy amplitudes: 0.3 % relative noise on the training values

The paper's code carries a `NoisyEvaluator` with 3e-3 relative Gaussian noise; the
paper does not report results with it. Here the noise is added to `log |M|²` and
passed to the tree as the per-point sigma (`--noise 0.003`); the test values are
exact.

| points | f1 tree, exact / noisy | f1 hybrid, exact / noisy | f3 tree, exact / noisy | f3 hybrid, exact / noisy |
|---|---|---|---|---|
| 4000 | 1.4e-2 / 1.5e-2 | 6.5e-3 / 6.8e-3 | 0.12 / 0.12 | 6.8e-2 / 7.3e-2 |
| 16 000 | 6.6e-3 / 7.4e-3 | 2.4e-3 / 3.5e-3 | 6.4e-2 / 6.4e-2 | 4.2e-2 / 4.1e-2 |
| 32 768 | 4.3e-3 / 5.1e-3 | 1.4e-3 / 2.5e-3 | 4.4e-2 / 4.4e-2 | 2.9e-2 / 2.5e-2 |

The tree loses nothing on f3 and 5–20 % on f1; the hybrid loses nothing on f3 and up
to 1.7 times on f1, where its exact-data error is below the noise level and the
residual the leaves fit is noise-limited. The hybrid's coverage rises to 0.84–0.94
with noise: the network's error budget is added to every leaf's sigma, and with
noise-dominated residuals the calibration leaves it in place, as on the gg → HH grid.
A local version of that budget is the obvious next improvement.

## 4. Uncertainty-gated integration of f1

`gated_integration_demo.py`: uniform Monte Carlo points in chunks of 500; for each
point the tree predicts `log |M|²` with its calibrated sigma; where sigma exceeds the
tolerance the exact amplitude is evaluated and fed to the tree, elsewhere the tree's
value is used. The exact value of every point is also computed, so the shift the
surrogate introduces into the integral is known.

| surrogate, tolerance | points | exact calls, all / last 5000 | integral shift from the surrogate | Monte Carlo error |
|---|---|---|---|---|
| f1, tree on the cube, 1 % | 100 000 | 13.7 % / 5.4 % | +0.002 % | 0.32 % |
| f1, tree on the cube, 0.3 % | 200 000 | 30.0 % / 17.6 % | +0.001 % | 0.23 % |
| f1, hybrid on the cube, 1 % | 100 000 | 3.4 % / 0.8 % | +0.044 % | 0.32 % |
| f1, tree on log invariants, RBF, 300-point leaves, 1 % | 100 000 | 0.6 % / 0.0 % | −0.026 % | 0.32 % |
| f3, tree on log invariants, RBF, 300-point leaves, 1 % | 100 000 | 6.8 % / 2.9 % | +0.007 % | 1.05 % |

With a 1 % tolerance the cube tree answers 95 % of the last points itself and moves
the integral by a hundredth of its Monte Carlo error; the cube hybrid answers 99 % of
them and moves it by a seventh of the Monte Carlo error. The hybrid's shift is a
coherent bias of the network, the tree's local errors average out, so for a fixed
error budget the tree can be gated more aggressively than its tolerance alone
suggests and the hybrid less. The improved tree of Section 6 (`--inputs invariants
--kernel rbf --nbar 300`) needs 600 exact evaluations of f1 in 100 000 points, none
at all after the first 10 000, for a shift a twelfth of the Monte Carlo error, and
6800 of f3 (3 % of the last batch) for a shift a hundred-and-fiftieth of it. In every
case the shift is far inside the tolerance, since only points the surrogate is sure
of are taken from it.

## 5. What this demonstrates

With the package defaults on the cube coordinates the dedicated networks of the paper
are the better interpolants by 30–50 ×. Section 6 shows that this is a matter of
representation and sampling, not of the method: on the log-invariants, with RBF
leaves of 300 points and the importance-sampled stream an integrator produces, the
plain tree matches the networks on f1 at 16 000 points and is ten times below them
at 131 072, and is within a factor 2.6 on f3 with its curve still falling. On top
of that the tree family offers what the networks do not: learning from the
evaluations as they are made with no retraining, a per-point calibrated uncertainty
that can gate when the exact amplitude is needed (Section 4), the choice of which
points to evaluate, and per-point noise in the training values. The real two-loop
case, where the evaluations are hours each and carry a QMC error, is in
`BENCHMARK_RESULTS_hhgrid_amplitude.md`.

## Reproduce

```bash
cd examples
python tth_amplitude_benchmark.py <data_dir> --channels gg,uu --configs tree,hybrid --tag exact
python tth_amplitude_benchmark.py <data_dir> --channels gg,uu --configs acq --nmax 32768 --tag acq
python tth_amplitude_benchmark.py <data_dir> --channels uu --configs hybrid --nmax 32768 \
    --net hidden=256,depth=4,steps=600,polish_steps=10000 --tag bignet_uu
python tth_amplitude_benchmark.py <data_dir> --channels uu --configs tree,hybrid --nmax 32768 --noise 0.003 --tag noise0.003_uu
python gated_integration_demo.py <path to tth_driver> --config tree --tol 0.01 --npoints 100000
# Section 6: the variants, e.g. the best plain tree and the best hybrid over the full stream
python tth_amplitude_benchmark.py <data_dir> --channels uu,gg --configs tree --inputs invariants --kernel rbf --train mixed --nbar 300 --tag sweep_finalTreeMix300
python tth_amplitude_benchmark.py <data_dir> --channels uu,gg --configs hybrid --inputs invariants --features invariants --train mixed \
    --net hidden=256,depth=4,steps=600,polish_steps=10000 --nmax 65536 --tag sweep_finalHybMixBig
python plot_tth_results.py
```
`<data_dir>` needs the `{ch}_train_unweighted.npz` sets of `tth_amplitude/README.md` for `--train mixed`.

## 6. Improving on the 5D benchmark: what moves the error

A controlled sweep on f1 and f3 to 16 384 points (seed 1, one run each; the hybrid
variants with `steps_per_update=8`), ε on the uniform test set at 4000, 16 000 and,
where run, 32 768 points. The baselines are the rows of Section 2.

* **Inputs.** `--inputs invariants` feeds the tree the ten `log(p_i · p_j)` of the
  external momenta (the inputs of the paper's networks) instead of the five cube
  coordinates; `--features invariants` gives them to the hybrid's network while the
  tree keeps the cube (the `feature_map` argument of `FeatureNetLearner`).
* **Training points.** `--train unweighted` draws them with probability proportional
  to `f = w |M|²` (65 536 such points per channel, generated by rejection), which is
  the stream an importance-sampling integrator produces; `mixed` alternates Sobol
  and f-weighted points.
* **Leaves.** `--kernel` Matérn 5/2, RBF or the additive + Matérn kernel; `--nbar 300`;
  `--split max_uncertainty`.
* **Network.** `--net hidden=256,depth=4,steps=600,polish_steps=10000`; the "one fit
  at the end" row trains the 4 × 256 network once, on the first 15 000 points, with
  3000 L-BFGS iterations and 30 000 Adam steps, to see what the amortised online
  training costs against a converged fit.

| variant (ε on the uniform test set) | f1 @4000 | f1 @16000 | f1 @32768 | f1 @131072 | f3 @4000 | f3 @16000 | f3 @32768 | f3 @131072 | ms/pt (f1 @16k) |
|---|---|---|---|---|---|---|---|---|---|
| tree (baseline: Matern 3/2, Nbar 100, Sobol) | 1.39e-02 | 6.59e-03 | 4.30e-03 | 2.17e-03 | 1.16e-01 | 6.36e-02 | 4.42e-02 | 2.27e-02 | 2.0 |
| tree, chooses its points (w|M|^2 sigma) | 1.10e-02 | 4.99e-03 | 3.45e-03 | – | 6.46e-02 | 3.53e-02 | 2.81e-02 | – | 2.8 |
| hybrid (baseline: 3x128 network on the cube) | 6.45e-03 | 2.38e-03 | 1.43e-03 | 5.07e-04 | 6.76e-02 | 4.16e-02 | 2.91e-02 | 1.46e-02 | 6.4 |
| hybrid, 4x256 network | 3.13e-03 | 1.25e-03 | 8.34e-04 | – | 5.04e-02 | 2.03e-02 | 1.31e-02 | – | 32.5 |
| tree, Matern 5/2 | 9.81e-03 | 4.21e-03 | – | – | 1.05e-01 | 6.24e-02 | – | – | 2.6 |
| tree, RBF | 9.55e-03 | 3.67e-03 | – | – | 1.10e-01 | 6.65e-02 | – | – | 2.7 |
| tree, additive + Matern | 9.40e-03 | 4.28e-03 | – | – | 1.14e-01 | 6.77e-02 | – | – | 13.1 |
| tree, Nbar 300 | 9.20e-03 | 4.27e-03 | – | – | 8.14e-02 | 4.98e-02 | – | – | 14.7 |
| tree, training points ~ f | 1.11e-02 | 5.20e-03 | – | – | 6.17e-02 | 3.21e-02 | – | – | 6.0 |
| tree, Sobol + f-weighted alternating | 1.19e-02 | 5.93e-03 | – | – | 6.83e-02 | 3.44e-02 | – | – | 5.9 |
| tree, inputs = 10 log invariants | 3.62e-03 | 1.82e-03 | – | – | 5.42e-02 | 2.29e-02 | – | – | 8.9 |
| tree, split by max_uncertainty | 1.35e-02 | 6.65e-03 | – | – | 1.10e-01 | 6.45e-02 | – | – | 7.6 |
| hybrid, network on log invariants | 1.58e-03 | 5.68e-04 | – | – | 2.42e-02 | 1.08e-02 | – | – | 17.6 |
| hybrid, Sobol + f-weighted | 5.12e-03 | 2.07e-03 | – | – | 4.16e-02 | 1.82e-02 | – | – | 14.2 |
| hybrid, invariants + 4x256 network | 8.93e-04 | 3.00e-04 | – | – | 1.53e-02 | 7.44e-03 | – | – | 75.2 |
| hybrid, invariants + mixed + 4x256 | 6.42e-04 | 2.34e-04 | 1.46e-04 | – | 4.50e-03 | 2.02e-03 | 2.19e-03 | – | 72.5 |
| hybrid, invariants, one 4x256 fit at the end (3000 L-BFGS + 30000 Adam) | 1.39e-02 | 3.82e-04 | – | – | 1.16e-01 | 5.44e-03 | – | – | 43.7 |
| tree, invariants + RBF | 1.04e-03 | 4.07e-04 | – | – | 8.15e-02 | 2.77e-02 | – | – | 7.8 |
| tree, invariants + Matern 5/2 | 1.69e-03 | 7.51e-04 | – | – | 4.88e-02 | 2.06e-02 | – | – | 10.5 |
| tree, inputs = cube + invariants (15D) | 3.56e-03 | 1.73e-03 | – | – | 5.72e-02 | 2.26e-02 | – | – | 13.6 |
| hybrid, tree and network on invariants | 1.58e-03 | 5.92e-04 | – | – | 2.54e-02 | 1.01e-02 | – | – | 20.1 |
| hybrid, tree and network on invariants, 4x256 | 8.77e-04 | 2.78e-04 | – | – | 2.23e-02 | 6.64e-03 | – | – | 48.1 |
| tree, invariants, training ~ f | – | – | – | – | 1.90e-02 | 9.74e-03 | – | – | nan |
| tree, invariants + RBF, training ~ f | 7.51e-04 | 2.19e-04 | – | – | 1.54e-02 | 6.61e-03 | – | – | 7.9 |
| hybrid, tree and network on invariants, training ~ f | – | – | – | – | 7.05e-03 | 2.66e-03 | – | – | nan |
| full stream: tree, invariants + RBF, Sobol | 1.04e-03 | 4.07e-04 | 2.48e-04 | 7.50e-05 | 6.18e-02 | 2.38e-02 | 2.00e-02 | 7.91e-03 | 3.8 |
| full stream: tree, invariants + RBF, Sobol + f-weighted | 7.97e-04 | 2.66e-04 | 1.45e-04 | 5.07e-05 | 1.90e-02 | 7.78e-03 | 5.15e-03 | 1.81e-03 | 7.6 |
| full stream: hybrid, invariants (both), Sobol + f-weighted | 1.10e-03 | 4.40e-04 | 2.62e-04 | 9.36e-05 | 1.02e-02 | 3.98e-03 | 2.42e-03 | 1.07e-03 | 17.0 |
| full stream: same, 4x256 | 5.91e-04 | 1.97e-04 | 1.15e-04 | – | 4.42e-03 | 1.87e-03 | 1.46e-03 | – | 40.9 |
| tree, invariants + RBF, mixed, Nbar 300 | 1.90e-04 | 5.42e-05 | – | – | 1.35e-02 | 4.61e-03 | – | – | 13.9 |
| hybrid, invariants (both), mixed, RBF leaves | 9.65e-04 | 2.94e-04 | – | – | 1.02e-02 | 3.75e-03 | – | – | 9.8 |
| tree, invariants + RBF, mixed, 0.3 % noise on the values | 3.05e-03 | 2.11e-03 | 1.86e-03 | – | 2.08e-02 | 9.57e-03 | 6.64e-03 | – | 2.9 |
| full stream: tree, invariants + RBF, Sobol + f-weighted, Nbar 300 | 1.90e-04 | 5.38e-05 | 2.64e-05 | 7.42e-06 | 1.35e-02 | 4.60e-03 | 2.48e-03 | 7.88e-04 | 16.7 |

Reading:

* **The representation is the lever.** In the log-invariants the amplitude is
  nearly a low-order rational function, and ten-dimensional leaves fit it far better
  than five-dimensional ones fit the cube: the plain tree gains 3.6 × on f1 and
  2.8 × on f3 from the inputs alone, and with RBF leaves 16 × on f1 (4.1e-4 at
  16 000 points, a median relative error of 1e-5, every test point within 1 %).
  f3 is rougher (collinear tops), where Matérn 5/2 beats RBF (2.1e-2 against
  2.8e-2) and the gain is 3 ×. Adding the cube coordinates to the invariants
  (15 inputs) changes nothing. For the hybrid the network on the invariants is
  worth 4 × on both channels, with the tree on the cube or on the invariants alike.
* **Points drawn where the integrand is large are the second lever.** Alone they
  give the tree 1.3 × on f1 and 2 × on f3 (the weighted error lives in the peaks);
  on top of the invariants and RBF leaves they bring the plain tree to 2.2e-4 on f1
  and 6.6e-3 on f3 at 16 000 points, 30 and 10 times the baseline, at 8 ms per point
  and with no network; the hybrid with everything on the invariants reaches 2.7e-3
  on f3, 16 times its baseline. Since an integrator's stream *is* importance
  sampled, this is the distribution the tree sees in its intended use, not a trick.
* **Kernels, leaf size, the split criterion.** On their own the smoother kernels
  give 1.6–1.8 × on f1 and nothing on f3; the additive kernel 1.5 × and nothing, at
  five times the cost; `Nbar = 300` 1.5 × and 1.3 × at seven times the cost; the
  `max_uncertainty` split nothing at three times the cost. These are second-order
  next to the representation and the sampling.
* **The network budget.** The 4 × 256 network halves the hybrid's error at four to
  six times the update cost, in every setting; with the invariants and mixed points
  it reaches 2.3e-4 at 16 000 and 1.5e-4 at 32 768 on f1. A single converged fit at
  the end of the same stream gives 3.8e-4 against 3.0e-4 for the amortised online
  training with the same network, so the online schedule costs little.
* **Leaf size in the invariants.** With ten inputs a 100-point leaf is short of
  points for its ten length scales; `Nbar = 300` on the invariants with RBF leaves
  and the mixed stream gives a further 4 × on f1 (5.4e-5 at 16 000, a median
  relative error of 1e-6) and 1.4 × on f3, at twice the update cost. In the cube
  the same change gave 1.5 ×.
* **Noise.** With 0.3 % noise on the training values the improved tree sits at
  1.9e-3 (f1) and 6.6e-3 (f3) at 32 768 points: on f1 its exact-data error is now
  far below the noise, so the noise sets the floor; on f3 the noisy and exact runs
  coincide.

### Full streams with the best configurations

`eps_vs_points_improved.png`; the "full stream" rows of the table.

| configuration (131 072 points, mixed stream unless noted) | f1 | f3 | update cost |
|---|---|---|---|
| tree on the cube, Matérn 3/2 leaves, Sobol (Section 2 baseline) | 2.2e-3 | 2.3e-2 | 2 ms |
| tree on log invariants, RBF leaves, Sobol | 7.5e-5 | 7.9e-3 | 4 ms |
| tree on log invariants, RBF leaves | 5.1e-5 | 1.8e-3 | 8 ms |
| tree on log invariants, RBF leaves, `Nbar = 300` | 7.4e-6 | 7.9e-4 | 16 ms |
| hybrid on log invariants, Matérn 3/2 leaves | 9.4e-5 | 1.1e-3 | 11 ms |
| hybrid on log invariants, 4 × 256 network (65 536 points) | 7.1e-5 | 9.4e-4 | 21 ms |
| MLP / L-GATr of arXiv:2412.09534 at 10⁵ points (Figure 13) | 7e-5 | 3e-4 | |

* **f1 is solved by the plain tree.** On the log-invariants with RBF leaves it
  matches the paper's networks at 131 072 points (7.5e-5 with Sobol points,
  5.1e-5 with the mixed stream), and with 300-point leaves it is ten times below
  them (7.4e-6, a median relative error of 4e-7, every test point within 1 %),
  online, at 16 ms per point, with no network. Its curve keeps falling as a power
  law where the networks' flattens.
* **f3 closes to a factor 2.6.** The tree with 300-point leaves reaches 7.9e-4 and
  the hybrids 9.4e-4 to 1.1e-3 against the networks' 3e-4, from 13 × worse at the
  baseline, with no sign of a plateau; the mixed stream alone is worth 4 × on f3
  at 131 072 points (7.9e-3 with Sobol against 1.8e-3), because the error lives in
  the collinear peaks the importance-sampled points populate.
* **The network no longer carries the result.** With the right inputs the plain
  tree with RBF leaves beats the hybrid with Matérn leaves on f1 and nearly matches
  it on f3; the hybrid's remaining advantage is on f3 at small point counts
  (4.4e-3 against 1.4e-2 at 4000). RBF leaves inside the hybrid give 2 × on f1
  (2.9e-4 against 5.9e-4 at 16 000) and nothing on f3.

### Where f3's error sits

`tth_error_diagnostic.py`, the tree above (invariants, RBF, 300-point leaves, mixed
stream) at 16 384 points, ε = 3.4e-3, broken down over the uniform test set:

* 61 % of the error sum comes from 1 % of the test points, whose relative errors
  run to 290 % while the median is 3e-5; 85 % of it from the top fifth in energy
  (β² > 0.8, which holds 62 % of the integral) and 59 % from φ_t > 0.8 π, where a
  top can line up with a beam after the boost.
* For those worst points the smallest invariant is a gluon–top one (p_g·p_t,
  80 % of cases) rather than a top–top or Higgs one: the t-channel top propagator,
  |M|² ∝ 1/(p_g·p_t)², whose peaks sharpen as (E_t/m_t)² with the energy. In the
  log-invariants the peak is a linear trend, −2 log(p_g·p_t), and a leaf with a
  constant prior mean extrapolates it badly at the edge of its data.
* Two remedies follow: factor the poles out of the target, learning
  log(|M|² Π s_gt^p) (`--target pole1`, `pole2`), and give the leaves a global
  additive model in the invariants (`--global additive`). REMEDY_RESULTS
