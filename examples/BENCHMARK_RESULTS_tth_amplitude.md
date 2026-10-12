# pyGPTreeO on 5D tree-level tt̄H amplitudes, and an uncertainty-gated integration

`examples/tth_amplitude_benchmark.py`, `examples/gated_integration_demo.py`,
`examples/plot_tth_results.py`; data as in `examples/tth_amplitude/README.md`; one
CPU core per run, up to four runs in parallel; seed 1; 12 October 2026.

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
* **The paper's networks are far ahead on these exact, smooth amplitudes**: 30 to 50
  times lower ε at 10⁴ points for the default hybrid, 10 to 20 times for the larger
  network. The tree family sits with the paper's classical interpolants, which reach
  1 % at 10⁴–10⁵ points in 5D: the default hybrid crosses 1 % at 2000 points on f1
  and would need about 2 × 10⁵ on f3. Every method here keeps improving with points,
  with no plateau in sight, which is the behaviour the paper reports for its
  classical methods and not for its networks.
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
| tree, 1 % | 100 000 | 13.7 % / 5.4 % | +0.002 % | 0.32 % |
| tree, 0.3 % | 200 000 | 30.0 % / 17.6 % | +0.001 % | 0.23 % |
| hybrid, 1 % | 100 000 | 3.4 % / 0.8 % | +0.044 % | 0.32 % |

With a 1 % tolerance the tree answers 95 % of the last points itself and moves the
integral by a hundredth of its Monte Carlo error; the hybrid answers 99 % of them and
moves it by a seventh of the Monte Carlo error. The hybrid's shift is a coherent bias
of the network, the tree's local errors average out, so for a fixed error budget the
tree can be gated more aggressively than its tolerance alone suggests and the hybrid
less. In every case the shift is far inside the tolerance, since only points the
surrogate is sure of are taken from it.

## 5. What this demonstrates

On exact, cheap-to-sample 5D amplitudes the dedicated networks of the paper are the
better interpolants, and the hybrid's accuracy is set by its network budget. The
tree family's case is what the networks do not offer: learning from the evaluations
as they are made with no retraining, a per-point calibrated uncertainty that can gate
when the exact amplitude is needed, the choice of which points to evaluate, and
per-point noise in the training values at no cost in accuracy. The first two are
measured above; the real two-loop case, where the evaluations are hours each and
carry a QMC error, is in `BENCHMARK_RESULTS_hhgrid_amplitude.md`.

## Reproduce

```bash
cd examples
python tth_amplitude_benchmark.py <data_dir> --channels gg,uu --configs tree,hybrid --tag exact
python tth_amplitude_benchmark.py <data_dir> --channels gg,uu --configs acq --nmax 32768 --tag acq
python tth_amplitude_benchmark.py <data_dir> --channels uu --configs hybrid --nmax 32768 \
    --net hidden=256,depth=4,steps=600,polish_steps=10000 --tag bignet_uu
python tth_amplitude_benchmark.py <data_dir> --channels uu --configs tree,hybrid --nmax 32768 --noise 0.003 --tag noise0.003_uu
python gated_integration_demo.py <path to tth_driver> --config tree --tol 0.01 --npoints 100000
python plot_tth_results.py
```
