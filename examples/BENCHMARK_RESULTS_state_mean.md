# Interaction-order state as global model: design and benchmark

Question: can one global model serve targets of different structure (additive, pairwise,
fully coupled) without being told which one it faces, on streams that do not sample
uniformly and on noisy data, and can it give the tree an uncertainty that knows where the
global model has data?

Short answer: yes, at six to eight times the run time of the plain tree at 4000 points.

* On two targets that couple all inputs, under four kinds of stream, the tree with the
  state has 0.29 of the plain tree's error without noise and 0.55 with noise (geometric
  mean over 24 cells each), and a lower error than the plain tree in all 48 cells. With
  network features it has 0.16 and 0.55. `global_mean='net'` has 0.31 and 1.34.
* On three targets with additive or pairwise structure it is two to four orders of
  magnitude below the plain tree without noise and 3 to 16 times below with noise, and
  below `'additive_gp'` in all six cases.
* Away from the stream its one-sigma coverage is 0.51 to 0.57 without noise, where the
  other modes have 0.30 to 0.37.
* A by-product, usable on its own: held-out stopping for the feature network
  (`holdout`), which takes `'net'` from 1.34 to 0.64 of the plain tree's error with noise.

The numbers are from one seed and 4000 points per cell.

![Error relative to the plain tree](results/state_mean/summary.png)

`results/state_mean/` also has `streams.png` (every cell), `coverage.png` and
`additive.png`; `plot_state_mean_results.py` draws them from the result lines.

## The model

### One prior for every order of interaction

For `d` inputs, with `k_j(x, x') = exp(-(x_j - x'_j)^2 / (2 l_j^2))`:

    k(x, x') = s1^2 / d * e_1  +  s2^2 / C(d,2) * e_2  +  sf^2 * prod_j k_j  (+ sig^2 for the noise)

`e_1 = sum_j k_j` and `e_2 = sum_{i<j} k_i k_j` are the first two elementary symmetric
polynomials of the `k_j` (the additive kernels of Duvenaud, Nickisch and Rasmussen, 2011).
A function drawn from the first term is a sum of functions of one input each, from the
second a sum of functions of two inputs, from the last a function in which all inputs
interact. The hyperparameters are the `d` length scales, shared by the three terms, the
weights `s1`, `s2`, `sf` and the noise level `sig`: `d + 4` numbers. They maximise the exact
marginal likelihood (closed-form gradient, L-BFGS-B, at most 60 evaluations per start) on a
coverage reservoir of at most 1000 points, with the targets standardised.
`StateGlobalMean.order_shares()` reports `s1^2 : s2^2 : sf^2` as fractions. `max_order=1`
drops the pairwise term; with two inputs it is dropped because it equals the last term.

### From the prior to a linear model

A GP with this prior fitted to all `N` points would cost `O(N^3)`. The state is a linear
model `f(x) = phi(x)^T w + e(x)` that keeps as much of the prior as a fixed number of
features can, and whose cost per solve is linear in `N`:

* **Additive term, sine basis.** On an interval `[c_j - L_j, c_j + L_j]` that contains the
  range of input `j` with a margin, the functions `sin(pi m (x_j - c_j + L_j) / (2 L_j)) /
  sqrt(L_j)`, `m = 1, 2, ...`, with independent coefficients of prior variance
  `s1^2 / d * sqrt(2 pi) l_j exp(-(l_j w_m)^2 / 2)` at frequency `w_m = pi m / (2 L_j)`,
  reproduce the RBF kernel in input `j` (Solin and Särkkä, 2020). Frequencies up to
  `7.5 / l_j` are kept, between 8 and 128 per input, and the interval follows the range
  explored so far. No point of the stream is singled out: the additive part of the target
  is learned from every point at a cost that depends on the number of basis functions only.
* **Higher terms, kernel centres.** For `k_rest = s2^2 / C(d,2) * e_2 + sf^2 * prod_j k_j`
  the features are `psi(x) = W^T k_rest(U, x)` with `U` the points of a second coverage
  reservoir (500) and `W` the whitening of `k_rest(U, U)`, so that `psi(x)^T psi(x')` is the
  Nyström approximation of `k_rest`.
* **Remainder, the leaves as blocks.** What the centres do not carry, `e(x)`, has the
  covariance `k_rest(x, x') - psi(x)^T psi(x')`. It is kept among the points that one leaf
  owns and dropped between leaves (the partially independent conditional approximation of
  Snelson and Ghahramani, 2007, with the tree's partition as its blocks). For a leaf with
  points `X_c`: `R_c = k_rest(X_c, X_c) - Psi_c Psi_c^T + diag(noise)`, the noise of a point
  being the larger of its declared variance and the fitted `sig^2`.
* **Solve.** With `Lambda` the prior variances of the coefficients,
  `M = Lambda^-1 + sum_c Phi_c^T R_c^-1 Phi_c` and `w = M^-1 sum_c Phi_c^T R_c^-1 t_c`: one
  pass over the leaves, `O(n_c^3 + n_c P^2)` per leaf with `P` features, and one `P x P`
  factorisation.
* **Network features (`net=True`).** A `FeatureNetLearner` (3 x 128 SiLU, the network of
  `global_mean='net'`) is refitted with held-out stopping (`holdout=0.2`) on the residual of
  the additive part alone, whenever the prior is due. Its last hidden layer, standardised,
  is appended to `phi`. The prior variance of these coefficients is the one of nine values
  between 1e-6 and 1e-1 with the highest evidence on the held-out points: features of a
  network that has only memorised its training points get a narrow prior and drop out.

### In the tree

1. Every observation goes to the two coverage reservoirs and, as always, to its leaf.
2. At 60 observations and at every doubling of the count the prior is due. Once the
   reservoir is full the fit starts from the previous hyperparameters only, and is skipped
   if less than a quarter of the reservoir has been replaced since the last one.
3. At these moments and whenever the count has grown by a quarter, the coefficients are
   solved again from the points in the leaves and published as a new snapshot.
4. A leaf that meets a newer snapshot refits its GP on the new residual with its
   hyperparameters kept (`GPRegressorInterface.refit_fixed`); the hyperparameters are
   optimised on the leaf's usual retrain schedule. Global models with rare snapshots
   (`'additive_gp'`, `'net'`) keep the full refit.
5. The prediction is `phi(x)^T w` plus the leaf GP's mean.

### The variance

A leaf GP's mean at `x` is `a(x)^T r`, with `r` its training residuals. As these residuals
are `y - Phi_train w`, the tree's prediction depends on the coefficients through
`q(x) = phi(x) - Phi_train^T a(x)`, and their uncertainty contributes `q^T M^-1 q`. Among
the leaf's points `a(x)` reproduces `phi`, `q` is small, and the leaf has absorbed the
state's error; away from them `q` tends to `phi(x)` and the full uncertainty of the
coefficients appears. The snapshot's `leaf_variance` returns

    max(leaf GP variance, remainder variance) + q^T M^-1 q

where the remainder variance is the prior variance of `e(x)` given the leaf's training
points. It takes the place of `error_scale`, the stream-wide error estimate that the other
global models add at every `x` alike. Per-leaf calibration (`use_calibrated_sigma`) is
applied on top as before.

## Benchmark

Setup (`examples/benchmark_global_mean_streams.py`):

* Targets `RotatedRosenbrock` and `GaussianPeaks` in 6 dimensions (every input coupled
  through a fixed rotation) under the four streams of
  `BENCHMARK_RESULTS_global_mean_streams.md` (uniform, focusing, sweeping, walker), 4000
  points, seed 1. Also `Rosenbrock`, `Rastrigin` and `Levy` in 6 dimensions (sums of terms
  in one or two coordinates) on the uniform stream.
* Without noise (the benchmark's nominal sigma of `1e-3 |y|`) and with Gaussian noise of
  0.2 standard deviations of the target (`--noise 0.2`), declared to the tree.
* Tree: `GPTree(Default_GPR(n_restarts_optimizer=1), Nbar=100, theta=1e-4,
  retrain_every_n_points=25, splitting_strategy='gradual', use_calibrated_sigma=True)`.
* Configurations: `tree`; `additive_gp` (`global_pkg`); `net` (`hybrid`); `net` with
  `holdout=0.2` (`hybrid_holdout`); `state`; `state` with `net=True` (`state_net`).
* Metrics as in the stream benchmark, all against the noise-free target: NRMSE = RMSE /
  range of the target; `prequential` on the stream after the first 666 points, `uniform
  test` on 3000 uniform points, `focus test` on 3000 points where the stream ended.
  Coverage is the fraction of errors within one predicted sigma (0.68 for a calibrated
  Gaussian).

### Without noise

| | geometric-mean error relative to the tree | cells with lower error than the tree | than `net` | largest ratio to the tree |
|---|---|---|---|---|
| tree | 1.00 |  | 2 of 24 |  |
| `additive_gp` | 0.69 | 23 of 24 | 3 of 24 | 1.67 (rotated_rosenbrock, sweeping, focus test) |
| `net` | 0.31 | 22 of 24 |  | 1.12 (gaussian_peaks, sweeping, focus test) |
| `net`, held-out stopping | 0.31 | 23 of 24 | 8 of 24 | 1.21 (gaussian_peaks, sweeping, focus test) |
| `state` | 0.29 | 24 of 24 | 15 of 24 | 0.84 (gaussian_peaks, uniform, focus test) |
| `state`, `net=True` | 0.16 | 24 of 24 | 18 of 24 | 0.50 (gaussian_peaks, focusing, prequential) |

Prequential NRMSE (bold: lowest of the row)

| target | stream | tree | `additive_gp` | `net` | `net`, held-out stopping | `state` | `state`, `net=True` |
|---|---|---|---|---|---|---|---|
| rotated_rosenbrock | uniform | 0.0187 | 0.0128 | 0.0037 | 0.0050 | 0.0019 | **0.0011** |
| rotated_rosenbrock | focusing | 0.0061 | 0.0049 | 0.0019 | 0.0032 | 6.50e-04 | **4.76e-04** |
| rotated_rosenbrock | sweeping | 0.0064 | 0.0057 | 0.0034 | 0.0052 | 0.0012 | **9.95e-04** |
| rotated_rosenbrock | walker | 0.0061 | 0.0042 | 0.0021 | 0.0024 | 0.0014 | **9.88e-04** |
| gaussian_peaks | uniform | 0.0162 | 0.0126 | **0.0056** | 0.0063 | 0.0123 | 0.0062 |
| gaussian_peaks | focusing | 0.0076 | 0.0058 | 0.0046 | 0.0055 | 0.0050 | **0.0038** |
| gaussian_peaks | sweeping | 0.0231 | 0.0121 | 0.0200 | 0.0171 | 0.0107 | **0.0098** |
| gaussian_peaks | walker | 0.0100 | 0.0056 | **0.0043** | 0.0052 | 0.0068 | 0.0049 |

Uniform-test NRMSE

| target | stream | tree | `additive_gp` | `net` | `net`, held-out stopping | `state` | `state`, `net=True` |
|---|---|---|---|---|---|---|---|
| rotated_rosenbrock | uniform | 0.0141 | 0.0089 | 0.0013 | 0.0019 | 0.0013 | **5.96e-04** |
| rotated_rosenbrock | focusing | 0.0308 | 0.0179 | 0.0037 | 0.0057 | 0.0032 | **0.0020** |
| rotated_rosenbrock | sweeping | 0.0613 | 0.0477 | 0.0258 | 0.0197 | **0.0099** | 0.0119 |
| rotated_rosenbrock | walker | 0.0921 | 0.0722 | 0.0446 | 0.0491 | **0.0279** | 0.0283 |
| gaussian_peaks | uniform | 0.0129 | 0.0106 | 0.0023 | **0.0018** | 0.0104 | 0.0029 |
| gaussian_peaks | focusing | 0.0213 | 0.0158 | **0.0051** | 0.0067 | 0.0164 | 0.0078 |
| gaussian_peaks | sweeping | 0.0842 | 0.0422 | 0.0525 | 0.0559 | 0.0427 | **0.0308** |
| gaussian_peaks | walker | 0.1056 | 0.0567 | 0.0277 | **0.0249** | 0.0789 | 0.0295 |

Focus-test NRMSE

| target | stream | tree | `additive_gp` | `net` | `net`, held-out stopping | `state` | `state`, `net=True` |
|---|---|---|---|---|---|---|---|
| rotated_rosenbrock | uniform | 0.0134 | 0.0097 | 0.0014 | 0.0020 | 0.0014 | **6.32e-04** |
| rotated_rosenbrock | focusing | 4.07e-05 | 2.89e-05 | 8.37e-06 | 4.42e-06 | 3.38e-06 | **1.07e-06** |
| rotated_rosenbrock | sweeping | 0.0048 | 0.0081 | 0.0052 | 0.0013 | 6.95e-04 | **3.83e-04** |
| rotated_rosenbrock | walker | 0.0025 | 9.02e-04 | 3.02e-04 | 4.46e-04 | 2.28e-04 | **7.49e-05** |
| gaussian_peaks | uniform | 0.0127 | 0.0109 | 0.0024 | **0.0018** | 0.0107 | 0.0028 |
| gaussian_peaks | focusing | 1.24e-04 | 1.01e-04 | 2.92e-05 | **2.04e-05** | 8.40e-05 | 2.44e-05 |
| gaussian_peaks | sweeping | 0.0085 | 0.0075 | 0.0095 | 0.0103 | 0.0056 | **0.0040** |
| gaussian_peaks | walker | 0.0049 | 0.0016 | 0.0011 | 0.0011 | 0.0012 | **5.90e-04** |

### With noise of 0.2 sd

| | geometric-mean error relative to the tree | cells with lower error than the tree | than `net` | largest ratio to the tree |
|---|---|---|---|---|
| tree | 1.00 |  | 19 of 24 |  |
| `additive_gp` | 0.81 | 20 of 24 | 21 of 24 | 1.16 (gaussian_peaks, sweeping, focus test) |
| `net` | 1.34 | 5 of 24 |  | 4.40 (rotated_rosenbrock, walker, prequential) |
| `net`, held-out stopping | 0.64 | 22 of 24 | 24 of 24 | 1.11 (rotated_rosenbrock, sweeping, focus test) |
| `state` | 0.55 | 24 of 24 | 24 of 24 | 0.90 (gaussian_peaks, focusing, focus test) |
| `state`, `net=True` | 0.55 | 23 of 24 | 24 of 24 | 1.01 (gaussian_peaks, focusing, focus test) |

Prequential NRMSE

| target | stream | tree | `additive_gp` | `net` | `net`, held-out stopping | `state` | `state`, `net=True` |
|---|---|---|---|---|---|---|---|
| rotated_rosenbrock | uniform | 0.0317 | 0.0226 | 0.0345 | 0.0168 | **0.0127** | 0.0131 |
| rotated_rosenbrock | focusing | 0.0138 | 0.0106 | 0.0199 | 0.0097 | **0.0082** | 0.0082 |
| rotated_rosenbrock | sweeping | 0.0152 | 0.0153 | 0.0286 | 0.0137 | 0.0121 | **0.0113** |
| rotated_rosenbrock | walker | 0.0140 | 0.0113 | 0.0614 | 0.0144 | **0.0096** | 0.0113 |
| gaussian_peaks | uniform | 0.0277 | 0.0220 | 0.0362 | 0.0199 | 0.0180 | **0.0169** |
| gaussian_peaks | focusing | 0.0150 | 0.0129 | 0.0239 | 0.0125 | **0.0098** | 0.0101 |
| gaussian_peaks | sweeping | 0.0440 | 0.0282 | 0.0497 | 0.0252 | **0.0216** | 0.0219 |
| gaussian_peaks | walker | 0.0293 | 0.0230 | 0.0746 | 0.0219 | **0.0156** | 0.0170 |

Uniform-test NRMSE

| target | stream | tree | `additive_gp` | `net` | `net`, held-out stopping | `state` | `state`, `net=True` |
|---|---|---|---|---|---|---|---|
| rotated_rosenbrock | uniform | 0.0267 | 0.0191 | 0.0278 | 0.0111 | **0.0094** | 0.0099 |
| rotated_rosenbrock | focusing | 0.0465 | 0.0269 | 0.0478 | 0.0238 | **0.0176** | 0.0213 |
| rotated_rosenbrock | sweeping | 0.0754 | 0.0565 | 0.0722 | 0.0434 | **0.0364** | 0.0381 |
| rotated_rosenbrock | walker | 0.1339 | 0.1214 | 0.1078 | **0.0859** | 0.1058 | 0.0936 |
| gaussian_peaks | uniform | 0.0247 | 0.0209 | 0.0320 | 0.0137 | 0.0153 | **0.0123** |
| gaussian_peaks | focusing | 0.0373 | 0.0256 | 0.0454 | 0.0249 | 0.0237 | **0.0208** |
| gaussian_peaks | sweeping | 0.1148 | 0.0765 | 0.1320 | 0.0603 | 0.0509 | **0.0459** |
| gaussian_peaks | walker | 0.2005 | 0.1255 | 0.1478 | **0.0611** | 0.1145 | 0.0737 |

Focus-test NRMSE

| target | stream | tree | `additive_gp` | `net` | `net`, held-out stopping | `state` | `state`, `net=True` |
|---|---|---|---|---|---|---|---|
| rotated_rosenbrock | uniform | 0.0254 | 0.0203 | 0.0307 | 0.0118 | **0.0098** | 0.0103 |
| rotated_rosenbrock | focusing | 0.0062 | 0.0062 | 0.0061 | 0.0060 | **0.0045** | 0.0061 |
| rotated_rosenbrock | sweeping | 0.0175 | 0.0187 | 0.0241 | 0.0194 | **0.0092** | 0.0099 |
| rotated_rosenbrock | walker | 0.0095 | 0.0080 | 0.0315 | 0.0073 | **0.0057** | 0.0062 |
| gaussian_peaks | uniform | 0.0253 | 0.0214 | 0.0325 | 0.0138 | 0.0160 | **0.0129** |
| gaussian_peaks | focusing | 0.0068 | 0.0073 | 0.0073 | **0.0060** | 0.0061 | 0.0069 |
| gaussian_peaks | sweeping | 0.0405 | 0.0469 | 0.0362 | 0.0224 | 0.0183 | **0.0169** |
| gaussian_peaks | walker | 0.0186 | 0.0136 | 0.0381 | 0.0102 | **0.0080** | 0.0091 |

Reading:

* **Without noise** the state alone is level with the feature network on average (0.29
  against 0.31 of the plain tree's error) but the two differ by target: on
  `RotatedRosenbrock` the state is ahead in all 12 cells, on `GaussianPeaks` the
  network is ahead in 9 of 12. With `net=True` the state has the lowest error of the six
  configurations in 15 of 24 cells and is within a factor of 1.6 of the lowest in the
  others; every other configuration is at least a factor of 6 from the lowest somewhere.
* **With noise** `'net'` is worse than the plain tree in 19 of 24 cells. The state is
  better than the plain tree in all 24 and better than `'net'` in all 24; `state` or
  `state, net=True` has the lowest error in 21 of 24 cells. The network features change
  nothing on average here (ratio 1.00, lower in 9 of 24 cells): their prior width is
  chosen on held-out points, and with this much noise it is mostly chosen small.
* **Against `'additive_gp'`**, the closest existing global model, the state has 0.41 of
  its error without noise (lower in 20 of 24 cells) and 0.69 with noise (lower in 24 of
  24), at the same run time. `'additive_gp'` fits its GP to 500 reservoir points; the
  state uses the same kind of prior on every point.

### Targets with additive or pairwise structure (uniform stream, uniform-test NRMSE)

| target | noise | tree | `additive_gp` | `net` | `net`, held-out stopping | `state` | `state`, `net=True` |
|---|---|---|---|---|---|---|---|
| rosenbrock | 0 | 0.0081 | 1.30e-04 | 0.0021 | 0.0024 | **6.16e-05** | 6.32e-05 |
| rosenbrock | 0.2 | 0.0245 | 0.0134 | 0.0421 | 0.0155 | **0.0083** | 0.0103 |
| rastrigin | 0 | 0.1166 | 0.0639 | 0.1514 | 0.0981 | **8.94e-06** | 9.16e-06 |
| rastrigin | 0.2 | 0.1171 | 0.0487 | 0.1507 | 0.0972 | 0.0072 | **0.0071** |
| levy | 0 | 0.1061 | 1.00e-04 | 0.1375 | 0.0789 | **3.06e-05** | 3.10e-05 |
| levy | 0.2 | 0.1057 | 0.0163 | 0.1423 | 0.0826 | **0.0077** | 0.0079 |

Division of the prior variance among the orders at the end of the noise-free uniform
stream (`order_shares()`):

| target | additive | pairwise | full interaction |
|---|---|---|---|
| rotated_rosenbrock | 0.33 | 0.33 | 0.33 |
| gaussian_peaks | 0.95 | 0.00 | 0.05 |
| rosenbrock | 0.99 | 0.01 | 0.00 |
| rastrigin | 1.00 | 0.00 | 0.00 |
| levy | 1.00 | 0.00 | 0.00 |

On these targets the fitted prior puts 0.99 to 1.00 of its variance into the additive
term, and the sine basis learns that part from all 4000 points. `'additive_gp'` reaches
1e-4 on two of the three and 0.06 on `Rastrigin`. `'net'` is worse than the plain tree on
`Rastrigin` and `Levy`, with and without noise. The network features neither help nor
hurt here.

The shares are weights of the prior that the evidence prefers; they are not a
decomposition of the target's variance. A large additive share on a target the state
fits to 1e-5 can be taken at face value. The other two rows cannot: for
`RotatedRosenbrock`, a polynomial, all three weights end at their upper bound with length
scales longer than the cube, and the shares say nothing; `GaussianPeaks` gets 0.95
although the state alone removes only a fifth of the plain tree's error there.

### One-sigma coverage (prequential / uniform test / focus test)

Without noise

| target | stream | tree | `additive_gp` | `net` | `net`, held-out stopping | `state` | `state`, `net=True` |
|---|---|---|---|---|---|---|---|
| rotated_rosenbrock | uniform | 0.68 / 0.67 / 0.66 | 0.69 / 0.65 / 0.64 | 0.64 / 0.61 / 0.61 | 0.66 / 0.63 / 0.62 | 0.67 / 0.71 / 0.69 | 0.68 / 0.72 / 0.72 |
| rotated_rosenbrock | focusing | 0.70 / 0.30 / 0.59 | 0.73 / 0.36 / 0.68 | 0.70 / 0.08 / 0.60 | 0.70 / 0.08 / 0.57 | 0.75 / 0.48 / 0.62 | 0.75 / 0.48 / 0.75 |
| rotated_rosenbrock | sweeping | 0.66 / 0.38 / 0.64 | 0.67 / 0.42 / 0.81 | 0.67 / 0.49 / 0.76 | 0.68 / 0.63 / 0.87 | 0.68 / 0.65 / 0.77 | 0.67 / 0.69 / 0.74 |
| rotated_rosenbrock | walker | 0.66 / 0.26 / 0.77 | 0.66 / 0.27 / 0.98 | 0.68 / 0.34 / 0.97 | 0.67 / 0.35 / 0.96 | 0.66 / 0.40 / 0.75 | 0.67 / 0.44 / 0.79 |
| gaussian_peaks | uniform | 0.68 / 0.68 / 0.68 | 0.68 / 0.68 / 0.68 | 0.66 / 0.67 / 0.68 | 0.66 / 0.76 / 0.76 | 0.67 / 0.66 / 0.66 | 0.69 / 0.77 / 0.79 |
| gaussian_peaks | focusing | 0.70 / 0.60 / 0.66 | 0.74 / 0.36 / 0.67 | 0.71 / 0.17 / 0.66 | 0.71 / 0.17 / 0.75 | 0.73 / 0.51 / 0.72 | 0.76 / 0.68 / 0.75 |
| gaussian_peaks | sweeping | 0.64 / 0.47 / 0.78 | 0.67 / 0.39 / 0.76 | 0.69 / 0.60 / 0.65 | 0.67 / 0.65 / 0.71 | 0.67 / 0.62 / 0.73 | 0.66 / 0.58 / 0.77 |
| gaussian_peaks | walker | 0.68 / 0.18 / 0.73 | 0.66 / 0.13 / 0.85 | 0.66 / 0.15 / 0.77 | 0.68 / 0.15 / 0.78 | 0.68 / 0.41 / 0.80 | 0.70 / 0.54 / 0.85 |
| **mean** | | 0.67 / 0.44 / 0.69 | 0.69 / 0.41 / 0.76 | 0.68 / 0.39 / 0.71 | 0.68 / 0.43 / 0.75 | 0.69 / 0.55 / 0.72 | 0.70 / 0.61 / 0.77 |

With noise of 0.2 sd

| target | stream | tree | `additive_gp` | `net` | `net`, held-out stopping | `state` | `state`, `net=True` |
|---|---|---|---|---|---|---|---|
| rotated_rosenbrock | uniform | 0.84 / 0.85 / 0.85 | 0.88 / 0.90 / 0.89 | 0.78 / 0.88 / 0.84 | 0.91 / 0.92 / 0.91 | 0.97 / 0.98 / 0.98 | 0.98 / 0.97 / 0.97 |
| rotated_rosenbrock | focusing | 0.95 / 0.78 / 0.99 | 0.97 / 0.63 / 1.00 | 0.90 / 0.03 / 0.98 | 0.96 / 0.44 / 0.99 | 0.99 / 0.99 / 1.00 | 0.99 / 0.97 / 1.00 |
| rotated_rosenbrock | sweeping | 0.93 / 0.75 / 0.94 | 0.90 / 0.78 / 0.96 | 0.80 / 0.53 / 0.88 | 0.91 / 0.73 / 0.92 | 0.97 / 0.93 / 0.92 | 0.97 / 0.93 / 0.97 |
| rotated_rosenbrock | walker | 0.93 / 0.40 / 0.96 | 0.95 / 0.56 / 1.00 | 0.69 / 0.46 / 0.83 | 0.92 / 0.70 / 1.00 | 0.98 / 0.59 / 1.00 | 0.96 / 0.65 / 0.99 |
| gaussian_peaks | uniform | 0.83 / 0.86 / 0.84 | 0.88 / 0.91 / 0.89 | 0.77 / 0.85 / 0.84 | 0.89 / 0.92 / 0.92 | 0.92 / 0.94 / 0.93 | 0.93 / 0.97 / 0.96 |
| gaussian_peaks | focusing | 0.96 / 0.85 / 1.00 | 0.97 / 0.65 / 1.00 | 0.90 / 0.01 / 0.95 | 0.96 / 0.29 / 0.92 | 0.98 / 0.94 / 1.00 | 0.98 / 0.92 / 1.00 |
| gaussian_peaks | sweeping | 0.91 / 0.76 / 0.92 | 0.91 / 0.87 / 0.91 | 0.78 / 0.74 / 0.88 | 0.90 / 0.76 / 0.95 | 0.95 / 0.89 / 0.94 | 0.95 / 0.92 / 0.94 |
| gaussian_peaks | walker | 0.86 / 0.31 / 0.93 | 0.89 / 0.34 / 0.93 | 0.71 / 0.44 / 0.88 | 0.89 / 0.55 / 0.98 | 0.96 / 0.74 / 0.99 | 0.95 / 0.93 / 0.98 |
| **mean** | | 0.90 / 0.70 / 0.93 | 0.92 / 0.70 / 0.95 | 0.79 / 0.49 / 0.88 | 0.92 / 0.67 / 0.95 | 0.96 / 0.88 / 0.97 | 0.96 / 0.91 / 0.98 |

On the stream (first number) every configuration is calibrated by the per-leaf factor.
The second number is the test of the variance itself: on the focusing, sweeping and walker
streams the uniform test set lies mostly away from the stream. There the mean coverage
without noise is 0.37 for the plain tree, 0.32 for `'additive_gp'`, 0.30 for `'net'`, 0.51
for the state and 0.57 with network features: `error_scale` adds the same amount at every
`x`, the state's variance adds more where its coefficients are poorly determined. It is
still below 0.68 on the walker and focusing streams (0.40 to 0.51 for the state, 0.44 to
0.68 with network features), where the prior itself was fitted to a small part of the cube.

With noise all configurations except `'net'` overcover the noise-free target on the stream
(0.90 to 0.96): the per-leaf factor is fitted to errors against noisy observations. Away
from the stream the state has 0.85 to 0.89, the plain tree 0.64, `'net'` 0.37.

### Held-out stopping for the feature network

`FeatureNetLearner(holdout=0.2)`, off by default, never trains on a fifth of the points
(membership is a hash of a point's coordinates, so it is the same at every refit), checks
the loss on them every 25 L-BFGS iterations or 250 Adam steps, stops a refit after three
checks without improvement and publishes the weights with the lowest held-out loss.
`global_mean='net', global_mean_kwargs=dict(holdout=0.2)` is the column `net`, held-out
stopping in the tables above.

* With noise it has 0.48 of the error of `'net'` (geometric mean; lower in 24 of 24 cells),
  which moves the hybrid from 1.34 to 0.64 of the plain tree's error. Coverage away from
  the stream rises from 0.37 to 0.58, and the network's training time falls from 24 to 7 s
  per stream. A refit continues from the current weights; without held-out points the
  early fits to a few hundred noisy points are carried through the stream.
* Without noise the two are level on average (ratio 1.02) with a spread from 0.24 to 1.66
  over the cells; held-out stopping is lower in 8 of 24. The network then gains from
  training on every point to the end of its budget.

The default therefore stays off. `StateGlobalMean(net=True)` turns it on for its own
network, which it needs in any case: the held-out points are also the ones on which the
prior width of the network features is chosen.

### Cost

| | seconds per 4,000-point stream (mean over the 8 streams) | of which prior fits | state solves | network |
|---|---|---|---|---|
| tree | 21 |  |  |  |
| `additive_gp` | 134 |  |  |  |
| `net` | 56 |  |  | 24 |
| `net`, held-out stopping | 51 |  |  | 17 |
| `state` | 138 | 96 | 3 | 0 |
| `state`, `net=True` | 165 | 95 | 4 | 20 |

The runs shared four cores four at a time, so the absolute times are somewhat high. Two
thirds of the state's time is the fit of the prior: one evaluation of the likelihood
factorises a 1000 x 1000 matrix, with up to 60 evaluations per start, at up to seven fits
per stream (60, 120, ..., 3840 points). The cost of a fit does not grow with the stream
and the number of fits grows with its logarithm. The 25 solves of the coefficients take 3
to 4 s together; the rest is the tree, the leaf refreshes and the features at prediction.
With noise the network of `net=True` stops early and costs 6 s instead of 20.

## Limits

* Single output. `StateGlobalMean` raises on a multi-output tree.
* One seed and 4000 points per cell. There are no runs at 40 000 points, where the
  reference numbers of the hybrid are, and none in more than 6 dimensions.
* Six to eight times the run time of the plain tree at this stream length, most of it in
  the prior fits.
* 500 kernel centres. Structure above order one that is finer than their spacing is left
  to the leaves, and with `net=True` to the network: on `GaussianPeaks` the state alone
  gains little over the plain tree (0.76 to 0.84 of its error on the uniform stream).
* The shares of `order_shares()` are not a variance decomposition (see above).
* Away from a narrow stream the coverage is better than with `error_scale` but still below
  nominal.
* Inputs outside the range explored so far are clipped to the basis interval: beyond its
  margin the additive part is constant until the next solve.
* The cheap leaf refresh and the leaf weights in the variance need `refit_fixed` and
  `mean_weights` from the GP backend. The scikit-learn adapter has both; with a backend
  that has neither, leaves do a full refit at every new state and the variance uses
  `phi(x)` in place of `q(x)`, which is larger.

## Reproduce

```bash
cd examples
for noise in 0 0.2; do
  for target in rotated_rosenbrock gaussian_peaks; do
    OMP_NUM_THREADS=1 python benchmark_global_mean_streams.py --target $target --calibrate --noise $noise \
        --streams uniform,focusing,sweeping,walker --configs tree,global_pkg,hybrid,hybrid_holdout,state,state_net
  done
  for target in rosenbrock rastrigin levy; do
    OMP_NUM_THREADS=1 python benchmark_global_mean_streams.py --target $target --calibrate --noise $noise \
        --streams uniform --configs tree,global_pkg,hybrid,hybrid_holdout,state,state_net
  done
done
python benchmark_global_mean_streams.py --summarize results/state_mean/streams_N4000_seed1.jsonl
```

The `RESULT` lines of the runs reported here are in `results/state_mean/streams_N4000_seed1.jsonl`.
