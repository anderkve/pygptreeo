# Iteration 00 — Baseline investigation (pre-review)

This directory captures the state of the benchmark **as it was** when we
started iterating. It contains the findings from the NLPD bug investigation
that motivate iteration 01.

## NLPD bug investigation (the question that kicked this off)

pygptreeo's saved NLPD values exhibit huge spikes on `rosenbrock_2d`
(O(10^12)) and `step_3d` (O(10^5)). I replayed a rosenbrock_2d run with a
per-point diagnostic and found:

| step | # test pts with std=0 | max single-point NLPD | max test-point std |
| ---- | --------------------- | --------------------- | ------------------ |
|  400 | 0                     | 1.97e+1               | 8.7e-1             |
|  600 | 0                     | 7.64e+0               | 2.3e-1             |
|  800 | **109 / 400**         | nan (inf after clip)  | 2.4e-1             |
| 1000 | **109 / 400**         | nan                   | 7.1e+2             |
| 1400 | 52 / 400              | nan                   | 1.7e+3             |
| 1600 | 0                     | 85                    | **3.29e+36**       |
| 1800 | 0                     | 88                    | **7.18e+37**       |
| 2000 | 33 / 400              | nan                   | 7.18e+37           |

**Two distinct failure modes in `GPTree.predict_recursive` / `GPNode.predict`:**

1. **Catastrophic cancellation in the MoE posterior variance.** In
   `gptree.py` around line 446 we have
   ```python
   var_DLGP[i, :] += ptilde * (sigma_leaf[0, :]**2 + mu_leaf[0, :]**2)
   var_DLGP[i, :] += -mean_DLGP[i, :]**2
   ```
   For rosenbrock, `mu_leaf` can be ~3×10^3, so `mu_leaf**2` is ~10^7 while
   the true cross-leaf variance is ~10^-3. In float64 the subtraction
   underflows to zero (or even a small negative number that we take the
   sqrt of — producing NaN or zero). This is the standard
   "naive variance formula" failure.

2. **`use_calibrated_sigma` can produce astronomic std values.** Once
   `update_sigma_scaler` converges on a large scaler for a leaf whose
   empirical residuals are much larger than its raw GP std, the output
   std gets multiplied up. We observed std ≈ 1×10^37 on a few test points
   — clearly non-physical. `gpnode.py` does cap `sigma_scaler_inits` at
   1e6 inside the bracketing loop but `sigma_scalers[i] * sigma_pred` can
   still explode when the raw sigma is also large after the y-space
   inverse transform.

**Implication for the benchmark (not for pygptreeo as a library):** our
harness's NLPD calculation clips std to 1e-8, so std=0 cases get `(err /
1e-8)**2` ≈ 10^12 per point, which is what we saw in the saved numbers.
The single-number mean NLPD is therefore mostly measuring *how bad the
worst few test points are* rather than the typical calibration.

## Baseline benchmark setup that produced those numbers

* 5 methods: `pygptreeo`, `sklearn_gp` (periodic refit), `gpytorch_svgp`,
  `random_forest` (periodic refit), `river_knn`.
* 3 problems: `smooth_sines_2d`, `rosenbrock_2d`, `step_3d`. All sampled
  i.i.d. from U[0,1]^d.
* `n_stream = 2000`, `checkpoint_every = 200`, `n_test = 400`.
* Metrics: RMSE, NRMSE, MAE, NLPD, 1-sigma coverage, cumulative update/predict time.
* Single seed (seed=0).

## Issues that motivate the first formal review

Beyond the NLPD bug, an honest look at the baseline shows several places
where the comparison is not yet paper-quality. These are deliberately left
for the iteration-01 reviewer to articulate:

* Is the per-method configuration fair? (e.g. we capped sklearn_gp's
  training set at 800 points but let pygptreeo use everything.)
* Are the problems diverse enough / representative of "expensive target
  functions" that a real emulation paper would use?
* Is a single seed enough to distinguish methods?
* Are the metrics the right ones for *continual emulation*?
* Is the plotting honest? (The mean NLPD blow-up plotted on symlog
  actually *hides* the real bug.)
