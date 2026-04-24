# Higher-precision MMD² with seed std and pairwise differences

MMD² is the median-heuristic-RBF unbiased estimator on a 2000-point sub-sample of each chain (post burn-in 10 %). Reported as mean ± std across seeds at four significant figures. Pairwise differences are seed-paired (same seed compared across methods).

## Per-cell mean ± std

| problem | method | kind | τ_σ | n_seeds | MMD² mean | MMD² std |
|---|---|---|---|---|---|---|
| rosenbrock_2d | pygptreeo_A | assisted | 3e-03 | 3 | 0.004247 | 0.001573 |
| rosenbrock_2d | pygptreeo_A | assisted | 1e-02 | 3 | 0.007224 | 0.007562 |
| rosenbrock_2d | pygptreeo_A | assisted | 3e-02 | 3 | 0.01676 | 0.01232 |
| rosenbrock_2d | pygptreeo_A | assisted | 1e-01 | 3 | 0.01949 | 0.002426 |
| rosenbrock_2d | pygptreeo_A | delayed | — | 3 | 0.007464 | 0.003523 |
| rosenbrock_2d | pygptreeo_D | assisted | 3e-03 | 3 | 0.003222 | 0.001878 |
| rosenbrock_2d | pygptreeo_D | assisted | 1e-02 | 3 | 0.005768 | 0.001002 |
| rosenbrock_2d | pygptreeo_D | assisted | 3e-02 | 3 | 0.01344 | 0.01265 |
| rosenbrock_2d | pygptreeo_D | assisted | 1e-01 | 3 | 0.01949 | 0.002426 |
| rosenbrock_2d | pygptreeo_D | delayed | — | 3 | 0.007254 | 0.004338 |
| rosenbrock_2d | gpytorch_svgp_A | assisted | 3e-03 | 3 | 0.002911 | 0.0018 |
| rosenbrock_2d | gpytorch_svgp_A | assisted | 1e-02 | 3 | 0.005149 | 0.00356 |
| rosenbrock_2d | gpytorch_svgp_A | assisted | 3e-02 | 3 | 0.003018 | 8.244e-05 |
| rosenbrock_2d | gpytorch_svgp_A | assisted | 1e-01 | 3 | 0.005182 | 0.004665 |
| rosenbrock_2d | gpytorch_svgp_A | delayed | — | 3 | 0.004126 | 0.003153 |
| borehole_8d | pygptreeo_A | assisted | 3e-03 | 3 | 0.009586 | 0.003643 |
| borehole_8d | pygptreeo_A | assisted | 1e-02 | 3 | 0.007142 | 0.001922 |
| borehole_8d | pygptreeo_A | assisted | 3e-02 | 3 | 0.04309 | 0.04956 |
| borehole_8d | pygptreeo_A | assisted | 1e-01 | 3 | 0.09611 | 0.01354 |
| borehole_8d | pygptreeo_A | delayed | — | 3 | 0.004359 | 0.001235 |
| borehole_8d | pygptreeo_D | assisted | 3e-03 | 3 | 0.008747 | 0.002435 |
| borehole_8d | pygptreeo_D | assisted | 1e-02 | 3 | 0.008125 | 0.001994 |
| borehole_8d | pygptreeo_D | assisted | 3e-02 | 3 | 0.04407 | 0.04885 |
| borehole_8d | pygptreeo_D | assisted | 1e-01 | 3 | 0.09611 | 0.01354 |
| borehole_8d | pygptreeo_D | delayed | — | 3 | 0.005733 | 0.001248 |
| borehole_8d | gpytorch_svgp_A | assisted | 3e-03 | 3 | 0.009245 | 0.002698 |
| borehole_8d | gpytorch_svgp_A | assisted | 1e-02 | 3 | 0.006563 | 0.0008288 |
| borehole_8d | gpytorch_svgp_A | assisted | 3e-02 | 3 | 0.006198 | 0.003013 |
| borehole_8d | gpytorch_svgp_A | assisted | 1e-01 | 3 | 0.02174 | 0.005055 |
| borehole_8d | gpytorch_svgp_A | delayed | — | 3 | 0.008218 | 0.002352 |
| banana_2d | pygptreeo_A | assisted | 3e-03 | 1 | 0.0007882 | 0 |
| banana_2d | pygptreeo_A | assisted | 1e-02 | 1 | 0.0007882 | 0 |
| banana_2d | pygptreeo_D | assisted | 3e-03 | 1 | 0.001813 | 0 |
| banana_2d | pygptreeo_D | assisted | 1e-02 | 1 | 0.0001401 | 0 |
| banana_2d | gpytorch_svgp_A | assisted | 3e-03 | 1 | 0.002282 | 0 |
| banana_2d | gpytorch_svgp_A | assisted | 1e-02 | 1 | 0.001784 | 0 |
| banana_5d | pygptreeo_A | assisted | 3e-03 | 1 | 0.0122 | 0 |
| banana_5d | pygptreeo_A | assisted | 1e-02 | 1 | 0.02279 | 0 |
| banana_5d | pygptreeo_D | assisted | 3e-03 | 1 | 0.003612 | 0 |
| banana_5d | pygptreeo_D | assisted | 1e-02 | 1 | 0.02279 | 0 |
| banana_5d | gpytorch_svgp_A | assisted | 3e-03 | 1 | 0.004325 | 0 |
| banana_5d | gpytorch_svgp_A | assisted | 1e-02 | 1 | 0.007148 | 0 |

## Seed-paired pairwise MMD² differences (vs pygptreeo_D) — assisted only

Sign convention: positive = the other method has a *higher* MMD² (worse fidelity) than pygptreeo_D. Each row averages over the seeds where both methods have a value at that (problem, τ_σ) cell.

| problem | method | τ_σ | n_pairs | Δ MMD² mean | Δ MMD² std |
|---|---|---|---|---|---|
| rosenbrock_2d | pygptreeo_A | 3e-03 | 3 | +0.001025 | 0.003401 |
| rosenbrock_2d | pygptreeo_A | 1e-02 | 3 | +0.001456 | 0.008414 |
| rosenbrock_2d | pygptreeo_A | 3e-02 | 3 | +0.003324 | 0.008209 |
| rosenbrock_2d | pygptreeo_A | 1e-01 | 3 | +0 | 0 |
| rosenbrock_2d | gpytorch_svgp_A | 3e-03 | 3 | -0.0003112 | 0.00235 |
| rosenbrock_2d | gpytorch_svgp_A | 1e-02 | 3 | -0.0006186 | 0.00442 |
| rosenbrock_2d | gpytorch_svgp_A | 3e-02 | 3 | -0.01042 | 0.01259 |
| rosenbrock_2d | gpytorch_svgp_A | 1e-01 | 3 | -0.01431 | 0.003248 |
| borehole_8d | pygptreeo_A | 3e-03 | 3 | +0.0008392 | 0.001856 |
| borehole_8d | pygptreeo_A | 1e-02 | 3 | -0.0009831 | 0.001366 |
| borehole_8d | pygptreeo_A | 3e-02 | 3 | -0.0009808 | 0.001521 |
| borehole_8d | pygptreeo_A | 1e-01 | 3 | +0 | 0 |
| borehole_8d | gpytorch_svgp_A | 3e-03 | 3 | +0.000498 | 0.004933 |
| borehole_8d | gpytorch_svgp_A | 1e-02 | 3 | -0.001562 | 0.00282 |
| borehole_8d | gpytorch_svgp_A | 3e-02 | 3 | -0.03788 | 0.04586 |
| borehole_8d | gpytorch_svgp_A | 1e-01 | 3 | -0.07437 | 0.009389 |
| banana_2d | pygptreeo_A | 3e-03 | 1 | -0.001024 | 0 |
| banana_2d | pygptreeo_A | 1e-02 | 1 | +0.0006481 | 0 |
| banana_2d | gpytorch_svgp_A | 3e-03 | 1 | +0.0004691 | 0 |
| banana_2d | gpytorch_svgp_A | 1e-02 | 1 | +0.001644 | 0 |
| banana_5d | pygptreeo_A | 3e-03 | 1 | +0.008587 | 0 |
| banana_5d | pygptreeo_A | 1e-02 | 1 | +0 | 0 |
| banana_5d | gpytorch_svgp_A | 3e-03 | 1 | +0.0007126 | 0 |
| banana_5d | gpytorch_svgp_A | 1e-02 | 1 | -0.01564 | 0 |
