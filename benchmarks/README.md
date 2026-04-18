# pygptreeo continual-emulation benchmark

This directory contains a benchmark comparing `pygptreeo` against four strong
alternative Python packages/algorithms on a suite of continual regression
problems. The same online interface is used for every method.

## Methods under test and their compute budget (iteration 01)

All methods run on a single CPU thread (`OMP_NUM_THREADS=1`). The budgets
below were chosen by the reviewer to give every method its honest best shot
on a ~3000-point stream without any method blowing past ~5 minutes per run.

| Key              | Description                                                                                          | Capacity knob                             | Refit / update cadence                |
| ---------------- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------- |
| `pygptreeo`      | Dynamically growing tree of local GPs (Matern-1.5 + anisotropic RQ, MoE aggregation, calibrated σ).  | `Nbar=200` points per leaf                | Online per-point + node retrain / 200 |
| `sklearn_gp`     | Global `GaussianProcessRegressor` (Matern-1.5), refit on a **uniform reservoir** over stream history.| `max_train_points=1500`                   | Refit every 200 stream points         |
| `gpytorch_svgp`  | Streaming sparse variational GP (Matern-1.5 ARD) with inducing points learned by SVI.               | `n_inducing=256`, `max_buffer=5000`        | Refit every 200 stream points, 60 SVI epochs @ lr=5e-3 |
| `random_forest`  | `RandomForestRegressor` with 300 trees; predictive σ = across-tree std.                             | `n_estimators=300`, `max_train_points=20000` | Refit every 200 stream points        |
| `river_knn`      | True online k-NN (`KNNRegressor`, k=8, sliding window 4000); σ = std of k-NN target values.         | window 4000                                | Per-point online, no refit            |

## Test problems

All inputs are in `[0, 1]^d` and internally mapped to each problem's natural
domain. The default iteration-01 problem set (`run_all.py` default):

| Problem           | Dim | Character                                                                    |
| ----------------- | --- | ---------------------------------------------------------------------------- |
| `smooth_sines_2d` | 2   | Smooth analytic; GPs should excel (sanity check).                            |
| `rosenbrock_2d`   | 2   | Curved valley, highly non-linear, output range O(10^3).                      |
| `friedman1_5d`    | 5   | Friedman-1 (10 sin(π x0 x1) + 20 (x2-0.5)^2 + 10 x3 + 5 x4). Tests ARD.      |
| `borehole_8d`     | 8   | Classic 8-D water-flow emulator. Output range O(100). Medium dim, smooth.    |

Kept in `problems.py` but **not** in the default set:
`piston_7d` (smooth 7-D physical simulator),
`eggholder_2d` (adversarially multi-modal),
`rastrigin_3d` (very many local minima),
`step_3d` (intentional discontinuity — a known-pathological case for smooth-kernel methods; useful only as a diagnostic).

The stream uses **independent RNGs** for the stream (`seed`) and the test set
(`seed + 10_000`), so test-set points cannot correlate with stream draws.

**Schedule options** (`--schedules` in `run_all.py`):
- `iid` (default): uniform U[0, 1]^d.
- `shift`: first half of the stream from U[0, 0.5]^d, second half from U[0.5, 1]^d — a covariate-shift stress test.
- `sobol`: scrambled Sobol low-discrepancy sequence.

## Metrics (iteration 01)

At every checkpoint the method is evaluated on the held-out test set
(default `n_test=1000`). All metrics are computed with a physically motivated
std floor of `1e-6 * (y_test_range)` and cap of `1e3 * y_test_range` so that
a single catastrophic std=0 prediction cannot destroy a mean NLPD.

Accuracy: `rmse`, `nrmse`, `mae`.

Probabilistic quality: `nlpd` (mean), `median_nlpd`, `nlpd_trimmed` (mean
after trimming the 5/95th percentiles of per-point NLPD), and `crps`
(closed-form CRPS for Gaussian predictives).

Calibration: `coverage_50`, `coverage_1sigma` (nominal 0.6827), `coverage_2sigma`
(0.9545), `coverage_90`, `coverage_95`.

Uncertainty hygiene: `frac_pathological_std` — fraction of test points whose
predicted σ is non-finite, ≤ 0, or above the cap. Reported separately rather
than silently hidden by the floor.

Compute: `cum_update_time`, `cum_predict_time` (wall-clock seconds).

## Reproducing

```bash
# One-shot: run everything and plot.
OMP_NUM_THREADS=1 python benchmarks/run_all.py
python benchmarks/make_plots.py

# Subset:
OMP_NUM_THREADS=1 python benchmarks/run_all.py \
    --methods pygptreeo gpytorch_svgp \
    --problems rosenbrock_2d \
    --seeds 0 1 \
    --n-stream 2000
```

Results are cached in `benchmarks/data/*.npz` and re-runs skip existing files
(use `--force` to recompute). Plots are written to `benchmarks/plots/`.

## Files

```
benchmarks/
├── adapters/                 # OnlineRegressor wrappers for each package
│   ├── base.py
│   ├── pygptreeo_adapter.py
│   ├── sklearn_gp_adapter.py
│   ├── gpytorch_svgp_adapter.py
│   ├── rf_adapter.py
│   └── river_knn_adapter.py
├── harness.py                # Streaming loop, metrics, result saving
├── problems.py               # Target functions and Problem dataclass
├── run_all.py                # Main driver
├── make_plots.py             # Reads data/, writes plots/
├── smoke_test.py             # Quick adapter sanity check
├── data/                     # Saved .npz results (one per run)
└── plots/                    # Generated PNG figures
```
