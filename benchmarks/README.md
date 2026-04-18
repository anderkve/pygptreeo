# pygptreeo continual-emulation benchmark

This directory contains a benchmark comparing `pygptreeo` against four strong
alternative Python packages/algorithms on a suite of continual regression
problems. The same online interface is used for every method.

## Methods under test

| Key              | Description                                                                                                                                                        |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `pygptreeo`      | Dynamically growing tree of local GPs from this package (`GPTree`, `Nbar=200`, gradual splitting, MoE aggregation, calibrated sigma, sklearn GP at the leaves).    |
| `sklearn_gp`     | Single global `sklearn.gaussian_process.GaussianProcessRegressor` with a Matern-1.5 kernel, refit every 400 new points. Training set capped at 800 points.          |
| `gpytorch_svgp`  | Streaming sparse variational GP in GPyTorch with 64 inducing points and a Matern-1.5 ARD kernel. Re-trained every 250 new points with 25 SVI epochs.                |
| `random_forest`  | `sklearn.ensemble.RandomForestRegressor` (100 trees) refit every 250 points. Predictive std is the across-tree standard deviation.                                  |
| `river_knn`      | `river.neighbors.KNNRegressor` (true online, 8 neighbours, sliding window of 4000). Predictive std is the std of the k nearest-neighbour target values.            |

## Test problems

All problems are sampled i.i.d. uniformly in `[0, 1]^d` and internally mapped to their natural domains. Every run uses the same test set (drawn from the same rng as the stream but *after* the stream).

| Problem           | Dim | Character                           |
| ----------------- | --- | ----------------------------------- |
| `smooth_sines_2d` | 2   | Smooth analytic; GPs should excel   |
| `rosenbrock_2d`   | 2   | Curved valley, moderate non-linearity |
| `step_3d`         | 3   | Quadratic bowl + sharp step (discontinuity), tests locality |

Two more problems (`eggholder_2d`, `rastrigin_3d`) are defined in `problems.py`
but not used in the default run to keep the total wall-clock time reasonable.

## Metrics

At every checkpoint (every 200 points), each method is evaluated on a held-out
test set of 400 points. We record:

* **NRMSE**: RMSE divided by the range of the true test targets.
* **MAE**: mean absolute error.
* **NLPD**: mean negative log predictive density under the Gaussian predictive
  the method reports.
* **1-sigma coverage**: fraction of test points for which `|y_true - mean| <= std`.
* **Cumulative update time**: wall-clock spent inside `update(x, y)`.
* **Cumulative predict time**: wall-clock spent inside `predict(X_test)`.

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
