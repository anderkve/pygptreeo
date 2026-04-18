# Iteration 04 review — algorithm settings critique

*Written by the critical reviewer. Reference: iteration_03 summary + figures.*

## Summary of asymmetries and inappropriate settings in iter-03 run

- **pygptreeo gets a strictly richer kernel** than all other GP-based methods: `Matern(1.5) + AnisotropicRationalQuadratic` with `n_restarts=1` per leaf, whereas `sklearn_gp` and `gpytorch_svgp` are Matern-1.5-only. Apples-to-oranges; a reviewer will call it out.
- **`sklearn_gp` is crippled by the 400-point cap** on 2-D problems where 3000 points are available. The current 31.5× NRMSE gap is largely a compute-budget artefact, not a modelling one.
- **SVGP is likely under-trained**: `n_epochs=60`, `max_steps_per_refit=500`, `n_inducing=256`. We publish "SVGP is 82× worse" while under-compute-ing it.
- **pygptreeo's retrain cadence (200) exactly equals its `Nbar`**. This coupling is not justified and makes pygptreeo fit only between splits.
- **Calibration saturation at 1.00** for batch methods at nominal 0.95 may be a std-floor artefact or genuine over-confidence from small-reservoir refits. Orthogonal to settings, but flagged.

## Per-method settings critique

### pygptreeo
`Nbar=200`, `retrain_every_n_points=200`, `Matern+RQ`, `n_restarts=1`, `use_calibrated_sigma=True`, `aggregation="moe"`, `max_n_pred_leaves=3`. The richer kernel is the biggest exposure. `Nbar=200` is defensible but unmotivated — no reading at smaller `Nbar` where locality should help more on rosenbrock's curved valley.

### sklearn_gp
`max_train_points=400` (d≤5) or `250` (d≥6), `n_restarts_optimizer=0`, `normalize_y=True`, Matern-1.5. On 2-D with 3000 stream points the 400-cap leaves 87 % of data unused. We need a "best-case exact GP" reading.

### gpytorch_svgp
`n_inducing=256`, `n_epochs=60`, `max_steps_per_refit=500`, `lr=5e-3`, `max_buffer=5000`, Matern-ARD. At 500 SVI steps on 5-D ARD lengthscales the hyperparameters have barely moved from init. Inducing-point init is random (code contradicts its own docstring claim of "k-means-like").

### random_forest
`n_estimators=300` — sensible. Variance-across-trees is a structural weakness not fixable by a hyperparameter variant. Skip.

### river_knn
`k=8`, `window_size=4000`. Window ≥ n_stream=3000, so the window never actually slides on iid — effectively full-history kNN. A smaller, more local variant is worth one slot.

## Proposed variants — P0 (must land)

Total 10 configs; 5 baselines (`-A`) + 5 stress/ablation (`-B` or `pygptreeo-C`):

| Variant | Changes vs `run_all.py` baseline |
| --- | --- |
| `pygptreeo_A` | baseline (`Nbar=200`, `retrain_step=200`, `Matern+RQ`). |
| `pygptreeo_B` | `Nbar=100`, `retrain_step=100`. Kernel unchanged. |
| `pygptreeo_C` | baseline `Nbar=200`, kernel = Matern-1.5 only (drop RQ). Apples-to-apples with sklearn_gp's kernel. |
| `sklearn_gp_A` | baseline (`max_train=400/250`, `n_restarts_optimizer=0`). |
| `sklearn_gp_B` | `max_train_points=1200` on d≤2, `600` on d=5, `n_restarts_optimizer=1`. `--max-wall-time 600`. Skip borehole_8d. |
| `svgp_A` | baseline. |
| `svgp_B` | `n_inducing=512`, `n_epochs=120`, `max_steps_per_refit=1500`. `--max-wall-time 300`. |
| `rf_A` | baseline (`n_estimators=300`). No `-B`. |
| `river_knn_A` | baseline. |
| `river_knn_B` | `n_neighbors=3`, `window_size=1000`. |

Implement by adding a second entry per method in `METHODS` (`pygptreeo_A`, `pygptreeo_B`, `pygptreeo_C`, `sklearn_gp_A`, …). Keep the original factory names as aliases for the `-A` variants so existing data is not invalidated.

## P0 — runs to execute (~40 min)

```
# 1. pygptreeo sensitivity: Nbar & kernel ablation (~10 min)
python benchmarks/run_all.py --methods pygptreeo_B pygptreeo_C \
    --problems rosenbrock_2d friedman1_5d --seeds 0 1 2 \
    --max-wall-time 120

# 2. sklearn_gp ceiling (~12 min)
python benchmarks/run_all.py --methods sklearn_gp_B \
    --problems rosenbrock_2d smooth_sines_2d --seeds 0 1 \
    --max-wall-time 600

# 3. SVGP heavy (~15 min)
python benchmarks/run_all.py --methods svgp_B \
    --problems rosenbrock_2d friedman1_5d --seeds 0 1 \
    --max-wall-time 300

# 4. river_knn local (~2 min)
python benchmarks/run_all.py --methods river_knn_B \
    --problems rosenbrock_2d friedman1_5d smooth_sines_2d --seeds 0 1 2 \
    --max-wall-time 60
```

Baselines `-A` are already on disk; no re-run needed.

## P1 — pending holes to close (defer if over budget)

- `pygptreeo_A` and `svgp_A` on `borehole_8d` seeds 0,1,2.
- Full shift sweep for `svgp_B` and `pygptreeo_B` on rosenbrock_2d.
- 3rd seed for the `-B` sweeps above.

## Out-of-scope

- Do NOT modify `pygptreeo/gptree.py`, `pygptreeo/gpnode.py`, `benchmarks/harness.py`, or `benchmarks/adapters/base.py`.
- Do NOT add a new problem.
- Do NOT add a new method.
- Do NOT change the std floor or the cap.
- Do NOT re-tune `theta`, `sigma_rel`, `splitting_strategy`, `aggregation`, or `max_n_pred_leaves`.
- Do NOT introduce a third variant beyond `-A`/`-B`/`pygptreeo-C`.
- Do NOT re-run the existing 55 iid `.npz` files.
- Do NOT raise `--n-stream` or `--n-test`.
- Do NOT change plotting panel layout beyond adding variant labels.

## Acceptance criteria

- 10 variant configs registered in `run_all.py`'s `METHODS` dict with per-variant doc comments; `benchmarks/README.md` method-budget table updated.
- ≥ 24 new `.npz` files on disk (target: 30+ from the runs above).
- `iteration_04/comparison.png` and `wilcoxon_per_problem.png` show the variants overlaid.
- `iteration_04/summary.md` reports, per (variant, problem): median NRMSE, ratio vs `pygptreeo_A`, wall-time, NLPD-sanity warnings.
- `pygptreeo_C` result is called out explicitly: if it still beats `sklearn_gp_B`, the kernel criticism is defused; if it loses, summary.md must document honestly.
- `pygptreeo_B` result establishes whether `Nbar=200` is near-optimal or clearly wrong.
- Every new pygptreeo_* run has `frac_pathological_std[-1] == 0.0` and zero NLPD sanity warnings.
- Total iter-04 runtime < 60 min.
