# Iteration 18 review — Local-approximate GP (laGP-style) comparator

## Goal

Close the single referee-2 item that requires new code: a streaming-
GP comparator on the same problems as iter 12. The candidates the
referee names (`GPvecchia`, NNGP, `laGP`) are all reasonable, but
the lowest-effort one to implement in pure-Python without a new
dependency is a **local-approximate GP** in the spirit of Gramacy &
Apley 2015 (laGP): for each prediction `x*`, find its `k` nearest
training neighbours and fit a fresh exact GP on those `k` points
only. We use `k = 200` to match the per-leaf budget of `pygptreeo
(A)`. This is a methodologically distinct alternative to
`pygptreeo`'s tree-of-GPs because the partition is implicit
(per-query) rather than learned, and it lets the reader judge whether
`pygptreeo`'s headline win is "tree-of-GPs vs single-fit GP" or
"tree-of-GPs vs all local-GP approaches".

## Plan

1. **`benchmarks/adapters/lagp_adapter.py`** — new
   `LocalApproxGPAdapter` exposing the standard `update` /
   `predict` interface:

   - `update(x, y)` appends to an internal buffer (no model
     training).
   - `predict(X)` for each row of `X`, finds the `k = 200` nearest
     buffered points (Euclidean), fits a fresh `sklearn`
     `GaussianProcessRegressor` (Matérn-1.5, no restarts), returns
     `(mean, std)`. Uses `sklearn.neighbors.KDTree` for the nearest-
     neighbour query so the per-query cost is `O(k³ + log n)`
     rather than `O(n²)`.
   - `supports_uncertainty = True`. `name = "lagp"`.
   - The "training" wall time charged is the buffer append (≈ 0).
     The "predict" wall time is dominated by the per-query GP fit
     and is reported separately, as in iter 16.

2. **`benchmarks/run_all.py`** — register the adapter as
   `_make_lagp_A` (k = 200) and `_make_lagp_B` (k = 100, the
   `pygptreeo (D)` analogue). Add to `METHODS`.

3. **Sweeps**.
   - **Static-stream**: `lagp_A` and `lagp_B` × `rosenbrock_2d` and
     `borehole_8d` × `de` and `mcmc` × seeds 0–2. n_stream = 4000,
     `--de-popsize 300`. **24 runs**.
   - **Trust-threshold deployment**: `lagp_A` × the same problems
     × `iid` and `mcmc` × τ_σ ∈ {1e-3, 3e-3, 1e-2, 3e-2}. Single
     seed (matches iter-13 protocol). **16 runs**.
   - We do *not* run the assisted-MCMC sweep on laGP — its per-
     query wall time at 20 000 chain steps would dominate the
     budget and provide little new headline beyond the iter-15 DA
     baseline.

4. **`make_plots.py`** — register `lagp_A` and `lagp_B` with
   colour, label, marker so the existing comparison panels include
   them automatically.

5. **`summary.md` requirements**.
   - Reliability invariant unchanged (laGP is a non-pygptreeo
     baseline; the invariant only applies to pygptreeo*).
   - Static-stream NRMSE table extended to include laGP rows
     alongside the iter-17 multi-seed pygptreeo / sklearn / SVGP
     numbers.
   - Trust-threshold table extended with laGP at the same τ_σ grid
     and an explicit "speedup vs predict-time" caveat: laGP's
     `update` is ~free but `predict` is expensive, so its trust-
     threshold "speedup" must be qualified.
   - One paragraph contrasting pygptreeo's *learned* partition
     with laGP's *per-query* partition: which approach wins when.

## Out-of-scope

- NNGP / GPvecchia: too much new code for one iter; documented as
  future work.
- Realistic global-fit posterior: future work.
- Adaptive MCMC, NN surrogates, acquisition baselines: future work.

## Acceptance criteria

- `benchmarks/adapters/lagp_adapter.py` exists and passes a smoke
  test on `rosenbrock_2d` at n=200.
- `_make_lagp_{A,B}` registered in `run_all.py`.
- `iteration_18/data/` contains 24 + 16 = 40 new `.npz`.
- `iteration_18/summary.md` extends the iter-17 NRMSE table with
  laGP rows and reports the trust-threshold sweep.
- `iteration_18/plots/` contains a regenerated `pareto.png` that
  includes the laGP markers and a regenerated `trust_speedup.png`
  with laGP curves.
- Sweep wall time < 90 min (laGP predict cost is the bottleneck;
  target 16 trust runs at ~ 4 min each = 64 min plus 24 static at
  ~ 1 min each).
