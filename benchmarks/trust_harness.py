"""Trust-threshold deployment harness.

Mirrors ``benchmarks.harness.run_online_benchmark`` but introduces the
deployment loop the package was actually built for: at each stream step
the emulator is queried for ``(μ, σ)``; if ``σ ≤ τ_σ`` the emulator's
prediction is *trusted* and the (expensive) true function is **not**
called — and so no training point is added — otherwise the truth is
fetched and the emulator is updated on it.

The truth is still computed at trusted steps, but **only** for
diagnostic accounting (so we can report the would-have-been error of
trusted predictions); it never enters the model. In a production
deployment that diagnostic call would be removed and the recorded
``n_trained / n_stream`` ratio is the directly-quantifiable speedup.

Per-step we accumulate, in 1000-step batches:

- ``n_trained_in_batch``, ``n_trusted_in_batch``
- ``trusted_err`` distribution (mean / median / p90 of ``|μ - f|``)
- ``frac_trusted_within_tau_y_grid`` for a fixed grid of absolute
  target-scale tolerances ``τ_y``

We also keep the standard checkpoint metrics on a held-out test set,
exactly as ``run_online_benchmark`` does, so the reader has a familiar
yardstick for "is the emulator any good".
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, List

import numpy as np

from benchmarks.harness import (  # type: ignore
    RunConfig as _BaseRunConfig,
    _metrics as _checkpoint_metrics,
    _seed_all,
)


# Default absolute target-scale tolerances at which we report
# `frac_trusted_within_tau_y`. Three orders of magnitude is enough to
# build a quality curve in the summary plot.
DEFAULT_TAU_Y_GRID = (1e-4, 1e-3, 1e-2, 1e-1)


@dataclass
class TrustRunConfig:
    method_name: str
    problem_name: str
    seed: int
    n_stream: int
    n_test: int
    checkpoint_every: int
    schedule: str
    tau_sigma: float           # the σ-trust threshold (relative to y-range)
    batch_size: int            # per-batch reporting bin (1000 is the default)
    tau_y_grid: tuple          # absolute target-scale tolerances
    max_wall_time_s: float = 3600.0


@dataclass
class TrustRunResult:
    config: TrustRunConfig

    # Per-checkpoint emulator quality on the held-out test set.
    checkpoints: List[int] = field(default_factory=list)
    nrmse: List[float] = field(default_factory=list)
    rmse: List[float] = field(default_factory=list)
    mae: List[float] = field(default_factory=list)
    nlpd: List[float] = field(default_factory=list)
    median_nlpd: List[float] = field(default_factory=list)
    coverage_1sigma: List[float] = field(default_factory=list)
    coverage_2sigma: List[float] = field(default_factory=list)
    crps: List[float] = field(default_factory=list)
    frac_pathological_std: List[float] = field(default_factory=list)

    # Per-batch deployment statistics.
    batch_index: List[int] = field(default_factory=list)
    batch_n_trained: List[int] = field(default_factory=list)
    batch_n_trusted: List[int] = field(default_factory=list)
    batch_trusted_err_med: List[float] = field(default_factory=list)
    batch_trusted_err_p90: List[float] = field(default_factory=list)
    # Each row of frac_within_tau_y is len(tau_y_grid) wide.
    batch_frac_within_tau_y: List[List[float]] = field(default_factory=list)

    # Cumulative tallies (final = total speedup).
    cum_n_trained: List[int] = field(default_factory=list)
    cum_n_trusted: List[int] = field(default_factory=list)

    cum_update_time: List[float] = field(default_factory=list)
    cum_predict_time: List[float] = field(default_factory=list)

    wall_time: float = 0.0
    aborted: bool = False
    notes: str = ""

    def to_npz_dict(self) -> dict:
        d = asdict(self)
        d.pop("config")
        for k, v in list(d.items()):
            d[k] = np.asarray(v) if isinstance(v, list) else v
        # Serialise config too, like the existing harness does.
        d["config_json"] = np.asarray(json.dumps(asdict(self.config)))
        return d


def _y_range_estimate(y_test: np.ndarray) -> float:
    """Robust scale of y, used to make the σ-threshold dimensionless."""
    return float(max(1e-12, np.ptp(y_test)))


def run_trust_threshold_benchmark(
    method_factory: Callable[[int], "OnlineRegressor"],
    problem,
    seed: int,
    n_stream: int,
    tau_sigma: float,
    schedule: str = "mcmc",
    n_test: int = 1000,
    checkpoint_every: int = 500,
    batch_size: int = 1000,
    tau_y_grid: tuple = DEFAULT_TAU_Y_GRID,
    max_wall_time_s: float = 3600.0,
    method_name: str | None = None,
    verbose: bool = True,
    save_every_checkpoint_to: str | None = None,
    schedule_kwargs: dict | None = None,
) -> TrustRunResult:
    """One run with the trust-threshold deployment loop.

    Returns a `TrustRunResult` containing per-checkpoint emulator
    quality, per-batch deployment statistics, and final cumulative
    counts of trained/trusted decisions.
    """
    _seed_all(seed)
    rng_stream = np.random.default_rng(seed)
    rng_test = np.random.default_rng(seed + 10_000)

    X_stream, y_stream = problem.sample_schedule(
        n_stream, rng_stream, schedule=schedule,
        **(schedule_kwargs or {}),
    )
    X_test, y_test = problem.sample(n_test, rng_test)
    y_range = _y_range_estimate(y_test)
    # σ-threshold is reported relative to the y-range; the predicate
    # below uses the absolute σ.
    tau_sigma_abs = tau_sigma * y_range

    method = method_factory(problem.dim)
    name = method_name or method.name
    cfg = TrustRunConfig(
        method_name=name, problem_name=problem.name, seed=seed,
        n_stream=n_stream, n_test=n_test, checkpoint_every=checkpoint_every,
        schedule=schedule, tau_sigma=tau_sigma, batch_size=batch_size,
        tau_y_grid=tuple(tau_y_grid), max_wall_time_s=max_wall_time_s,
    )
    result = TrustRunResult(config=cfg)

    n_trained = 0
    n_trusted = 0
    # In-flight batch accumulators.
    batch_n_trained = 0
    batch_n_trusted = 0
    batch_trusted_errs = []  # list of |μ - y_true| values
    batch_within = np.zeros(len(tau_y_grid), dtype=int)
    batch_within_total = 0   # = number of trusted decisions in the batch
    cur_batch_idx = 0

    cum_predict = 0.0
    cum_update = 0.0
    t_start = time.time()

    for i in range(n_stream):
        x = X_stream[i:i + 1]
        y_true = float(y_stream[i])

        # 1) Predict. Always.
        t0 = time.time()
        mean, std = method.predict(x)
        cum_predict += time.time() - t0
        mu = float(np.asarray(mean).ravel()[0])
        sigma = float(np.asarray(std).ravel()[0])
        # Pathological / non-finite std => never trust (must train).
        is_finite_sigma = np.isfinite(sigma)

        # 2) Decide.
        trust = is_finite_sigma and (sigma <= tau_sigma_abs)
        if trust:
            n_trusted += 1
            batch_n_trusted += 1
            err = abs(mu - y_true)
            batch_trusted_errs.append(err)
            batch_within_total += 1
            for j, tau_y in enumerate(tau_y_grid):
                if err <= tau_y:
                    batch_within[j] += 1
        else:
            # Train on the truth.
            n_trained += 1
            batch_n_trained += 1
            t0 = time.time()
            method.update(x, np.array([[y_true]]))
            cum_update += time.time() - t0

        # 3) Per-batch flush.
        if (i + 1) % batch_size == 0:
            result.batch_index.append(cur_batch_idx)
            result.batch_n_trained.append(int(batch_n_trained))
            result.batch_n_trusted.append(int(batch_n_trusted))
            if batch_trusted_errs:
                arr = np.asarray(batch_trusted_errs)
                med = float(np.median(arr))
                p90 = float(np.percentile(arr, 90))
            else:
                med = float("nan"); p90 = float("nan")
            result.batch_trusted_err_med.append(med)
            result.batch_trusted_err_p90.append(p90)
            if batch_within_total > 0:
                fracs = (batch_within / batch_within_total).tolist()
            else:
                fracs = [float("nan")] * len(tau_y_grid)
            result.batch_frac_within_tau_y.append(fracs)
            result.cum_n_trained.append(int(n_trained))
            result.cum_n_trusted.append(int(n_trusted))
            cur_batch_idx += 1
            batch_n_trained = 0
            batch_n_trusted = 0
            batch_trusted_errs = []
            batch_within = np.zeros(len(tau_y_grid), dtype=int)
            batch_within_total = 0

        # 4) Held-out checkpoint metrics.
        step = i + 1
        if step % checkpoint_every == 0 or step == n_stream:
            t0 = time.time()
            mean_t, std_t = method.predict(X_test)
            cum_predict += time.time() - t0
            m = _checkpoint_metrics(y_test, mean_t, std_t)
            result.checkpoints.append(step)
            for key in ("rmse", "nrmse", "mae", "nlpd", "median_nlpd",
                        "crps", "coverage_1sigma", "coverage_2sigma",
                        "frac_pathological_std"):
                getattr(result, key).append(m[key])
            result.cum_update_time.append(cum_update)
            result.cum_predict_time.append(cum_predict)
            if verbose:
                print(
                    f"  [{name:>20s} | {problem.name:>15s} | "
                    f"τσ={tau_sigma:.0e}] step {step:>6d} | "
                    f"nrmse {m['nrmse']:.3e} | trained={n_trained} "
                    f"trusted={n_trusted} "
                    f"speedup={n_stream / max(1, n_trained):.2f}x"
                )
            if save_every_checkpoint_to:
                result.wall_time = time.time() - t_start
                np.savez(save_every_checkpoint_to, **result.to_npz_dict())
            if time.time() - t_start > max_wall_time_s:
                result.aborted = True
                result.notes = (
                    f"aborted after {step} / {n_stream} steps "
                    f"(wall-time {time.time() - t_start:.0f}s)"
                )
                break

    result.wall_time = time.time() - t_start
    method.close()
    if save_every_checkpoint_to:
        np.savez(save_every_checkpoint_to, **result.to_npz_dict())
    return result
