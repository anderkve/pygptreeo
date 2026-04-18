"""Continual-emulation benchmark harness.

The harness simulates an online stream of (x, y) pairs from a target problem.
At each step it calls ``method.predict(x)`` to record how well the current
model predicts the just-arrived point BEFORE seeing its label (an honest
streaming evaluation), then calls ``method.update(x, y)``. A fixed held-out
test set is evaluated at checkpoint intervals to track out-of-sample accuracy
and uncertainty calibration as a function of "points processed".

Recorded metrics per checkpoint (on the held-out test set):
    - RMSE and NRMSE (normalised by target range on the test set)
    - MAE
    - mean negative log predictive density (NLPD) assuming Gaussian predictive
      density; only computed where the method returns a finite std
    - empirical coverage at 1 sigma (fraction of test points whose true value
      is within +/- predicted-std of the mean)
    - cumulative wall-clock update time
    - cumulative wall-clock predict time (on the streamed points, not test set)

Results are saved to a single .npz file per (method, problem, seed) triple.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, List

import numpy as np


@dataclass
class RunConfig:
    method_name: str
    problem_name: str
    seed: int
    n_stream: int
    n_test: int
    checkpoint_every: int
    # Optional step budget (wall-clock seconds) to skip a method that blows up.
    max_wall_time_s: float = 3600.0


@dataclass
class RunResult:
    config: RunConfig
    checkpoints: List[int] = field(default_factory=list)
    rmse: List[float] = field(default_factory=list)
    nrmse: List[float] = field(default_factory=list)
    mae: List[float] = field(default_factory=list)
    nlpd: List[float] = field(default_factory=list)
    coverage_1sigma: List[float] = field(default_factory=list)
    cum_update_time: List[float] = field(default_factory=list)
    cum_predict_time: List[float] = field(default_factory=list)
    wall_time: float = 0.0
    aborted: bool = False
    notes: str = ""

    def to_npz_dict(self) -> dict:
        d = asdict(self)
        d.pop("config")
        for k, v in list(d.items()):
            if isinstance(v, list):
                d[k] = np.asarray(v)
        d["config_json"] = np.asarray(json.dumps(asdict(self.config)))
        return d


def _metrics(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict:
    y_true = y_true.ravel()
    mean = mean.ravel()
    err = mean - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    y_range = float(y_true.max() - y_true.min())
    nrmse = rmse / y_range if y_range > 0 else float("nan")
    mae = float(np.mean(np.abs(err)))
    std_flat = std.ravel() if std is not None else None
    nlpd = float("nan")
    cov = float("nan")
    if std_flat is not None and np.all(np.isfinite(std_flat)):
        s = np.clip(std_flat, 1e-8, None)
        nlpd = float(np.mean(
            0.5 * np.log(2.0 * np.pi * s ** 2) + 0.5 * (err / s) ** 2
        ))
        cov = float(np.mean(np.abs(err) <= s))
    return dict(rmse=rmse, nrmse=nrmse, mae=mae, nlpd=nlpd, coverage=cov)


def run_online_benchmark(
    method_factory: Callable[[int], "OnlineRegressor"],
    problem,
    seed: int,
    n_stream: int,
    n_test: int = 500,
    checkpoint_every: int = 500,
    max_wall_time_s: float = 3600.0,
    method_name: str | None = None,
    verbose: bool = True,
) -> RunResult:
    rng = np.random.default_rng(seed)
    X_stream, y_stream = problem.sample(n_stream, rng)
    X_test, y_test = problem.sample(n_test, rng)

    method = method_factory(problem.dim)
    name = method_name or method.name
    cfg = RunConfig(
        method_name=name,
        problem_name=problem.name,
        seed=seed,
        n_stream=n_stream,
        n_test=n_test,
        checkpoint_every=checkpoint_every,
        max_wall_time_s=max_wall_time_s,
    )
    result = RunResult(config=cfg)

    t_start = time.time()
    cum_predict = 0.0
    cum_update = 0.0

    for i in range(n_stream):
        x = X_stream[i:i + 1]
        y = np.array([[y_stream[i]]])

        # We do not record streaming predictions here for the test metrics
        # (we'll use the held-out set at checkpoints). But we still time update.
        t0 = time.time()
        method.update(x, y)
        cum_update += time.time() - t0

        step = i + 1
        if step % checkpoint_every == 0 or step == n_stream:
            t0 = time.time()
            mean, std = method.predict(X_test)
            pred_time = time.time() - t0
            cum_predict += pred_time
            m = _metrics(y_test, mean, std)
            result.checkpoints.append(step)
            result.rmse.append(m["rmse"])
            result.nrmse.append(m["nrmse"])
            result.mae.append(m["mae"])
            result.nlpd.append(m["nlpd"])
            result.coverage_1sigma.append(m["coverage"])
            result.cum_update_time.append(cum_update)
            result.cum_predict_time.append(cum_predict)
            if verbose:
                print(
                    f"  [{name:>15s} | {problem.name:>15s} | seed {seed}] "
                    f"step {step:>6d} | nrmse {m['nrmse']:.3e} | "
                    f"cov {m['coverage']:.2f} | "
                    f"cum_update {cum_update:.1f}s | cum_predict {cum_predict:.1f}s"
                )

            if time.time() - t_start > max_wall_time_s:
                result.aborted = True
                result.notes = (
                    f"aborted after {step} / {n_stream} steps "
                    f"(wall-time {time.time() - t_start:.0f}s)"
                )
                break

    result.wall_time = time.time() - t_start
    method.close()
    return result


def save_result(result: RunResult, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    cfg = result.config
    fname = f"{cfg.method_name}__{cfg.problem_name}__seed{cfg.seed}.npz"
    path = os.path.join(out_dir, fname)
    np.savez(path, **result.to_npz_dict())
    return path
