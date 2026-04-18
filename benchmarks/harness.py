"""Continual-emulation benchmark harness (iteration 01).

The harness simulates an online stream of (x, y) pairs from a target problem.
At each step it calls ``method.update(x, y)`` and times that update. At fixed
checkpoints it evaluates the method on a held-out test set and records a
battery of metrics suitable for a paper comparison. Results are saved to a
single ``.npz`` per (method, problem, seed) triple.

Metrics at each checkpoint (on the held-out test set):

Accuracy:
    - ``rmse``, ``nrmse`` (normalised by test-target range), ``mae``

Probabilistic quality (Gaussian predictive assumed):
    - ``nlpd`` (mean negative log predictive density with a physical std floor)
    - ``median_nlpd`` (more robust to outlier predictions)
    - ``nlpd_trimmed`` (mean after discarding the top/bottom 5 % of per-point NLPD)
    - ``crps`` (mean continuous ranked probability score, closed form)

Calibration:
    - ``coverage_1sigma`` (fraction within ±1 std, i.e. nominal 0.6827)
    - ``coverage_2sigma`` (nominal 0.9545)
    - ``coverage_90``, ``coverage_95`` (via z = 1.645, 1.960)

Uncertainty hygiene:
    - ``frac_pathological_std``: fraction of test points whose predicted std is
      below the physical floor (std_floor = 1e-3 * y-range), above a huge-std
      cap (1e3 * y-range), or non-finite. This exposes catastrophic MoE /
      calibration failures that would otherwise be hidden by the floor.

Compute:
    - ``cum_update_time``, ``cum_predict_time``

Reproducibility: every run seeds numpy / random / torch with the run seed at
the start, and the test set is drawn from an independent RNG
(``default_rng(seed + 10_000)``) so it cannot correlate with the stream.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, List

import numpy as np


# Z-scores for common nominal coverage levels.
_Z_50 = 0.6744897501960817
_Z_68 = 1.0   # 1-sigma (nominal 0.6827)
_Z_90 = 1.6448536269514722
_Z_95 = 1.959963984540054
_Z_2S = 2.0   # 2-sigma (nominal 0.9545)


@dataclass
class RunConfig:
    method_name: str
    problem_name: str
    seed: int
    n_stream: int
    n_test: int
    checkpoint_every: int
    schedule: str = "iid"
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
    median_nlpd: List[float] = field(default_factory=list)
    nlpd_trimmed: List[float] = field(default_factory=list)
    crps: List[float] = field(default_factory=list)
    coverage_1sigma: List[float] = field(default_factory=list)
    coverage_2sigma: List[float] = field(default_factory=list)
    coverage_50: List[float] = field(default_factory=list)
    coverage_90: List[float] = field(default_factory=list)
    coverage_95: List[float] = field(default_factory=list)
    frac_pathological_std: List[float] = field(default_factory=list)
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
                d[k] = np.asarray(v, dtype=float) if v else np.asarray(v)
        d["config_json"] = np.asarray(json.dumps(asdict(self.config)))
        return d


def _gaussian_crps(err: np.ndarray, sigma: np.ndarray) -> float:
    """Closed-form CRPS for a Gaussian predictive distribution.

    For a Gaussian N(mu, sigma^2), CRPS(F, y) has the exact form

        CRPS = sigma * ( z * (2 * Phi(z) - 1)  +  2 * phi(z) - 1/sqrt(pi) )

    with z = (y - mu) / sigma, phi the standard-normal pdf, Phi its cdf.
    """
    from math import erf, sqrt, pi

    z = err / sigma
    # Vectorised phi / Phi.
    phi = np.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)
    # scipy-free Phi via erf
    Phi = 0.5 * (1.0 + np.vectorize(erf)(z / math.sqrt(2.0)))
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    return float(np.mean(crps))


def _metrics(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict:
    """Compute the full metric battery for one checkpoint.

    The physical std floor prevents single pathological predictions (std=0)
    from destroying mean NLPD. The pathology is tracked separately in
    ``frac_pathological_std`` so the hiding is honest.
    """
    y_true = y_true.ravel()
    mean = mean.ravel()
    err = mean - y_true

    rmse = float(np.sqrt(np.mean(err ** 2)))
    y_range = float(y_true.max() - y_true.min())
    nrmse = rmse / y_range if y_range > 0 else float("nan")
    mae = float(np.mean(np.abs(err)))

    out = dict(
        rmse=rmse, nrmse=nrmse, mae=mae,
        nlpd=float("nan"), median_nlpd=float("nan"), nlpd_trimmed=float("nan"),
        crps=float("nan"),
        coverage_1sigma=float("nan"), coverage_2sigma=float("nan"),
        coverage_50=float("nan"), coverage_90=float("nan"), coverage_95=float("nan"),
        frac_pathological_std=float("nan"),
    )
    if std is None:
        return out

    std_flat = np.asarray(std).ravel()

    # Physical std floor: 0.0001 % of observed target range. Tight enough
    # that typical well-behaved stds are unaffected, but large enough that
    # a catastrophic std=0 cannot blow mean NLPD up to 1e12. The cap is
    # 1000 x the y-range.
    if y_range > 0:
        std_floor = 1e-6 * y_range
        std_cap = 1e3 * y_range
    else:
        std_floor = 1e-8
        std_cap = np.inf

    # "Pathological" = clearly broken: non-finite, non-positive, or huge.
    # Merely under-confident predictions (std positive but tiny) are captured
    # by the coverage / NLPD metrics directly.
    bad = (~np.isfinite(std_flat)) | (std_flat <= 0) | (std_flat > std_cap)
    out["frac_pathological_std"] = float(np.mean(bad))

    # Clip for downstream formulas.
    s = np.where(np.isfinite(std_flat), std_flat, std_floor)
    s = np.clip(s, std_floor, std_cap)

    # Per-point NLPD.
    per_point_nlpd = 0.5 * np.log(2.0 * np.pi * s ** 2) + 0.5 * (err / s) ** 2
    out["nlpd"] = float(np.mean(per_point_nlpd))
    out["median_nlpd"] = float(np.median(per_point_nlpd))
    if per_point_nlpd.size >= 20:
        lo, hi = np.percentile(per_point_nlpd, [5, 95])
        trimmed = per_point_nlpd[(per_point_nlpd >= lo) & (per_point_nlpd <= hi)]
        out["nlpd_trimmed"] = float(np.mean(trimmed))
    else:
        out["nlpd_trimmed"] = out["nlpd"]

    # CRPS.
    out["crps"] = _gaussian_crps(err, s)

    # Coverage at nominal levels.
    ae = np.abs(err)
    out["coverage_1sigma"] = float(np.mean(ae <= _Z_68 * s))
    out["coverage_2sigma"] = float(np.mean(ae <= _Z_2S * s))
    out["coverage_50"] = float(np.mean(ae <= _Z_50 * s))
    out["coverage_90"] = float(np.mean(ae <= _Z_90 * s))
    out["coverage_95"] = float(np.mean(ae <= _Z_95 * s))

    return out


def _seed_all(seed: int) -> None:
    """Deterministically seed every RNG that the methods might touch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass


def run_online_benchmark(
    method_factory: Callable[[int], "OnlineRegressor"],
    problem,
    seed: int,
    n_stream: int,
    n_test: int = 1000,
    checkpoint_every: int = 500,
    schedule: str = "iid",
    max_wall_time_s: float = 3600.0,
    method_name: str | None = None,
    verbose: bool = True,
    save_every_checkpoint_to: str | None = None,
) -> RunResult:
    """Run one (method, problem, seed) benchmark.

    The test set is drawn from an **independent** RNG (``seed + 10_000``) so
    it cannot correlate with whatever the stream rng consumes.

    If ``save_every_checkpoint_to`` is set, write ``result`` to that path
    after every checkpoint. The driver uses this together with a subprocess
    timeout: if the child is killed mid-``method.update()``, whatever was
    already flushed is the surviving record.
    """
    _seed_all(seed)

    rng_stream = np.random.default_rng(seed)
    rng_test = np.random.default_rng(seed + 10_000)

    X_stream, y_stream = problem.sample_schedule(
        n_stream, rng_stream, schedule=schedule,
    )
    X_test, y_test = problem.sample(n_test, rng_test)

    method = method_factory(problem.dim)
    name = method_name or method.name
    cfg = RunConfig(
        method_name=name,
        problem_name=problem.name,
        seed=seed,
        n_stream=n_stream,
        n_test=n_test,
        checkpoint_every=checkpoint_every,
        schedule=schedule,
        max_wall_time_s=max_wall_time_s,
    )
    result = RunResult(config=cfg)

    t_start = time.time()
    cum_predict = 0.0
    cum_update = 0.0

    for i in range(n_stream):
        x = X_stream[i:i + 1]
        y = np.array([[y_stream[i]]])

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
            for key in ("rmse", "nrmse", "mae", "nlpd", "median_nlpd",
                        "nlpd_trimmed", "crps", "coverage_1sigma",
                        "coverage_2sigma", "coverage_50", "coverage_90",
                        "coverage_95", "frac_pathological_std"):
                getattr(result, key).append(m[key])
            result.cum_update_time.append(cum_update)
            result.cum_predict_time.append(cum_predict)
            if verbose:
                print(
                    f"  [{name:>15s} | {problem.name:>15s} | seed {seed}] "
                    f"step {step:>6d} | nrmse {m['nrmse']:.3e} | "
                    f"medNLPD {m['median_nlpd']:.3f} | "
                    f"cov68 {m['coverage_1sigma']:.2f} | "
                    f"badσ {m['frac_pathological_std']:.2f} | "
                    f"upd {cum_update:.1f}s"
                )
            if save_every_checkpoint_to:
                # Partial save so a subprocess timeout still leaves a record.
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


def save_result(result: RunResult, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    cfg = result.config
    fname = f"{cfg.method_name}__{cfg.problem_name}__seed{cfg.seed}.npz"
    path = os.path.join(out_dir, fname)
    np.savez(path, **result.to_npz_dict())
    return path
