"""Reference and emulator-assisted random-walk Metropolis chains.

In iter 11 we used MCMC purely as a *stream generator*: the chain
produced inputs, the emulator was trained on the truth, and the chain
itself never saw the emulator. Iter 14 closes the loop: the emulator's
prediction can replace the true likelihood call, so the emulator's
mistakes can in principle steer the chain into regions it otherwise
would have rejected.

Two chain types live here:

- ``run_reference_chain``: standard random-walk Metropolis on the true
  log-likelihood ``log L(x) = -β · (f(x) - f_min) / f_scale``.
- ``run_assisted_chain``: same proposal kernel, but at every
  likelihood-evaluation step the emulator is queried first; if
  ``σ_emu ≤ τ_σ`` the chain accepts the emulator's mean as the
  likelihood and skips the true call. Otherwise the truth is fetched,
  the emulator is updated on it, and the chain uses the truth.

Both chains start from the same seed-derived initial point and use the
same proposal RNG, so the two trajectories are "as paired as possible"
when ``τ_σ`` is small.

Output of each call: dict with keys
    ``samples``, ``logL``, ``n_accept``, ``n_proposals``,
    ``n_used_emu``, ``n_used_true``, ``trusted_err`` (only assisted),
    ``wall_time``.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np


def _logL_factory(problem, f_min: float, f_scale: float, beta: float = 1.0):
    """Negative-mapped log-likelihood from the target function ``f``.

    `log L(x) = -beta · (f(x) - f_min) / f_scale`.
    Higher f => lower likelihood, so the chain concentrates on the
    region where f is small. The shape of f is what's interesting; the
    constants only set the temperature.
    """
    def _logL(X):
        return -beta * (problem.fn(np.atleast_2d(X)) - f_min) / f_scale
    return _logL


def estimate_f_scale(problem, rng: np.random.Generator,
                     n_pre: int = 200) -> tuple[float, float]:
    """Cheap pre-pass: f_min and f_scale on a Sobol/uniform pre-pass."""
    X = rng.uniform(0.0, 1.0, size=(n_pre, problem.dim))
    y = problem.fn(X)
    return float(np.min(y)), float(np.ptp(y))


def run_reference_chain(
    problem,
    seed: int,
    n_steps: int,
    proposal_sigma: float = 0.05,
    beta: float = 1.0,
    f_min_scale: tuple | None = None,
):
    rng = np.random.default_rng(seed)
    if f_min_scale is None:
        f_min, f_scale = estimate_f_scale(problem, rng)
    else:
        f_min, f_scale = f_min_scale
    logL = _logL_factory(problem, f_min, max(1e-12, 0.1 * f_scale), beta=beta)

    x = rng.uniform(0.0, 1.0, size=problem.dim)
    lp = float(logL(x.reshape(1, -1)).ravel()[0])
    samples = np.empty((n_steps, problem.dim), dtype=float)
    logLs = np.empty(n_steps, dtype=float)
    n_accept = 0

    t0 = time.time()
    for i in range(n_steps):
        prop = x + rng.normal(0.0, proposal_sigma, size=problem.dim)
        if np.all((prop >= 0.0) & (prop <= 1.0)):
            lp_prop = float(logL(prop.reshape(1, -1)).ravel()[0])
            if np.log(rng.random() + 1e-300) < (lp_prop - lp):
                x = prop
                lp = lp_prop
                n_accept += 1
        samples[i] = x
        logLs[i] = lp
    wall = time.time() - t0
    return {
        "samples": samples, "logL": logLs,
        "n_accept": int(n_accept), "n_proposals": int(n_steps),
        "n_used_emu": 0, "n_used_true": int(n_steps),
        "wall_time": float(wall),
        "f_min_scale": (f_min, f_scale),
    }


def run_assisted_chain(
    method_factory,
    problem,
    seed: int,
    n_steps: int,
    tau_sigma: float,
    proposal_sigma: float = 0.05,
    beta: float = 1.0,
    f_min_scale: tuple | None = None,
):
    """Emulator-assisted chain. Reuses the same seed/proposal RNG as the
    reference so the two chains are as comparable as we can make them.
    """
    rng = np.random.default_rng(seed)
    if f_min_scale is None:
        f_min, f_scale = estimate_f_scale(problem, rng)
    else:
        f_min, f_scale = f_min_scale
    f_scale_eff = max(1e-12, 0.1 * f_scale)

    def _true_logL(X):
        return -beta * (problem.fn(np.atleast_2d(X)) - f_min) / f_scale_eff

    method = method_factory(problem.dim)

    # The emulator is trained on the **logL value**, not on f directly,
    # so its σ is in logL units and the τ_σ threshold is dimensionless
    # against the y-range of logL on the test set we'll inspect later.
    # logL has range β (since (f - f_min)/f_scale_eff in [0, 10]).
    tau_sigma_abs = tau_sigma * (10.0 * beta)  # rough natural y-range

    x = rng.uniform(0.0, 1.0, size=problem.dim)
    lp_init = float(_true_logL(x.reshape(1, -1)).ravel()[0])
    method.update(x.reshape(1, -1), np.array([[lp_init]]))
    lp = lp_init

    samples = np.empty((n_steps, problem.dim), dtype=float)
    logLs = np.empty(n_steps, dtype=float)
    n_accept = 0
    n_used_emu = 0
    n_used_true = 1   # for the init
    trusted_errs = []

    t0 = time.time()
    for i in range(n_steps):
        prop = x + rng.normal(0.0, proposal_sigma, size=problem.dim)
        if np.all((prop >= 0.0) & (prop <= 1.0)):
            mean, std = method.predict(prop.reshape(1, -1))
            mu = float(np.asarray(mean).ravel()[0])
            sigma = float(np.asarray(std).ravel()[0])
            if np.isfinite(sigma) and sigma <= tau_sigma_abs:
                lp_prop = mu
                n_used_emu += 1
                # Diagnostic: track |μ - true| for trusted picks.
                lp_true = float(_true_logL(prop.reshape(1, -1)).ravel()[0])
                trusted_errs.append(abs(mu - lp_true))
            else:
                lp_prop = float(_true_logL(prop.reshape(1, -1)).ravel()[0])
                n_used_true += 1
                method.update(prop.reshape(1, -1), np.array([[lp_prop]]))

            if np.log(rng.random() + 1e-300) < (lp_prop - lp):
                x = prop
                lp = lp_prop
                n_accept += 1
        samples[i] = x
        logLs[i] = lp
    wall = time.time() - t0
    method.close()
    return {
        "samples": samples, "logL": logLs,
        "n_accept": int(n_accept), "n_proposals": int(n_steps),
        "n_used_emu": int(n_used_emu), "n_used_true": int(n_used_true),
        "trusted_err": np.asarray(trusted_errs),
        "wall_time": float(wall),
        "f_min_scale": (f_min, f_scale),
    }


def wasserstein1_marginals(s_ref: np.ndarray, s_alt: np.ndarray) -> float:
    """Mean (over dimensions) Wasserstein-1 distance between the
    1-D marginals of two sample sets. Closed form via sorted-sample
    integrated absolute difference of empirical CDFs.
    """
    assert s_ref.shape[1] == s_alt.shape[1]
    d = s_ref.shape[1]
    out = 0.0
    n = min(s_ref.shape[0], s_alt.shape[0])
    sr = np.sort(s_ref[:n], axis=0)
    sa = np.sort(s_alt[:n], axis=0)
    for j in range(d):
        out += float(np.mean(np.abs(sr[:, j] - sa[:, j])))
    return out / d
