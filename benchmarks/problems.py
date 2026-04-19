"""Benchmark target functions and data-stream generators (iteration 01).

All problems accept inputs in ``[0, 1]^d`` and internally map them to their
natural domains. The Problem dataclass exposes both i.i.d. sampling and a
covariate-shift schedule used to stress-test continual learners.

Included emulation-community standards:
    - Borehole (8-D) — classic water-flow emulator
    - Friedman-1 (5-D)
    - Piston simulation (7-D)
    - Smooth sines (2-D) — an "easy GP" sanity check
    - Rosenbrock (2-D) — the curved valley
    - Rastrigin (3-D) — many local minima, kept as an optional hard case
    - Eggholder (2-D) — kept as optional, not in default run
    - Step (3-D) — deliberately pathological discontinuity, kept as a
      diagnostic; NOT in the default set since smooth-kernel methods cannot
      reasonably be expected to approximate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def _scale(X: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + X * (hi - lo)


# ---------- optimisation test functions (kept from iteration 0) ------------

def eggholder(X: np.ndarray) -> np.ndarray:
    Z = _scale(X, -512.0, 512.0)
    out = np.zeros(Z.shape[0])
    for i in range(Z.shape[1] - 1):
        a, b = Z[:, i], Z[:, i + 1]
        term1 = -(b + 47.0) * np.sin(np.sqrt(np.abs(b + a / 2.0 + 47.0)))
        term2 = -a * np.sin(np.sqrt(np.abs(a - (b + 47.0))))
        out += term1 + term2
    return out + 959.6407 * (Z.shape[1] - 1)


def rosenbrock(X: np.ndarray) -> np.ndarray:
    Z = _scale(X, -2.0, 2.0)
    out = np.zeros(Z.shape[0])
    for i in range(Z.shape[1] - 1):
        out += 100.0 * (Z[:, i + 1] - Z[:, i] ** 2) ** 2 + (1.0 - Z[:, i]) ** 2
    return out


def rastrigin(X: np.ndarray) -> np.ndarray:
    Z = _scale(X, -5.12, 5.12)
    d = Z.shape[1]
    return 10.0 * d + np.sum(Z ** 2 - 10.0 * np.cos(2.0 * np.pi * Z), axis=1)


def smooth_sines(X: np.ndarray) -> np.ndarray:
    Z = _scale(X, -np.pi, np.pi)
    out = np.zeros(Z.shape[0])
    for i in range(Z.shape[1]):
        out += np.sin(Z[:, i]) * np.cos(0.5 * Z[:, (i + 1) % Z.shape[1]])
    return out


def discontinuous_step(X: np.ndarray) -> np.ndarray:
    Z = _scale(X, -1.0, 1.0)
    base = np.sum(Z ** 2, axis=1)
    step = 3.0 * (X[:, 0] > 0.5).astype(float)
    return base + step


# ---------- emulation-community benchmarks ---------------------------------

def borehole(X: np.ndarray) -> np.ndarray:
    """Classic 8-D borehole water-flow emulator.

    Inputs (all in [0, 1]) are mapped to
        rw in [0.05, 0.15]          radius of borehole
        r  in [100, 50000]           radius of influence
        Tu in [63070, 115600]        transmissivity of upper aquifer
        Hu in [990, 1110]            potentiometric head of upper aquifer
        Tl in [63.1, 116]            transmissivity of lower aquifer
        Hl in [700, 820]             potentiometric head of lower aquifer
        L  in [1120, 1680]           length of borehole
        Kw in [9855, 12045]          hydraulic conductivity of borehole
    See Morris, Mitchell & Ylvisaker (1993).
    """
    bounds = np.array([
        [0.05, 0.15],
        [100.0, 50_000.0],
        [63_070.0, 115_600.0],
        [990.0, 1110.0],
        [63.1, 116.0],
        [700.0, 820.0],
        [1120.0, 1680.0],
        [9855.0, 12_045.0],
    ])
    Z = bounds[:, 0] + X * (bounds[:, 1] - bounds[:, 0])
    rw, r, Tu, Hu, Tl, Hl, L, Kw = [Z[:, i] for i in range(8)]
    num = 2.0 * np.pi * Tu * (Hu - Hl)
    den = np.log(r / rw) * (
        1.0
        + (2.0 * L * Tu) / (np.log(r / rw) * rw ** 2 * Kw)
        + Tu / Tl
    )
    return num / den


def friedman1(X: np.ndarray) -> np.ndarray:
    """Friedman-1 (5-D) used extensively in regression literature.

    f(x) = 10 sin(pi x0 x1) + 20 (x2 - 0.5)^2 + 10 x3 + 5 x4.
    X is expected to have d >= 5 columns; extra columns (if any) are ignored
    by the function but act as noise dimensions that test the emulator's
    ability to ignore irrelevant features.
    """
    x0, x1, x2, x3, x4 = [X[:, i] for i in range(5)]
    return (
        10.0 * np.sin(np.pi * x0 * x1)
        + 20.0 * (x2 - 0.5) ** 2
        + 10.0 * x3
        + 5.0 * x4
    )


def piston(X: np.ndarray) -> np.ndarray:
    """7-D piston-simulation emulator (Kenett & Zacks, 1998).

    Inputs (all in [0, 1]) are mapped to
        M  in [30, 60]      piston weight (kg)
        S  in [0.005, 0.020]  surface area (m^2)
        V0 in [0.002, 0.010]  initial gas volume (m^3)
        k  in [1000, 5000]   spring coefficient (N/m)
        P0 in [90000, 110000] atmospheric pressure (N/m^2)
        Ta in [290, 296]     ambient temperature (K)
        T0 in [340, 360]     filling-gas temperature (K)
    Output is the cycle time in seconds.
    """
    bounds = np.array([
        [30.0, 60.0],
        [0.005, 0.020],
        [0.002, 0.010],
        [1000.0, 5000.0],
        [90_000.0, 110_000.0],
        [290.0, 296.0],
        [340.0, 360.0],
    ])
    Z = bounds[:, 0] + X * (bounds[:, 1] - bounds[:, 0])
    M, S, V0, k, P0, Ta, T0 = [Z[:, i] for i in range(7)]
    A = P0 * S + 19.62 * M - k * V0 / S
    V = (S / (2.0 * k)) * (np.sqrt(A ** 2 + 4.0 * k * (P0 * V0 / T0) * Ta) - A)
    C = 2.0 * np.pi * np.sqrt(M / (k + S ** 2 * (P0 * V0 * Ta) / (T0 * V ** 2)))
    return C


# ---------- Problem dataclass ----------------------------------------------

@dataclass
class Problem:
    name: str
    dim: int
    fn: Callable[[np.ndarray], np.ndarray]

    def sample(self, n: int, rng: np.random.Generator):
        X = rng.uniform(0.0, 1.0, size=(n, self.dim))
        y = self.fn(X)
        return X, y

    def sample_schedule(self, n: int, rng: np.random.Generator,
                        schedule: str = "iid",
                        likelihood: str = "bimodal_gauss",
                        de_popsize: int = 100):
        """Draw the stream under a particular sampling schedule.

        - ``iid``: uniform U[0, 1]^d (the default / no distribution shift).
        - ``shift``: first half from U[0, 0.5]^d, second half from
          U[0.5, 1]^d. A covariate-shift stress test.
        - ``sobol``: scrambled Sobol low-discrepancy sequence.
        - ``lhs``: Latin hypercube sampling — space-filling design, draws
          one point per per-dimension stratum, scrambled with ``rng``.
          Mirrors the "design-of-experiments" workflow where an
          experimenter picks a one-shot grid of emulator training points
          rather than drawing iid.
        """
        if schedule == "iid":
            return self.sample(n, rng)
        if schedule == "shift":
            half = n // 2
            X1 = rng.uniform(0.0, 0.5, size=(half, self.dim))
            X2 = rng.uniform(0.5, 1.0, size=(n - half, self.dim))
            X = np.vstack([X1, X2])
            y = self.fn(X)
            return X, y
        if schedule == "sobol":
            try:
                from scipy.stats import qmc
                eng = qmc.Sobol(d=self.dim, scramble=True,
                                seed=int(rng.integers(2**31 - 1)))
                X = eng.random(n)
            except ImportError:
                return self.sample(n, rng)
            y = self.fn(X)
            return X, y
        if schedule == "lhs":
            from scipy.stats import qmc
            eng = qmc.LatinHypercube(
                d=self.dim, scramble=True,
                seed=int(rng.integers(2**31 - 1)),
            )
            X = eng.random(n)
            y = self.fn(X)
            return X, y
        if schedule == "de":
            X = _sample_differential_evolution(
                n=n, dim=self.dim, rng=rng, likelihood=likelihood,
                popsize=de_popsize,
            )
            y = self.fn(X)
            return X, y
        if schedule == "mcmc":
            X = _sample_mcmc(
                n=n, dim=self.dim, rng=rng, likelihood=likelihood,
            )
            y = self.fn(X)
            return X, y
        raise ValueError(f"unknown schedule '{schedule}'")


def _sample_differential_evolution(n: int, dim: int,
                                   rng: np.random.Generator,
                                   likelihood: str,
                                   popsize: int = 100) -> np.ndarray:
    """Run DE on the chosen auxiliary log-likelihood and log every
    function evaluation in visit order. Returns the first ``n``.

    DE minimises; since ``log L`` is the quantity we want to *maximise*,
    the DE objective is ``-log L``. ``popsize`` is scipy's per-dimension
    multiplier — the actual population per generation is ``popsize *
    dim``. Larger ``popsize`` => slower convergence and broader cube
    coverage in the early generations (good for "map the 2σ region of
    the profile likelihood" workloads, not just "find the best fit").
    """
    from scipy.optimize import differential_evolution
    from benchmarks.likelihoods import LIKELIHOODS

    log_lik = LIKELIHOODS[likelihood]
    bounds = [(0.0, 1.0)] * dim
    visits = []

    def _objective(x):
        x = np.asarray(x).reshape(1, dim)
        val = -float(log_lik(x).ravel()[0])
        # Clip x into the unit cube before we store it, so that what the
        # emulator sees is a lawful input. DE's native mutation can
        # step slightly outside the bounds during internal work; the
        # optimiser re-clips but we also store the clipped value.
        x_clipped = np.clip(x.ravel(), 0.0, 1.0)
        visits.append(x_clipped)
        return val

    # We need roughly n / (popsize * dim) iterations of DE to accumulate
    # n visits. `maxiter` is a cap on that, with a safety factor.
    evals_per_iter = popsize * dim
    maxiter = max(5, int(np.ceil(n / max(1, evals_per_iter)) * 2 + 5))

    # Use DE with polish=False to avoid a late-stage L-BFGS burst that
    # would suddenly add many local-search visits.
    differential_evolution(
        _objective,
        bounds,
        seed=int(rng.integers(2 ** 31 - 1)),
        popsize=popsize,
        mutation=(0.5, 1.5),
        recombination=0.9,
        tol=0.0,          # run until maxiter; do not early-stop
        init="sobol",
        polish=False,
        maxiter=maxiter,
        updating="deferred",
    )

    if len(visits) < n:
        # Pad with uniform fills if DE converged before we had enough.
        extra = n - len(visits)
        visits.extend(list(rng.uniform(0.0, 1.0, size=(extra, dim))))
    X = np.asarray(visits[:n], dtype=float)
    return X


def _sample_mcmc(n: int, dim: int, rng: np.random.Generator,
                 likelihood: str, proposal_sigma: float = 0.1) -> np.ndarray:
    """Random-walk Metropolis from a uniform start. Every step
    (accepted or rejected) contributes its *current state* to the
    stream — so the emulator sees realistic chain autocorrelation.
    """
    from benchmarks.likelihoods import LIKELIHOODS

    log_lik = LIKELIHOODS[likelihood]
    x = rng.uniform(0.0, 1.0, size=dim)
    lp = float(log_lik(x.reshape(1, dim)).ravel()[0])
    visits = np.empty((n, dim), dtype=float)
    for i in range(n):
        prop = x + rng.normal(0.0, proposal_sigma, size=dim)
        lp_prop = float(log_lik(prop.reshape(1, dim)).ravel()[0])
        if np.log(rng.random() + 1e-300) < (lp_prop - lp):
            x = prop
            lp = lp_prop
        # Clip into the cube for storage (the penalty in the likelihood
        # will have already rejected most out-of-box proposals).
        visits[i] = np.clip(x, 0.0, 1.0)
    return visits


PROBLEMS = {
    "smooth_sines_2d": Problem("smooth_sines_2d", 2, smooth_sines),
    "rosenbrock_2d": Problem("rosenbrock_2d", 2, rosenbrock),
    "borehole_8d": Problem("borehole_8d", 8, borehole),
    "friedman1_5d": Problem("friedman1_5d", 5, friedman1),
    "piston_7d": Problem("piston_7d", 7, piston),
    # Kept but not in the default benchmark set.
    "eggholder_2d": Problem("eggholder_2d", 2, eggholder),
    "rastrigin_3d": Problem("rastrigin_3d", 3, rastrigin),
    "step_3d": Problem("step_3d", 3, discontinuous_step),
}
