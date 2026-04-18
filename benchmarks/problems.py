"""Benchmark target functions and data-stream generators.

All problems expose:
    - ``dim``: input dimension
    - ``domain``: (n_dims, 2) array of [low, high] bounds for each axis
    - ``__call__(X)`` where X has shape (n, dim) and returns shape (n,)

We standardise by sampling X uniformly from [0, 1]^d inside the benchmark
harness (not in the problem itself), and each problem maps that to its natural
domain. This mirrors how the pygptreeo examples are set up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --- 2D / 3D classic optimisation test functions ---------------------------

def _scale(X: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + X * (hi - lo)


def eggholder(X: np.ndarray) -> np.ndarray:
    """N-dim Eggholder (sum of 2D terms). X in [0, 1]^d, mapped to [-512, 512]."""
    Z = _scale(X, -512.0, 512.0)
    out = np.zeros(Z.shape[0])
    for i in range(Z.shape[1] - 1):
        a, b = Z[:, i], Z[:, i + 1]
        term1 = -(b + 47.0) * np.sin(np.sqrt(np.abs(b + a / 2.0 + 47.0)))
        term2 = -a * np.sin(np.sqrt(np.abs(a - (b + 47.0))))
        out += term1 + term2
    return out + 959.6407 * (Z.shape[1] - 1)


def rosenbrock(X: np.ndarray) -> np.ndarray:
    """N-dim Rosenbrock. X in [0, 1]^d, mapped to [-2, 2]."""
    Z = _scale(X, -2.0, 2.0)
    out = np.zeros(Z.shape[0])
    for i in range(Z.shape[1] - 1):
        out += 100.0 * (Z[:, i + 1] - Z[:, i] ** 2) ** 2 + (1.0 - Z[:, i]) ** 2
    return out


def rastrigin(X: np.ndarray) -> np.ndarray:
    """N-dim Rastrigin. X in [0, 1]^d, mapped to [-5.12, 5.12]."""
    Z = _scale(X, -5.12, 5.12)
    d = Z.shape[1]
    return 10.0 * d + np.sum(Z ** 2 - 10.0 * np.cos(2.0 * np.pi * Z), axis=1)


def smooth_sines(X: np.ndarray) -> np.ndarray:
    """Smooth analytic function: sum of low-frequency sinusoids.

    Intended as an "easy" emulation target where GPs should shine.
    X in [0, 1]^d, mapped to [-pi, pi].
    """
    Z = _scale(X, -np.pi, np.pi)
    out = np.zeros(Z.shape[0])
    for i in range(Z.shape[1]):
        out += np.sin(Z[:, i]) * np.cos(0.5 * Z[:, (i + 1) % Z.shape[1]])
    return out


def discontinuous_step(X: np.ndarray) -> np.ndarray:
    """Smooth bowl with a sharp step: stress-tests locality of the emulator.

    f(x) = |x|^2 + 3 * 1[x_0 > 0.5]. X in [0, 1]^d, mapped to [-1, 1].
    """
    Z = _scale(X, -1.0, 1.0)
    base = np.sum(Z ** 2, axis=1)
    step = 3.0 * (X[:, 0] > 0.5).astype(float)
    return base + step


@dataclass
class Problem:
    name: str
    dim: int
    fn: callable  # (N, d) -> (N,)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        X = rng.uniform(0.0, 1.0, size=(n, self.dim))
        y = self.fn(X)
        return X, y


PROBLEMS = {
    "smooth_sines_2d": Problem("smooth_sines_2d", 2, smooth_sines),
    "rosenbrock_2d": Problem("rosenbrock_2d", 2, rosenbrock),
    "eggholder_2d": Problem("eggholder_2d", 2, eggholder),
    "rastrigin_3d": Problem("rastrigin_3d", 3, rastrigin),
    "step_3d": Problem("step_3d", 3, discontinuous_step),
}
