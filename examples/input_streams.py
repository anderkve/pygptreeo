"""Input-stream generators shared by the benchmark scripts.

    uniform    independent uniform samples over the unit cube
    focusing   differential-evolution-like: a broad uniform phase, then a
               Gaussian cluster around the target's minimum whose width shrinks
               geometrically (0.3 -> 0.02)
    sweeping   a Gaussian cluster (width 0.12) whose centre moves along a smooth
               path through the cube
    walker     MCMC-like: every proposal of a Metropolis random walk on
               exp(-f / T), accepted or not

``make_stream`` returns the stream inputs and a 3000-point "focus" test set
drawn where the stream ended up (another uniform sample for ``uniform``).
Same definitions as in ``benchmark_global_mean_streams.py``.
"""

import numpy as np
from scipy.optimize import minimize


def find_minimum(target, d, rng):
    X0 = rng.rand(400, d)
    f0 = target(X0.T)
    best = None
    for i in np.argsort(f0)[:5]:
        res = minimize(lambda x: float(target(x)), X0[i], method='L-BFGS-B', bounds=[(0, 1)] * d)
        if best is None or res.fun < best.fun:
            best = res
    return np.clip(best.x, 0, 1)


def make_stream(kind, target, d, N, rng, n_test=3000):
    if kind == 'uniform':
        return rng.rand(N, d), rng.rand(n_test, d)

    if kind == 'focusing':
        n_broad = max(500, N // 6)
        xmin = find_minimum(target, d, rng)
        width = 0.3 * (0.02 / 0.3) ** (np.clip(np.arange(N) - n_broad, 0, None) / (N - n_broad))
        broad = rng.rand(N, d)
        cluster = np.clip(xmin + width[:, None] * rng.randn(N, d), 0, 1)
        X = np.where((np.arange(N) < n_broad)[:, None], broad, cluster)
        return X, np.clip(xmin + 0.03 * rng.randn(n_test, d), 0, 1)

    if kind == 'sweeping':
        s = np.linspace(0, 1, N)
        centre = 0.5 + 0.35 * np.column_stack([np.sin(2 * np.pi * (s * 0.75 + 0.2 * j)) for j in range(d)])
        X = np.clip(centre + 0.12 * rng.randn(N, d), 0, 1)
        s_end = rng.uniform(0.9, 1.0, n_test)
        c_end = 0.5 + 0.35 * np.column_stack([np.sin(2 * np.pi * (s_end * 0.75 + 0.2 * j)) for j in range(d)])
        return X, np.clip(c_end + 0.12 * rng.randn(n_test, d), 0, 1)

    if kind == 'walker':
        X0 = rng.rand(400, d)
        f0 = target(X0.T)
        T = 0.05 * (np.percentile(f0, 90) - f0.min())
        x = X0[np.argmin(f0)].copy(); fx = float(target(x))
        X = np.empty((N, d)); visited = []
        for i in range(N):
            prop = np.clip(x + 0.05 * rng.randn(d), 0, 1)
            fp = float(target(prop))
            X[i] = prop
            if np.log(rng.rand()) < -(fp - fx) / T:
                x, fx = prop, fp
            visited.append(x.copy())
        visited = np.array(visited[N // 2:])
        idx = rng.randint(0, visited.shape[0], n_test)
        return X, np.clip(visited[idx] + 0.02 * rng.randn(n_test, d), 0, 1)

    raise ValueError(f"unknown stream kind '{kind}'")
