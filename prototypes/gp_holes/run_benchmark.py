"""Run the GP-hole search on the target functions.

    python run_benchmark.py [--seeds 3] [--n 100] [--noise 0.01] [--kernel matern52] [--only NAME ...]
"""
import argparse

import numpy as np

from gpholes import search
from targets import TARGETS, sample


def run(target, n, noise, seed, verbose, **kw):
    rng = np.random.default_rng(seed)
    X = sample(target, n, rng)
    f = target[2](X)
    ystd = f.std()
    y = f / ystd + noise * rng.standard_normal(n)
    model, trace, info = search(X, y, seed=seed, verbose=verbose, **kw)
    err = {}
    for name, widen in (('in', 0.0), ('out', 0.25)):
        Xs = sample(target, 2000, rng, widen)
        with np.errstate(all='ignore'):
            fs = target[2](Xs)
        ok = np.isfinite(fs)
        err[name] = np.sqrt(np.mean((model.predict(Xs[ok]) - fs[ok] / ystd) ** 2))
    return model.text(ystd), model.n_holes(), err, trace, info


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--noise', type=float, default=0.01)
    ap.add_argument('--beam', type=int, default=4)
    ap.add_argument('--kernel', default='matern52')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--node-cost', type=float, default=3.0)
    ap.add_argument('--power-cost', type=float, default=10.0)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    summary = []
    for target in TARGETS:
        if a.only and target[0] not in a.only:
            continue
        print(f"\n{target[0]}:  {target[1]}", flush=True)
        runs = []
        for seed in range(a.seeds):
            text, holes, err, trace, info = run(target, a.n, a.noise, seed, a.verbose, beam=a.beam,
                                                 kernel=a.kernel, workers=a.workers, node_cost=a.node_cost,
                                                 power_cost=a.power_cost)
            print(f"  seed {seed}: {text}\n          holes {holes}, rmse in {err['in']:.4f} "
                  f"out {err['out']:.4f}, {len(trace) - 1} steps, {info['n_fits']} fits, "
                  f"{info['seconds']:.0f} s", flush=True)
            runs.append((holes, err['out'], info['seconds']))
        summary.append((target[0], runs))
    # A formula without holes counts as right if it also holds outside the training box.
    print(f"\n{'target':12s} {'right':>5s} {'holes':>5s} {'other':>5s} {'seconds':>8s}")
    for name, runs in summary:
        right = sum(h == 0 and e < a.noise for h, e, _ in runs)
        holes = sum(h > 0 for h, _, _ in runs)
        print(f"{name:12s} {right:5d} {holes:5d} {len(runs) - right - holes:5d} "
              f"{np.mean([s for _, _, s in runs]):8.0f}")
