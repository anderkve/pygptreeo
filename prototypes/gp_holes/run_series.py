"""The GP-hole search on targets that have a series expansion and no closed form.

    python run_series.py [--noise 0.1 0.01 0.001] [--library full powers poly full+poly] [--seeds 2]

For reference, the least-squares fit of the first K terms of the series is
printed first: its error, and its coefficients beside those of the series.
"""
import argparse

import numpy as np

from run_benchmark import run
from targets import SERIES, sample


LIBRARIES = {
    'full': ('pow', 'log', 'exp', 'aff', 'sin'),
    'powers': ('pow',),
    'poly': ('pow', 'poly'),
    'full+poly': ('pow', 'log', 'exp', 'aff', 'sin', 'poly'),
}


def reference(target, n_terms):
    """Error and coefficients of the least-squares fit of the first terms of the series."""
    X = sample(target, 20000, np.random.default_rng(0))
    f = target[2](X)
    A = np.column_stack([target[7](X) * X[:, target[4]] ** p for p in target[5][:n_terms]])
    c = np.linalg.lstsq(A, f, rcond=None)[0]
    return np.sqrt(np.mean((A @ c - f) ** 2)) / f.std(), c


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--noise', type=float, nargs='*', default=[0.1, 0.01, 0.001])
    ap.add_argument('--library', nargs='*', default=list(LIBRARIES))
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    for target in SERIES:
        if a.only and target[0] not in a.only:
            continue
        print(f"\n{target[0]}:  {target[1]}")
        print("  series coefficients    " + ' '.join(f"{c:9.4g}" for c in target[6]))
        for k in range(1, len(target[5]) + 1):
            err, c = reference(target, k)
            print(f"  best {k} terms: error {err:7.1e}  " + ' '.join(f"{v:9.4g}" for v in c))
        for library in a.library:
            for noise in a.noise:
                for seed in range(a.seeds):
                    text, holes, err, trace, info = run(target, a.n, noise, seed, False, max_power=9,
                                                        library=LIBRARIES[library], workers=a.workers)
                    print(f"  {library:9s} noise {noise:<6g} seed {seed}: {text}\n          holes {holes}, "
                          f"rmse in {err['in']:.4f} out {err['out']:.4f}, {info['seconds']:.0f} s", flush=True)
