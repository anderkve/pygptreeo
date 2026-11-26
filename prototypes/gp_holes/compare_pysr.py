"""PySR on the same data as run_benchmark.py and run_series.py.

    python compare_pysr.py [--targets benchmark|series] [--seconds 60] [--seeds 3] [--only NAME ...]

Needs `pip install pysr`. A result counts as right if its error outside the
training box is below the noise level, as in run_benchmark.py. "front" counts
the runs in which some equation of PySR's Pareto front is right, selected or not.
"""
import argparse
import time

import numpy as np
from pysr import PySRRegressor

from run_benchmark import data, errors
from targets import SERIES, TARGETS


def fit(X, y, seconds):
    model = PySRRegressor(
        binary_operators=['+', '-', '*', '/'],
        unary_operators=['sin', 'cos', 'exp', 'log', 'sqrt', 'square'],
        niterations=1000000, timeout_in_seconds=seconds,
        verbosity=0, progress=False, temp_equation_file=True)
    return model.fit(X, y)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default='benchmark')
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--n', type=int, nargs='*', default=[100])
    ap.add_argument('--noise', type=float, nargs='*', default=[0.01])
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    fit(rng.uniform(1, 2, (20, 2)), rng.uniform(1, 2, 20), 20)       # compiles the search
    summary = []
    for target in (SERIES if a.targets == 'series' else TARGETS):
        if a.only and target[0] not in a.only:
            continue
        print(f"\n{target[0]}:  {target[1]}", flush=True)
        for n in a.n:
            for noise in a.noise:
                right = front = 0
                for seed in range(a.seeds):
                    X, y, ystd, tests = data(target, n, noise, seed)
                    t0 = time.time()
                    model = fit(X, y, a.seconds)
                    with np.errstate(all='ignore'):
                        err = errors(model.predict, tests)
                        best = np.nanmin([errors(lambda Z: model.predict(Z, index=i), tests)['out']
                                          for i in range(len(model.equations_))])
                    right += err['out'] < noise
                    front += best < noise
                    print(f"  n {n} noise {noise:g} seed {seed}: {model.get_best()['equation']}\n"
                          f"          rmse in {err['in']:.4f} out {err['out']:.4f}, best on the "
                          f"front {best:.4f}, {time.time() - t0:.0f} s", flush=True)
                summary.append((target[0], n, noise, right, front))
    print(f"\n{'target':12s} {'n':>5s} {'noise':>6s} {'right':>5s} {'front':>5s}")
    for row in summary:
        print("{:12s} {:5d} {:6g} {:5d} {:5d}".format(*row))
