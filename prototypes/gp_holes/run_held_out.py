"""The GP-hole search and PySR on targets that were not used to develop the search.

    python run_held_out.py --set docs|empirical --code gp_holes|pysr [--seeds 3] [--seconds 60]

docs: the examples of PySR's documentation, with the data drawn as there.
empirical: EmpiricalBench of the PySR paper (empirical_bench.json, written by
make_empirical_bench.py), where the data are fixed and the seed is that of the search.
PySR gets the settings of the documentation or of the paper, and names the inputs from x0.
"""
import argparse
import json
import os
import time

import numpy as np

from targets import PYSR_DOCS

# The settings of PySR in the benchmark of its paper, but for the run time.
PAPER = dict(
    binary_operators=['+', '-', '*', '/'], unary_operators=['square', 'cube', 'exp', 'log', 'sqrt'],
    elementwise_loss='L1DistLoss()', maxsize=30, maxdepth=20, precision=64, warmup_maxsize_by=0.002,
    constraints={'square': 13, 'cube': 13, 'exp': 13, 'log': 13, 'sqrt': 13, '/': (-1, 13),
                 '*': (-1, -1), '+': (-1, -1), '-': (-1, -1)},
    nested_constraints={'/': {'/': 2}, 'exp': {'exp': 0, 'log': 1, 'sqrt': 0},
                        'square': {'square': 1, 'cube': 1, 'log': 0, 'sqrt': 0},
                        'cube': {'cube': 1, 'square': 1}, 'log': {'log': 0, 'exp': 1},
                        'sqrt': {'sqrt': 1, 'exp': 1, 'square': 0}})
DOCS = dict(binary_operators=['+', '-', '*', '/'],
            unary_operators=['sin', 'cos', 'exp', 'log', 'sqrt', 'square'])


def docs_data(target, seed):
    """The data of an example of the documentation, and test sets from the same
    and from 1.5 times wider inputs."""
    name, text, f, spec = target
    rng = np.random.default_rng(seed)
    X = spec['scale'] * rng.standard_normal((spec['n'], spec['d']))
    y = f(X) + spec['noise'] * rng.standard_normal(spec['n'])
    tests = {}
    for key, wide in (('in', 1.0), ('out', 1.5)):
        Xs = wide * spec['scale'] * rng.standard_normal((2000, spec['d']))
        tests[key] = (Xs, f(Xs))
    note = ''
    if 'select' in spec:                # the selection of inputs that the documentation uses
        from pysr.feature_selection import run_feature_selection
        keep = run_feature_selection(X, y, spec['select'], random_state=np.random.RandomState(seed))
        X, tests = X[:, keep], {k: (Xs[:, keep], fs) for k, (Xs, fs) in tests.items()}
        note = 'inputs kept: ' + ', '.join(f"x{j + 1}" for j in np.flatnonzero(keep))
    return X, y, tests, note


def problems(which, seeds):
    """(name, formula, seed, X, y, test sets, note, settings for PySR) for each run."""
    if which == 'docs':
        for target in PYSR_DOCS:
            for seed in range(seeds):
                X, y, tests, note = docs_data(target, seed)
                yield (target[0], target[1], seed, X, y, tests, note,
                       dict(DOCS, denoise=bool(target[3].get('denoise'))))
    else:
        for p in json.load(open(os.path.join(os.path.dirname(__file__), 'empirical_bench.json'))):
            for seed in range(seeds):
                yield (p['key'], f"{p['target']}   ({p['source']}, inputs {', '.join(p['inputs'])})",
                       seed, np.array(p['X']), np.array(p['y']), {}, '', PAPER)


def errors(predict, tests, ystd):
    """Rms and median absolute error on each test set, in units of the scale of the target."""
    out = []
    for key, (Xs, fs) in tests.items():
        with np.errstate(all='ignore'):
            e = np.abs(predict(Xs) - fs) / ystd
        out.append(f"{key}: rms {np.sqrt(np.mean(e ** 2)):.4f}, median {np.median(e):.4f}")
    return '; '.join(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='docs')
    ap.add_argument('--code', default='gp_holes')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    if a.code == 'pysr':
        from pysr import PySRRegressor
    last = None
    for name, text, seed, X, y, tests, note, settings in problems(a.set, a.seeds):
        if a.only and name not in a.only:
            continue
        if name != last:
            print(f"\n{name}:  {text}   [{len(y)} points, {X.shape[1]} inputs]", flush=True)
            last = name
        ystd, t0 = np.std(y), time.time()
        if a.code == 'gp_holes':
            from gpholes import search
            model, trace, info = search(X, y / ystd, seed=seed, verbose=False, workers=a.workers)
            result, front = model.text(ystd), [f"holes {model.n_holes()}"]
            predict = lambda Z: model.predict(Z) * ystd
        else:
            model = PySRRegressor(niterations=1000000, timeout_in_seconds=a.seconds, verbosity=0,
                                  progress=False, temp_equation_file=True, **settings).fit(X, y)
            result, predict = model.get_best()['equation'], model.predict
            front = [f"front {int(c):3d}  {e}" for c, e in zip(model.equations_['complexity'],
                                                              model.equations_['equation'])]
        print(f"  seed {seed}: {result}\n          {errors(predict, tests, ystd)}  "
              f"{time.time() - t0:.0f} s  {note}", flush=True)
        for line in front:
            print(f"          {line}", flush=True)
