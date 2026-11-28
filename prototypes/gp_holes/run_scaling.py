"""The GP-hole search on larger data sets.

    python run_scaling.py NAME [--set benchmark|srsd|srsd2|srsd3] [--n 10000] [--noise 0.01 0.1 0.3]
                          [--m 100] [--local 0] [--only TARGET ...] [--seeds 1]

For each target, n points with noise are drawn, and a fifth of them is held
out. The noise of the benchmark is of the same size at every point, in units
of the spread of the target; that of the SRSD equations is in proportion to
the target. Ways to use the other points (see bigdata.py and local.py):

    subsample   the search on 100 of the points, and its best model fitted to all
    validated   the same search, and of its 10 best models, fitted to all the
                points, the one with the best score on the held-out points (at
                most 2 of them with a hole, and each model with a hole also
                without it)
    distilled   the search on the predictions of a GPTree of all the points at
                m of them, and its 10 best models validated in the same way
    local       with --local N, in place of the three others: the search on N
                of the points with holes of local GPs, and its 10 best models
                validated in the same way

A result is right if it has no hole and, for the benchmark, its rms error
outside the box of the data is below 1% of the spread of the target, whatever
the noise; for an SRSD equation, if its median relative error for the wider
ranges is below 1%. Writes results/scaling_NAME.txt and .json, and goes on
where an earlier call stopped.
"""
import argparse
import copy
import json
import os
import time

import numpy as np

import run_held_out
from bigdata import distil, validate
from gpholes import search
from local import LocalModel
from targets import TARGETS, sample

HERE = os.path.dirname(os.path.abspath(__file__))
TOLERANCE = 0.01
KEEP = 10           # the best models of a search that are validated
HOLES = 2           # ... and how many of them can have a hole


def best(models, data, rng):
    """The validated models, the best first: those without a hole, the first
    `HOLES` of those with one, whose trees take most of the time, and every
    model with a hole also without its holes, so that a hole has to earn its
    place on the held-out points."""
    with_hole = [m for m in models if m.n_holes()]
    stripped = []
    for m in with_hole:
        s = copy.deepcopy(m)
        s.terms = [t for t in s.terms if t.feats is None]
        stripped.append(s)
    chosen = {m.key(): m for m in [m for m in models if not m.n_holes()] + with_hole[:HOLES] + stripped}
    fitted = [f for f in (validate(m, *data, rng) for m in chosen.values()) if f is not None]
    return sorted(fitted, key=lambda f: -f.score)


def problems(which, n, noises, seeds, only):
    """(name, formula, noise, seed, X, y, the spread of the target, test inputs
    and targets, whether the error of a result is relative) for each run."""
    if which == 'benchmark':
        for target in TARGETS:
            for noise in noises:
                for seed in range(seeds):
                    if only and target[0] not in only:
                        continue
                    rng = np.random.default_rng(seed)
                    X = sample(target, n, rng)
                    f = target[2](X)
                    y = f + noise * f.std() * rng.standard_normal(n)
                    Xo = sample(target, 2000, rng, 0.25)
                    with np.errstate(all='ignore'):
                        fo = target[2](Xo)
                    yield target[0], target[1], noise, seed, X, y, f.std(), Xo[np.isfinite(fo)], fo[np.isfinite(fo)], False
    else:
        from make_srsd_sample import sample_file
        sample3 = json.load(open(sample_file(int(which[4:] or 1))))
        for eq in sample3['equations']:
            for noise in noises:
                for seed in range(seeds):
                    if only and eq['name'] not in only:
                        continue
                    X, y, tests = run_held_out.srsd_data(eq, sample3['constants'], seed, n, noise)
                    yield eq['name'], eq['formula'], noise, seed, X, y, np.std(y), *tests['out'], True


def run(X, y, ystd, Xo, fo, relative, seed, m, local, workers):
    rng = np.random.default_rng(seed)
    hold = len(y) // 5
    data = (X[hold:], y[hold:], X[:hold], y[:hold])
    out = {}

    def error(fit):
        with np.errstate(all='ignore'):
            e = np.abs(fit.predict(Xo) - fo)
            if relative:
                e = e / np.abs(fo)
                return float(np.median(e[np.isfinite(e)])) if np.isfinite(e).any() else np.inf
            return float(np.sqrt(np.mean(e ** 2)) / ystd)

    def entry(fitted, seconds):
        """The first of the fitted models as a result, and whether any of them is right."""
        if not fitted:
            return dict(result='no model could be fitted', right=False, any_right=False, seconds=seconds)
        fit, right = fitted[0], [f.holes == 0 and error(f) < TOLERANCE for f in fitted]
        return dict(result=fit.text, holes=fit.holes, score=float(fit.score), error=error(fit),
                    right=bool(right[0]), any_right=bool(any(right)), seconds=seconds)

    t0 = time.time()
    if local:
        _, _, info = search(data[0][:local], data[1][:local] / ystd, y_scale=ystd, seed=seed, verbose=False,
                            workers=workers, keep=KEEP, model=LocalModel)
        t1 = time.time()
        out['local'] = entry(best(info['candidates'], data, rng), time.time() - t0)
        out['local'].update(fits=info['n_fits'], search_seconds=t1 - t0)
        return out
    Xs, ys = data[0][:100], data[1][:100]
    _, _, info = search(Xs, ys / ystd, y_scale=ystd, seed=seed, verbose=False, workers=workers, keep=KEEP)
    t1 = time.time()
    first = validate(info['candidates'][0], *data, rng)
    out['subsample'] = entry([first] * (first is not None), t1 - t0)
    out['subsample']['fits'] = info['n_fits']
    out['validated'] = entry(best(info['candidates'], data, rng), time.time() - t0)
    t2 = time.time()
    Xd, yd = distil(data[0], data[1], m, rng)
    t3 = time.time()
    _, _, info = search(Xd, yd / ystd, y_scale=ystd, seed=seed, verbose=False, workers=workers, keep=KEEP)
    out['distilled'] = entry(best(info['candidates'], data, rng), time.time() - t2)
    out['distilled'].update(fits=info['n_fits'], tree_seconds=t3 - t2)
    return out


def report(runs):
    lines, last = [], None
    for key, r in runs.items():
        name, n, noise, seed = key.split('/')[-4:]
        if name != last:
            lines.append(f"\n{name}:  {r['target']}")
            last = name
        lines.append(f"  n {n}, noise {noise}, seed {seed}")
        for way in [w for w in ('subsample', 'validated', 'distilled', 'local') if w in r]:
            e = r[way]
            lines.append(f"    {way:9s} {'RIGHT' if e['right'] else '     '} error {e.get('error', np.nan):.4f}, "
                         f"{e['seconds']:.0f} s:  {e['result']}")
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--set', default='benchmark')
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--m', type=int, default=100)
    ap.add_argument('--local', type=int, default=0)
    ap.add_argument('--noise', nargs='*', type=float, default=[0.01, 0.1, 0.3])
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--seeds', type=int, default=1)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    path = os.path.join(HERE, 'results', f"scaling_{a.name}")
    runs = json.load(open(path + '.json')) if os.path.exists(path + '.json') else {}
    for name, text, noise, seed, *problem in problems(a.set, a.n, a.noise, a.seeds, a.only):
        key = f"{name}/{a.n}/{noise:g}/{seed}"
        if key not in runs:
            runs[key] = dict(target=text, **run(*problem, seed, a.m, a.local, a.workers))
            json.dump(runs, open(path + '.json', 'w'), indent=1)
            open(path + '.txt', 'w').write(report(runs))
        print(key, {w: e['right'] for w, e in runs[key].items() if w != 'target'}, flush=True)
