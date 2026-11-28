"""The search on every set of targets, with the score of the right formula
beside each result.

    python run_baseline.py NAME [--sets benchmark srsd srsd2 srsd3 empirical docs] [--only TARGET ...] [--check]
                           [--symmetry off|add|replace|first]

Writes results/baseline_NAME.json, with one entry per run, and the log
results/baseline_NAME.txt. A call with a name that has been used goes on where
the earlier one stopped.

benchmark: the targets of run_benchmark.py, 100 points, 1% noise. The other sets
are those of run_held_out.py. A result of benchmark is right if it has no hole
and its rms error outside the training box is below the noise, one of the srsd
sets if it has no hole and its median relative error is below 1% for the wider
ranges. The results of empirical and docs are judged by the form of the formula.

With --check only the 41 runs of CHECK are done, which take about half an hour
in place of the hours of all 150: the runs that the changes to the search have
been tried on.

The right formula is that of truth.py, fitted to the same data. A result that
is not right is put down to the search if the right formula scores higher by
more than 1, and to the score if not.

The alternatives of a result are the models that score less than WITHIN below
it. Each is judged as the result is, and the report counts the runs in which
the result or one of its alternatives is right, also for the distances GAPS.
"""
import argparse
import json
import os

import numpy as np

import run_benchmark
import run_held_out
from gpholes import CFG, search
from targets import TARGETS
from truth import truth

SEEDS = dict(benchmark=3, srsd=2, srsd2=2, srsd3=2, empirical=3, docs=3)
NOISE = 0.01            # of the benchmark, and the largest relative error of a right result of srsd
WITHIN = CFG['within']  # how far below the score of the result an alternative can be
GAPS = (1, 3, 5, 10)    # the distances that the report counts the alternatives for
# The check runs: the seeds of some targets of each set.
CHECK = dict(
    benchmark={'toy': (0,), 'I.14.3': (0,), 'I.12.2': (0,), 'I.6.2': (0,), 'I.12.11': (0,), 'I.37.4': (0, 1),
               'I.50.26': (0, 1), 'I.10.7': (0,), 'I.30.5': (0,), 'I.8.14': (0, 1), 'I.18.4': (0, 1, 2),
               'II.11.28': (0, 1, 2)},
    srsd={'i.6.20b': (0, 1), 'i.12.4': (0,), 'i.14.4': (0,), 'i.27.6': (0, 1), 'i.39.11': (0,), 'ii.11.20': (0,),
          'ii.13.34': (0,), 'iii.10.19': (0,), 'bonus.18': (0,)},
    srsd2={'i.34.14': (0,), 'i.47.23': (0,), 'ii.8.7': (0,), 'bonus.7': (0, 1), 'bonus.19': (0,), 'bonus.20': (0,)},
    empirical={'newton': (0,), 'schechter': (0,), 'bode': (0,), 'rydberg': (0,)})
HERE = os.path.dirname(os.path.abspath(__file__))


def problems(which):
    """(name, formula, seed, X, y, scale of y, test sets, inputs) for each run of
    a set. y and the targets of the test sets are in units of the scale. inputs
    has the numbers of the inputs that were kept, if not all were."""
    if which == 'benchmark':
        for target in TARGETS:
            for seed in range(SEEDS[which]):
                X, y, ystd, tests = run_benchmark.data(target, 100, NOISE, seed)
                yield target[0], target[1], seed, X, y, ystd, tests, None
    else:
        for name, text, seed, X, y, tests, note, _ in run_held_out.problems(which, SEEDS[which]):
            ystd = np.std(y)
            inputs = [int(s[1:]) for s in note.split(': ')[1].split(', ')] if note else None
            yield (name, text, seed, X, y / ystd, ystd, {k: (Xs, fs / ystd) for k, (Xs, fs) in tests.items()},
                   inputs)


def errors(predict, tests):
    """The rms error and the median relative error on each test set."""
    out = {}
    for key, (Xs, fs) in tests.items():
        with np.errstate(all='ignore'):
            e = np.abs(predict(Xs) - fs)
            rel = (e / np.abs(fs))[np.isfinite(e / np.abs(fs))]
        out[key] = dict(rms=float(np.sqrt(np.mean(e ** 2))), rel=float(np.median(rel)) if len(rel) else np.inf)
    return out


def is_right(which, holes, err):
    """Whether a result is right, or None for a set that is judged by the form."""
    if which == 'benchmark':
        return bool(holes == 0 and err['out']['rms'] < NOISE)
    if which.startswith('srsd'):
        return bool(holes == 0 and err['out']['rel'] < NOISE)
    return None


def run(which, X, y, ystd, tests, seed, workers, **cfg):
    """The search on one data set, as an entry of the results. cfg replaces
    entries of the settings of the search."""
    t0 = sum(os.times()[:4])
    try:
        model, trace, info = search(X, y, y_scale=ystd, seed=seed, verbose=False, workers=workers,
                                    within=max(GAPS), **cfg)
    except Exception as error:                  # a run that fails counts as not right
        return dict(crashed=f"{type(error).__name__}: {error}", right=is_right(which, 1, None))
    err = errors(model.predict, tests)
    return dict(result=model.text(ystd), score=model.score, holes=model.n_holes(), key=model.key(),
                err=err, right=is_right(which, model.n_holes(), err),
                alternatives=[(m.score, m.n_holes(), m.text(ystd),
                               is_right(which, m.n_holes(), errors(m.predict, tests)))
                              for m in info['alternatives'][1:]],
                steps=len(trace) - 1, fits=info['n_fits'], seconds=info['seconds'],
                cpu=sum(os.times()[:4]) - t0)


def verdict(r):
    """'right', or what a result that is not right is put down to."""
    t = r.get('truth')
    if r['right'] or 'crashed' in r or t is None:
        return 'right' if r['right'] else 'crashed' if 'crashed' in r else 'no formula'
    if t['score'] > r['score'] + 1:
        return 'search'
    return 'by eye' if r['right'] is None else 'score'


def kept(r, gap=WITHIN):
    """The alternatives of a result that score less than gap below it."""
    return [a for a in r.get('alternatives', []) if r['score'] - a[0] < gap]


def report(runs):
    """The log of the runs, as text."""
    lines, last = [], None
    for key, r in runs.items():
        which, name, seed = key.split('/')
        if (which, name) != last:
            lines.append(f"\n{which} {name}:  {r['target']}   [{r['n']} points, {r['d']} inputs]")
            last = (which, name)
        if 'crashed' in r:
            lines.append(f"  seed {seed}: crashed: {r['crashed']}")
            continue
        err = '; '.join(f"{k}: rms {e['rms']:.2g}, relative {e['rel']:.2g}" for k, e in r['err'].items())
        lines.append(f"  seed {seed}: {r['result']}")
        lines.append(f"          score {r['score']:.1f}, holes {r['holes']}{', RIGHT' if r['right'] else ''}; "
                     f"{err}; {r['steps']} steps, {r['fits']} fits, {r['seconds']:.0f} s")
        lines += [f"          alternative {s - r['score']:+.1f}{', RIGHT' if right else ''}: {text}"
                  for s, _, text, right in kept(r)]
        if r.get('truth'):
            t = r['truth']
            lines.append(f"          right formula {t['score'] - r['score']:+.1f}: {t['text']}")
    # For each target the number of runs of each verdict, and the score of the
    # result less that of the right formula, which has a * if it has a hole.
    # "kept" has the runs in which the result or one of its alternatives is right.
    kinds = ('right', 'search', 'score', 'by eye', 'no formula', 'crashed')
    lines.append(f"\n{'set':10s} {'target':12s} " + ' '.join(f"{k:>10s}" for k in kinds) +
                 f" {'kept':>5s} {'seconds':>8s}  result - right formula")
    groups = {}
    for key, r in runs.items():
        groups.setdefault(tuple(key.split('/')[:2]), []).append(r)
    for (which, name), rs in groups.items():
        counts = [sum(verdict(r) == k for r in rs) for k in kinds]
        diffs = ', '.join(f"{r['score'] - r['truth']['score']:+.1f}{'*' if r['truth']['holes'] else ''}"
                          for r in rs if r.get('truth') and 'score' in r)
        among = sum(bool(r['right'] or any(a[3] for a in kept(r))) for r in rs)
        lines.append(f"{which:10s} {name:12s} " + ' '.join(f"{c or '':>10}" for c in counts) +
                     f" {among or '':>5} {np.mean([r.get('seconds', np.nan) for r in rs]):8.0f}  {diffs}")
    judged = [r for r in runs.values() if r['right'] is not None and 'score' in r]
    lines.append(f"\n{len(judged)} runs judged by number: the result is right in {sum(r['right'] for r in judged)}")
    for gap in GAPS:
        among = sum(bool(r['right'] or any(a[3] for a in kept(r, gap))) for r in judged)
        lines.append(f"  the result or an alternative within {gap:2d}: {among}, "
                     f"with {np.mean([1 + len(kept(r, gap)) for r in judged]):.1f} models kept on average")
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--sets', nargs='*', default=list(SEEDS))
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--symmetry', default='off')
    a = ap.parse_args()
    path = os.path.join(HERE, 'results', f"baseline_{a.name}")
    runs = json.load(open(path + '.json')) if os.path.exists(path + '.json') else {}
    for which in [w for w in a.sets if w in CHECK] if a.check else a.sets:
        for name, text, seed, X, y, ystd, tests, inputs in problems(which):
            key = f"{which}/{name}/{seed}"
            if ((key in runs and 'truth' in runs[key]) or (a.only and name not in a.only)
                    or (a.check and seed not in CHECK[which].get(name, ()))):
                continue
            if key not in runs:
                runs[key] = dict(target=text, n=len(y), d=X.shape[1],
                                 **run(which, X, y, ystd, tests, seed, a.workers, symmetry=a.symmetry))
            runs[key]['truth'] = truth(name, X, y, ystd, inputs)
            print(key, runs[key].get('result', runs[key].get('crashed')), flush=True)
            json.dump(runs, open(path + '.json', 'w'), indent=1)
            open(path + '.txt', 'w').write(report(runs))
