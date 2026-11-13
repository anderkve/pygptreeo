"""Benchmark: GPTree with and without a global model on particle-physics-style targets.

The targets are those of ``particle_physics_targets.py``. A run streams ``N`` points
into a tree, predicting each point before it is used, and then predicts a uniform test
set. Errors are against the noise-free target, in the target's own unit (``rmse``) and
over the range of the target on the test set (``nrmse``). ``curve`` is the prequential
error over the range, batch by batch of 1000 points.

Configurations:
    tree       plain GPTree
    additive   ``GPTree(global_mean='additive')``
    net        ``GPTree(global_mean='net')``: the feature network beside the additive model
    net_only   ``global_mean='net'`` with ``additive=False``: the network alone
    net_only_holdout   the network alone with the training of ``net``: a fifth of the
               points held out, the weights with the lowest error on them published
               (``additive=False, holdout=0.2, patience=None``)
    net_only_cheap     ``net_only_holdout`` whose leaves take up a new version of the
               network by a refit with their kernel hyperparameters kept, as in ``net``

Streams: ``uniform`` (a random scan of the unit cube) and, for the likelihood target,
``sampler`` (every proposal of a Metropolis random walk on the likelihood; the test set
is then drawn around the second half of the walk).

Usage:
    OMP_NUM_THREADS=1 python examples/benchmark_particle_physics.py --targets relic_density --configs tree,additive,net --seeds 1
    python examples/benchmark_particle_physics.py --summarize results/particle_physics/*.jsonl

Each run prints one JSON line prefixed with ``RESULT ``; ``--summarize`` turns such
lines into markdown tables.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from pygptreeo import GPTree, Default_GPR
from particle_physics_targets import TARGETS

CONFIGS = ('tree', 'additive', 'net', 'net_only', 'net_only_holdout', 'net_only_cheap')
CURVE_BATCH = 1000      # points per value of the prequential learning curve
NAMES = {'tree': 'plain tree', 'additive': "`'additive'`", 'net': "`'net'`", 'net_only': "`'net'`, network alone",
         'net_only_holdout': "network alone, held-out selection", 'net_only_cheap': "network alone, held-out selection, cheap refit"}


def make_stream(kind, target, N, rng):
    """Stream inputs and test inputs."""
    d = target.d
    if kind == 'uniform':
        return rng.rand(N, d), rng.rand(3000, d)
    if kind == 'sampler':
        X0 = rng.rand(2000, d); f0 = target(X0)
        x = X0[np.argmax(f0)].copy(); fx = float(f0.max())
        X = np.empty((N, d)); visited = np.empty((N, d))
        for i in range(N):
            proposal = np.clip(x + 0.04 * rng.randn(d), 0, 1); fp = float(target(proposal)[0]); X[i] = proposal
            if np.log(rng.rand()) < fp - fx:
                x, fx = proposal, fp
            visited[i] = x
        centres = visited[N // 2:][rng.randint(0, N - N // 2, 3000)]
        return X, np.clip(centres + 0.02 * rng.randn(3000, d), 0, 1)
    raise ValueError(f"unknown stream '{kind}'")


def make_tree(config, seed):
    kw = dict(GPR=Default_GPR(n_restarts_optimizer=1), Nbar=100, theta=1e-4, retrain_every_n_points=25,
              splitting_strategy='gradual', use_calibrated_sigma=True)
    if config == 'tree':
        return GPTree(**kw)
    if config == 'additive':
        return GPTree(global_mean='additive', **kw)
    if config in ('net', 'net_only', 'net_only_holdout', 'net_only_cheap'):
        net = dict(additive=config == 'net', steps_per_update=8, random_state=seed)
        if config in ('net_only_holdout', 'net_only_cheap'):
            net.update(holdout=0.2, patience=None)
        gpt = GPTree(global_mean='net', global_mean_kwargs=net, **kw)
        if config == 'net_only_cheap':
            gpt.global_mean.cheap_leaf_refresh = True
        return gpt
    raise ValueError(config)


def run_one(target_name, config, seed, N, stream='uniform'):
    target = TARGETS[target_name]
    rng = np.random.RandomState(seed)
    X, X_test = make_stream(stream, target, N, rng)
    y, sigma, truth = target.sample(X, rng)
    truth_test = target(X_test); y_range = truth_test.max() - truth_test.min()
    np.random.seed(seed)
    gpt = make_tree(config, seed)
    err = np.empty(N); sd = np.empty(N); t_predict = 0.0; t_update = 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(N):
            xi = X[i:i + 1]
            t0 = time.perf_counter(); mu, s = gpt.predict(xi); t1 = time.perf_counter()
            gpt.update_tree(xi, np.array([[y[i]]]), np.array([[sigma[i]]])); t2 = time.perf_counter()
            err[i] = mu[0, 0] - truth[i]; sd[i] = s[0, 0]; t_predict += t1 - t0; t_update += t2 - t1
        P, S = gpt.predict(X_test, mode='loop')
    e_test = P[:, 0] - truth_test; half = slice(N // 2, N)
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
    out = {
        'target': target_name, 'd': target.d, 'unit': target.unit, 'config': config, 'seed': seed, 'N': N, 'stream': stream,
        'test_rmse': rmse(e_test), 'test_nrmse': rmse(e_test) / y_range,
        'test_median': float(np.median(np.abs(e_test))), 'test_q95': float(np.percentile(np.abs(e_test), 95)),
        'test_max': float(np.abs(e_test).max()),
        'prequential_rmse': rmse(err[half]), 'prequential_nrmse': rmse(err[half]) / y_range,
        'coverage_test': float(np.mean(np.abs(e_test) <= S[:, 0])), 'coverage_prequential': float(np.mean(np.abs(err[half]) <= sd[half])),
        'update_ms': 1e3 * t_update / N, 'predict_ms': 1e3 * t_predict / N, 'seconds': round(t_update + t_predict, 1),
        'leaves': len(gpt.root.leaves),
        'curve': [rmse(err[i:i + CURVE_BATCH]) / y_range for i in range(0, N - CURVE_BATCH + 1, CURVE_BATCH)],
    }
    gm = gpt.global_mean
    if config == 'net':
        out['weights'] = [float(w) for w in gm.weights]
    additive = gm if config == 'additive' else getattr(gm, 'additive', None)
    if additive is not None:
        out['additive_share'] = additive.additive_share()
    return out


def summarize(lines):
    rows = {}
    for line in lines:
        if line.startswith('RESULT '):
            r = json.loads(line[7:]); rows.setdefault((r['stream'], r['N'], r['target']), {}).setdefault(r['config'], []).append(r)
    geo = lambda v: float(np.exp(np.mean(np.log(v))))
    fmt = lambda v: f"{v:.2e}" if v < 1e-3 else f"{v:.4f}"
    out = []
    for stream, N in sorted({k[:2] for k in rows}):
        sel = {k[2]: v for k, v in rows.items() if k[:2] == (stream, N)}
        configs = [c for c in CONFIGS if any(c in v for v in sel.values())]
        for key, title in (('test_nrmse', 'test RMSE / range'), ('test_rmse', "test RMSE in the target's unit"), ('prequential_nrmse', 'prequential RMSE / range'),
                           ('coverage_test', 'one-sigma coverage on the test set'), ('seconds', 'seconds')):
            out += [f"\n{stream} stream, {N} points: {title}\n", "| target | " + " | ".join(NAMES[c] for c in configs) + " |", "|---|" + "---|" * len(configs)]
            for t in TARGETS:
                if t not in sel:
                    continue
                agg = np.mean if key in ('coverage_test', 'seconds') else geo
                vals = [agg([r[key] for r in sel[t][c]]) if c in sel[t] else None for c in configs]
                best = min(v for v in vals if v is not None) if key.endswith('rmse') else None
                cell = lambda v: "" if v is None else (f"{v:.2f}" if key == 'coverage_test' else f"{v:.0f}" if key == 'seconds'
                                                        else f"**{fmt(v)}**" if v == best else fmt(v))
                out.append(f"| {t} ({TARGETS[t].d}D, {TARGETS[t].unit}) | " + " | ".join(cell(v) for v in vals) + " |")
            if key == 'test_nrmse' and 'tree' in configs:
                for c in configs[1:]:
                    r = [a[key] / b[key] for t in sel if c in sel[t] and 'tree' in sel[t]
                         for a in sel[t][c] for b in sel[t]['tree'] if a['seed'] == b['seed']]
                    if r:
                        out.append(f"\n{NAMES[c]} relative to the plain tree: geometric mean {geo(r):.2f}, lower in {sum(x < 1 for x in r)} of {len(r)} runs, "
                                   f"{min(r):.2g} to {max(r):.2f}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--targets', default='all', help='comma-separated target names, or all')
    ap.add_argument('--configs', default='tree,additive,net')
    ap.add_argument('--seeds', default='1')
    ap.add_argument('--N', type=int, default=10000)
    ap.add_argument('--stream', default='uniform', choices=('uniform', 'sampler'))
    ap.add_argument('--summarize', nargs='*', help='turn RESULT lines from these files (or stdin) into tables')
    args = ap.parse_args()
    if args.summarize is not None:
        lines = []
        for f in (args.summarize or ['-']):
            lines += (sys.stdin if f == '-' else open(f)).readlines()
        print(summarize(lines))
        return
    targets = list(TARGETS) if args.targets == 'all' else args.targets.split(',')
    for seed in [int(s) for s in args.seeds.split(',')]:
        for target in targets:
            for config in args.configs.split(','):
                print("RESULT " + json.dumps(run_one(target, config, seed, args.N, args.stream)), flush=True)


if __name__ == '__main__':
    main()
