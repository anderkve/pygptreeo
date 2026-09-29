"""Benchmark: Bayesian-quadratic-leaf tree (BQTree) vs GPTree, with and without the global model.

Configurations (``--configs``):
    tree      GPTree with the ARD Matern default leaf kernel
    global    the same tree with ``global_mean='additive_gp'`` (global additive GP + residual leaves)
    bq        BQTree with quadratic leaves (rss split criterion)
    bq:<k>=<v>+<k>=<v>   BQTree with overridden constructor arguments, e.g. ``bq:Nbar=200+degree=1``
    tree:<k>=<v>+...     GPTree with overridden Nbar / theta / retrain_every_n_points

Per (target, stream, config, seed) one JSON line (prefixed ``RESULT ``) is printed and
appended to ``--out``: prequential NRMSE after warm-up, NRMSE on a uniform test set and
on a focus test set drawn where the stream ended, 1-sigma coverage and RMS sigma / RMSE
for the three, per-block prequential NRMSE (learning curve), mean update and single-point
predict times, leaves and wall time.

Usage:
    python examples/benchmark_bqtree.py --target rotated_rosenbrock --streams uniform,sweeping \\
        --configs tree,global,bq --seeds 1,2,3 --out results/bqtree/rr.jsonl
    python examples/benchmark_bqtree.py --summarize results/bqtree/*.jsonl
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

from pygptreeo import GPTree, Default_GPR, AdditiveGPGlobalMean, BQTree
import target_functions as tf
from input_streams import make_stream

TARGETS = {
    'rotated_rosenbrock': tf.RotatedRosenbrock,
    'gaussian_peaks': tf.GaussianPeaks,
    'rosenbrock': tf.Rosenbrock,
    'rastrigin': tf.Rastrigin,
    'levy': tf.Levy,
    'eggholder': tf.Eggholder,
    'himmelblau': tf.Himmelblau,
}


def _parse_overrides(spec):
    out = {}
    if not spec:
        return out
    for item in spec.split('+'):
        k, v = item.split('=')
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        out[k] = v
    return out


def build_model(config, seed, nbar, calibrate):
    name, _, spec = config.partition(':')
    over = _parse_overrides(spec)
    if name in ('tree', 'global'):
        kw = dict(Nbar=over.pop('Nbar', nbar), theta=over.pop('theta', 1e-4),
                  retrain_every_n_points=over.pop('retrain_every_n_points', 25),
                  splitting_strategy='gradual', use_calibrated_sigma=calibrate)
        if name == 'global':
            kw['global_mean'] = AdditiveGPGlobalMean(reservoir_size=over.pop('reservoir_size', 500),
                                                     min_points=200, min_turnover=0.25,
                                                     n_restarts_optimizer=over.pop('global_restarts', 2),
                                                     random_state=seed)
        if over:
            raise ValueError(f"unknown overrides for {name}: {over}")
        return GPTree(GPR=Default_GPR(n_restarts_optimizer=1), **kw)
    if name == 'bq':
        kw = dict(Nbar=nbar, theta=1e-4, degree=2, split_criterion='rss', rebuild_every=10,
                  use_calibrated_sigma=calibrate)
        kw.update(over)
        return BQTree(**kw)
    raise ValueError(config)


def run_one(target_name, stream, config, seed, d, N, nbar, calibrate=True, block=250):
    target = TARGETS[target_name]
    rng = np.random.RandomState(seed)
    X, X_focus = make_stream(stream, target, d, N, rng)
    y = target(X.T); sig = np.maximum(1e-3 * np.abs(y), 1e-6)
    X_uni = rng.rand(3000, d); y_uni = target(X_uni.T); y_focus = target(X_focus.T)
    yrange = y_uni.max() - y_uni.min()
    warmup = max(500, N // 6)

    np.random.seed(seed)
    model = build_model(config, seed, nbar, calibrate)
    errs = np.empty(N); sds = np.empty(N); t_upd = np.empty(N); t_pred = np.empty(N)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(N):
            xi = X[i:i + 1]
            ta = time.perf_counter()
            mu, sd = model.predict(xi)
            tb = time.perf_counter()
            errs[i] = mu[0, 0] - y[i]; sds[i] = sd[0, 0]
            model.update_tree(xi, np.array([[y[i]]]), np.array([[sig[i]]]))
            tc = time.perf_counter()
            t_pred[i] = tb - ta; t_upd[i] = tc - tb
        P_uni, S_uni = model.predict(X_uni, mode='loop')
        P_focus, S_focus = model.predict(X_focus, mode='loop')
    elapsed = time.time() - t0
    e_uni = P_uni[:, 0] - y_uni; e_focus = P_focus[:, 0] - y_focus
    nblocks = N // block
    curve = [float(np.sqrt(np.mean(errs[k * block:(k + 1) * block] ** 2)) / yrange) for k in range(nblocks)]
    out = {
        'target': target_name, 'stream': stream, 'config': config, 'seed': seed, 'd': d, 'N': N, 'nbar': nbar,
        'prequential_nrmse': float(np.sqrt(np.mean(errs[warmup:] ** 2)) / yrange),
        'uniform_nrmse': float(np.sqrt(np.mean(e_uni ** 2)) / yrange),
        'focus_nrmse': float(np.sqrt(np.mean(e_focus ** 2)) / yrange),
        'coverage_prequential': float(np.mean(np.abs(errs[warmup:]) <= sds[warmup:])),
        'coverage_uniform': float(np.mean(np.abs(e_uni) <= S_uni[:, 0])),
        'coverage_focus': float(np.mean(np.abs(e_focus) <= S_focus[:, 0])),
        'sigma_over_rmse_prequential': float(np.sqrt(np.mean(sds[warmup:] ** 2)) / np.sqrt(np.mean(errs[warmup:] ** 2))),
        'sigma_over_rmse_uniform': float(np.sqrt(np.mean(S_uni[:, 0] ** 2)) / np.sqrt(np.mean(e_uni ** 2))),
        'sigma_over_rmse_focus': float(np.sqrt(np.mean(S_focus[:, 0] ** 2)) / np.sqrt(np.mean(e_focus ** 2))),
        'curve_block': block, 'curve_nrmse': curve,
        'update_ms': float(1e3 * np.mean(t_upd)), 'update_ms_max': float(1e3 * np.max(t_upd)),
        'predict_ms': float(1e3 * np.mean(t_pred)),
        'leaves': len(model.root.leaves), 'seconds': round(elapsed, 1),
    }
    gm = getattr(model, 'global_mean', None)
    if gm is not None:
        out.update(refits=gm.n_refits)
    return out


def summarize(lines, curves=False):
    rows = [json.loads(l[len('RESULT '):] if l.startswith('RESULT ') else l) for l in lines if l.strip()]
    groups = {}
    for r in rows:
        groups.setdefault((r['target'], r['stream'], r['config']), []).append(r)

    def cell(vals, fmt='{:.4f}'):
        vals = list(vals)
        m = np.mean(vals)
        return fmt.format(m) if len(vals) == 1 else (fmt + ' (' + fmt + '..' + fmt + ')').format(m, min(vals), max(vals))

    out = ["| target | stream | config | seeds | prequential NRMSE | uniform-test NRMSE | focus-test NRMSE | "
           "coverage preq / uni / focus | sigma/RMSE preq / uni / focus | update ms | predict ms | leaves | time [s] |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (tgt, strm, cfg), rs in sorted(groups.items()):
        cov = ' / '.join(cell((r[k] for r in rs), '{:.2f}') for k in ('coverage_prequential', 'coverage_uniform', 'coverage_focus'))
        rat = ' / '.join(cell((r[k] for r in rs), '{:.2f}') for k in ('sigma_over_rmse_prequential', 'sigma_over_rmse_uniform', 'sigma_over_rmse_focus'))
        out.append(f"| {tgt} | {strm} | {cfg} | {len(rs)} | {cell(r['prequential_nrmse'] for r in rs)} | "
                   f"{cell(r['uniform_nrmse'] for r in rs)} | {cell(r['focus_nrmse'] for r in rs)} | {cov} | {rat} | "
                   f"{np.mean([r['update_ms'] for r in rs]):.2f} | {np.mean([r['predict_ms'] for r in rs]):.2f} | "
                   f"{np.mean([r['leaves'] for r in rs]):.0f} | {np.mean([r['seconds'] for r in rs]):.0f} |")
    if curves:
        out.append("")
        out.append("Learning curves: prequential NRMSE per block of points (mean over seeds)")
        for (tgt, strm, cfg), rs in sorted(groups.items()):
            c = np.mean([r['curve_nrmse'] for r in rs], axis=0)
            out.append(f"{tgt} | {strm} | {cfg} | block={rs[0]['curve_block']} | " + ' '.join(f"{v:.4f}" for v in c))
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target', default='rotated_rosenbrock', choices=sorted(TARGETS))
    ap.add_argument('--streams', default='uniform,focusing,sweeping,walker')
    ap.add_argument('--configs', default='tree,global,bq')
    ap.add_argument('--seeds', default='1')
    ap.add_argument('--d', type=int, default=6)
    ap.add_argument('--N', type=int, default=4000)
    ap.add_argument('--nbar', type=int, default=100)
    ap.add_argument('--no-calibrate', action='store_true')
    ap.add_argument('--out', default=None, help='append RESULT lines to this file')
    ap.add_argument('--summarize', nargs='*', help='aggregate RESULT lines from these files (or stdin) into a table')
    ap.add_argument('--curves', action='store_true', help='with --summarize: also print learning curves')
    args = ap.parse_args()

    if args.summarize is not None:
        lines = []
        if args.summarize:
            for f in args.summarize:
                with open(f) as fh:
                    lines += fh.readlines()
        else:
            lines = sys.stdin.readlines()
        print(summarize(lines, curves=args.curves))
        return

    for stream in args.streams.split(','):
        for config in args.configs.split(','):
            for seed in [int(s) for s in args.seeds.split(',')]:
                res = run_one(args.target, stream, config, seed, args.d, args.N, args.nbar,
                              calibrate=not args.no_calibrate)
                line = 'RESULT ' + json.dumps(res)
                print(line, flush=True)
                if args.out:
                    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                    with open(args.out, 'a') as fh:
                        fh.write(line + '\n')


if __name__ == '__main__':
    main()
