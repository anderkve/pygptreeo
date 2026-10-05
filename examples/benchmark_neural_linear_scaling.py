"""Benchmark: how update and prediction times scale with the length of the stream.

Streams N points (default 100 000) from a target into a plain GPTree and into
neural-linear trees, timing every ``update_tree`` and every per-point ``predict``
call, and at geometric checkpoints the batch prediction on a fixed uniform test
set. The question is whether the neural-linear tree keeps pygptreeo's bounded
per-point cost on a very long stream, and what the fixed-cost network refits
and the bounded reservoir cost in accuracy.

Configurations:
    tree              plain GPTree (ARD Matern leaves)
    neural            neural-linear tree, network trained on every point seen
    neural_cov<k>     network trained on a maximin coverage reservoir of k points
    neural_uni<k>     network trained on a uniform reservoir sample of k points
    neural_cap<k>     every point kept, but a refit at least every k points (as well
                      as at each doubling); shows the cost of refitting more often
    neural_amort<s>   refits spread over the following updates, s Adam steps per
                      update, published when complete (no latency spike)
    Parts combine: neural_cov4000_amort8.

Usage:
    OMP_NUM_THREADS=1 python examples/benchmark_neural_linear_scaling.py --target gaussian_peaks \\
        --stream walker --d 6 --N 100000 --configs tree,neural,neural_cov4000 --seed 1

Prints one ``RESULT`` JSON line per (config, checkpoint): mean and maximum
update time and mean per-point prediction time over the window since the
previous checkpoint (ms), the batch prediction time per point on the test set,
the test NRMSE, the prequential NRMSE in the window, the number of leaves and
the learner's refit count and fit time.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

from pygptreeo import GPTree, Default_GPR, FeatureNetLearner, NeuralLinearGPR
from benchmark_global_mean_streams import make_stream, TARGETS


def make_model(config, seed, nbar):
    learner = None
    if config == 'tree':
        gpr = Default_GPR(n_restarts_optimizer=1)
    elif config.startswith('neural'):
        kw = dict(random_state=seed)
        for part in [p for p in config[len('neural'):].split('_') if p]:
            if part.startswith('cov'):
                kw.update(reservoir_size=int(part[3:]), reservoir='coverage', min_turnover=0.25)
            elif part.startswith('uni'):
                kw.update(reservoir_size=int(part[3:]), reservoir='uniform', min_turnover=0.25)
            elif part.startswith('cap'):
                kw.update(refit_cap=int(part[3:]))
            elif part.startswith('amort'):
                kw.update(steps_per_update=int(part[5:]))
            else:
                raise ValueError(config)
        learner = FeatureNetLearner(**kw)
        gpr = NeuralLinearGPR(learner)
    else:
        raise ValueError(config)
    gpt = GPTree(GPR=gpr, Nbar=nbar, theta=1e-4, retrain_every_n_points=25,
                 splitting_strategy='gradual', use_calibrated_sigma=True)
    return gpt, learner


def run_one(target_name, stream, config, seed, d, N, nbar, checkpoints):
    target = TARGETS[target_name]
    rng = np.random.RandomState(seed)
    X, _ = make_stream(stream, target, d, N, rng)
    y = target(X.T); sig = np.maximum(1e-3 * np.abs(y), 1e-6)
    X_test = rng.rand(1000, d); y_test = target(X_test.T)
    yrange = y_test.max() - y_test.min()
    np.random.seed(seed)
    gpt, learner = make_model(config, seed, nbar)
    t_start = time.time()
    t_upd, t_pred, errs = [], [], []
    cp = iter(checkpoints); next_cp = next(cp)
    out = []
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(N):
            xi = X[i:i + 1]
            t0 = time.perf_counter(); mu, _ = gpt.predict(xi); t1 = time.perf_counter()
            gpt.update_tree(xi, np.array([[y[i]]]), np.array([[sig[i]]])); t2 = time.perf_counter()
            t_pred.append(t1 - t0); t_upd.append(t2 - t1); errs.append(mu[0, 0] - y[i])
            if i + 1 == next_cp:
                t3 = time.perf_counter(); P, _ = gpt.predict(X_test, mode='loop'); t4 = time.perf_counter()
                r = {'target': target_name, 'stream': stream, 'config': config, 'seed': seed, 'd': d,
                     'n_points': i + 1, 'elapsed_s': round(time.time() - t_start, 1),
                     'update_ms_mean': 1e3 * float(np.mean(t_upd)), 'update_ms_max': 1e3 * float(np.max(t_upd)),
                     'update_ms_p99': 1e3 * float(np.percentile(t_upd, 99)),
                     'predict_point_ms_mean': 1e3 * float(np.mean(t_pred)),
                     'predict_batch_ms_per_point': 1e3 * (t4 - t3) / X_test.shape[0],
                     'window_prequential_nrmse': float(np.sqrt(np.mean(np.square(errs))) / yrange),
                     'test_nrmse': float(np.sqrt(np.mean((P[:, 0] - y_test) ** 2)) / yrange),
                     'leaves': len(gpt.root.leaves)}
                if learner is not None:
                    r.update(refits=learner.n_refits, fit_seconds=round(learner.fit_seconds, 1),
                             sample_size=learner.sample.n,
                             leaf_solves=sum(l.my_GPR.n_solves for l in gpt.root.leaves))
                out.append(r); print('RESULT ' + json.dumps(r), file=sys.__stdout__, flush=True)
                t_upd, t_pred, errs = [], [], []
                try:
                    next_cp = next(cp)
                except StopIteration:
                    break
    return out


def summarize(paths):
    """Markdown tables, one per (target, stream, config), from RESULT lines in the files."""
    rows = []
    for path in paths:
        for line in open(path):
            if line.startswith('RESULT '):
                rows.append(json.loads(line[len('RESULT '):]))
    keys = sorted({(r['target'], r['stream'], r['config']) for r in rows})
    out = []
    for key in keys:
        rs = sorted([r for r in rows if (r['target'], r['stream'], r['config']) == key], key=lambda r: r['n_points'])
        out.append(f"**{key[2]}** ({key[0]}, {key[1]} stream, d = {rs[0]['d']})\n")
        neural = 'refits' in rs[0]
        out.append("| points | update ms: mean / p99 / max | predict ms per point: single / batch | window prequential NRMSE | test NRMSE | leaves |"
                   + (" refits | fit s | sample | leaf solves |" if neural else "") + " elapsed s |")
        out.append("|---|---|---|---|---|---|" + ("---|---|---|---|" if neural else "") + "---|")
        for r in rs:
            line = (f"| {r['n_points']} | {r['update_ms_mean']:.2f} / {r['update_ms_p99']:.1f} / {r['update_ms_max']:.0f} | "
                    f"{r['predict_point_ms_mean']:.2f} / {r['predict_batch_ms_per_point']:.2f} | {r['window_prequential_nrmse']:.4f} | "
                    f"{r['test_nrmse']:.4f} | {r['leaves']} |")
            if neural:
                line += f" {r['refits']} | {r['fit_seconds']} | {r['sample_size']} | {r['leaf_solves']} |"
            out.append(line + f" {r['elapsed_s']} |")
        out.append("")
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summarize', nargs='*', metavar='FILE', help='aggregate RESULT lines from files into tables')
    ap.add_argument('--target', default='gaussian_peaks'); ap.add_argument('--stream', default='walker')
    ap.add_argument('--configs', default='tree,neural,neural_cov4000'); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--d', type=int, default=6); ap.add_argument('--N', type=int, default=100000)
    ap.add_argument('--nbar', type=int, default=100)
    ap.add_argument('--checkpoints', default=None, help='comma-separated; default 1000,2000,4000,... up to N')
    a = ap.parse_args()
    if a.summarize is not None:
        summarize(a.summarize); return
    if a.checkpoints:
        cps = [int(c) for c in a.checkpoints.split(',')]
    else:
        cps = []; c = 1000
        while c < a.N:
            cps.append(c); c *= 2
        cps.append(a.N)
    for cfg in a.configs.split(','):
        run_one(a.target, a.stream, cfg, a.seed, a.d, a.N, a.nbar, cps)


if __name__ == '__main__':
    main()
