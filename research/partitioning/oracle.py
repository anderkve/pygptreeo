"""Ceiling experiment: what a GP on an ideal neighbourhood of each query would achieve.

For a sample of the last quarter of a stream, each query is predicted from the
points before it in the stream by a GP fitted, like a leaf's, to a chosen subset:

    knn{k}       the k nearest points (inputs scaled by their spread over the stream so far)
    ard{k}       the k nearest in the metric of the length scales fitted to knn{k}
    ms{k}        multi-scale: the k/2 nearest and k/2 at geometrically growing ranks
    recent{k}    the k most recent points

The baseline tree's prediction of the same queries comes from its stored record.

    python oracle.py --target gaussian_peaks --stream sweeping --N 20000
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench                                                    # noqa: E402


def gp_predict(Xtr, ytr, sig, xq, return_ls=False):
    """A leaf-style GP: standardised inputs and targets, ARD Matern 3/2, declared noise."""
    mx, sx = Xtr.mean(0), Xtr.std(0)
    sx = np.where(sx > 0, sx, 1.0)
    my, sy = ytr.mean(), ytr.std()
    sy = sy if sy > 0 else 1.0
    gp = GaussianProcessRegressor(ConstantKernel() * Matern(nu=1.5, length_scale=np.ones(Xtr.shape[1])),
                                  alpha=(sig / sy) ** 2)
    gp.fit((Xtr - mx) / sx, (ytr - my) / sy)
    mu, sd = gp.predict((xq - mx) / sx, return_std=True)
    if return_ls:
        return mu * sy + my, sd * sy, gp.kernel_.k2.length_scale * sx
    return mu * sy + my, sd * sy


def neighbours(X, x, k, scale):
    d2 = (((X - x) / scale) ** 2).sum(1)
    return np.argpartition(d2, min(k, len(d2) - 1))[:k], d2


def multiscale(X, x, k, scale):
    """k/2 nearest, and k/2 at ranks spaced geometrically up to the whole set."""
    d2 = (((X - x) / scale) ** 2).sum(1)
    order = np.argsort(d2)
    near = k // 2
    ranks = np.unique(np.round(np.geomspace(near, len(order) - 1, k - near)).astype(int))
    return np.concatenate((order[:near], order[ranks]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True)
    ap.add_argument('--stream', required=True)
    ap.add_argument('--N', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--n_queries', type=int, default=300)
    ap.add_argument('--methods', default='knn100,ard100,knn200,knn400,ms100,ms200,recent100')
    a = ap.parse_args()

    data = bench.make_data(a.target, a.stream, a.N, a.seed)
    X, y, y_obs, sigma = data['X'], data['y'], data['y_obs'], data['sigma']
    rec = np.load(bench.record_path('base', a.target, a.stream, a.N, a.seed))
    rng = np.random.RandomState(0)
    queries = np.sort(rng.choice(np.arange(int(0.75 * a.N), a.N), a.n_queries, replace=False))
    methods = a.methods.split(',')
    pred = {m: np.empty(len(queries)) for m in methods}
    std = {m: np.empty(len(queries)) for m in methods}
    t0 = time.time()
    for j, i in enumerate(queries):
        Xp, yp, sp = X[:i], y_obs[:i], sigma[:i]
        scale = Xp.std(0)
        xq = X[i:i + 1]
        cache = {}
        for m in methods:
            kind = m.rstrip('0123456789')
            k = int(m[len(kind):])
            if kind == 'knn':
                idx, _ = neighbours(Xp, xq, k, scale)
                mu, sd, ls = gp_predict(Xp[idx], yp[idx], sp[idx], xq, return_ls=True)
                cache[k] = ls
            elif kind == 'ard':
                ls = cache[k] if k in cache else gp_predict(*[arr[neighbours(Xp, xq, k, scale)[0]] for arr in (Xp, yp, sp)], xq, return_ls=True)[2]
                idx, _ = neighbours(Xp, xq, k, np.minimum(ls, 10 * scale))
                mu, sd = gp_predict(Xp[idx], yp[idx], sp[idx], xq)
            elif kind == 'ms':
                idx = multiscale(Xp, xq, k, scale)
                mu, sd = gp_predict(Xp[idx], yp[idx], sp[idx], xq)
            elif kind == 'recent':
                idx = np.arange(i - k, i)
                mu, sd = gp_predict(Xp[idx], yp[idx], sp[idx], xq)
            else:
                raise ValueError(m)
            pred[m][j], std[m][j] = mu[0], sd[0]
    out = dict(target=a.target, stream=a.stream, N=a.N, seed=a.seed, seconds=time.time() - t0)
    truth = y[queries]
    e_tree = np.abs(rec['pred'][queries] - truth)
    out['tree'] = dict(rmse=float(np.sqrt(np.mean(e_tree ** 2))), mae=float(e_tree.mean()), med=float(np.median(e_tree)))
    line = f"{a.target:24s} {a.stream:9s} tree rmse {out['tree']['rmse']:.3g}  | ratios rmse/mae/med:"
    for m in methods:
        e = np.abs(pred[m] - truth)
        out[m] = dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(e.mean()), med=float(np.median(e)),
                      cover=float(np.mean(e <= std[m])))
        line += f"  {m} {out[m]['rmse'] / out['tree']['rmse']:.2f}/{out[m]['mae'] / out['tree']['mae']:.2f}/{out[m]['med'] / out['tree']['med']:.2f}"
    print(line, flush=True)
    os.makedirs(os.path.join(bench.RESULTS, '_oracle'), exist_ok=True)
    with open(os.path.join(bench.RESULTS, '_oracle', f"{a.target}__{a.stream}__N{a.N}__s{a.seed}.json"), 'w') as f:
        json.dump(out, f)


if __name__ == '__main__':
    main()
