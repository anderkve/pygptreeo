"""Rules for the size of the neighbour set and the fallback to the leaf's GP, on the replays of diag_knn.py."""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench                                                    # noqa: E402
from diag_knn import KS                                          # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
folder = sys.argv[2] if len(sys.argv) > 2 else '_diag2'
runs = {}
for path in sorted(glob.glob(os.path.join(bench.RESULTS, folder, f"*__N{N}__s*.npz"))):
    target, stream, _, seed = os.path.basename(path)[:-4].split('__')
    d = dict(np.load(path))
    d['chi2'] = np.cumsum(d['v'].astype(float) ** 2, axis=1)[:, KS - 1] / KS      # per prefix size
    runs[(target, stream, seed)] = d


def gmean(x):
    return float(np.exp(np.mean(np.log(x))))


def col(k):
    return int(np.where(KS == k)[0][0])


def evaluate(name, rule, quiet=False):
    """rule(d) -> column of KS to use per query, or -1 for the leaf's GP."""
    ratios, med, share, by_stream, cover = [], [], [], {}, []
    for key, d in runs.items():
        n = len(d['y'])
        j = rule(d)
        pred = np.where(j >= 0, d['knn'][np.arange(n), np.maximum(j, 0)], d['leaf'])
        e, el = np.abs(pred - d['y']), np.abs(d['leaf'] - d['y'])
        r = np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(el ** 2))
        ratios.append(r); med.append(np.median(e) / np.median(el)); share.append(np.mean(j >= 0))
        by_stream.setdefault(key[1], []).append(r)
    ratios = np.array(ratios)
    if not quiet:
        print(f"{name:44s} rmse {gmean(ratios):.3f}  med {gmean(med):.3f}  worst {ratios.max():.2f}  best {ratios.min():.2f}  "
              f">2% worse {np.sum(ratios > 1.02):2d}/{len(ratios)}  knn {100 * np.mean(share):3.0f}%   "
              + "  ".join(f"{s[:3]} {gmean(by_stream[s]):.3f}" for s in sorted(by_stream)))
    return gmean(ratios), ratios


def fixed(k, tau):
    j = col(k)
    return lambda d: np.where(d['chi2'][:, j] <= tau, j, -1)


def adaptive(tau, k_min, k_max):
    """The largest neighbour set up to k_max whose chi2 is within tau; the leaf below k_min."""
    cols = [j for j, k in enumerate(KS) if k_min <= k <= k_max]

    def rule(d):
        j = np.full(len(d['y']), -1)
        for c in cols:                      # ascending, so the largest consistent one remains
            j = np.where(d['chi2'][:, c] <= tau, c, j)
        return j
    return rule


def innovation(tau_v, k_min, k_max):
    """Neighbours are added nearest first until one is predicted worse than tau_v sigma by the nearer ones."""
    def rule(d):
        bad = d['v'].astype(float) ** 2 > tau_v ** 2
        stop = np.where(bad.any(axis=1), bad.argmax(axis=1), d['v'].shape[1])     # consistent neighbours
        stop = np.minimum(stop, k_max)
        j = np.searchsorted(KS, stop, side='right') - 1
        return np.where((j >= 0) & (KS[np.maximum(j, 0)] >= k_min), j, -1)
    return rule


print(f"{len(runs)} runs at N={N}; ratios against the leaf's own GP (the current tree)")
print("-- fixed k, never falling back")
for k in KS:
    evaluate(f"k={k}", fixed(k, np.inf))
print("-- fixed k, leaf GP where chi2 > tau")
for k in (50, 100, 200):
    for tau in (5, 10, 30, 100):
        evaluate(f"k={k}, chi2 <= {tau}", fixed(k, tau))
print("-- largest consistent neighbour set")
for k_max in (100, 200):
    for k_min in (10, 20, 50):
        for tau in (3, 5, 10, 30):
            evaluate(f"adaptive k in [{k_min},{k_max}], chi2 <= {tau}", adaptive(tau, k_min, k_max))
print("-- neighbours added until one is badly predicted")
for k_max in (100, 200):
    for k_min in (10, 20, 50):
        for tau_v in (5, 10, 20, 50):
            evaluate(f"innovation k in [{k_min},{k_max}], |v| <= {tau_v}", innovation(tau_v, k_min, k_max))
