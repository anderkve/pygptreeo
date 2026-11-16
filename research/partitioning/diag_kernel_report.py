"""Rules for the kernel of a neighbour-conditioned prediction, on the replays of diag_kernel.py."""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench                                                    # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
runs = {}
for path in sorted(glob.glob(os.path.join(bench.RESULTS, '_diagk', f"*__N{N}__s*.npz"))):
    target, stream, _, seed = os.path.basename(path)[:-4].split('__')
    runs[(target, stream, seed)] = dict(np.load(path))


def gmean(x):
    return float(np.exp(np.mean(np.log(x))))


def evaluate(name, rule):
    ratios, by_stream = [], {}
    for key, d in runs.items():
        e = np.abs(rule(d) - d['y'])
        el = np.abs(d['leaf'] - d['y'])
        r = np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(el ** 2))
        ratios.append(r)
        by_stream.setdefault(key[1], []).append(r)
    ratios = np.array(ratios)
    print(f"{name:58s} rmse {gmean(ratios):.3f}  worst {ratios.max():.2f}  >2% worse {np.sum(ratios > 1.02):2d}/{len(ratios)}   "
          + "  ".join(f"{s[:3]} {gmean(v):.3f}" for s, v in sorted(by_stream.items())))


TAU = 10.0
print(f"{len(runs)} runs at N={N}; ratios against the leaf's own GP")
print(f"the leaf with most neighbours is the query's own for {100 * np.mean([d['same'].mean() for d in runs.values()]):.0f}% of the queries; "
      f"the own leaf holds {100 * np.mean([d['share_own'].mean() for d in runs.values()]):.0f}% of the neighbours, "
      f"the leaf with most {100 * np.mean([d['share_major'].mean() for d in runs.values()]):.0f}%")
evaluate("own leaf's kernel, guarded", lambda d: np.where(d['own_chi2'] <= TAU, d['own'], d['leaf']))
evaluate("kernel of the leaf with most neighbours, guarded", lambda d: np.where(d['maj_chi2'] <= TAU, d['maj'], d['leaf']))
evaluate("the kernel with the higher likelihood, guarded",
         lambda d: np.where(d['maj_ll'] > d['own_ll'], np.where(d['maj_chi2'] <= TAU, d['maj'], d['leaf']),
                            np.where(d['own_chi2'] <= TAU, d['own'], d['leaf'])))
evaluate("own; the other kernel where the own one fails the guard",
         lambda d: np.where(d['own_chi2'] <= TAU, d['own'], np.where(d['maj_chi2'] <= TAU, d['maj'], d['leaf'])))
evaluate("the kernel with the lower chi2, guarded",
         lambda d: np.where(d['maj_chi2'] < d['own_chi2'], np.where(d['maj_chi2'] <= TAU, d['maj'], d['leaf']),
                            np.where(d['own_chi2'] <= TAU, d['own'], d['leaf'])))
evaluate("mean of the two kernels' predictions, guarded",
         lambda d: np.where((d['own_chi2'] <= TAU) & (d['maj_chi2'] <= TAU), 0.5 * (d['own'] + d['maj']),
                            np.where(d['own_chi2'] <= TAU, d['own'], d['leaf'])))
evaluate("best of own-kernel, other-kernel and leaf (oracle)",
         lambda d: np.where(np.abs(d['own'] - d['y']) < np.abs(d['maj'] - d['y']),
                            np.where(np.abs(d['own'] - d['y']) < np.abs(d['leaf'] - d['y']), d['own'], d['leaf']),
                            np.where(np.abs(d['maj'] - d['y']) < np.abs(d['leaf'] - d['y']), d['maj'], d['leaf'])))
