"""Rules for choosing between the nearest-neighbour GP and the leaf's GP, evaluated on the replays."""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench                                                    # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
runs = {}
for path in sorted(glob.glob(os.path.join(bench.RESULTS, '_diag', f"*__N{N}__s*.npz"))):
    target, stream, _, seed = os.path.basename(path)[:-4].split('__')
    runs[(target, stream, seed)] = dict(np.load(path))


def gmean(x):
    return float(np.exp(np.mean(np.log(x))))


def evaluate(name, choose_knn):
    """choose_knn(d) -> boolean array: where the nearest-neighbour GP predicts."""
    ratios, med_ratios, share, by_stream = [], [], [], {}
    for key, d in runs.items():
        ek, el = np.abs(d['knn'] - d['y']), np.abs(d['leaf'] - d['y'])
        use = choose_knn(d)
        e = np.where(use, ek, el)
        r = np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(el ** 2))
        ratios.append(r)
        med_ratios.append(np.median(e) / np.median(el))
        share.append(use.mean())
        by_stream.setdefault(key[1], []).append(r)
    ratios = np.array(ratios)
    print(f"{name:34s} rmse {gmean(ratios):.3f}  med {gmean(med_ratios):.3f}  worst {ratios.max():.2f}  best {ratios.min():.2f}  "
          f">2% worse {np.sum(ratios > 1.02):2d}/{len(ratios)}   knn used {100 * np.mean(share):3.0f}%   "
          + "  ".join(f"{s} {gmean(by_stream[s]):.3f}" for s in sorted(by_stream)))


print(f"{len(runs)} runs at N={N}; all ratios against the leaf's own GP (the current tree)")
evaluate('knn always', lambda d: np.ones(len(d['y']), bool))
evaluate('best of both (oracle)', lambda d: np.abs(d['knn'] - d['y']) < np.abs(d['leaf'] - d['y']))
for stat in ('chi2', 'z5', 'z10', 'z20', 'zall'):
    for tau in (1.5, 2, 3, 5, 10, 30, 100):
        evaluate(f"knn where {stat} <= {tau}", lambda d, stat=stat, tau=tau: d[stat] <= tau)
for stat in ('z10', 'z20', 'zall'):
    evaluate(f"smaller sigma, knn x max(1,{stat})", lambda d, stat=stat: d['knn_sd'] * np.maximum(1.0, d[stat]) < d['leaf_sd'])
evaluate('smaller raw sigma', lambda d: d['knn_sd'] < d['leaf_sd'])
