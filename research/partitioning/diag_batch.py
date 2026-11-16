"""Run diag_knn.replay for a set of targets and streams in parallel."""
import itertools
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                              # noqa: E402
import bench                                                    # noqa: E402
from run_matrix import SETS                                     # noqa: E402


def job(args):
    from diag_knn import replay
    target, stream, N, seed = args
    path = os.path.join(bench.RESULTS, '_diag2', f"{target}__{stream}__N{N}__s{seed}.npz")
    if not os.path.exists(path):
        out = replay(target, stream, N, seed)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, **out)
    return args


if __name__ == '__main__':
    targets = SETS[sys.argv[1]] if sys.argv[1] in SETS else sys.argv[1].split(',')
    N = int(sys.argv[2])
    seeds = [int(s) for s in sys.argv[3].split(',')] if len(sys.argv) > 3 else [1]
    jobs = [(t, s, N, seed) for t, s, seed in itertools.product(targets, ('uniform', 'walker', 'sweeping'), seeds)]
    with Pool(4) as pool:
        for k, a in enumerate(pool.imap_unordered(job, jobs), 1):
            print(k, len(jobs), a, flush=True)
