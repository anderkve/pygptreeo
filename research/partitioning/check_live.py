"""Runs of the repository's own package against the records of the base tree and of the frozen copy:

    python check_live.py N target:stream [target:stream ...]
"""
import os
import sys
from multiprocessing import Pool

os.environ['PYGPTREEO_LIVE'] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                              # noqa: E402
import bench                                                    # noqa: E402
import report                                                   # noqa: E402

VARIANT = os.environ.get('CHECK_VARIANT', 'nb')


def job(args):
    from variants import make_model
    target, stream, N = args
    data = bench.make_data(target, stream, N, 1)
    np.random.seed(1)
    model = make_model(VARIANT, data['d'], 1)
    rec = bench.run_stream(model, data, 1)
    own = np.mean([leaf.prefers_own for leaf in model.root.leaves])
    k = int(0.75 * N)
    err = np.abs(rec['pred'][k:] - rec['y'][k:])
    out = dict(rmse=np.sqrt(np.mean(err ** 2)), cover=np.mean(err <= rec['std'][k:]), t_pred=1e3 * rec['t_pred'][k:].mean(),
               t_upd=1e3 * rec['t_upd'][k:].mean(), own=own)
    return args, out


if __name__ == '__main__':
    N = int(sys.argv[1])
    jobs = [tuple(a.split(':')) + (N,) for a in sys.argv[2:]]
    B, A = report.load('base', N), report.load('nb', N)
    with Pool(min(len(jobs), int(os.environ.get('CHECK_PROCS', 2)))) as pool:
        for (target, stream, _), out in pool.imap_unordered(job, jobs):
            key = (target, stream, 1)
            base = report.metrics(B[key])['rmse'] if key in B else np.nan
            old = report.metrics(A[key])['rmse'] if key in A else np.nan
            print(f"{target:24s} {stream:9s} {VARIANT}: {out['rmse'] / base:.3f} of the base tree's error (the recorded run {old / base:.3f}), "
                  f"coverage {out['cover']:.2f}, leaves on their own GP {100 * out['own']:.0f}%, predict {out['t_pred']:.2f} ms, update {out['t_upd']:.2f} ms", flush=True)
