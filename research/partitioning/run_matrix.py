"""Run a matrix of (variant, target, stream, seed) in parallel; finished runs are skipped.

    python run_matrix.py --variants "base;oblique:dir=pca" --set dev --N 5000 --seeds 1
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # before numpy is loaded
import argparse
import itertools
import os
import sys
import time
import traceback
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench                                                    # noqa: E402

SETS = {
    # quick look: two artificial and two physics targets
    'mini': ['gaussian_peaks', 'eggholder', 'resonance_xsec', 'loop_decay_width'],
    # early testing
    'dev': ['gaussian_peaks', 'rotated_rosenbrock', 'eggholder', 'chirp', 'diagonal_ridge',
            'resonance_xsec', 'loop_decay_width', 'relic_density', 'eft_signal_strength', 'neutralino_mass',
            'oscillation_probability'],
    'artificial': sorted(bench.ANALYTIC),
    'physics': sorted(bench.PHYSICS),
    'all': bench.ALL_TARGETS,
    # long streams
    'long': ['gaussian_peaks', 'eggholder', 'resonance_xsec', 'eft_signal_strength'],
}


def _job(args):
    variant, target, stream, N, seed, overwrite = args
    t0 = time.time()
    try:
        bench.run_one(variant, target, stream, N, seed, overwrite=overwrite)
        return args, time.time() - t0, None
    except Exception:
        return args, time.time() - t0, traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', required=True)
    ap.add_argument('--set', default='dev')
    ap.add_argument('--targets', default=None, help='comma-separated, instead of --set')
    ap.add_argument('--streams', default='uniform,walker,sweeping')
    ap.add_argument('--N', type=int, default=5000)
    ap.add_argument('--seeds', default='1')
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()

    targets = a.targets.split(',') if a.targets else SETS[a.set]
    variants = a.variants.split(';')
    jobs = [(v, t, s, a.N, int(seed), a.overwrite)
            for t, s, seed, v in itertools.product(targets, a.streams.split(','), a.seeds.split(','), variants)]
    todo = [j for j in jobs if a.overwrite or not os.path.exists(bench.record_path(*j[:5]))]
    print(f"{len(jobs)} runs, {len(todo)} to do", flush=True)
    t0 = time.time()
    with Pool(a.procs) as pool:
        for k, (args, dt, err) in enumerate(pool.imap_unordered(_job, todo), 1):
            status = 'FAILED' if err else 'ok'
            print(f"[{k}/{len(todo)}] {time.time() - t0:6.0f}s  {args[0]:40s} {args[1]:24s} {args[2]:9s} s{args[4]}  {dt:6.1f}s  {status}",
                  flush=True)
            if err:
                print(err, flush=True)


if __name__ == '__main__':
    main()
