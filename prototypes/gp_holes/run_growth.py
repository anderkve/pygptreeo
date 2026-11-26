"""How the expression found for one target changes with the number of points.

    python run_growth.py [--target toy] [--n 10 15 25 40 60 100] [--seeds 3]
"""
import argparse

from run_benchmark import run
from targets import TARGETS

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='toy')
    ap.add_argument('--n', type=int, nargs='*', default=[10, 15, 25, 40, 60, 100])
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--noise', type=float, default=0.01)
    ap.add_argument('--kernel', default='matern52')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    target = [t for t in TARGETS if t[0] == a.target][0]
    print(f"{target[0]}:  {target[1]}")
    for n in a.n:
        for seed in range(a.seeds):
            text, holes, err, trace, info = run(target, n, a.noise, seed, False, kernel=a.kernel,
                                                workers=a.workers)
            print(f"  n {n:4d} seed {seed}: {text}\n          holes {holes}, rmse in {err['in']:.4f} "
                  f"out {err['out']:.4f}, {info['seconds']:.0f} s", flush=True)
