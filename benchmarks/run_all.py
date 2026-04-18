"""Driver: run every (method, problem, seed, schedule) combination and save.

Results are stored as compressed ``.npz`` files in ``benchmarks/data/`` with
the pattern ``{method}__{problem}__{schedule}__seed{seed}.npz``. Re-runs skip
combinations whose file already exists (use ``--force`` to recompute).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.adapters import (
    PyGPTreeOAdapter,
    SklearnGPAdapter,
    GPyTorchSVGPAdapter,
    RandomForestAdapter,
    RiverKNNAdapter,
)
from benchmarks.harness import run_online_benchmark, save_result
from benchmarks.problems import PROBLEMS


# Fair-budget defaults (iteration 01). Rationale:
#  - pygptreeo: leaf-size 200, retrain every 200 pts, calibrated sigma.
#  - sklearn_gp: 1500-pt reservoir, refit every 200 pts, Matern-1.5.
#  - gpytorch_svgp: 256 inducing points, 60 SVI epochs per refit, 5000 buffer.
#  - random_forest: 300 trees, refit every 200 pts, 20000 buffer cap.
#  - river_knn: k=8, window 4000. True online, no refit.
METHODS = {
    "pygptreeo": lambda d: PyGPTreeOAdapter(
        d, Nbar=200, retrain_step=200, theta=1e-4, sigma_rel=1e-3),
    "sklearn_gp": lambda d: SklearnGPAdapter(
        d, retrain_every=200, max_train_points=1500),
    "gpytorch_svgp": lambda d: GPyTorchSVGPAdapter(
        d, retrain_every=200, n_epochs=60, n_inducing=256, max_buffer=5000, lr=5e-3),
    "random_forest": lambda d: RandomForestAdapter(
        d, retrain_every=200, n_estimators=300, max_train_points=20000),
    "river_knn": lambda d: RiverKNNAdapter(
        d, n_neighbors=8, window_size=4000),
}


DEFAULT_PROBLEMS = [
    "smooth_sines_2d", "rosenbrock_2d", "friedman1_5d", "borehole_8d",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=list(METHODS.keys()))
    p.add_argument("--problems", nargs="+", default=DEFAULT_PROBLEMS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--schedules", nargs="+", default=["iid"],
                   help="Stream schedule(s): iid, shift, sobol")
    p.add_argument("--n-stream", type=int, default=3000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--checkpoint-every", type=int, default=300)
    p.add_argument("--max-wall-time", type=float, default=900.0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out-dir", default="benchmarks/data")
    p.add_argument("--quiet-adapter-stdout", action="store_true", default=True)
    return p.parse_args()


@contextlib.contextmanager
def _maybe_silence(enabled: bool):
    if not enabled:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def _fname(method_name, problem_name, schedule, seed):
    if schedule == "iid":
        return f"{method_name}__{problem_name}__seed{seed}.npz"
    return f"{method_name}__{problem_name}__{schedule}__seed{seed}.npz"


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")

    total_start = time.time()
    for method_name in args.methods:
        if method_name not in METHODS:
            print(f"[skip] unknown method {method_name}")
            continue
        factory = METHODS[method_name]
        for problem_name in args.problems:
            if problem_name not in PROBLEMS:
                print(f"[skip] unknown problem {problem_name}")
                continue
            problem = PROBLEMS[problem_name]
            for schedule in args.schedules:
                for seed in args.seeds:
                    out_file = os.path.join(
                        args.out_dir,
                        _fname(method_name, problem_name, schedule, seed),
                    )
                    if os.path.exists(out_file) and not args.force:
                        print(f"[exists] {out_file}")
                        continue
                    print(f"\n==> {method_name} | {problem_name} | "
                          f"sched={schedule} | seed {seed}")
                    t0 = time.time()
                    with _maybe_silence(args.quiet_adapter_stdout):
                        result = run_online_benchmark(
                            factory,
                            problem,
                            seed=seed,
                            n_stream=args.n_stream,
                            n_test=args.n_test,
                            checkpoint_every=args.checkpoint_every,
                            schedule=schedule,
                            max_wall_time_s=args.max_wall_time,
                            method_name=method_name,
                            verbose=False,
                        )
                    path = save_result(result, args.out_dir)
                    last = -1 if result.checkpoints else None
                    if last is not None:
                        print(
                            f"    done in {time.time() - t0:.1f}s | "
                            f"nrmse={result.nrmse[last]:.3e} | "
                            f"medNLPD={result.median_nlpd[last]:.3f} | "
                            f"cov68={result.coverage_1sigma[last]:.2f} | "
                            f"badσ={result.frac_pathological_std[last]:.2f} | "
                            f"aborted={result.aborted} | saved={path}"
                        )
                    else:
                        print(f"    no checkpoints; aborted={result.aborted}")
    print(f"\nAll done in {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
