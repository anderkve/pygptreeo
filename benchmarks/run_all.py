"""Driver: run every (method, problem, seed) combination and save results.

Results are stored as compressed ``.npz`` files in ``benchmarks/data/`` with the
pattern ``{method}__{problem}__seed{seed}.npz``. Re-running skips combinations
whose file already exists (use ``--force`` to recompute).
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


METHODS = {
    "pygptreeo": lambda d: PyGPTreeOAdapter(
        d, Nbar=200, retrain_step=200, theta=1e-4, sigma_rel=1e-3),
    "sklearn_gp": lambda d: SklearnGPAdapter(
        d, retrain_every=400, max_train_points=800),
    "gpytorch_svgp": lambda d: GPyTorchSVGPAdapter(
        d, retrain_every=250, n_epochs=25, n_inducing=64, max_buffer=4000),
    "random_forest": lambda d: RandomForestAdapter(
        d, retrain_every=250, n_estimators=100, max_train_points=8000),
    "river_knn": lambda d: RiverKNNAdapter(
        d, n_neighbors=8, window_size=4000),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=list(METHODS.keys()))
    p.add_argument("--problems", nargs="+",
                   default=["smooth_sines_2d", "rosenbrock_2d", "step_3d"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--n-stream", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--max-wall-time", type=float, default=900.0,
                   help="Abort a single run after this many seconds")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if the output file already exists")
    p.add_argument("--out-dir", default="benchmarks/data")
    p.add_argument("--quiet-adapter-stdout", action="store_true", default=True,
                   help="Silence prints coming from adapters (e.g. pygptreeo "
                        "node-creation messages)")
    return p.parse_args()


@contextlib.contextmanager
def _maybe_silence(enabled: bool):
    if not enabled:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


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
            for seed in args.seeds:
                out_file = os.path.join(
                    args.out_dir,
                    f"{method_name}__{problem_name}__seed{seed}.npz",
                )
                if os.path.exists(out_file) and not args.force:
                    print(f"[exists] {out_file}")
                    continue
                print(f"\n==> {method_name} | {problem_name} | seed {seed}")
                t0 = time.time()
                with _maybe_silence(args.quiet_adapter_stdout):
                    result = run_online_benchmark(
                        factory,
                        problem,
                        seed=seed,
                        n_stream=args.n_stream,
                        n_test=args.n_test,
                        checkpoint_every=args.checkpoint_every,
                        max_wall_time_s=args.max_wall_time,
                        method_name=method_name,
                        verbose=False,
                    )
                path = save_result(result, args.out_dir)
                last = -1 if result.checkpoints else None
                print(
                    f"    done in {time.time() - t0:.1f}s | "
                    f"final nrmse={result.nrmse[last]:.3e} | "
                    f"coverage={result.coverage_1sigma[last]:.2f} | "
                    f"aborted={result.aborted} | saved={path}"
                )
    print(f"\nAll done in {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
