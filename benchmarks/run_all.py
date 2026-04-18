"""Driver: run every (method, problem, seed, schedule) combination in an
isolated subprocess, with a hard per-run wall-clock timeout.

The harness's internal `max_wall_time_s` check only fires between
checkpoints, so a long ``sklearn GPR.fit()`` or a wedged ``torch`` backward
can block for an entire run. Iteration 02 wraps every run in a child
`multiprocessing.Process`; if the child does not exit within a deadline, the
parent terminates it and the partial `.npz` that the child has already
flushed (via ``save_every_checkpoint_to``) becomes the record. A run that
makes no checkpoints at all gets a stub `.npz` written by the parent so the
plot script can display an "aborted" marker.

Results pattern in `benchmarks/data/`:

    ``{method}__{problem}__seed{seed}.npz``           (iid schedule)
    ``{method}__{problem}__{schedule}__seed{seed}.npz`` (non-iid)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import multiprocessing as mp
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
from benchmarks.harness import (
    run_online_benchmark, save_result, RunConfig, RunResult,
)
from benchmarks.problems import PROBLEMS


def _make_pygptreeo(d: int):
    return PyGPTreeOAdapter(d, Nbar=200, retrain_step=200, theta=1e-4,
                            sigma_rel=1e-3)


def _make_sklearn_gp(d: int):
    # sklearn exact GP is O(N^3) per refit; give high-d problems a smaller
    # training cap so borehole_8d (and any 6+ -D problem) completes in a
    # reasonable time. Under 6 dimensions, keep the bigger cap.
    max_train = 1500 if d <= 5 else 500
    return SklearnGPAdapter(d, retrain_every=200,
                            max_train_points=max_train)


def _make_svgp(d: int):
    return GPyTorchSVGPAdapter(d, retrain_every=200, n_epochs=60,
                               n_inducing=256, max_buffer=5000, lr=5e-3,
                               max_steps_per_refit=500)


def _make_rf(d: int):
    return RandomForestAdapter(d, retrain_every=200, n_estimators=300,
                               max_train_points=20000)


def _make_river_knn(d: int):
    return RiverKNNAdapter(d, n_neighbors=8, window_size=4000)


METHODS = {
    "pygptreeo": _make_pygptreeo,
    "sklearn_gp": _make_sklearn_gp,
    "gpytorch_svgp": _make_svgp,
    "random_forest": _make_rf,
    "river_knn": _make_river_knn,
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
    p.add_argument("--max-wall-time", type=float, default=1200.0,
                   help="Per-run wall-time ceiling in seconds (hard timeout).")
    p.add_argument("--force", action="store_true")
    p.add_argument("--out-dir", default="benchmarks/data")
    p.add_argument("--no-subprocess", action="store_true",
                   help="Run each benchmark in the parent process (debug only).")
    return p.parse_args()


def _fname(method_name, problem_name, schedule, seed):
    if schedule == "iid":
        return f"{method_name}__{problem_name}__seed{seed}.npz"
    return f"{method_name}__{problem_name}__{schedule}__seed{seed}.npz"


def _child_target(method_name, problem_name, schedule, seed,
                  n_stream, n_test, checkpoint_every, max_wall_time, out_file):
    """Child-process entry. Runs one benchmark and writes results to disk."""
    warnings.filterwarnings("ignore")
    # Silence adapter stdout (pygptreeo prints a lot during tree splits).
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        factory = METHODS[method_name]
        problem = PROBLEMS[problem_name]
        run_online_benchmark(
            factory,
            problem,
            seed=seed,
            n_stream=n_stream,
            n_test=n_test,
            checkpoint_every=checkpoint_every,
            schedule=schedule,
            max_wall_time_s=max_wall_time,
            method_name=method_name,
            verbose=False,
            save_every_checkpoint_to=out_file,
        )


def _write_stub_aborted(out_file, method_name, problem_name, schedule, seed,
                        n_stream, n_test, checkpoint_every, max_wall_time,
                        wall_time, notes):
    """Parent emergency save when the child died before any checkpoint."""
    cfg = RunConfig(
        method_name=method_name, problem_name=problem_name, seed=seed,
        n_stream=n_stream, n_test=n_test, checkpoint_every=checkpoint_every,
        schedule=schedule, max_wall_time_s=max_wall_time,
    )
    r = RunResult(config=cfg)
    r.wall_time = float(wall_time)
    r.aborted = True
    r.notes = notes
    np.savez(out_file, **r.to_npz_dict())


def _run_one(method_name, problem_name, schedule, seed, args):
    """Run a single benchmark in a subprocess with a hard timeout."""
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(
        args.out_dir, _fname(method_name, problem_name, schedule, seed),
    )
    if os.path.exists(out_file) and not args.force:
        print(f"[exists] {out_file}")
        return
    print(f"\n==> {method_name} | {problem_name} | sched={schedule} | seed {seed}")
    t0 = time.time()

    if args.no_subprocess:
        _child_target(
            method_name, problem_name, schedule, seed,
            args.n_stream, args.n_test, args.checkpoint_every,
            args.max_wall_time, out_file,
        )
    else:
        ctx = mp.get_context("spawn")
        # Grace period: checkpoint saves can take a bit longer than the
        # harness's own wall-time ceiling on very slow methods.
        deadline = args.max_wall_time + 30.0
        proc = ctx.Process(
            target=_child_target,
            args=(method_name, problem_name, schedule, seed,
                  args.n_stream, args.n_test, args.checkpoint_every,
                  args.max_wall_time, out_file),
        )
        proc.start()
        proc.join(timeout=deadline)
        if proc.is_alive():
            print(f"    [TIMEOUT] hard-killing after {deadline:.0f}s")
            proc.terminate()
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.kill()
                proc.join()
            # If the child never wrote any partial file, drop a stub.
            if not os.path.exists(out_file):
                _write_stub_aborted(
                    out_file, method_name, problem_name, schedule, seed,
                    args.n_stream, args.n_test, args.checkpoint_every,
                    args.max_wall_time, time.time() - t0,
                    notes=f"process-level timeout at {deadline:.0f}s",
                )

    wall = time.time() - t0
    # Report final row from disk.
    if os.path.exists(out_file):
        data = np.load(out_file, allow_pickle=True)
        n_ck = int(len(np.atleast_1d(data["checkpoints"])))
        if n_ck > 0:
            last = int(data["checkpoints"][-1])
            nrmse = float(data["nrmse"][-1])
            mednlpd = float(data["median_nlpd"][-1])
            cov68 = float(data["coverage_1sigma"][-1])
            bad = float(data["frac_pathological_std"][-1])
            aborted = bool(data["aborted"]) if "aborted" in data.files else False
            print(
                f"    done in {wall:.1f}s | step {last} | nrmse={nrmse:.3e} | "
                f"medNLPD={mednlpd:.3f} | cov68={cov68:.2f} | badσ={bad:.2f} | "
                f"aborted={aborted}"
            )
        else:
            print(f"    no checkpoints reached (stub written) in {wall:.1f}s")
    else:
        print(f"    [FAIL] no output file after {wall:.1f}s")


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    total_start = time.time()
    for method_name in args.methods:
        if method_name not in METHODS:
            print(f"[skip] unknown method {method_name}")
            continue
        for problem_name in args.problems:
            if problem_name not in PROBLEMS:
                print(f"[skip] unknown problem {problem_name}")
                continue
            for schedule in args.schedules:
                for seed in args.seeds:
                    _run_one(method_name, problem_name, schedule, seed, args)
    print(f"\nAll done in {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
