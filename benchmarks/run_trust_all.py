"""Driver: sweep (method, problem, schedule, τ_σ) for the trust harness.

Filenames: ``<method>__<problem>__<schedule>__tau{tau_sigma}__seed{N}.npz``
where ``tau_sigma`` is rendered as ``1e-3`` etc.

The runs are subprocess-isolated with a hard wall-time exactly like
``run_all.py``. Output goes into the iter directory's ``data/``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import multiprocessing as mp
import os
import time
import warnings

import numpy as np

from benchmarks.problems import PROBLEMS
from benchmarks.run_all import METHODS  # reuse the registry
from benchmarks.trust_harness import (
    DEFAULT_TAU_Y_GRID,
    TrustRunConfig,
    TrustRunResult,
    run_trust_threshold_benchmark,
)


def _fname(method, problem, schedule, tau_sigma, seed):
    tau_str = f"{tau_sigma:g}".replace("+", "")  # "0.001" or "1e-05"
    return (
        f"{method}__{problem}__{schedule}__tau{tau_str}__seed{seed}.npz"
    )


def _child_target(method_name, problem_name, schedule, seed,
                  n_stream, n_test, checkpoint_every, batch_size,
                  tau_sigma, max_wall_time, out_file, schedule_kwargs):
    warnings.filterwarnings("ignore")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        factory = METHODS[method_name]
        problem = PROBLEMS[problem_name]
        run_trust_threshold_benchmark(
            factory, problem,
            seed=seed, n_stream=n_stream, tau_sigma=tau_sigma,
            schedule=schedule, n_test=n_test,
            checkpoint_every=checkpoint_every, batch_size=batch_size,
            max_wall_time_s=max_wall_time, method_name=method_name,
            verbose=False, save_every_checkpoint_to=out_file,
            schedule_kwargs=schedule_kwargs,
        )


def _write_stub_aborted(out_file, method, problem, schedule, seed,
                        n_stream, n_test, checkpoint_every, batch_size,
                        tau_sigma, max_wall_time, wall_time, notes):
    cfg = TrustRunConfig(
        method_name=method, problem_name=problem, seed=seed,
        n_stream=n_stream, n_test=n_test,
        checkpoint_every=checkpoint_every, schedule=schedule,
        tau_sigma=tau_sigma, batch_size=batch_size,
        tau_y_grid=tuple(DEFAULT_TAU_Y_GRID),
        max_wall_time_s=max_wall_time,
    )
    r = TrustRunResult(config=cfg)
    r.wall_time = float(wall_time)
    r.aborted = True
    r.notes = notes
    np.savez(out_file, **r.to_npz_dict())


def _run_one(method, problem, schedule, tau_sigma, seed, args):
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(
        args.out_dir, _fname(method, problem, schedule, tau_sigma, seed),
    )
    if os.path.exists(out_file) and not args.force:
        print(f"[exists] {out_file}")
        return
    print(
        f"\n==> {method} | {problem} | sched={schedule} | "
        f"τσ={tau_sigma:g} | seed {seed}"
    )
    t0 = time.time()
    schedule_kwargs = {"de_popsize": args.de_popsize}

    if args.no_subprocess:
        _child_target(
            method, problem, schedule, seed,
            args.n_stream, args.n_test, args.checkpoint_every,
            args.batch_size, tau_sigma, args.max_wall_time, out_file,
            schedule_kwargs,
        )
    else:
        ctx = mp.get_context("spawn")
        deadline = args.max_wall_time + 30.0
        proc = ctx.Process(
            target=_child_target,
            args=(method, problem, schedule, seed,
                  args.n_stream, args.n_test, args.checkpoint_every,
                  args.batch_size, tau_sigma, args.max_wall_time,
                  out_file, schedule_kwargs),
        )
        proc.start()
        proc.join(timeout=deadline)
        if proc.is_alive():
            print(f"    [TIMEOUT] hard-killing after {deadline:.0f}s")
            proc.terminate()
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.kill(); proc.join()
            elapsed = time.time() - t0
            if not os.path.exists(out_file):
                _write_stub_aborted(
                    out_file, method, problem, schedule, seed,
                    args.n_stream, args.n_test, args.checkpoint_every,
                    args.batch_size, tau_sigma, args.max_wall_time,
                    elapsed, f"hard-killed after {elapsed:.0f}s",
                )

    elapsed = time.time() - t0
    if os.path.exists(out_file):
        try:
            d = np.load(out_file, allow_pickle=True)
            if d["nrmse"].shape[0] == 0:
                print(f"    [no checkpoints in {elapsed:.1f}s]")
            else:
                print(
                    f"    done in {elapsed:.1f}s | "
                    f"nrmse_last={float(d['nrmse'][-1]):.3e} | "
                    f"trained={int(d['cum_n_trained'][-1])} "
                    f"trusted={int(d['cum_n_trusted'][-1])} "
                    f"speedup={args.n_stream/max(1,int(d['cum_n_trained'][-1])):.2f}x"
                )
        except Exception as exc:
            print(f"    [load failed: {exc}]")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--problems", nargs="+", required=True)
    p.add_argument("--schedules", nargs="+", default=["mcmc"])
    p.add_argument("--tau-sigmas", nargs="+", type=float,
                   default=[1e-3, 5e-3, 1e-2, 5e-2])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--n-stream", type=int, default=8000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--max-wall-time", type=float, default=900.0)
    p.add_argument("--de-popsize", type=int, default=300)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--no-subprocess", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    for method in args.methods:
        for problem in args.problems:
            for schedule in args.schedules:
                for tau in args.tau_sigmas:
                    for seed in args.seeds:
                        _run_one(method, problem, schedule, tau, seed, args)
    print(f"\nAll done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
