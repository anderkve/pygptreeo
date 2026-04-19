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


# Iteration 04 introduces per-method variants. Suffix convention:
#   "_A"      — baseline (matches the hyperparameters used in iter 03)
#   "_B"      — stress variant (usually more compute or a different
#                regime — either shows "more resources wouldn't rescue
#                the alternative" or "pygptreeo is robust across
#                hyperparameters")
#   "pygptreeo_C" — kernel ablation: pygptreeo with only Matern-1.5
#                (drops AnisotropicRQ so it matches the other GP-based
#                methods' kernel).  Addresses the "pygptreeo gets a
#                richer kernel" fairness concern.
# The bare legacy names (`pygptreeo`, `sklearn_gp`, …) are aliased to
# `_A` so that .npz files produced before iter 04 still load cleanly.


# ---------- pygptreeo --------------------------------------------------

def _make_pygptreeo_A(d: int):
    """Baseline: Nbar=200, retrain_every=200, Matern+RQ kernel."""
    return PyGPTreeOAdapter(
        d, Nbar=200, retrain_step=200, theta=1e-4, sigma_rel=1e-3,
        kernel_spec="matern+rq",
    )


def _make_pygptreeo_B(d: int):
    """Smaller leaves, faster retrain cadence, same kernel as baseline.

    Tests whether Nbar=200 is near-optimal or whether more, smaller
    leaves would help on curved problems like rosenbrock.
    """
    return PyGPTreeOAdapter(
        d, Nbar=100, retrain_step=100, theta=1e-4, sigma_rel=1e-3,
        kernel_spec="matern+rq",
    )


def _make_pygptreeo_C(d: int):
    """Kernel ablation: Nbar=200, retrain=200, but Matern-1.5 only.

    Gives an apples-to-apples comparison with `sklearn_gp`'s kernel.
    If `pygptreeo_C` still beats `sklearn_gp_B`, the "richer kernel"
    criticism is defused.
    """
    return PyGPTreeOAdapter(
        d, Nbar=200, retrain_step=200, theta=1e-4, sigma_rel=1e-3,
        kernel_spec="matern",
    )


def _make_pygptreeo_D(d: int):
    """Fast-adapting variant: Nbar=100, retrain_step=100.

    Targets streaming workloads where the input distribution is
    non-stationary (DE early generations, MCMC mode hops). Smaller
    leaves split sooner and the more frequent retrain cadence keeps
    each leaf's kernel hyperparameters in sync with the local data.
    Same `matern+rq` kernel as `_A` so the difference is purely the
    leaf-size / retrain-cadence pair.
    """
    return PyGPTreeOAdapter(
        d, Nbar=100, retrain_step=100, theta=1e-4, sigma_rel=1e-3,
        kernel_spec="matern+rq",
    )


def _make_pygptreeo_poe(d: int):
    """Aggregation ablation: same baseline as _A but PoE (product of
    experts) instead of the default MoE (mixture of experts).

    The adapter's GPTree takes an ``aggregation`` kwarg; PoE tends to
    give sharper but potentially over-confident predictions when leaves
    disagree. We include this variant so the paper can report the
    MoE-vs-PoE trade-off empirically rather than just citing the DLGP
    defaults.
    """
    from benchmarks.adapters.pygptreeo_adapter import (
        PyGPTreeOAdapter, _make_configured_gpr_class,
    )
    from pygptreeo import GPTree
    from pygptreeo.adapters import SklearnGPAdapter as _SklearnGPAdapter

    class _PoEAdapter(PyGPTreeOAdapter):
        def __init__(self, n_dims, Nbar=200, theta=1e-4,
                     retrain_step=200, sigma_rel=1e-3,
                     kernel_spec="matern+rq"):
            # We can't easily override a single kwarg on the GPTree
            # constructor from the parent __init__, so we rebuild the
            # tree here with aggregation="poe".
            self.n_dims = n_dims
            self.sigma_rel = sigma_rel
            self.kernel_spec = kernel_spec
            gpr_cls = _make_configured_gpr_class(n_dims, kernel_spec)
            self.tree = GPTree(
                GPR=_SklearnGPAdapter(gpr_cls()),
                Nbar=Nbar, theta=theta,
                split_position_method="median",
                split_dimension_criteria="max_uncertainty",
                retrain_every_n_points=retrain_step,
                use_calibrated_sigma=True,
                splitting_strategy="gradual",
                max_n_pred_leaves=3,
                aggregation="poe",
                use_hyperparameter_inheritance=False,
                use_standard_scaling=True,
                enable_point_rejection=False,
                enable_point_merging=False,
                enable_split_evaluation=True,
                n_split_candidates=4,
                split_eval_train_fraction=0.4,
                split_eval_min_points=20,
            )
            self._seen = 0
    return _PoEAdapter(d)


# ---------- sklearn_gp -------------------------------------------------

def _make_sklearn_gp_A(d: int):
    """Baseline: max_train=400 (d<=5) or 250 (d>=6), no optimiser restarts."""
    max_train = 400 if d <= 5 else 250
    return SklearnGPAdapter(
        d, retrain_every=200, max_train_points=max_train,
        n_restarts_optimizer=0,
    )


def _make_sklearn_gp_B(d: int):
    """Best-case: larger reservoir + (on 2-D) one optimiser restart.

    Use with --max-wall-time 600.  Not meant for borehole_8d — exact GP
    is hopeless there at these scales even without a budget cap.

    On 5-D problems the optimiser restart alone blows past the 300 s
    per-fit budget, so for d >= 5 we keep the bigger reservoir (600
    points) but drop restarts to 0 — the paper point of the `_B`
    variant is "what does a bigger reservoir do?", not "what does
    one more bfgs restart do?".
    """
    if d <= 2:
        max_train = 1200
        n_restarts = 1
    elif d <= 5:
        max_train = 600
        n_restarts = 0
    else:
        max_train = 250
        n_restarts = 0
    return SklearnGPAdapter(
        d, retrain_every=200, max_train_points=max_train,
        n_restarts_optimizer=n_restarts,
    )


# ---------- gpytorch SVGP ---------------------------------------------

def _make_svgp_A(d: int):
    """Baseline."""
    return GPyTorchSVGPAdapter(
        d, retrain_every=200, n_epochs=60,
        n_inducing=256, max_buffer=5000, lr=5e-3,
        max_steps_per_refit=500,
    )


def _make_svgp_B(d: int):
    """Heavy variant: 2x inducing, 2x epochs, 3x inner-loop step cap."""
    return GPyTorchSVGPAdapter(
        d, retrain_every=200, n_epochs=120,
        n_inducing=512, max_buffer=5000, lr=5e-3,
        max_steps_per_refit=1500,
    )


# ---------- random_forest ---------------------------------------------

def _make_rf_A(d: int):
    return RandomForestAdapter(
        d, retrain_every=200, n_estimators=300,
        max_train_points=20000,
    )


# ---------- river_knn -------------------------------------------------

def _make_river_knn_A(d: int):
    return RiverKNNAdapter(d, n_neighbors=8, window_size=4000)


def _make_river_knn_B(d: int):
    """Local variant: smaller k, shorter sliding window."""
    return RiverKNNAdapter(d, n_neighbors=3, window_size=1000)


METHODS = {
    # Variant-explicit names (preferred going forward).
    "pygptreeo_A": _make_pygptreeo_A,
    "pygptreeo_B": _make_pygptreeo_B,
    "pygptreeo_C": _make_pygptreeo_C,
    "pygptreeo_D": _make_pygptreeo_D,
    "pygptreeo_poe": _make_pygptreeo_poe,
    "sklearn_gp_A": _make_sklearn_gp_A,
    "sklearn_gp_B": _make_sklearn_gp_B,
    "gpytorch_svgp_A": _make_svgp_A,
    "gpytorch_svgp_B": _make_svgp_B,
    "random_forest_A": _make_rf_A,
    "river_knn_A": _make_river_knn_A,
    "river_knn_B": _make_river_knn_B,
    # Legacy aliases (point to the `_A` baselines so pre-iter-04 .npz
    # files produced under the bare names are still comparable).
    "pygptreeo": _make_pygptreeo_A,
    "sklearn_gp": _make_sklearn_gp_A,
    "gpytorch_svgp": _make_svgp_A,
    "random_forest": _make_rf_A,
    "river_knn": _make_river_knn_A,
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
    p.add_argument("--de-popsize", type=int, default=100,
                   help="DE popsize per dimension (scipy convention). "
                        "Larger => slower convergence, wider coverage.")
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
                  n_stream, n_test, checkpoint_every, max_wall_time, out_file,
                  schedule_kwargs):
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
            schedule_kwargs=schedule_kwargs,
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

    schedule_kwargs = {"de_popsize": args.de_popsize}
    if args.no_subprocess:
        _child_target(
            method_name, problem_name, schedule, seed,
            args.n_stream, args.n_test, args.checkpoint_every,
            args.max_wall_time, out_file, schedule_kwargs,
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
                  args.max_wall_time, out_file, schedule_kwargs),
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
            # Cheap trip-wire for the MoE-variance upstream regression: if
            # median NLPD ever shoots past 10^3, print a warning so we see
            # it in the driver log (bug we fixed in iter 02 pushed this to
            # ~10^12 on rosenbrock_2d).
            if abs(mednlpd) > 1e3:
                print(
                    f"    [WARN] NLPD sanity: {method_name}/{problem_name}/"
                    f"seed{seed} medNLPD={mednlpd:.2e} magnitude > 1e3 "
                    f"— possible upstream regression"
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
