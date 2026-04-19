"""Driver: paired reference vs emulator-assisted MCMC sweep (iter 14)."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from benchmarks.problems import PROBLEMS
from benchmarks.run_all import METHODS
from benchmarks.mcmc_assisted import (
    estimate_f_scale,
    run_assisted_chain,
    run_reference_chain,
    wasserstein1_marginals,
)


def _ref_fname(problem, seed):
    return f"reference__{problem}__seed{seed}.npz"


def _asst_fname(method, problem, tau_sigma, seed):
    return f"assisted__{method}__{problem}__tau{tau_sigma:g}__seed{seed}.npz"


def _save(out_path, payload, cfg):
    payload = dict(payload)
    payload["config_json"] = np.asarray(json.dumps(cfg))
    np.savez(out_path, **payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--problems", nargs="+", required=True)
    ap.add_argument("--tau-sigmas", nargs="+", type=float,
                    default=[1e-3, 1e-2, 1e-1])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--n-steps", type=int, default=20000)
    ap.add_argument("--proposal-sigma", type=float, default=0.05)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    for problem_name in args.problems:
        problem = PROBLEMS[problem_name]
        for seed in args.seeds:
            ref_file = os.path.join(args.out_dir, _ref_fname(problem_name, seed))
            if os.path.exists(ref_file) and not args.force:
                print(f"[exists] {ref_file}")
                ref = dict(np.load(ref_file, allow_pickle=True))
                f_min = float(ref["f_min"]); f_scale = float(ref["f_scale"])
            else:
                rng_pre = np.random.default_rng(seed + 99_999)
                f_min, f_scale = estimate_f_scale(problem, rng_pre)
                print(
                    f"\n==> reference | {problem_name} | seed {seed} "
                    f"(f_min={f_min:.3f}, f_scale={f_scale:.3f})"
                )
                ref = run_reference_chain(
                    problem, seed=seed, n_steps=args.n_steps,
                    proposal_sigma=args.proposal_sigma,
                    f_min_scale=(f_min, f_scale),
                )
                cfg = {
                    "kind": "reference", "problem_name": problem_name,
                    "seed": seed, "n_steps": args.n_steps,
                    "proposal_sigma": args.proposal_sigma,
                }
                _save(ref_file, {**ref, "f_min": np.float64(f_min),
                                 "f_scale": np.float64(f_scale)}, cfg)
                print(
                    f"    accept={ref['n_accept']/args.n_steps:.2f} | "
                    f"wall={ref['wall_time']:.1f}s"
                )

            for method in args.methods:
                for tau in args.tau_sigmas:
                    out_file = os.path.join(
                        args.out_dir,
                        _asst_fname(method, problem_name, tau, seed),
                    )
                    if os.path.exists(out_file) and not args.force:
                        print(f"[exists] {out_file}")
                        continue
                    print(
                        f"\n==> assisted | {method} | {problem_name} | "
                        f"τσ={tau:g} | seed {seed}"
                    )
                    t = time.time()
                    asst = run_assisted_chain(
                        METHODS[method], problem, seed=seed,
                        n_steps=args.n_steps, tau_sigma=tau,
                        proposal_sigma=args.proposal_sigma,
                        f_min_scale=(f_min, f_scale),
                    )
                    # W1 distance against reference.
                    w1 = wasserstein1_marginals(ref["samples"], asst["samples"])
                    cfg = {
                        "kind": "assisted",
                        "method_name": method, "problem_name": problem_name,
                        "seed": seed, "n_steps": args.n_steps,
                        "tau_sigma": tau,
                        "proposal_sigma": args.proposal_sigma,
                    }
                    _save(out_file, {
                        **asst, "w1_marginals": np.float64(w1),
                        "f_min": np.float64(f_min),
                        "f_scale": np.float64(f_scale),
                    }, cfg)
                    speedup = args.n_steps / max(1, asst["n_used_true"])
                    print(
                        f"    accept={asst['n_accept']/args.n_steps:.2f} | "
                        f"emu_used={asst['n_used_emu']} "
                        f"true_used={asst['n_used_true']} "
                        f"speedup={speedup:.2f}x | "
                        f"W1={w1:.4f} | wall={asst['wall_time']:.1f}s"
                    )
    print(f"\nAll done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
