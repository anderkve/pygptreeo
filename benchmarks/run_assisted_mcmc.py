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
    energy_2d_01,
    ks_marginals_max,
    mmd_rbf_joint,
    run_assisted_chain,
    run_delayed_acceptance_chain,
    run_reference_chain,
    wasserstein1_marginals,
)

# Per-problem tuning, chosen in the iter 14 review.
_BETA_PER_PROBLEM = {"rosenbrock_2d": 0.5, "borehole_8d": 2.0}
_SIGMA_PER_PROBLEM = {"rosenbrock_2d": 0.04, "borehole_8d": 0.08}


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
    ap.add_argument("--mode", default="assisted",
                    choices=["assisted", "delayed"],
                    help="assisted: σ-gated; delayed: Christen-Fox DA")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    for problem_name in args.problems:
        problem = PROBLEMS[problem_name]
        beta = _BETA_PER_PROBLEM.get(problem_name, 1.0)
        proposal_sigma = _SIGMA_PER_PROBLEM.get(
            problem_name, args.proposal_sigma,
        )
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
                    f"(β={beta} σprop={proposal_sigma} "
                    f"f_min={f_min:.3f}, f_scale={f_scale:.3f})"
                )
                ref = run_reference_chain(
                    problem, seed=seed, n_steps=args.n_steps,
                    proposal_sigma=proposal_sigma, beta=beta,
                    f_min_scale=(f_min, f_scale),
                )
                cfg = {
                    "kind": "reference", "problem_name": problem_name,
                    "seed": seed, "n_steps": args.n_steps,
                    "proposal_sigma": proposal_sigma, "beta": beta,
                }
                _save(ref_file, {**ref, "f_min": np.float64(f_min),
                                 "f_scale": np.float64(f_scale),
                                 "beta": np.float64(beta)}, cfg)
                print(
                    f"    accept={ref['n_accept']/args.n_steps:.2f} | "
                    f"wall={ref['wall_time']:.1f}s"
                )

            for method in args.methods:
                tau_list = args.tau_sigmas if args.mode == "assisted" else [None]
                for tau in tau_list:
                    if args.mode == "delayed":
                        out_file = os.path.join(
                            args.out_dir,
                            f"delayed__{method}__{problem_name}__seed{seed}.npz",
                        )
                    else:
                        out_file = os.path.join(
                            args.out_dir,
                            _asst_fname(method, problem_name, tau, seed),
                        )
                    if os.path.exists(out_file) and not args.force:
                        print(f"[exists] {out_file}")
                        continue
                    tag = "delayed" if args.mode == "delayed" else "assisted"
                    tau_tag = "" if tau is None else f" | τσ={tau:g}"
                    print(
                        f"\n==> {tag} | {method} | {problem_name}{tau_tag} | seed {seed}"
                    )
                    t = time.time()
                    if args.mode == "delayed":
                        asst = run_delayed_acceptance_chain(
                            METHODS[method], problem, seed=seed,
                            n_steps=args.n_steps,
                            proposal_sigma=proposal_sigma, beta=beta,
                            f_min_scale=(f_min, f_scale),
                        )
                    else:
                        asst = run_assisted_chain(
                            METHODS[method], problem, seed=seed,
                            n_steps=args.n_steps, tau_sigma=tau,
                            proposal_sigma=proposal_sigma, beta=beta,
                            f_min_scale=(f_min, f_scale),
                        )
                    # Fidelity metrics. Burn-in 10 %.
                    burn = args.n_steps // 10
                    ref_burn = ref["samples"][burn:]
                    asst_burn = asst["samples"][burn:]
                    w1 = wasserstein1_marginals(ref_burn, asst_burn)
                    ks = ks_marginals_max(ref_burn, asst_burn)
                    e2d = energy_2d_01(ref_burn, asst_burn)
                    mmd = mmd_rbf_joint(ref_burn, asst_burn)
                    cfg = {
                        "kind": args.mode,
                        "method_name": method, "problem_name": problem_name,
                        "seed": seed, "n_steps": args.n_steps,
                        "proposal_sigma": proposal_sigma, "beta": beta,
                    }
                    if args.mode == "assisted":
                        cfg["tau_sigma"] = tau
                    payload = dict(asst)
                    payload.update({
                        "w1_marginals": np.float64(w1),
                        "ks_marginals_max": np.float64(ks),
                        "energy_2d_01": np.float64(e2d),
                        "mmd_rbf_joint": np.float64(mmd),
                        "f_min": np.float64(f_min),
                        "f_scale": np.float64(f_scale),
                        "beta": np.float64(beta),
                    })
                    _save(out_file, payload, cfg)
                    # n_true_evals field varies by mode.
                    n_true = int(asst.get("n_true_evals",
                                          asst.get("n_used_true", 0)))
                    speedup = args.n_steps / max(1, n_true)
                    print(
                        f"    accept={asst['n_accept']/args.n_steps:.2f} | "
                        f"n_true={n_true} speedup={speedup:.2f}x | "
                        f"W1={w1:.4f} KS={ks:.3f} MMD={mmd:.4f} | "
                        f"wall={asst['wall_time']:.1f}s"
                    )
    print(f"\nAll done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
