"""Plots for the iter-14 emulator-assisted MCMC sweep."""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.make_plots import METHOD_COLOR, METHOD_LABEL, METHOD_ORDER


def load_all(data_dir: str) -> dict:
    """Returns (assisted, references) where:
    - assisted: {(method, problem, tau, seed): payload}
    - references: {(problem, seed): payload}
    """
    assisted = {}
    refs = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        try:
            d = dict(np.load(path, allow_pickle=True))
            cfg = json.loads(str(d["config_json"]))
        except Exception:
            continue
        if cfg["kind"] == "reference":
            refs[(cfg["problem_name"], int(cfg["seed"]))] = d
        elif cfg["kind"] == "assisted":
            assisted[(cfg["method_name"], cfg["problem_name"],
                      float(cfg["tau_sigma"]), int(cfg["seed"]))] = d
    return assisted, refs


def plot_marginals(assisted, refs, problems, out_path,
                   tau_pick=1e-2, n_dims_max=4, burn=2000):
    """Reference vs assisted 1-D marginals, one column per problem,
    one row per dim. Each column only plots as many dim-rows as its
    own problem actually has (others left blank).
    """
    if not refs:
        return
    n_dims_per_problem = {}
    for p in problems:
        key = next(((pp, s) for (pp, s) in refs if pp == p), None)
        if key is None:
            continue
        n_dims_per_problem[p] = min(
            refs[key]["samples"].shape[1], n_dims_max,
        )
    if not n_dims_per_problem:
        return
    n_dims_show = max(n_dims_per_problem.values())
    fig, axes = plt.subplots(
        n_dims_show, len(problems),
        figsize=(4.5 * len(problems), 2.6 * n_dims_show),
        squeeze=False, sharex=False,
    )
    handles = {}
    for pi, problem in enumerate(problems):
        ref_key = next(((p, s) for (p, s) in refs if p == problem), None)
        if ref_key is None:
            for di in range(n_dims_show):
                axes[di][pi].axis("off")
            continue
        ref = refs[ref_key]
        ref_samp = ref["samples"][burn:]
        d_here = n_dims_per_problem[problem]
        for di in range(n_dims_show):
            ax = axes[di][pi]
            if di >= d_here:
                ax.axis("off")
                continue
            ax.hist(ref_samp[:, di], bins=60, density=True,
                    color="black", histtype="step", linewidth=2,
                    label="reference (truth)")
            for (m, p, tau, seed), asst in assisted.items():
                if p != problem:
                    continue
                if abs(tau - tau_pick) / tau_pick > 0.05:
                    continue
                if seed != 0:
                    continue  # show seed-0 chain for readability
                samp = asst["samples"][burn:]
                color = METHOD_COLOR.get(m, "#888")
                label = METHOD_LABEL.get(m, m)
                ax.hist(samp[:, di], bins=60, density=True,
                        color=color, histtype="step", linewidth=1.5,
                        alpha=0.85, label=label)
                handles.setdefault(label, color)
            if di == 0:
                ax.set_title(problem)
            ax.set_xlabel(f"x[{di}]")
            ax.set_ylabel("density")
            if pi == 0 and di == 0:
                ax.legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle(
        f"1-D posterior marginals: reference vs assisted (τ_σ = {tau_pick:g})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_corner(assisted, refs, problems, out_path,
                tau_pick=1e-2, method_pick="pygptreeo_A", burn=2000):
    """2-D scatter of dims (0,1): reference + one chosen assisted method."""
    fig, axes = plt.subplots(
        1, len(problems), figsize=(4.5 * len(problems), 4.5), squeeze=False,
    )
    for pi, problem in enumerate(problems):
        ax = axes[0][pi]
        ref_key = next(((p, s) for (p, s) in refs if p == problem), None)
        if ref_key is None:
            continue
        ref = refs[ref_key]
        ref_samp = ref["samples"][burn:]
        ax.scatter(ref_samp[:, 0], ref_samp[:, 1], s=4, alpha=0.25,
                   color="black", label="reference")
        for (m, p, tau, seed), asst in assisted.items():
            if p != problem or m != method_pick:
                continue
            if abs(tau - tau_pick) / tau_pick > 0.05:
                continue
            samp = asst["samples"][burn:]
            ax.scatter(samp[:, 0], samp[:, 1], s=4, alpha=0.25,
                       color=METHOD_COLOR.get(m, "#888"),
                       label=METHOD_LABEL.get(m, m))
        ax.set_title(problem)
        ax.set_xlabel("x[0]"); ax.set_ylabel("x[1]")
        if pi == 0:
            ax.legend(fontsize=8, frameon=False, loc="best", markerscale=3)
    fig.suptitle(
        f"Corner: reference vs {method_pick} (τ_σ = {tau_pick:g})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fidelity_vs_speedup(assisted, refs, problems, out_path):
    fig, axes = plt.subplots(
        1, len(problems), figsize=(4.4 * len(problems), 4.0), squeeze=False,
    )
    handles_by_label = {}
    for pi, problem in enumerate(problems):
        ax = axes[0][pi]
        for (m, p, tau, seed), asst in assisted.items():
            if p != problem:
                continue
            n_total = int(asst["n_used_emu"]) + int(asst["n_used_true"])
            n_true = int(asst["n_used_true"])
            speedup = n_total / max(1, n_true)
            w1 = float(asst["w1_marginals"])
            color = METHOD_COLOR.get(m, "#888")
            label = METHOD_LABEL.get(m, m)
            ax.scatter(speedup, w1, s=80, color=color,
                       edgecolor="black", linewidth=0.5)
            ax.annotate(
                f"{tau:g}", (speedup, w1),
                fontsize=7, xytext=(4, 4), textcoords="offset points",
                color=color,
            )
            if label not in handles_by_label:
                from matplotlib.lines import Line2D
                handles_by_label[label] = Line2D(
                    [0], [0], marker="o", color="white",
                    markerfacecolor=color, markeredgecolor="black",
                    markersize=10, linewidth=0, label=label,
                )
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("speedup (n_steps / n_true_evals)")
        if pi == 0:
            ax.set_ylabel("W1 distance to reference (avg over dims)")
        ax.set_title(problem)
        ax.grid(True, which="both", alpha=0.3)
        if pi == 0 and handles_by_label:
            ax.legend(handles=list(handles_by_label.values()),
                      labels=list(handles_by_label.keys()),
                      fontsize=8, frameon=False, loc="best")
    fig.suptitle("Posterior fidelity vs deployment speedup", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trusted_err_hist(assisted, problems, out_path,
                          tau_pick=1e-2, burn=2000):
    fig, axes = plt.subplots(
        1, len(problems), figsize=(4.5 * len(problems), 3.4),
        squeeze=False,
    )
    for pi, problem in enumerate(problems):
        ax = axes[0][pi]
        for (m, p, tau, seed), asst in assisted.items():
            if p != problem or seed != 0:
                continue
            if abs(tau - tau_pick) / tau_pick > 0.05:
                continue
            err = np.asarray(asst.get("trusted_err", []), dtype=float)
            err = err[np.isfinite(err)]
            if err.size == 0:
                continue
            color = METHOD_COLOR.get(m, "#888")
            label = METHOD_LABEL.get(m, m)
            # log-scale histogram of |mu - true_logL|.
            bins = np.logspace(-5, 2, 45)
            ax.hist(err, bins=bins, histtype="step", linewidth=1.8,
                    color=color, label=label)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("|μ_emu − log L_true| on trusted steps")
        if pi == 0:
            ax.set_ylabel("count (seed 0)")
        ax.set_title(problem)
        ax.grid(True, which="both", alpha=0.3)
        if pi == 0:
            ax.legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle(f"Per-step trusted-prediction error (τ_σ = {tau_pick:g})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_accept_rate(assisted, refs, problems, out_path):
    from collections import defaultdict
    fig, axes = plt.subplots(
        1, len(problems), figsize=(4.5 * len(problems), 3.6),
        squeeze=False,
    )
    for pi, problem in enumerate(problems):
        ax = axes[0][pi]
        by_method = defaultdict(list)
        for (m, p, tau, seed), asst in assisted.items():
            if p != problem:
                continue
            rate = int(asst["n_accept"]) / max(1, int(asst["n_proposals"]))
            by_method[m].append((tau, rate))
        for m, xy in sorted(
            by_method.items(),
            key=lambda kv: METHOD_ORDER.index(kv[0]) if kv[0] in METHOD_ORDER else 99,
        ):
            by_tau = defaultdict(list)
            for tau, r in xy:
                by_tau[tau].append(r)
            xs = sorted(by_tau.keys())
            ys = [np.mean(by_tau[t]) for t in xs]
            es = [np.std(by_tau[t]) for t in xs]
            ax.errorbar(xs, ys, yerr=es, marker="o",
                        color=METHOD_COLOR.get(m, "#888"),
                        label=METHOD_LABEL.get(m, m),
                        linewidth=2, markersize=6, capsize=3)
        # Reference accept rate as a horizontal band.
        ref_rates = []
        for (p, _), ref in refs.items():
            if p == problem:
                ref_rates.append(int(ref["n_accept"]) / max(1, int(ref["n_proposals"])))
        if ref_rates:
            ax.axhline(np.mean(ref_rates), color="black",
                       linestyle="--", linewidth=1.5,
                       label=f"reference ({np.mean(ref_rates):.2f})")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\tau_\sigma$")
        if pi == 0:
            ax.set_ylabel("acceptance rate (mean ± std over seeds)")
        ax.set_title(problem)
        ax.grid(True, which="both", alpha=0.3)
        if pi == 0:
            ax.legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle("Emulator-induced acceptance-rate drift", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--problems", nargs="+",
                    default=["rosenbrock_2d", "borehole_8d"])
    ap.add_argument("--tau-picks", nargs="+", type=float,
                    default=[1e-2, 3e-2])
    ap.add_argument("--burn", type=int, default=2000)
    args = ap.parse_args()
    assisted, refs = load_all(args.data_dir)
    print(f"Loaded {len(assisted)} assisted runs, {len(refs)} references.")
    os.makedirs(args.out_dir, exist_ok=True)
    for tau in args.tau_picks:
        plot_marginals(
            assisted, refs, args.problems,
            os.path.join(args.out_dir, f"assisted_marginals_tau{tau:g}.png"),
            tau_pick=tau, burn=args.burn,
        )
    # Corner plot — default method (pygptreeo_A); second row for _D.
    plot_corner(
        assisted, refs, args.problems,
        os.path.join(args.out_dir, "assisted_corner.png"),
        tau_pick=args.tau_picks[0], method_pick="pygptreeo_A",
        burn=args.burn,
    )
    plot_fidelity_vs_speedup(
        assisted, refs, args.problems,
        os.path.join(args.out_dir, "assisted_fidelity_vs_speedup.png"),
    )
    plot_trusted_err_hist(
        assisted, args.problems,
        os.path.join(args.out_dir, "assisted_trusted_err_hist.png"),
        tau_pick=args.tau_picks[0], burn=args.burn,
    )
    plot_accept_rate(
        assisted, refs, args.problems,
        os.path.join(args.out_dir, "assisted_accept_rate.png"),
    )
    print(f"Wrote plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
