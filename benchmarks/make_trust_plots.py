"""Plots for the iter-13 trust-threshold deployment sweep."""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.make_plots import (
    METHOD_COLOR, METHOD_LABEL, METHOD_ORDER, _LEGACY_TO_A,
)


def load_all(data_dir: str) -> dict:
    """Return {(method, problem, schedule, tau_sigma): [runs]}."""
    results = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        try:
            d = np.load(path, allow_pickle=True)
            cfg = json.loads(str(d["config_json"]))
        except Exception:
            continue
        key = (cfg["method_name"], cfg["problem_name"],
               cfg["schedule"], float(cfg["tau_sigma"]))
        r = {"seed": cfg["seed"]}
        for k in d.files:
            if k != "config_json":
                r[k] = d[k]
        results[key].append(r)
    return results


def _final_speedup(r, n_stream):
    if "cum_n_trained" in r and len(r["cum_n_trained"]) > 0:
        n_trained = int(r["cum_n_trained"][-1])
        if n_trained == 0:
            return float("inf")
        return n_stream / float(n_trained)
    return float("nan")


def plot_speedup_vs_threshold(results, problems, out_path,
                              schedules=("mcmc", "iid")):
    """Two-row figure: speedup vs τ_σ, one row per schedule, one column
    per problem, one line per method, shared legend.
    """
    fig, axes = plt.subplots(
        len(schedules), len(problems),
        figsize=(4.2 * len(problems), 3.4 * len(schedules)),
        squeeze=False, sharex=True, sharey=True,
    )
    handles_by_label = {}
    for si, schedule in enumerate(schedules):
        for pi, problem in enumerate(problems):
            ax = axes[si][pi]
            by_method = defaultdict(list)
            for (m, p, s, tau), runs in results.items():
                if p != problem or s != schedule or not runs:
                    continue
                r = runs[0]
                if "cum_n_trained" not in r or len(r["cum_n_trained"]) == 0:
                    continue
                n_stream = int(r["cum_n_trained"][-1] + r["cum_n_trusted"][-1])
                speedup = _final_speedup(r, n_stream)
                by_method[m].append((tau, speedup))
            for m, xy in sorted(
                by_method.items(),
                key=lambda kv: METHOD_ORDER.index(kv[0])
                if kv[0] in METHOD_ORDER else 99,
            ):
                xy = sorted(xy)
                xs = [t for (t, _) in xy]
                ys = [s for (_, s) in xy]
                color = METHOD_COLOR.get(m, "#888")
                label = METHOD_LABEL.get(m, m)
                ax.plot(xs, ys, "o-", color=color, label=label,
                        linewidth=2, markersize=7)
                handles_by_label.setdefault(label, color)
            ax.set_xscale("log"); ax.set_yscale("log")
            if si == len(schedules) - 1:
                ax.set_xlabel(r"trust threshold $\tau_\sigma$"
                              " (relative to y-range)")
            if pi == 0:
                ax.set_ylabel(f"speedup [{schedule}]")
            if si == 0:
                ax.set_title(problem)
            ax.grid(True, which="both", alpha=0.3)
    if handles_by_label:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color=c, marker="o", linewidth=2, label=l)
            for l, c in handles_by_label.items()
        ]
        fig.legend(handles=handles, labels=list(handles_by_label.keys()),
                   loc="upper right", bbox_to_anchor=(0.99, 0.97),
                   ncol=1, fontsize=9, frameon=False)
    fig.suptitle("Trust-threshold deployment: speedup vs τ_σ", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.83, 0.94])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trained_vs_batch(results, problems, out_path,
                          schedules=("mcmc", "iid"),
                          tau_sigma_pick=1e-2):
    """Cumulative n_trained as a function of stream step, one panel per
    (problem, schedule) cell. Makes the MCMC-revisit plateau visible.
    """
    fig, axes = plt.subplots(
        len(schedules), len(problems),
        figsize=(4.4 * len(problems), 3.4 * len(schedules)),
        squeeze=False, sharex=True, sharey=False,
    )
    handles_by_label = {}
    for si, schedule in enumerate(schedules):
        for pi, problem in enumerate(problems):
            ax = axes[si][pi]
            for (m, p, s, tau), runs in results.items():
                if p != problem or s != schedule or not runs:
                    continue
                if abs(tau - tau_sigma_pick) / tau_sigma_pick > 0.1:
                    continue
                r = runs[0]
                cum_t = np.asarray(r.get("cum_n_trained", []), dtype=float)
                cum_s = np.asarray(r.get("cum_n_trusted", []), dtype=float)
                if cum_t.size == 0:
                    continue
                # The cumulative arrays are at batch boundaries.
                batch_size = int(cum_t[0] + cum_s[0]) if (cum_t.size > 0) else 1000
                xs = (np.arange(1, cum_t.size + 1)) * batch_size
                color = METHOD_COLOR.get(m, "#888")
                label = METHOD_LABEL.get(m, m)
                ax.plot(xs, cum_t, "-", color=color, linewidth=2,
                        label=label)
                handles_by_label.setdefault(label, color)
            # Reference y=x line: every stream step trains.
            xs_ref = ax.get_xlim()
            ax.plot([0, max(xs_ref)], [0, max(xs_ref)],
                    "--", color="black", linewidth=0.8, alpha=0.5,
                    label="trains-on-everything (no skipping)")
            handles_by_label.setdefault(
                "trains-on-everything (no skipping)", "black"
            )
            if si == len(schedules) - 1:
                ax.set_xlabel("stream step")
            if pi == 0:
                ax.set_ylabel(f"cum. n_trained [{schedule}]")
            if si == 0:
                ax.set_title(problem)
            ax.grid(True, alpha=0.3)
    if handles_by_label:
        from matplotlib.lines import Line2D
        handles = []
        for label, c in handles_by_label.items():
            ls = "--" if c == "black" else "-"
            handles.append(Line2D([0], [0], color=c, linewidth=2,
                                  linestyle=ls, label=label))
        fig.legend(handles=handles, labels=list(handles_by_label.keys()),
                   loc="upper right", bbox_to_anchor=(0.99, 0.97),
                   ncol=1, fontsize=8, frameon=False)
    fig.suptitle(
        f"Cumulative true-function calls (τ_σ = {tau_sigma_pick:g})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 0.81, 0.94])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_quality_per_batch(results, problems, out_path,
                           tau_sigma_pick=1e-2, schedule="mcmc",
                           tau_y_col=1):
    """Bar chart: per-1000-step batch fraction of trusted predictions that
    landed within τ_y = tau_y_grid[tau_y_col]. One panel per problem, bars
    by method side-by-side within each batch.
    """
    fig, axes = plt.subplots(
        1, len(problems), figsize=(5.2 * len(problems), 3.8), squeeze=False,
    )
    for pi, problem in enumerate(problems):
        ax = axes[0][pi]
        # Find all methods that have a run at this τ_σ.
        methods = []
        for (m, p, s, tau), runs in sorted(results.items()):
            if p != problem or s != schedule or not runs:
                continue
            if abs(tau - tau_sigma_pick) / tau_sigma_pick < 0.05:
                methods.append(m)
        methods = [m for m in METHOD_ORDER if m in methods]
        if not methods:
            ax.text(0.5, 0.5, "no runs", ha="center", va="center")
            ax.set_title(problem); continue

        # Gather per-batch fractions.
        n_batches = 0
        method_curves = {}
        for m in methods:
            runs = results[(m, problem, schedule, tau_sigma_pick)]
            r = runs[0]
            fracs = r["batch_frac_within_tau_y"]  # (n_batches, n_grid)
            fracs = np.asarray(fracs, dtype=float)
            if fracs.ndim != 2:
                continue
            col = fracs[:, tau_y_col]
            method_curves[m] = col
            n_batches = max(n_batches, col.size)

        width = 0.8 / max(1, len(methods))
        x = np.arange(n_batches)
        for mi, m in enumerate(methods):
            col = method_curves.get(m)
            if col is None:
                continue
            offset = (mi - (len(methods) - 1) / 2.0) * width
            # Plot NaN as 0-height empty bar; edgecolor indicates "no trusted picks".
            heights = np.where(np.isfinite(col), col, 0.0)
            ax.bar(
                x + offset, heights, width,
                color=METHOD_COLOR.get(m, "#888"),
                edgecolor="black", linewidth=0.4,
                label=METHOD_LABEL.get(m, m),
                alpha=0.9,
            )
        ax.set_xlabel("1000-step batch")
        if pi == 0:
            ax.set_ylabel(r"fraction of trusted picks with |μ − f| ≤ τ_y")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(f"{problem}  (τ_σ = {tau_sigma_pick:g})")
        ax.grid(True, axis="y", alpha=0.3)
        if pi == 0:
            ax.legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle("Trusted-prediction quality per 1000-step batch", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trust_pareto(results, problems, out_path, schedule="mcmc"):
    """Accuracy (NRMSE on held-out test) vs Speedup Pareto."""
    fig, axes = plt.subplots(
        1, len(problems) + 1, figsize=(4.2 * (len(problems) + 1), 3.8),
        squeeze=False,
    )
    handles_by_label = {}
    for pi, problem in enumerate(problems):
        ax = axes[0][pi]
        for (m, p, s, tau), runs in results.items():
            if p != problem or s != schedule or not runs:
                continue
            r = runs[0]
            if "cum_n_trained" not in r or len(r["cum_n_trained"]) == 0:
                continue
            n_trained = int(r["cum_n_trained"][-1])
            n_stream = int(r["cum_n_trained"][-1] + r["cum_n_trusted"][-1])
            speedup = n_stream / max(1, n_trained)
            nrmse = float(r["nrmse"][-1]) if len(r["nrmse"]) > 0 else float("nan")
            color = METHOD_COLOR.get(m, "#888")
            label = METHOD_LABEL.get(m, m)
            ax.scatter(speedup, nrmse, s=70, color=color,
                       edgecolor="black", linewidth=0.5)
            # Annotate τ_σ on the point.
            ax.annotate(f"{tau:g}", (speedup, nrmse),
                        fontsize=7, xytext=(3, 3), textcoords="offset points",
                        color=color)
            if label not in handles_by_label:
                from matplotlib.lines import Line2D
                handles_by_label[label] = Line2D(
                    [0], [0], marker="o", color="white",
                    markerfacecolor=color, markeredgecolor="black",
                    markersize=10, linewidth=0, label=label,
                )
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("speedup (n_stream / n_trained)")
        if pi == 0:
            ax.set_ylabel("final NRMSE on held-out test")
        ax.set_title(problem)
        ax.grid(True, which="both", alpha=0.3)
    leg_ax = axes[0][-1]
    leg_ax.axis("off")
    if handles_by_label:
        leg_ax.legend(
            handles=list(handles_by_label.values()),
            labels=list(handles_by_label.keys()),
            loc="center", frameon=False, fontsize=10,
            title="methods (labels = τ_σ)", title_fontsize=11,
        )
    fig.suptitle(
        "Deployment Pareto — accuracy vs speedup across trust thresholds",
        fontsize=13,
    )
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
    ap.add_argument("--schedules", nargs="+", default=["mcmc", "iid"])
    ap.add_argument("--tau-sigma-pick", type=float, default=1e-2)
    ap.add_argument("--tau-y-picks", nargs="+", type=float,
                    default=[1e-3, 1e-2])
    args = ap.parse_args()
    results = load_all(args.data_dir)
    print(f"Loaded {sum(len(v) for v in results.values())} runs, "
          f"{len(results)} (method, problem, schedule, τ_σ) cells.")
    os.makedirs(args.out_dir, exist_ok=True)
    plot_speedup_vs_threshold(
        results, args.problems,
        os.path.join(args.out_dir, "trust_speedup.png"),
        schedules=tuple(args.schedules),
    )
    # Map a τ_y CLI pick to its column in DEFAULT_TAU_Y_GRID.
    from benchmarks.trust_harness import DEFAULT_TAU_Y_GRID
    grid = list(DEFAULT_TAU_Y_GRID)
    for tau_y in args.tau_y_picks:
        col = int(np.argmin(np.abs(np.asarray(grid) - tau_y)))
        out = os.path.join(
            args.out_dir, f"trust_quality_per_batch_tau{tau_y:g}.png",
        )
        plot_quality_per_batch(
            results, args.problems, out,
            tau_sigma_pick=args.tau_sigma_pick,
            schedule=args.schedules[0],
            tau_y_col=col,
        )
    for sched in args.schedules:
        out = os.path.join(args.out_dir, f"trust_pareto_{sched}.png")
        plot_trust_pareto(
            results, args.problems, out, schedule=sched,
        )
    plot_trained_vs_batch(
        results, args.problems,
        os.path.join(args.out_dir, "trained_vs_batch.png"),
        schedules=tuple(args.schedules),
        tau_sigma_pick=args.tau_sigma_pick,
    )
    print(f"Wrote plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
