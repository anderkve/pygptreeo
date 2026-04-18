"""Generate publication-grade comparison plots from saved benchmark results.

Reads every ``*.npz`` file in ``benchmarks/data/``, aggregates across seeds,
and writes three figures to ``benchmarks/plots/``:

    comparison.png — grid of (problem × metric) over points-processed with
        median + IQR shading over seeds.

    summary.png   — final-checkpoint bar charts (median + IQR error bars).

    pareto.png    — final NRMSE vs cumulative update time scatter, one panel
        per problem; shows the accuracy-compute Pareto frontier.

    calibration.png — empirical coverage vs nominal coverage (reliability
        diagram) at the final checkpoint, one panel per problem.

Plots can be regenerated without re-running the benchmark.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = [
    "pygptreeo",
    "sklearn_gp",
    "gpytorch_svgp",
    "random_forest",
    "river_knn",
]

METHOD_LABEL = {
    "pygptreeo": "pygptreeo",
    "sklearn_gp": "sklearn GP (refit)",
    "gpytorch_svgp": "GPyTorch SVGP",
    "random_forest": "RandomForest (refit)",
    "river_knn": "River kNN",
}

METHOD_COLOR = {
    "pygptreeo": "#d7263d",
    "sklearn_gp": "#1b9e77",
    "gpytorch_svgp": "#7570b3",
    "random_forest": "#e6ab02",
    "river_knn": "#666666",
}

METHOD_LS = {
    "pygptreeo": "-",
    "sklearn_gp": "--",
    "gpytorch_svgp": "-.",
    "random_forest": ":",
    "river_knn": (0, (3, 1, 1, 1)),
}


def load_all(data_dir: str) -> dict:
    """Return {(method, problem, schedule): [per-seed dicts]}."""
    results = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        data = np.load(path, allow_pickle=True)
        cfg = json.loads(str(data["config_json"]))
        key = (cfg["method_name"], cfg["problem_name"], cfg.get("schedule", "iid"))
        r = {"seed": cfg["seed"]}
        for k in data.files:
            if k == "config_json":
                continue
            r[k] = data[k]
        results[key].append(r)
    return dict(results)


def _stack_over_seeds(runs: list, key: str):
    arrays = [np.asarray(r[key], dtype=float) for r in runs if key in r]
    if not arrays:
        return None, None
    max_len = max(len(a) for a in arrays)
    padded = np.full((len(arrays), max_len), np.nan)
    for i, a in enumerate(arrays):
        padded[i, :len(a)] = a
    xref = np.asarray(
        max(runs, key=lambda r: len(r["checkpoints"]))["checkpoints"],
        dtype=float,
    )
    return xref, padded


def _draw_metric(ax, results, problem, key, schedule="iid"):
    for method in METHOD_ORDER:
        runs = results.get((method, problem, schedule))
        if not runs:
            continue
        x, y = _stack_over_seeds(runs, key)
        if x is None or np.all(np.isnan(y)):
            continue
        med = np.nanmedian(y, axis=0)
        if y.shape[0] > 1:
            q1 = np.nanpercentile(y, 25, axis=0)
            q3 = np.nanpercentile(y, 75, axis=0)
        else:
            q1 = q3 = med
        ax.plot(x, med, color=METHOD_COLOR[method],
                linestyle=METHOD_LS[method], linewidth=2.0,
                label=METHOD_LABEL[method])
        if y.shape[0] > 1:
            ax.fill_between(x, q1, q3, color=METHOD_COLOR[method],
                            alpha=0.18, linewidth=0)


def plot_main_comparison(results, problems, out_path, schedule="iid"):
    metrics = [
        ("nrmse",                 "NRMSE on held-out test set",      "log"),
        ("median_nlpd",           "Median NLPD",                     "linear"),
        ("crps",                  "CRPS",                            "log"),
        ("coverage_1sigma",       "Empirical 1-sigma coverage",      "linear"),
        ("frac_pathological_std", "Fraction pathological std",       "linear"),
        ("cum_update_time",       "Cumulative update time [s]",      "log"),
    ]
    nrows = len(problems)
    ncols = len(metrics)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows),
        sharex="col", squeeze=False,
    )
    for i, problem in enumerate(problems):
        for j, (key, title, yscale) in enumerate(metrics):
            ax = axes[i][j]
            _draw_metric(ax, results, problem, key, schedule=schedule)
            if yscale == "log":
                ax.set_yscale("log")
            if j == 0:
                ax.set_ylabel(problem, fontsize=11, fontweight="bold")
            if i == 0:
                ax.set_title(title, fontsize=10)
            if i == nrows - 1:
                ax.set_xlabel("points processed")
            if key == "coverage_1sigma":
                ax.axhline(0.6827, color="black", linestyle=":",
                           linewidth=1, alpha=0.7)
                ax.set_ylim(-0.05, 1.05)
            if key == "frac_pathological_std":
                ax.set_ylim(-0.02, max(0.5, ax.get_ylim()[1]))
            ax.grid(True, alpha=0.3)
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi); labels.append(li)
        if labels:
            break
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)),
               bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.suptitle("pygptreeo vs. alternatives — continual emulation benchmark",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(results, problems, out_path, schedule="iid"):
    """Reliability diagram at the final checkpoint: empirical vs nominal cov."""
    nominal = np.array([0.50, 0.6827, 0.90, 0.95])
    cov_keys = ["coverage_50", "coverage_1sigma", "coverage_90", "coverage_95"]
    fig, axes = plt.subplots(
        1, len(problems), figsize=(3.4 * len(problems), 3.6),
        squeeze=False, sharey=True,
    )
    for i, problem in enumerate(problems):
        ax = axes[0][i]
        ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1)
        for method in METHOD_ORDER:
            runs = results.get((method, problem, schedule))
            if not runs:
                continue
            vals = []
            for k in cov_keys:
                arrs = [np.asarray(r[k])[-1] for r in runs if k in r and len(r[k]) > 0]
                if not arrs:
                    continue
                vals.append(np.nanmedian(arrs))
            if len(vals) != len(nominal):
                continue
            ax.plot(nominal, vals, "o-", color=METHOD_COLOR[method],
                    linewidth=2.0, markersize=7, label=METHOD_LABEL[method])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("nominal coverage")
        if i == 0:
            ax.set_ylabel("empirical coverage")
        ax.set_title(problem)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Calibration (final checkpoint) — closer to diagonal is better",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_bars(results, problems, out_path, schedule="iid"):
    fig, axes = plt.subplots(
        3, len(problems), figsize=(3.7 * len(problems), 8.5), squeeze=False,
    )
    rows = [
        ("nrmse",       "final NRMSE",       True),
        ("median_nlpd", "final median NLPD", False),
        ("cum_update_time", "total update time [s]", True),
    ]
    for i, problem in enumerate(problems):
        for r_idx, (key, title, logy) in enumerate(rows):
            ax = axes[r_idx][i]
            methods, meds, los, his, colors, labels = [], [], [], [], [], []
            for method in METHOD_ORDER:
                runs = results.get((method, problem, schedule))
                if not runs:
                    continue
                vals = np.array([
                    r[key][-1] for r in runs if key in r and len(r[key]) > 0
                ], dtype=float)
                if vals.size == 0 or np.all(np.isnan(vals)):
                    continue
                med = float(np.nanmedian(vals))
                q1 = float(np.nanpercentile(vals, 25)) if vals.size > 1 else med
                q3 = float(np.nanpercentile(vals, 75)) if vals.size > 1 else med
                methods.append(method); meds.append(med)
                los.append(med - q1); his.append(q3 - med)
                colors.append(METHOD_COLOR[method])
                labels.append(METHOD_LABEL[method])
            if not methods:
                ax.set_visible(False); continue
            ax.bar(range(len(methods)), meds, color=colors,
                   yerr=[los, his], capsize=4)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            if logy:
                ax.set_yscale("log")
            if r_idx == 0:
                ax.set_title(problem, fontsize=11, fontweight="bold")
            if i == 0:
                ax.set_ylabel(title)
            ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        "Final-checkpoint summary (median + IQR error bars over seeds)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(results, problems, out_path, schedule="iid"):
    fig, axes = plt.subplots(
        1, len(problems), figsize=(4.0 * len(problems), 3.8), squeeze=False,
    )
    for i, problem in enumerate(problems):
        ax = axes[0][i]
        for method in METHOD_ORDER:
            runs = results.get((method, problem, schedule))
            if not runs:
                continue
            xs, ys = [], []
            for r in runs:
                if len(r.get("cum_update_time", [])) == 0:
                    continue
                xs.append(float(r["cum_update_time"][-1]))
                ys.append(float(r["nrmse"][-1]))
            if not xs:
                continue
            ax.scatter(xs, ys, s=70, color=METHOD_COLOR[method],
                       edgecolor="black", linewidth=0.5,
                       label=METHOD_LABEL[method], zorder=3)
            ax.scatter([np.median(xs)], [np.median(ys)], s=160,
                       color=METHOD_COLOR[method], edgecolor="black",
                       linewidth=1.2, marker="D", zorder=4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("total update time [s]");
        if i == 0: ax.set_ylabel("final NRMSE")
        ax.set_title(problem)
        ax.grid(True, which="both", alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle("Accuracy vs compute budget — Pareto front "
                 "(diamonds = per-method medians)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="benchmarks/data")
    ap.add_argument("--plots-dir", default="benchmarks/plots")
    ap.add_argument("--problems", nargs="+",
                    default=["smooth_sines_2d", "rosenbrock_2d",
                             "friedman1_5d", "borehole_8d"])
    ap.add_argument("--schedule", default="iid")
    args = ap.parse_args()

    results = load_all(args.data_dir)
    if not results:
        print(f"No results in {args.data_dir}.", file=sys.stderr)
        sys.exit(1)
    n_runs = sum(len(v) for v in results.values())
    print(f"Loaded {n_runs} runs across {len(results)} "
          f"(method, problem, schedule) combinations.")

    plot_main_comparison(results, args.problems,
                         os.path.join(args.plots_dir, "comparison.png"),
                         schedule=args.schedule)
    plot_summary_bars(results, args.problems,
                      os.path.join(args.plots_dir, "summary.png"),
                      schedule=args.schedule)
    plot_pareto(results, args.problems,
                os.path.join(args.plots_dir, "pareto.png"),
                schedule=args.schedule)
    plot_calibration(results, args.problems,
                     os.path.join(args.plots_dir, "calibration.png"),
                     schedule=args.schedule)
    print(f"Wrote plots to {args.plots_dir}/")


if __name__ == "__main__":
    main()
