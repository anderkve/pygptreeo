"""Generate a multi-panel comparison figure from saved benchmark results.

Reads every ``*.npz`` file in ``benchmarks/data/``, aggregates across seeds,
and produces a single figure at ``benchmarks/plots/comparison.png`` with:

    Rows = problems,
    Columns = (NRMSE vs. points, 1-sigma coverage vs. points,
               cumulative update time vs. points, NLPD vs. points).

A separate summary figure ``benchmarks/plots/summary.png`` shows, for each
problem, a final-step bar chart of NRMSE and cumulative update time per method.

Re-running the plotting script does NOT re-run the benchmark; it only reads
from disk.
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
    "pygptreeo": "#d7263d",        # red
    "sklearn_gp": "#1b9e77",       # teal
    "gpytorch_svgp": "#7570b3",    # purple
    "random_forest": "#e6ab02",    # yellow
    "river_knn": "#666666",        # grey
}

METHOD_LS = {
    "pygptreeo": "-",
    "sklearn_gp": "--",
    "gpytorch_svgp": "-.",
    "random_forest": ":",
    "river_knn": (0, (3, 1, 1, 1)),
}


def load_all(data_dir: str) -> dict:
    """Return {(method, problem): [results...]} indexed by seed."""
    results = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        data = np.load(path, allow_pickle=True)
        cfg = json.loads(str(data["config_json"]))
        key = (cfg["method_name"], cfg["problem_name"])
        results[key].append({
            "seed": cfg["seed"],
            "checkpoints": data["checkpoints"],
            "rmse": data["rmse"],
            "nrmse": data["nrmse"],
            "mae": data["mae"],
            "nlpd": data["nlpd"],
            "coverage": data["coverage_1sigma"],
            "cum_update_time": data["cum_update_time"],
            "cum_predict_time": data["cum_predict_time"],
        })
    return dict(results)


def _stack_over_seeds(runs: list, key: str):
    """Stack the metric across seeds, left-aligned; pad shorter runs with NaN."""
    arrays = [np.asarray(r[key], dtype=float) for r in runs]
    if not arrays:
        return None, None
    max_len = max(len(a) for a in arrays)
    padded = np.full((len(arrays), max_len), np.nan)
    for i, a in enumerate(arrays):
        padded[i, :len(a)] = a
    # Reference checkpoints = from whichever run went the furthest.
    xref = np.asarray(
        max(runs, key=lambda r: len(r["checkpoints"]))["checkpoints"],
        dtype=float,
    )
    return xref, padded


def plot_main_comparison(results: dict, problems: list, out_path: str):
    ncols = 4
    nrows = len(problems)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows),
        sharex="col", squeeze=False,
    )

    metrics = [
        ("nrmse", "NRMSE on held-out test set", "log"),
        ("coverage", "Empirical 1-sigma coverage", "linear"),
        ("cum_update_time", "Cumulative update time [s]", "log"),
        ("nlpd", "Mean NLPD on held-out test set", "symlog"),
    ]

    for i, problem in enumerate(problems):
        for j, (key, ylabel, yscale) in enumerate(metrics):
            ax = axes[i][j]
            for method in METHOD_ORDER:
                runs = results.get((method, problem))
                if not runs:
                    continue
                x, y = _stack_over_seeds(runs, key)
                if x is None:
                    continue
                if np.all(np.isnan(y)):
                    continue
                med = np.nanmedian(y, axis=0)
                lo = np.nanmin(y, axis=0)
                hi = np.nanmax(y, axis=0)
                ax.plot(
                    x, med,
                    color=METHOD_COLOR[method], linestyle=METHOD_LS[method],
                    label=METHOD_LABEL[method], linewidth=2.0,
                )
                if y.shape[0] > 1:
                    ax.fill_between(
                        x, lo, hi, color=METHOD_COLOR[method],
                        alpha=0.15, linewidth=0,
                    )
            if yscale == "log":
                ax.set_yscale("log")
            elif yscale == "symlog":
                ax.set_yscale("symlog", linthresh=1.0)
            if j == 0:
                ax.set_ylabel(problem, fontsize=11, fontweight="bold")
            if i == 0:
                ax.set_title(ylabel, fontsize=11)
            if i == nrows - 1:
                ax.set_xlabel("points processed")
            if key == "coverage":
                ax.axhline(0.68, color="black", linestyle=":",
                           linewidth=1, alpha=0.7)
                ax.set_ylim(-0.05, 1.05)
            if key == "nlpd":
                # Clip to robust range across methods to avoid spike domination.
                all_vals = []
                for method in METHOD_ORDER:
                    runs = results.get((method, problem))
                    if not runs:
                        continue
                    for r in runs:
                        v = np.asarray(r["nlpd"], dtype=float)
                        v = v[np.isfinite(v)]
                        all_vals.append(v)
                if all_vals:
                    flat = np.concatenate(all_vals)
                    if flat.size > 0:
                        p_lo = np.percentile(flat, 5)
                        p_hi = np.percentile(flat, 95)
                        pad = max(1.0, 0.2 * (p_hi - p_lo + 1e-6))
                        ax.set_ylim(p_lo - pad, p_hi + pad)
            ax.grid(True, alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(labels), 5), bbox_to_anchor=(0.5, -0.01),
               frameon=False)
    fig.suptitle(
        "Continual emulation benchmark: pygptreeo vs. alternatives",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_bars(results: dict, problems: list, out_path: str):
    fig, axes = plt.subplots(
        2, len(problems), figsize=(4.2 * len(problems), 6.0),
        squeeze=False,
    )
    for i, problem in enumerate(problems):
        # Collect final-step values.
        methods, nrmse_final, ttime_final = [], [], []
        for method in METHOD_ORDER:
            runs = results.get((method, problem))
            if not runs:
                continue
            nrmse_vals = [r["nrmse"][-1] for r in runs
                          if len(r["nrmse"]) > 0]
            time_vals = [r["cum_update_time"][-1] for r in runs
                         if len(r["cum_update_time"]) > 0]
            if not nrmse_vals:
                continue
            methods.append(method)
            nrmse_final.append(np.nanmean(nrmse_vals))
            ttime_final.append(np.nanmean(time_vals))
        colors = [METHOD_COLOR[m] for m in methods]
        labels = [METHOD_LABEL[m] for m in methods]

        ax = axes[0][i]
        ax.bar(range(len(methods)), nrmse_final, color=colors)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yscale("log")
        ax.set_title(f"{problem}: final NRMSE")
        ax.grid(True, axis="y", alpha=0.3)

        ax = axes[1][i]
        ax.bar(range(len(methods)), ttime_final, color=colors)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(f"{problem}: total update time [s]")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Final-step summary across methods", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="benchmarks/data")
    ap.add_argument("--plots-dir", default="benchmarks/plots")
    ap.add_argument("--problems", nargs="+",
                    default=["smooth_sines_2d", "rosenbrock_2d", "step_3d"])
    args = ap.parse_args()

    results = load_all(args.data_dir)
    if not results:
        print(f"No results found in {args.data_dir}. Run the benchmark first.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {sum(len(v) for v in results.values())} runs "
          f"across {len(results)} (method, problem) combinations.")

    plot_main_comparison(
        results, args.problems,
        os.path.join(args.plots_dir, "comparison.png"),
    )
    plot_summary_bars(
        results, args.problems,
        os.path.join(args.plots_dir, "summary.png"),
    )
    print(f"Wrote plots to {args.plots_dir}/")


if __name__ == "__main__":
    main()
