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
        # De-emphasise the weak baseline visually — it anchors the bottom
        # of the story but shouldn't dominate the eye or the y-limits.
        is_weak = method == "river_knn"
        lw = 1.0 if is_weak else 2.0
        alpha = 0.6 if is_weak else 1.0
        ax.plot(x, med, color=METHOD_COLOR[method],
                linestyle=METHOD_LS[method], linewidth=lw, alpha=alpha,
                label=METHOD_LABEL[method])
        if y.shape[0] > 1 and not is_weak:
            ax.fill_between(x, q1, q3, color=METHOD_COLOR[method],
                            alpha=0.18, linewidth=0)


def plot_main_comparison(results, problems, out_path, schedule="iid"):
    # Iteration 02 panel set: the `frac_pathological_std` panel is now
    # provably zero after the upstream pygptreeo fix, so it's dropped in
    # favour of a `coverage_95` panel (the over-coverage behaviour of RF
    # and SVGP is where the calibration story lives). `nlpd_trimmed` is
    # plotted on a symlog axis so river_knn's ~1e4 does not flatten the
    # other curves.
    metrics = [
        ("nrmse",           "NRMSE on held-out test set", "log"),
        ("nlpd_trimmed",    "Trimmed NLPD (5–95%)",       "symlog"),
        ("crps",            "CRPS",                        "log"),
        ("coverage_1sigma", "Empirical 1σ coverage",       "linear"),
        ("coverage_95",     "Empirical 95% coverage",      "linear"),
        ("cum_update_time", "Cumulative update time [s]",  "log"),
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
            elif yscale == "symlog":
                ax.set_yscale("symlog", linthresh=1.0)
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
            if key == "coverage_95":
                ax.axhline(0.95, color="black", linestyle=":",
                           linewidth=1, alpha=0.7)
                ax.set_ylim(-0.05, 1.05)
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


def plot_shift_vs_iid(results, problems, out_path,
                      schedule_a="iid", schedule_b="shift"):
    """Grouped bar chart: final NRMSE under iid vs shift, per method."""
    fig, axes = plt.subplots(
        1, len(problems), figsize=(3.8 * len(problems), 4.0),
        squeeze=False,
    )
    width = 0.35
    for p_i, problem in enumerate(problems):
        ax = axes[0][p_i]
        methods_present = [m for m in METHOD_ORDER
                           if results.get((m, problem, schedule_a)) or
                              results.get((m, problem, schedule_b))]
        x = np.arange(len(methods_present))
        a_vals = []
        b_vals = []
        for m in methods_present:
            def _final(schedule):
                runs = results.get((m, problem, schedule))
                if not runs:
                    return np.nan
                vals = [r["nrmse"][-1] for r in runs
                        if len(r.get("nrmse", [])) > 0]
                return float(np.nanmedian(vals)) if vals else np.nan
            a_vals.append(_final(schedule_a))
            b_vals.append(_final(schedule_b))
        ax.bar(x - width / 2, a_vals, width, label=schedule_a,
               color=[METHOD_COLOR[m] for m in methods_present],
               edgecolor="black", linewidth=0.4)
        ax.bar(x + width / 2, b_vals, width, label=schedule_b,
               color=[METHOD_COLOR[m] for m in methods_present],
               edgecolor="black", linewidth=0.4, hatch="///", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABEL[m] for m in methods_present],
                           rotation=35, ha="right", fontsize=8)
        ax.set_yscale("log")
        ax.set_title(problem)
        ax.grid(True, axis="y", alpha=0.3)
        if p_i == 0:
            ax.set_ylabel("final NRMSE")
            ax.legend(fontsize=9, frameon=False, loc="best")
    fig.suptitle(f"Distribution-shift stress test: {schedule_a} (solid) "
                 f"vs {schedule_b} (hatched)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_wilcoxon_table(results, problems, out_path,
                        baseline="pygptreeo", metric="nrmse", schedule="iid"):
    """Wilcoxon signed-rank p-values + median NRMSE ratios vs baseline.

    We pair by (problem, seed) across all default problems to get enough
    samples for meaningful p-values — 3 seeds per cell is too few per problem.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("scipy not installed; skipping Wilcoxon table.")
        return

    method_rows = []
    for m in METHOD_ORDER:
        if m == baseline or m == "river_knn":
            continue
        pairs_base, pairs_m = [], []
        for problem in problems:
            base_runs = results.get((baseline, problem, schedule), [])
            m_runs = results.get((m, problem, schedule), [])
            # Pair by seed.
            base_by_seed = {int(r["seed"]): r for r in base_runs}
            m_by_seed = {int(r["seed"]): r for r in m_runs}
            common = sorted(set(base_by_seed) & set(m_by_seed))
            for s in common:
                a = base_by_seed[s].get(metric)
                b = m_by_seed[s].get(metric)
                if a is None or b is None or len(a) == 0 or len(b) == 0:
                    continue
                av = float(a[-1])
                bv = float(b[-1])
                if not (np.isfinite(av) and np.isfinite(bv)):
                    continue
                pairs_base.append(av)
                pairs_m.append(bv)
        pairs_base = np.asarray(pairs_base)
        pairs_m = np.asarray(pairs_m)
        n = pairs_base.size
        if n >= 3 and not np.all(pairs_base == pairs_m):
            try:
                stat, pval = wilcoxon(pairs_base, pairs_m,
                                      alternative="less")
            except ValueError:
                stat, pval = float("nan"), float("nan")
        else:
            stat, pval = float("nan"), float("nan")
        ratio = (float(np.median(pairs_m / pairs_base))
                 if n > 0 else float("nan"))
        method_rows.append((m, n, ratio, pval))

    # Draw as a matplotlib table.
    fig, ax = plt.subplots(figsize=(7.5, 0.45 + 0.4 * len(method_rows)))
    ax.axis("off")
    header = [f"method (vs {baseline})", "#pairs",
              f"median {metric} ratio (alt/base)",
              f"Wilcoxon p (alt > base?)"]
    cells = [[METHOD_LABEL[m],
              f"{n}",
              "—" if not np.isfinite(r) else f"{r:.3g}",
              "—" if not np.isfinite(p) else f"{p:.3g}"]
             for (m, n, r, p) in method_rows]
    tbl = ax.table(cellText=[header] + cells, cellLoc="center",
                   loc="center",
                   colWidths=[0.32, 0.12, 0.28, 0.28])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.45)
    for j in range(len(header)):
        tbl[(0, j)].set_facecolor("#dddddd")
        tbl[(0, j)].set_text_props(weight="bold")
    fig.suptitle(
        f"Paired Wilcoxon signed-rank on final {metric.upper()} "
        f"(pooled over {len(problems)} problems)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="benchmarks/data")
    ap.add_argument("--plots-dir", default="benchmarks/plots")
    ap.add_argument("--iter-dir", default=None,
                    help="If set, also copy every plot into this directory.")
    ap.add_argument("--problems", nargs="+",
                    default=["smooth_sines_2d", "rosenbrock_2d",
                             "friedman1_5d", "borehole_8d"])
    ap.add_argument("--schedule", default="iid")
    ap.add_argument("--no-shift-plot", action="store_true",
                    help="Skip shift-vs-iid plot (no shift data).")
    args = ap.parse_args()

    results = load_all(args.data_dir)
    if not results:
        print(f"No results in {args.data_dir}.", file=sys.stderr)
        sys.exit(1)
    n_runs = sum(len(v) for v in results.values())
    print(f"Loaded {n_runs} runs across {len(results)} "
          f"(method, problem, schedule) combinations.")

    out_dirs = [args.plots_dir]
    if args.iter_dir:
        out_dirs.append(args.iter_dir)

    def _write_all(basename, drawer, *drawer_args, **drawer_kwargs):
        for d in out_dirs:
            drawer(*drawer_args, out_path=os.path.join(d, basename),
                   **drawer_kwargs)

    _write_all("comparison.png", plot_main_comparison, results, args.problems,
               schedule=args.schedule)
    _write_all("summary.png", plot_summary_bars, results, args.problems,
               schedule=args.schedule)
    _write_all("pareto.png", plot_pareto, results, args.problems,
               schedule=args.schedule)
    _write_all("calibration.png", plot_calibration, results, args.problems,
               schedule=args.schedule)
    _write_all("wilcoxon_nrmse.png", plot_wilcoxon_table,
               results, args.problems, metric="nrmse", schedule=args.schedule)
    if not args.no_shift_plot:
        # Only attempt if shift data exists for any (method, problem).
        have_shift = any(
            schedule == "shift"
            for (_m, _p, schedule) in results.keys()
        )
        if have_shift:
            _write_all("shift_vs_iid.png", plot_shift_vs_iid,
                       results, args.problems)
    for d in out_dirs:
        print(f"Wrote plots to {d}/")


if __name__ == "__main__":
    main()
