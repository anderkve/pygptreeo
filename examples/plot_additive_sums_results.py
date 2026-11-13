"""Figures for the benchmarks of the additive model from running sums.

    streams.png   the stream benchmark (``benchmark_global_mean_streams.py``): error
                  relative to the plain tree for every target and stream, from the RESULT
                  lines in ``results/additive_sums/*.jsonl``
    long.png      the 40 000-point streams (``compare_tree_vs_hybrid.py --plot --overlay
                  tree,hybrid,additive,combined --overlay-name additive``): error over the
                  last 10 000 points and time for the whole stream, from
                  ``results/compare_tree_vs_neural/*_additive_batches.csv``

With several seeds a marker is the geometric mean over the seeds and a line their range.

    python examples/plot_additive_sums_results.py [--out DIR]
"""

import argparse
import collections
import csv
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
AQUA, VIOLET = "#1baf7a", "#6250d6"
# config, colour, marker, filled, label
SERIES = [("tree", AQUA, "^", True, "plain tree"), ("global_pkg", AQUA, "s", False, "'additive_gp'"),
          ("hybrid", AQUA, "^", False, "'net', network alone"), ("additive", VIOLET, "D", False, "'additive'"),
          ("combined", VIOLET, "D", True, "'net' (network beside the additive model)")]
MAIN = ["rotated_rosenbrock", "gaussian_peaks"]
STREAMS = ["uniform", "focusing", "sweeping", "walker"]
METRICS = ["prequential_nrmse", "uniform_nrmse", "focus_nrmse"]
NOISES = [(0.0, "no noise"), (0.2, "noise of 0.2 sd of the target")]

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.8,
                     "legend.frameon": False})


def geo(v):
    return float(np.exp(np.mean(np.log(v))))


def load_streams():
    """{(target, stream, config, noise): {seed: result}}"""
    R = collections.defaultdict(dict)
    for fn in sorted(glob.glob(os.path.join(HERE, "results", "additive_sums", "*.jsonl"))):
        for line in open(fn):
            if line.startswith("RESULT "):
                r = json.loads(line[7:])
                R[(r["target"], r["stream"], r["config"], float(r.get("noise", 0.0)))][r["seed"]] = r
    return R


def load_long():
    """{run label: {config: (error over the last five batches, seconds for the stream)}}"""
    out = {}
    for fn in sorted(glob.glob(os.path.join(HERE, "results", "compare_tree_vs_neural", "*_additive_batches.csv"))):
        m = re.match(r"(.+)_d(\d+)_(\w+?)(_noise[\d.]+)?_N(\d+)_seed\d+_additive_batches\.csv", os.path.basename(fn))
        label = f"{m.group(1)}, {m.group(2)}D" + ("" if m.group(3) == "uniform" else f", {m.group(3)}") + (", noise" if m.group(4) else "")
        rows = collections.defaultdict(list)
        for r in csv.DictReader(open(fn)):
            rows[r["config"]].append(r)
        batch = int(rows["tree"][0]["points"])
        out[label] = {c: (float(np.mean([float(r["nrmse"]) for r in rs[-5:]])),
                          batch * sum(float(r["predict_time"]) + float(r["update_time"]) for r in rs)) for c, rs in rows.items()}
    return out


def rows(ax, values, labels, series):
    """values[i][config] = (centre, low, high) or missing."""
    dodge = np.linspace(-0.22, 0.22, len(series))
    for i, v in enumerate(values):
        pts = [(v[c], col, mk, filled, i + dy) for (c, col, mk, filled, _), dy in zip(series, dodge) if c in v]
        if pts:
            ax.plot([min(p[0][0] for p in pts), max(p[0][0] for p in pts)], [i, i], color=GRID, lw=1.5, zorder=1)
        for (g, lo, hi), col, mk, filled, y in pts:
            if hi > lo:
                ax.plot([lo, hi], [y, y], color=col, lw=1.2, alpha=0.6, zorder=2)
            ax.plot(g, y, marker=mk, color=col, mfc=col if filled else SURFACE, mew=1.6, ms=7.5, ls="none", zorder=3)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.grid(True, which="major", axis="x"); ax.set_axisbelow(True); ax.tick_params(length=0)


def handles(series):
    return [Line2D([], [], marker=mk, color=col, mfc=col if filled else SURFACE, mew=1.6, ms=7.5, ls="none", label=lab) for _, col, mk, filled, lab in series]


def streams(R, out):
    """Error relative to the plain tree: per cell, the geometric mean over the three test sets."""
    series = SERIES[1:]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), sharey=True)
    for ax, (noise, title) in zip(axes, NOISES):
        vals, labels = [], []
        for t in MAIN:
            for s in STREAMS:
                tree = R.get((t, s, "tree", noise), {}); v = {}
                for c, *_ in series:
                    per_seed = [geo([r[m] / tree[seed][m] for m in METRICS]) for seed, r in R.get((t, s, c, noise), {}).items() if seed in tree]
                    if per_seed:
                        v[c] = (geo(per_seed), min(per_seed), max(per_seed))
                vals.append(v); labels.append(f"{t}, {s}")
        rows(ax, vals, labels, series); ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0))); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.axvline(1.0, color=INK2, lw=1.0, zorder=1); ax.set_title(title, loc="left")
        ax.set_xlabel("error / error of the plain tree (geometric mean of the three test sets)")
    fig.legend(handles=handles(series), loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("Streams of 4,000 points in 6D: global models relative to the plain tree (marker: geometric mean over seeds, line: range)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95)); fig.savefig(os.path.join(out, "streams.png"), dpi=170); plt.close(fig)


def long(L, out):
    series = [s for s in SERIES if s[0] != "global_pkg"]
    labels = list(L)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 0.46 * len(labels) + 2.2), sharey=True)
    rows(axes[0], [{c: (v[0],) * 3 for c, v in L[k].items()} for k in labels], labels, series)
    axes[0].set_xscale("log"); axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_title("error over the last 10,000 points", loc="left"); axes[0].set_xlabel("RMSE / range")
    rows(axes[1], [{c: (v[1],) * 3 for c, v in L[k].items()} for k in labels], labels, series); axes[1].set_xlim(left=0)
    axes[1].set_title("time for the whole stream", loc="left"); axes[1].set_xlabel("seconds (prediction and update of 40,000 points)")
    fig.legend(handles=handles(series), loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("Streams of 40,000 points (one seed)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95)); fig.savefig(os.path.join(out, "long.png"), dpi=170); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "additive_sums"), help="directory for the figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R = load_streams()
    if R:
        streams(R, args.out)
    L = load_long()
    if L:
        long(L, args.out)
    print("figures in", args.out)
