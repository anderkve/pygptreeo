"""Figures for the stream benchmark of the interaction-order state.

Reads the RESULT lines in ``results/state_mean/*.jsonl`` (written by
``benchmark_global_mean_streams.py``) and writes four figures:

    summary.png    error relative to the plain tree: every cell and the geometric mean
    streams.png    NRMSE per target, stream and test set
    coverage.png   one-sigma coverage away from the stream
    additive.png   targets with additive or pairwise structure

With several seeds a marker is the geometric mean over the seeds (the mean for
coverage) and a whisker their range.

    python examples/plot_state_mean_results.py [--out DIR]
"""

import argparse
import collections
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
AQUA, VIOLET = "#1baf7a", "#6250d6"          # existing modes, the state
# config, colour, marker, filled, label
SERIES = [
    ("tree", AQUA, "^", True, "plain tree"),
    ("global_pkg", AQUA, "s", False, "'additive_gp'"),
    ("hybrid", AQUA, "^", False, "'net' (feature network)"),
    ("hybrid_holdout", AQUA, "v", False, "'net' with held-out stopping"),
    ("state", VIOLET, "D", False, "'state'"),
    ("state_net", VIOLET, "D", True, "'state' with network features"),
]
MAIN = ["rotated_rosenbrock", "gaussian_peaks"]
EXTRA = ["rosenbrock", "rastrigin", "levy"]
STREAMS = ["uniform", "focusing", "sweeping", "walker"]
METRICS = [("prequential_nrmse", "prequential (predict, then update)"), ("uniform_nrmse", "uniform test set"),
           ("focus_nrmse", "where the stream ended")]
NOISES = [(0.0, "no noise"), (0.2, "noise of 0.2 sd of the target")]

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.8,
                     "legend.frameon": False})


def load():
    """{(target, stream, config, noise): {seed: result}}"""
    R = collections.defaultdict(dict)
    for fn in sorted(glob.glob(os.path.join(HERE, "results", "state_mean", "*.jsonl"))):
        for line in open(fn):
            if line.startswith("RESULT "):
                r = json.loads(line[7:])
                R[(r["target"], r["stream"], r["config"], float(r.get("noise", 0.0)))][r["seed"]] = r
    return R


def geo(v):
    return float(np.exp(np.mean(np.log(v))))


def style(ax, log=True):
    ax.grid(True, which="major", axis="x"); ax.set_axisbelow(True); ax.tick_params(length=0)
    if log:
        ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter())


def marker(ax, x, y, col, mk, filled, ms=7.5, **kw):
    ax.plot(x, y, marker=mk, color=col, mfc=col if filled else SURFACE, mew=1.6, ms=ms, ls="none", zorder=3, **kw)


def handles(series=SERIES):
    return [Line2D([], [], marker=mk, color=col, mfc=col if filled else SURFACE, mew=1.6, ms=7.5, ls="none", label=lab)
            for _, col, mk, filled, lab in series]


def seeds_label(R):
    seeds = sorted({s for k, v in R.items() if k[0] in MAIN for s in v})
    return "one seed" if len(seeds) == 1 else f"{len(seeds)} seeds"


def summary(R, out):
    """Error of each configuration over the plain tree's, per cell (target, stream, test set)."""
    configs = SERIES[1:]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True, sharex=True)
    rng = np.random.RandomState(0)
    for ax, (noise, title) in zip(axes, NOISES):
        for i, (c, col, mk, filled, _) in enumerate(configs):
            per_seed = collections.defaultdict(list); cells = []
            for t in MAIN:
                for s in STREAMS:
                    a, b = R.get((t, s, c, noise), {}), R.get((t, s, "tree", noise), {})
                    for m, _ in METRICS:
                        ratios = {sd: a[sd][m] / b[sd][m] for sd in a if sd in b}
                        if ratios:
                            cells.append(geo(list(ratios.values())))
                            for sd, v in ratios.items():
                                per_seed[sd].append(v)
            if not cells:
                continue
            ax.plot(cells, i + rng.uniform(-0.2, 0.2, len(cells)), marker="o", ms=3.6, ls="none", color=col, alpha=0.45, mew=0, zorder=2)
            g = [geo(v) for v in per_seed.values()]
            if len(g) > 1:
                ax.plot([min(g), max(g)], [i, i], color=INK, lw=1.4, zorder=3)
            marker(ax, geo(g), i, col, mk, filled, ms=10)
            ax.text(1.02, i, f"{geo(g):.2f}", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=10)
        ax.axvline(1.0, color=INK2, lw=1.0, zorder=1)
        ax.text(1.0, -0.72, "plain tree", ha="center", va="bottom", fontsize=9, color=INK2)
        ax.text(1.02, -0.72, "geometric\nmean", transform=ax.get_yaxis_transform(), va="bottom", ha="left", fontsize=8.5, color=INK2)
        style(ax); ax.set_title(title, loc="left", pad=22)
        ax.set_xlabel("error / error of the plain tree")
    axes[0].set_yticks(range(len(configs))); axes[0].set_yticklabels([lab for *_, lab in configs])
    axes[0].set_ylim(len(configs) - 0.5, -0.5)
    fig.suptitle(f"Global models against the plain tree: 2 targets x 4 streams x 3 test sets (small dots), 6D, 4,000 points, {seeds_label(R)}",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.98, 0.95), w_pad=5.0); fig.savefig(os.path.join(out, "summary.png"), dpi=170); plt.close(fig)


def rows(ax, R, keys, metric, noise, series=SERIES):
    """One line per key (target, stream), one marker per configuration. The markers
    of a line are spread a little vertically so that equal values stay visible."""
    dodge = np.linspace(-0.2, 0.2, len(series))
    for i, (t, s) in enumerate(keys):
        pts = []
        for (c, col, mk, filled, _), dy in zip(series, dodge):
            v = [r[metric] for r in R.get((t, s, c, noise), {}).values()]
            if v:
                pts.append((geo(v), min(v), max(v), col, mk, filled, i + dy))
        if not pts:
            continue
        ax.plot([min(p[0] for p in pts), max(p[0] for p in pts)], [i, i], color=GRID, lw=1.5, zorder=1)
        for g, lo, hi, col, mk, filled, y in pts:
            if hi > lo:
                ax.plot([lo, hi], [y, y], color=col, lw=1.2, alpha=0.6, zorder=2)
            marker(ax, g, y, col, mk, filled)


def streams(R, out):
    keys = [(t, s) for t in MAIN for s in STREAMS]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.2), sharey=True)
    for row, (noise, title) in enumerate(NOISES):
        for ax, (metric, mtitle) in zip(axes[row], METRICS):
            rows(ax, R, keys, metric, noise); style(ax)
            ax.set_title(f"{title}: {mtitle}", loc="left")
            if row == 1:
                ax.set_xlabel("RMSE / range of the target, against the noise-free target")
        axes[row, 0].set_yticks(range(len(keys))); axes[row, 0].set_yticklabels([f"{t}, {s}" for t, s in keys], fontsize=9)
        axes[row, 0].set_ylim(len(keys) - 0.5, -0.5)
    fig.legend(handles=handles(), loc="lower center", ncol=6, fontsize=9)
    fig.suptitle(f"GPTree with and without a global model on four kinds of stream, 6D, 4,000 points, {seeds_label(R)}",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96)); fig.savefig(os.path.join(out, "streams.png"), dpi=170); plt.close(fig)


def coverage(R, out):
    """One-sigma coverage on the uniform test set of the three streams that do not cover the cube."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), sharey=True, sharex=True)
    rng = np.random.RandomState(0)
    for ax, (noise, title) in zip(axes, NOISES):
        for i, (c, col, mk, filled, _) in enumerate(SERIES):
            off = [np.mean([r["coverage_uniform"] for r in R[(t, s, c, noise)].values()])
                   for t in MAIN for s in STREAMS[1:] if R.get((t, s, c, noise))]
            on = [np.mean([r["coverage_prequential"] for r in R[(t, s, c, noise)].values()])
                  for t in MAIN for s in STREAMS if R.get((t, s, c, noise))]
            if not off:
                continue
            ax.plot(off, i + rng.uniform(-0.2, 0.2, len(off)), marker="o", ms=3.6, ls="none", color=col, alpha=0.45, mew=0, zorder=2)
            ax.plot(np.mean(on), i, marker="|", ms=13, mew=1.8, color=INK2, ls="none", zorder=2)
            marker(ax, np.mean(off), i, col, mk, filled, ms=10)
            ax.text(1.02, i, f"{np.mean(off):.2f}", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=10)
        ax.axvline(0.68, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.text(0.68, -0.72, "0.68: calibrated", ha="center", va="bottom", fontsize=9, color=INK2)
        ax.text(1.02, -0.72, "mean", transform=ax.get_yaxis_transform(), va="bottom", ha="left", fontsize=8.5, color=INK2)
        style(ax, log=False); ax.set_xlim(0, 1); ax.set_title(title, loc="left", pad=22)
        ax.set_xlabel("fraction of errors within one predicted sigma")
    axes[0].set_yticks(range(len(SERIES))); axes[0].set_yticklabels([lab for *_, lab in SERIES])
    axes[0].set_ylim(len(SERIES) - 0.5, -0.5)
    legend = [Line2D([], [], marker="o", ms=3.6, ls="none", color=INK2, alpha=0.6, mew=0, label="away from the stream: one target and stream (uniform test set; focusing, sweeping, walker)"),
              Line2D([], [], marker="D", ms=8, ls="none", color=INK2, mfc=SURFACE, mew=1.6, label="their mean (marker of the configuration)"),
              Line2D([], [], marker="|", ms=13, mew=1.8, ls="none", color=INK2, label="on the stream (prequential), mean")]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(f"One-sigma coverage of the noise-free target, 6D, 4,000 points, {seeds_label(R)}",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 0.98, 0.95), w_pad=5.0); fig.savefig(os.path.join(out, "coverage.png"), dpi=170); plt.close(fig)


def additive(R, out):
    keys = [(t, "uniform") for t in EXTRA]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.6), sharey=True, sharex=True)
    for ax, (noise, title) in zip(axes, NOISES):
        rows(ax, R, keys, "uniform_nrmse", noise); style(ax)
        ax.set_title(title, loc="left"); ax.set_xlabel("RMSE / range of the target on a uniform test set")
    axes[0].set_yticks(range(len(keys))); axes[0].set_yticklabels([t for t, _ in keys]); axes[0].set_ylim(len(keys) - 0.5, -0.5)
    fig.legend(handles=handles(), loc="lower center", ncol=6, fontsize=9)
    fig.suptitle("Targets that are sums of terms in one or two inputs, 6D, uniform stream of 4,000 points, one seed",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 0.93)); fig.savefig(os.path.join(out, "additive.png"), dpi=170); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "state_mean"), help="directory for the figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R = load()
    for figure in (summary, streams, coverage, additive):
        figure(R, args.out)
    print(f"{sum(len(v) for v in R.values())} runs, figures in {args.out}")
