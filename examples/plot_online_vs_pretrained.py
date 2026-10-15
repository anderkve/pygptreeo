"""Figure: learning during the run against pretraining, from the gated integration runs
(gated_online_{ch}.json, gated_frozen_gated_{ch}.json, gated_frozen_{ch}.json of
gated_integration_demo.py). Top: exact amplitude evaluations spent as the Monte Carlo
run proceeds. Middle: the integral's deviation from the exact one in units of its
Monte Carlo error. Bottom (``--absolute``): the same deviation and the Monte Carlo
error themselves, both relative to the integral.

Usage: python plot_online_vs_pretrained.py [results_dir] [--channels uu,gg] [--suffix 200k] [--absolute]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"uu": r"$f_1$: $q\bar q \to t\bar t H$", "gg": r"$f_3$: $gg \to t\bar t H$"}
SERIES = [  # (file tag, label, colour, line style)
    ("online", "learns during the run, gated", "#2a78d6", "-"),
    ("frozen_gated", "pretrained on the same budget, gated", "#1baf7a", (0, (5, 1.5, 1, 1.5))),
    ("frozen", "pretrained on the same budget, ungated", "#eb6834", (0, (4, 2))),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
THIN = chr(0x2009)


def count(n):
    return f"{n:,}".replace(",", THIN)


def load(out_dir, tag, ch):
    with open(os.path.join(out_dir, f"gated_{tag}_{ch}.json")) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default=os.path.join(HERE, "results", "tth_amplitude"))
    ap.add_argument("--channels", default="uu,gg")
    ap.add_argument("--suffix", default="", help="file tag suffix of the runs, e.g. 200k")
    ap.add_argument("--absolute", action="store_true", help="add the row of absolute errors")
    args = ap.parse_args()
    chans = args.channels.split(",")
    nrows = 3 if args.absolute else 2
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "xtick.direction": "out",
                         "ytick.direction": "out", "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.titlesize": 8.5,
                         "legend.fontsize": 7.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(nrows, len(chans), figsize=(3.6 * len(chans), 2.2 * nrows), sharex="col", squeeze=False)
    for j, ch in enumerate(chans):
        runs = {tag: load(args.results_dir, tag + args.suffix, ch) for tag, _, _, _ in SERIES}
        budget = runs["online"]["checkpoints"][-1]["n_exact"]
        n_max = runs["online"]["series"][-1]["N"] / 1e3
        top, bot = axes[0, j], axes[1, j]
        for tag, label, color, ls in SERIES:
            s = runs[tag]["series"]
            N = np.array([r["N"] for r in s]) / 1e3
            n_exact = np.array([r["n_exact"] for r in s])
            start = 0 if tag == "online" else budget
            top.plot(np.concatenate([[0], N]), np.concatenate([[start], n_exact]), ls=ls, color=color, lw=1.6, label=label,
                     solid_capstyle="round", dash_capstyle="round", zorder=3 if tag == "online" else 2)
            if tag != "frozen":      # the ungated pretrained tree ends where the learning one does, by construction
                extra = f"  ({n_exact[-1] / budget:.1f} ×)" if tag == "frozen_gated" else ""
                top.annotate(count(n_exact[-1]) + extra, (N[-1], n_exact[-1]), xytext=(4, 0), textcoords="offset points",
                             va="center", ha="left", fontsize=7, color=INK)
            m = N >= 2
            bot.plot(N[m], np.array([r["shift_over_mc"] for r in s])[m], ls=ls, color=color, lw=1.6,
                     solid_capstyle="round", dash_capstyle="round", zorder=3 if tag == "online" else 2)
            if args.absolute:
                ab = axes[2, j]
                if tag == "online":      # the Monte Carlo error is the same for every strategy: the same points
                    mc = 100 * np.array([r["mc_rel_err"] for r in s])[m]
                    ab.fill_between(N[m], 1e-9, mc, color="#898781", alpha=0.10, lw=0, zorder=0)
                    ab.plot(N[m], mc, color="#898781", lw=1.0, zorder=1)
                ab.plot(N[m], 100 * np.abs(np.array([r["shift"] for r in s]))[m], ls=ls, color=color, lw=1.6,
                        solid_capstyle="round", dash_capstyle="round", zorder=3 if tag == "online" else 2)
        top.set_title(NAMES[ch], loc="left", color=INK)
        top.text(0.97 * n_max, budget * 1.04, f"pretrained on {count(budget)}", fontsize=7, color=MUTED, va="bottom", ha="right")
        top.set_ylim(0, None); top.set_xlim(0, 1.24 * n_max)
        top.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:g}k" if v >= 1e3 else f"{v:g}"))
        bot.axhspan(-1, 1, color="#898781", alpha=0.10, lw=0, zorder=0)
        bot.axhline(0, color="#c3c2b7", lw=0.6, zorder=1)
        bot.set_ylim(-1.15, 1.15)
        axes[-1, j].set_xlabel("Monte Carlo points (thousands)")
        axes[-1, j].set_xticks(np.linspace(0, n_max, 5))
        if args.absolute:
            ab = axes[2, j]
            ab.set_yscale("log"); ab.set_ylim(1e-3, 5)
            ab.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g} %"))
            k = len(runs["online"]["series"]) // 4
            ab.text(0.27 * n_max, 100 * runs["online"]["series"][k]["mc_rel_err"] * 1.25, "Monte Carlo error", fontsize=7,
                    color=MUTED, va="bottom", ha="left")
        for ax in axes[:, j]:
            ax.grid(True, axis="y", color=GRID, lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
        if j == 0:
            top.set_ylabel("exact amplitude evaluations")
            bot.set_ylabel("(integral $-$ exact) / MC error")
            bot.text(0.02 * n_max, 0.95, "$\\pm$1 Monte Carlo error", fontsize=7, color=MUTED, va="top")
            if args.absolute:
                axes[2, j].set_ylabel("|integral $-$ exact| / exact")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, handlelength=2.6, columnspacing=1.6,
               handletextpad=0.6, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(h_pad=1.0, w_pad=1.2, rect=(0, 0, 1, 1 - 0.3 / fig.get_figheight()))
    for ext in ("pdf", "png"):
        path = os.path.join(args.results_dir, f"online_vs_pretrained{args.suffix}.{ext}")
        fig.savefig(path, dpi=300 if ext == "png" else None)
        print("wrote", path)


if __name__ == "__main__":
    main()
