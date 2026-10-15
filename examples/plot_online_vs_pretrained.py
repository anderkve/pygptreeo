"""Figure: learning during the run against pretraining, from the gated integration runs
(gated_online_{ch}.json, gated_frozen_gated_{ch}.json, gated_frozen_{ch}.json of
gated_integration_demo.py). Top: exact amplitude evaluations spent as the Monte Carlo
run proceeds. Bottom: the integral's deviation from the exact one in units of its
Monte Carlo error.

Usage: python plot_online_vs_pretrained.py [results_dir] [--channels uu,gg]
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
    ("online", "learning during the run, gated", "#2a78d6", "-"),
    ("frozen_gated", "pretrained on the same budget, gated", "#1baf7a", (0, (5, 1.5, 1, 1.5))),
    ("frozen", "pretrained on the same budget, used everywhere", "#eb6834", (0, (4, 2))),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def load(out_dir, tag, ch):
    with open(os.path.join(out_dir, f"gated_{tag}_{ch}.json")) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default=os.path.join(HERE, "results", "tth_amplitude"))
    ap.add_argument("--channels", default="uu,gg")
    args = ap.parse_args()
    chans = args.channels.split(",")
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "xtick.direction": "out",
                         "ytick.direction": "out", "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.titlesize": 8.5,
                         "legend.fontsize": 7.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, len(chans), figsize=(3.4 * len(chans), 4.4), sharex="col", squeeze=False)
    for j, ch in enumerate(chans):
        runs = {tag: load(args.results_dir, tag, ch) for tag, _, _, _ in SERIES}
        budget = runs["online"]["checkpoints"][-1]["n_exact"]
        top, bot = axes[0, j], axes[1, j]
        for tag, label, color, ls in SERIES:
            s = runs[tag]["series"]
            N = np.array([r["N"] for r in s]) / 1e3
            n_exact = np.array([r["n_exact"] for r in s])
            start = 0 if tag == "online" else budget
            top.plot(np.concatenate([[0], N]), np.concatenate([[start], n_exact]), ls=ls, color=color, lw=1.6, label=label,
                     solid_capstyle="round", dash_capstyle="round", zorder=3 if tag == "online" else 2)
            top.annotate(f"{n_exact[-1]:,}".replace(",", " "), (N[-1], n_exact[-1]), xytext=(4, 0), textcoords="offset points",
                         va="center", ha="left", fontsize=7, color=INK)
            m = N >= 2
            bot.plot(N[m], np.array([r["shift_over_mc"] for r in s])[m], ls=ls, color=color, lw=1.6,
                     solid_capstyle="round", dash_capstyle="round", zorder=3 if tag == "online" else 2)
        top.set_title(f"{NAMES[ch]}, {budget:,} exact evaluations for the learning tree".replace(",", " "), loc="left", color=INK)
        top.set_ylim(0, None); top.set_xlim(0, 100)
        top.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:g}k" if v >= 1e3 else f"{v:g}"))
        bot.axhspan(-1, 1, color="#898781", alpha=0.10, lw=0, zorder=0)
        bot.axhline(0, color="#c3c2b7", lw=0.6, zorder=1)
        bot.set_ylim(-1.15, 1.15); bot.set_xlim(0, 100)
        bot.set_xlabel("Monte Carlo points (thousands)")
        for ax in (top, bot):
            ax.grid(True, axis="y", color=GRID, lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
        if j == 0:
            top.set_ylabel("exact amplitude evaluations")
            bot.set_ylabel("(integral $-$ exact) / MC error")
            bot.text(1.5, 0.93, "$\\pm$1 Monte Carlo error", fontsize=7, color=MUTED, va="top")
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(0, 1.0), frameon=False, handlelength=2.6, borderaxespad=0.2)
    top_right_xlim = axes[0, -1].get_xlim(); axes[0, -1].set_xlim(top_right_xlim[0], top_right_xlim[1] * 1.18)   # room for end labels
    axes[0, 0].set_xlim(0, 118)
    fig.tight_layout(h_pad=1.0, w_pad=1.2)
    for ext in ("pdf", "png"):
        path = os.path.join(args.results_dir, f"online_vs_pretrained.{ext}")
        fig.savefig(path, dpi=300 if ext == "png" else None)
        print("wrote", path)


if __name__ == "__main__":
    main()
