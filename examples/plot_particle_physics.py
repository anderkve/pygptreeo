"""Figures for the particle-physics-style targets and the benchmark on them.

    gallery.png   a two-input slice of every target (the other inputs at mid-range)
    results.png   test error of every configuration on every target
    curves.png    prequential error along the longest streams

The last two are drawn from the RESULT lines in results/particle_physics/*.jsonl.

    python examples/plot_particle_physics.py [--out DIR]
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from particle_physics_targets import TARGETS

HERE = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
AQUA, VIOLET = "#1baf7a", "#6250d6"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#f4f7fc", "#a9c6ee", "#2a78d6", "#0c2f5e"])
# config, colour, marker, filled, label
SERIES = [("tree", AQUA, "^", True, "plain tree"),
          ("net_no_holdout", VIOLET, "v", False, "'net', every point trained on, full leaf refit"),
          ("net", VIOLET, "D", True, "'net'")]
# target: (input on the horizontal axis, its label, input on the vertical axis, its label)
SLICES = {
    "resonance_xsec": (0, "mediator mass", 1, "dark-matter mass"),
    "dilepton_spectrum": (0, "dilepton mass", 1, "Z' mass"),
    "jet_spectrum": (0, "p_T", 1, "rapidity / its limit"),
    "oscillation_probability": (0, "energy", 3, "delta_CP"),
    "branching_ratio": (0, "scalar mass", 4, "dark fermion mass"),
    "loop_decay_width": (0, "scalar mass", 1, "top coupling"),
    "relic_density": (0, "dark-matter mass", 1, "mediator mass"),
    "neutralino_mass": (0, "M_1", 2, "mu"),
    "eft_signal_strength": (0, "coefficient 1", 1, "coefficient 2"),
    "signal_yield_mc": (0, "parent mass", 1, "daughter / parent mass"),
    "global_fit_loglike": (0, "dark-matter mass", 1, "coupling g"),
}

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.8,
                     "legend.frameon": False})


def gallery(out, n=220):
    fig, axes = plt.subplots(3, 4, figsize=(15.5, 11.2))
    u = (np.arange(n) + 0.5) / n; U, V = np.meshgrid(u, u)
    rng = np.random.RandomState(0)
    for ax, (name, (i, xlabel, j, ylabel)) in zip(axes.ravel(), SLICES.items()):
        target = TARGETS[name]
        X = np.full((n * n, target.d), 0.5); X[:, i] = U.ravel(); X[:, j] = V.ravel()
        if name == "global_fit_loglike":
            X[:, 2] = 0.2                                            # second coupling small, so that the allowed region shows
        Z = target.sample(X, rng)[0].reshape(n, n)                   # with its Monte Carlo noise where it has any
        im = ax.imshow(Z, origin="lower", extent=(0, 1, 0, 1), cmap=SEQUENTIAL, aspect="auto", interpolation="nearest")
        bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); bar.outline.set_visible(False); bar.ax.tick_params(length=0, labelsize=8)
        ax.set_title(f"{name}  ({target.d}D, {target.unit})", loc="left"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.tick_params(length=0)
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(0.0, 0.95, "Each panel: the target on a slice through\nthe unit cube, the other inputs at 0.5.\n"
                          "Axes are in unit-cube coordinates\n(masses and energies are logarithmic).\nDarker is larger.",
                          va="top", fontsize=10, color=INK2, transform=axes.ravel()[-1].transAxes)
    fig.suptitle("Particle-physics-style test targets", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(os.path.join(out, "gallery.png"), dpi=150); plt.close(fig)


def load():
    R = collections.defaultdict(lambda: collections.defaultdict(list))
    for fn in sorted(glob.glob(os.path.join(HERE, "results", "particle_physics", "*.jsonl"))):
        for line in open(fn):
            if line.startswith("RESULT "):
                r = json.loads(line[7:]); R[(r["stream"], r["N"], r["target"])][r["config"]].append(r)
    return R


def geo(v):
    return float(np.exp(np.mean(np.log(v))))


def results(R, out, stream="uniform", N=10000):
    names = [t for t in TARGETS if (stream, N, t) in R]
    if not names:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.5 * len(names) + 2.4), sharey=True)
    dodge = np.linspace(-0.2, 0.2, len(SERIES))
    for ax, relative in zip(axes, (False, True)):
        for i, t in enumerate(names):
            runs = R[(stream, N, t)]; pts = []
            tree = {r["seed"]: r["test_nrmse"] for r in runs.get("tree", [])}
            for (c, col, mk, filled, _), dy in zip(SERIES, dodge):
                v = [r["test_nrmse"] / (tree[r["seed"]] if relative else 1.0) for r in runs.get(c, []) if not relative or r["seed"] in tree]
                if v and not (relative and c == "tree"):
                    pts.append((geo(v), min(v), max(v), col, mk, filled, i + dy))
            if pts:
                ax.plot([min(p[0] for p in pts), max(p[0] for p in pts)], [i, i], color=GRID, lw=1.5, zorder=1)
            for g, lo, hi, col, mk, filled, y in pts:
                if hi > lo:
                    ax.plot([lo, hi], [y, y], color=col, lw=1.2, alpha=0.6, zorder=2)
                ax.plot(g, y, marker=mk, color=col, mfc=col if filled else SURFACE, mew=1.6, ms=7.5, ls="none", zorder=3)
        ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter()); ax.grid(True, which="major", axis="x"); ax.set_axisbelow(True); ax.tick_params(length=0)
        if relative:
            ax.axvline(1.0, color=INK2, lw=1.0, zorder=1)
            ax.set_title("relative to the plain tree", loc="left"); ax.set_xlabel("test error / test error of the plain tree")
        else:
            ax.set_title("test error", loc="left"); ax.set_xlabel("RMSE / range of the target on a uniform test set")
    axes[0].set_yticks(range(len(names))); axes[0].set_yticklabels([f"{t} ({TARGETS[t].d}D)" for t in names]); axes[0].set_ylim(len(names) - 0.5, -0.5)
    handles = [Line2D([], [], marker=mk, color=col, mfc=col if filled else SURFACE, mew=1.6, ms=7.5, ls="none", label=lab) for _, col, mk, filled, lab in SERIES]
    fig.legend(handles=handles, loc="lower center", ncol=len(SERIES), fontsize=9)
    seeds = sorted({r["seed"] for t in names for rs in R[(stream, N, t)].values() for r in rs})
    fig.suptitle(f"GPTree with and without the feature network on particle-physics-style targets: {N:,} points, {len(seeds)} seed{'s' if len(seeds) > 1 else ''} "
                 "(marker: geometric mean, line: range)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95)); fig.savefig(os.path.join(out, "results.png"), dpi=170); plt.close(fig)


def curves(R, out, stream="uniform"):
    """Prequential error per batch of points, for the longest streams that have it."""
    Ns = sorted({k[1] for k, v in R.items() if k[0] == stream and any("curve" in r for rs in v.values() for r in rs)})
    if not Ns:
        return
    N = Ns[-1]; names = [t for t in TARGETS if (stream, N, t) in R]
    fig, axes = plt.subplots(3, 4, figsize=(15.5, 10.4), sharex=True)
    for ax, t in zip(axes.ravel(), names):
        for c, col, mk, filled, _ in SERIES:
            for r in [r for r in R[(stream, N, t)].get(c, []) if "curve" in r][:1]:
                y = np.array(r["curve"]); x = (np.arange(len(y)) + 1) * (N // len(y))
                ax.plot(x, y, color=col, lw=2.0 if filled else 1.4, ls="-" if filled else (0, (4, 2)), zorder=3 if filled else 2)
                ax.plot(x[-1], y[-1], marker=mk, color=col, mfc=col if filled else SURFACE, mew=1.6, ms=7, ls="none", zorder=4, clip_on=False)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, which="major"); ax.set_axisbelow(True); ax.tick_params(length=0)
        ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{t}  ({TARGETS[t].d}D)", loc="left")
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("points seen")
    for ax in axes[:, 0]:
        ax.set_ylabel("RMSE / range, per 1,000 points")
    handles = [Line2D([], [], marker=mk, color=col, mfc=col if filled else SURFACE, mew=1.6, ms=7, lw=2.0 if filled else 1.4,
                      ls="-" if filled else (0, (4, 2)), label=lab) for _, col, mk, filled, lab in SERIES]
    axes.ravel()[-1].legend(handles=handles, loc="center left", fontsize=10, handlelength=3.2)
    fig.suptitle(f"Error on each point before it is used, along a uniform stream of {N:,} points (one seed)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(os.path.join(out, "curves.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "particle_physics"), help="directory for the figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    gallery(args.out)
    R = load()
    results(R, args.out)
    curves(R, args.out)
    print("figures in", args.out)
