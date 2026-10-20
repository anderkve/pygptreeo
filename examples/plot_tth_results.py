"""Figures for the 5D ttH benchmark: eps against the number of training points, from the
result files of tth_amplitude_benchmark.py, with the networks of arXiv:2412.09534
(values read off their Figure 16, on LO-unweighted test points) as reference.

eps_vs_points.png: the package defaults (tree, tree choosing its points, hybrid, larger network).
eps_vs_points_improved.png: log-invariant inputs, RBF leaves and f-weighted training points.
eps_vs_points_paper_weights.png: the same with the paper's weights (flux and pseudo-PDF factor), on
the networks' metric, the mean relative error on LO-unweighted test points.
eps_vs_points_targets.png: paper's weights, the log and the linear target, streamed to 524 288 points,
choosing the points from a pool of 131 072 or 524 288 candidates, or letting the leaves choose; and the
1000-point leaves refit only at splits.

eps_vs_points_loop.png: the one-loop test functions f2 and f4, learned as the ratio to the Born.
Usage: python plot_tth_results.py [results_dir]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "tth_amplitude")
NAMES = {"uu": r"$f_1$: $q\bar q \to t\bar t H$", "gg": r"$f_3$: $gg \to t\bar t H$"}
# MLP(s) and L-GATr of arXiv:2412.09534, Figure 16: the mean relative error on LO-unweighted test points
# (their eq. 30), read off the plot by pixel position; the two networks agree to 10-20 %, the mean is given
REFERENCE = {"uu": [(1000, 3.5e-3), (3000, 8e-4), (1e4, 2.2e-4), (3e4, 9.5e-5), (1e5, 6e-5), (1e6, 5e-5)],
             "gg": [(1000, 2.5e-2), (3000, 8e-3), (1e4, 1.7e-3), (3e4, 5e-4), (1e5, 2e-4), (1e6, 1.1e-4)]}
# the one-loop test functions f2 (uu) and f4 (gg): the better of MLP(s) and L-GATr on a / a_LO, Figure 16
REFERENCE_LOOP = {"uu": [(1000, 6e-3), (3000, 2.5e-3), (1e4, 7e-4), (3e4, 2.7e-4), (1e5, 1.6e-4), (1e6, 9.5e-5)],
                  "gg": [(1000, 2.1e-2), (3000, 8e-3), (1e4, 2.3e-3), (3e4, 8e-4), (1e5, 3e-4), (1e6, 9e-5)]}
NAMES_LOOP = {"uu": r"$f_2$: $q\bar q \to t\bar t H$, one loop", "gg": r"$f_4$: $gg \to t\bar t H$, one loop"}
# the metric each figure plots: the paper-weights figure uses the networks' metric
METRIC = {"eps_vs_points_paper_weights.png": ("eps_unweighted", r"mean relative error on LO-unweighted test points"),
          "eps_vs_points_targets.png": ("eps_unweighted", r"mean relative error on LO-unweighted test points"),
          "eps_vs_points_loop.png": ("eps_unweighted", r"$\epsilon$ on LO-unweighted test points, $\sum|\tilde r - r| / \sum |r|$, $r = a / a_{\rm LO}$")}
XLIM = {"eps_vs_points_targets.png": (500, 1.5e6), "eps_vs_points_loop.png": (500, 1.5e6)}
LEGEND_LOC = {"eps_vs_points_targets.png": "below", "eps_vs_points_loop.png": "below"}
DEFAULT_METRIC = ("eps_uniform", r"$\epsilon = \sum|\tilde f - f| \,/\, \sum f$ (uniform test points)")
COLORS = ["#2a78d6", "#1baf7a", "#eb6834", "#8e44ad", "#eda100", "#5c6bc0", "#00897b", "#c0392b", "#7f8c8d"]

# (file tag, key suffix, label) per series; {ch} is the channel, "a|b" tries a then b
FIGURES = {
    "eps_vs_points.png": [
        ("exact_{ch}", "tree", "GPTree, Sobol order"),
        ("acq", "acq", "GPTree, chooses its points"),
        ("exact_{ch}", "hybrid", "hybrid (GP leaves on a network's residual)"),
        ("bignet_{ch}", "hybrid", "hybrid, 4 x 256 network, longer training"),
    ],
    "eps_vs_points_improved.png": [
        ("sweep_finalTreeSobol_{ch}", "tree", "GPTree on log invariants, RBF leaves, Sobol"),
        ("sweep_finalTreeMix_{ch}", "tree", "same, half the points drawn with probability ~ f"),
        ("sweep_finalTreeMix300_{ch}", "tree", "same, 300-point leaves"),
        ("sweep_nbar1000split_{ch}", "tree", "same, 1000-point leaves, refit only at splits"),
        ("sweep_finalHybMixBig_{ch}", "hybrid", "hybrid on log invariants, half the points ~ f, 4 x 256 network"),
    ],
    "eps_vs_points_paper_weights.png": [
        ("sweep_rhoDefault_{ch}", "tree", "GPTree, package defaults"),
        ("sweep_rhoMix300_{ch}", "tree", "GPTree on log invariants, RBF leaves, 300-point leaves, mixed stream"),
        ("sweep_rhoSplit1000_{ch}", "tree", "same, 1000-point leaves, refit only at splits"),
        ("sweep_rhoAcq300_{ch}", "acq", "same as the 300-point tree, choosing its points"),
        ("sweep_fullMix300_{ch}", "tree", "300-point tree on the full polar-angle domain"),
    ],
    "eps_vs_points_targets.png": [
        ("sweep_rhoMix300big_{ch}|sweep_rhoMix300_{ch}", "tree", "log target, mixed stream"),
        ("sweep_scaleLinBig_{ch}|sweep_scaleLin64k_{ch}", "tree", "linear target, mixed stream"),
        ("sweep_rhoAcq300_{ch}", "acq", "log target, choosing its points from a pool of 131k"),
        ("sweep_rhoAcq300big_{ch}", "acq", "log target, choosing its points from a pool of 524k"),
        ("sweep_scaleLinAcqBig2_{ch}", "acq", "linear target, choosing its points from a pool of 524k"),
        ("sweep_leafSum75_{ch}", "leafacq", "leaves choose, fresh points from the amplitude code (linear target on f3, log on f1)"),
        ("sweep_knNbar1000splitBig_{ch}|sweep_rhoSplit1000_{ch}", "tree", "1000-point leaves refit at splits, mixed stream (linear target on f3, log on f1)"),
        ("sweep_knNbar1000splitAcq_{ch}", "acq", "1000-point leaves refit at splits, choosing its points from a pool of 524k (linear target on f3, log on f1)"),
    ],
    "eps_vs_points_loop.png": [
        ("sweep_loopMix300_{ch}", "tree", r"ratio $a / a_{\rm LO}$: 300-point RBF leaves refit every 25 points, log invariants and $\beta_{t\bar t}$, mixed stream"),
        ("sweep_loopSplit1000_{ch}", "tree", "ratio: 1000-point leaves refit only at splits, mixed stream"),
        ("sweep_loopInv1000_{ch}", "tree", r"ratio: same, without $\beta_{t\bar t}$"),
        ("sweep_loopAcq1000_{ch}", "acq", "ratio: 1000-point leaves refit at splits, choosing its points from the 131k pool"),
        ("sweep_loopLinA1000_{ch}", "tree", "amplitude $a$: 1000-point leaves refit at splits, mixed stream"),
        ("sweep_loopLinAcq1000_{ch}", "acq", "amplitude: same, choosing its points from the 131k pool"),
    ],
}
# The first two figures use the phase-space measure alone as the weight (no pseudo-PDF factor), the third the paper's f


def rows_of(tag, ch, key):
    for t in tag.split("|"):
        path = os.path.join(OUT_DIR, f"results_{t.format(ch=ch)}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f).get(f"{ch}_{key}")
    return None


for name, series in FIGURES.items():
    metric, ylabel = METRIC.get(name, DEFAULT_METRIC)
    below = LEGEND_LOC.get(name) == "below"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.6 if below else 3.9))
    plotted = []
    for ax, ch in zip(axes, ["uu", "gg"]):
        n = 0
        for (tag, key, label), color in zip(series, COLORS):
            rows = rows_of(tag, ch, key)
            if rows:
                ax.plot([r["N"] for r in rows], [r.get(metric, r["eps_uniform"]) for r in rows], "-o", color=color, lw=2, ms=4, label=label)
                n += 1
        plotted.append(n)
        loop = name == "eps_vs_points_loop.png"
        ref = (REFERENCE_LOOP if loop else REFERENCE)[ch]
        ax.plot([n for n, _ in ref], [e for _, e in ref], "--", color="#52514e", lw=1.2,
                label=("the better of MLP(s) and L-GATr" if loop else "MLP / L-GATr") + " of arXiv:2412.09534 (Fig. 16, LO-unweighted test points)")
        ax.axhline(0.01, color="#52514e", lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*XLIM.get(name, (500, 2e5)))
        ax.set_xlabel("training points"); ax.set_ylabel(ylabel)
        ax.set_title((NAMES_LOOP if loop else NAMES)[ch], fontsize=10); ax.grid(alpha=0.25)
    ax_leg = axes[0 if plotted[0] >= plotted[1] else 1]
    if below:   # the legend under both panels, two columns
        handles, labels = ax_leg.get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, fontsize=7.5, loc="lower center", ncol=2)
        fig.tight_layout(rect=(0, 0.26, 1, 1))
    else:
        ax_leg.legend(frameon=False, fontsize=7.5, loc=LEGEND_LOC.get(name, "lower left"))
        fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150)
    plt.close(fig)
    print("wrote", os.path.join(OUT_DIR, name))
