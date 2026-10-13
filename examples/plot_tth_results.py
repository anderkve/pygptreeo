"""Figures for the 5D ttH benchmark: eps against the number of training points, from the
result files of tth_amplitude_benchmark.py, with the networks of arXiv:2412.09534
(values read off their Figure 13) as reference.

eps_vs_points.png: the package defaults (tree, tree choosing its points, hybrid, larger network).
eps_vs_points_improved.png: log-invariant inputs, RBF leaves and f-weighted training points.
eps_vs_points_paper_weights.png: the same with the paper's weights (flux and pseudo-PDF factor).

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
# MLP and L-GATr of arXiv:2412.09534, Figure 13, read off the plot (uniform test sampling)
REFERENCE = {"uu": [(10, 0.3), (100, 0.03), (300, 0.01), (1000, 1e-3), (3000, 3e-4), (1e4, 1e-4), (1e5, 7e-5)],
             "gg": [(10, 0.5), (100, 0.1), (1000, 1e-2), (1e4, 1e-3), (1e5, 3e-4)]}
COLORS = ["#2a78d6", "#1baf7a", "#eb6834", "#8e44ad", "#eda100"]

# (file tag, key suffix, label) per series; {ch} is the channel
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
}
# The first two figures use the phase-space measure alone as the weight (no pseudo-PDF factor), the third the paper's f


def rows_of(tag, ch, key):
    path = os.path.join(OUT_DIR, f"results_{tag.format(ch=ch)}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get(f"{ch}_{key}")


for name, series in FIGURES.items():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for ax, ch in zip(axes, ["uu", "gg"]):
        for (tag, key, label), color in zip(series, COLORS):
            rows = rows_of(tag, ch, key)
            if rows:
                ax.plot([r["N"] for r in rows], [r["eps_uniform"] for r in rows], "-o", color=color, lw=2, ms=4, label=label)
        ref = REFERENCE[ch]
        ax.plot([n for n, _ in ref], [e for _, e in ref], "--", color="#52514e", lw=1.2,
                label="MLP / L-GATr of arXiv:2412.09534 (Fig. 13)")
        ax.axhline(0.01, color="#52514e", lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(500, 2e5)
        ax.set_xlabel("training points"); ax.set_ylabel(r"$\epsilon = \sum|\tilde f - f| \,/\, \sum f$")
        ax.set_title(NAMES[ch], fontsize=10); ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150)
    plt.close(fig)
    print("wrote", os.path.join(OUT_DIR, name))
