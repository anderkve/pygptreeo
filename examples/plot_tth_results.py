"""Combined figure for the 5D ttH benchmark: eps against the number of training points
for the tree, the hybrid and the tree with acquisition, from the result files of
tth_amplitude_benchmark.py, with the networks of arXiv:2412.09534 (values read off
their Figure 13) as reference.

Usage: python plot_tth_results.py [results_dir]
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "tth_amplitude")
COLORS = {"tree": "#2a78d6", "hybrid": "#eb6834", "acq": "#1baf7a", "hybrid-big": "#eda100"}
LABELS = {"tree": "GPTree, Sobol order", "hybrid": "hybrid (GP leaves on a network's residual)",
          "acq": "GPTree, chooses its points", "hybrid-big": "hybrid, 4 x 256 network, longer training"}
NAMES = {"uu": r"$f_1$: $q\bar q \to t\bar t H$", "gg": r"$f_3$: $gg \to t\bar t H$"}
# MLP and L-GATr of arXiv:2412.09534, Figure 13, read off the plot (uniform test sampling)
REFERENCE = {"uu": [(10, 0.3), (100, 0.03), (300, 0.01), (1000, 1e-3), (3000, 3e-4), (1e4, 1e-4), (1e5, 7e-5)],
             "gg": [(10, 0.5), (100, 0.1), (1000, 1e-2), (1e4, 1e-3), (1e5, 3e-4)]}

results = {}
for path in sorted(glob.glob(os.path.join(OUT_DIR, "results_*.json"))):
    tag = os.path.basename(path)[len("results_"):-len(".json")]
    if tag.startswith("noise") or tag == "smoke":
        continue
    with open(path) as f:
        for key, rows in json.load(f).items():
            ch, cfg = key.split("_")
            results[(ch, cfg + ("-big" if tag.startswith("bignet") else ""))] = rows

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
for ax, ch in zip(axes, ["uu", "gg"]):
    for cfg in ("tree", "acq", "hybrid", "hybrid-big"):
        rows = results.get((ch, cfg))
        if rows:
            ax.plot([r["N"] for r in rows], [r["eps_uniform"] for r in rows], "-o", color=COLORS[cfg], lw=2, ms=4,
                    label=LABELS[cfg])
    ref = REFERENCE[ch]
    ax.plot([n for n, _ in ref], [e for _, e in ref], "--", color="#52514e", lw=1.2,
            label="MLP / L-GATr of arXiv:2412.09534 (Fig. 13)")
    ax.axhline(0.01, color="#52514e", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(500, 2e5)
    ax.set_xlabel("training points"); ax.set_ylabel(r"$\epsilon = \sum|\tilde f - f| \,/\, \sum f$")
    ax.set_title(NAMES[ch], fontsize=10); ax.grid(alpha=0.25)
axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "eps_vs_points.png"), dpi=150)
print("wrote", os.path.join(OUT_DIR, "eps_vs_points.png"))
