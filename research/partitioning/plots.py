"""Figures of the partitioning study from the records: python plots.py <output folder>"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # before numpy is loaded
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import bench
import report
from tables import gmean, ratios

OUT = sys.argv[1]
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, AQUA, YELLOW, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e34948'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2,
    'axes.edgecolor': AXIS, 'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2,
    'ytick.labelcolor': INK2, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.axisbelow': True, 'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(path)


# 1. every alternative against the current tree, 5000-point streams
rows = [
    ("Neighbours, 200 points", 'knn:metric=std,fallback=10,k=200'),
    ("Neighbours, 150 points", 'nb:n_neighbours=150'),
    ("Neighbours, 100 points (as shipped)", 'nbf'),
    ("Neighbours, 100 points, exact search", 'nb'),
    ("Neighbours, without the consistency test", 'knn:metric=std'),
    ("Oblique cuts + decorrelated leaf inputs", 'oblique:dir=curv,pos=gsplit,whiten=True'),
    ("Current tree with 200-point leaves", 'base:Nbar=200,retrain_every_n_points=50'),
    ("Oblique cuts at the change of the GP gradient", 'oblique:dir=curv,pos=gsplit'),
    ("Current tree with a wide overlap (theta = 0.2)", 'base:theta=0.2'),
    ("Axis cuts placed at the gradient change", 'oblique:dir=axis,pos=gsplit'),
    ("Oblique cuts, GP curvature direction", 'oblique:dir=curv'),
    ("Oblique cuts, GP gradient direction", 'oblique:dir=grad'),
    ("Axis cuts by largest spread (max_spread)", 'base:split_dimension_criteria=max_spread'),
    ("Oblique cuts, two clusters", 'oblique:dir=twomeans'),
    ("Oblique cuts, largest spread (PCA)", 'oblique:dir=pca'),
    ("Oblique cuts, random direction", 'oblique:dir=random,scaled=True'),
    ("A model in every node (multi-resolution)", 'mr'),
]
data = []
for label, variant in rows:
    r = np.array([v[0] for v in ratios(variant, 'base', 5000).values()])
    if len(r):
        data.append((label, gmean(r), int(np.sum(r > 1.02)), len(r)))
data.sort(key=lambda t: t[1])
fig, ax = plt.subplots(figsize=(9.2, 6.0))
y = np.arange(len(data))[::-1]
for yi, (label, g, worse, n) in zip(y, data):
    ax.barh(yi, g - 1.0, left=1.0, height=0.62, color=BLUE if g < 1 else RED, zorder=3)
    ax.text(g + (0.012 if g >= 1 else -0.012), yi, f"{g:.2f}", va='center', ha='left' if g >= 1 else 'right',
            fontsize=9.5, color=INK, fontweight='bold')
    ax.text(1.66, yi, f"{worse} of {n}", va='center', ha='right', fontsize=9, color=INK2)
ax.axvline(1.0, color=INK2, linewidth=1.0, zorder=4)
ax.set_yticks(y)
ax.set_yticklabels([d[0] for d in data], fontsize=9.5, color=INK)
ax.set_xlim(0.58, 1.66)
ax.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4])
ax.set_xlabel("Prediction error relative to the current tree (geometric mean over the runs; below 1 is better)")
ax.text(1.66, len(data) - 0.45, "runs more than\n2% worse", ha='right', va='bottom', fontsize=8.5, color=MUTED)
ax.grid(axis='y', visible=False)
ax.tick_params(axis='y', length=0)
ax.set_title("Predicting from a query's nearest points beats every change to the cuts\n\n", pad=14)
ax.text(0.0, 1.085, "5000-point streams, 11 targets (artificial and physics) x uniform, walker and sweeping streams, 33 runs each",
        transform=ax.transAxes, fontsize=9, color=INK2)
save(fig, "1_alternatives_5k.png")

# 2. the new option on 20 000-point streams, run by run
r = ratios('nbf', 'base', 20000, seeds=(1, 2))
streams = [s for s in ('uniform', 'walker', 'sweeping', 'focusing') if any(k[1] == s for k in r)]
targets = sorted({k[0] for k in r}, key=lambda t: gmean(v[0] for k, v in r.items() if k[0] == t))
fig, axs = plt.subplots(1, len(streams), figsize=(11.5, 0.33 * len(targets) + 1.6), sharey=True)
for ax, s in zip(np.atleast_1d(axs), streams):
    for i, t in enumerate(targets):
        vals = np.array([v[0] for k, v in r.items() if k[0] == t and k[1] == s])
        ax.plot(vals[vals >= 0.2], [i] * int(np.sum(vals >= 0.2)), 'o', color=BLUE, markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.2, zorder=3)
        for v in vals[vals < 0.2]:          # off the scale: an arrow at the edge with the value
            ax.plot([0.215], [i], '<', color=BLUE, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
            ax.text(0.235, i, f"{v:.2f}", va='center', ha='left', fontsize=8, color=INK2)
    ax.axvline(1.0, color=INK2, linewidth=1.0)
    ax.set_xscale('log')
    ax.set_xlim(0.2, 1.6)
    ax.set_xticks([0.25, 0.5, 1.0])
    ax.set_xticklabels(['0.25', '0.5', '1'])
    ax.minorticks_off()
    vals = [v[0] for k, v in r.items() if k[1] == s]
    ax.set_title(f"{s}: {gmean(vals):.2f}", fontsize=10.5, fontweight='bold', loc='left')
    ax.grid(axis='y', color=GRID, linewidth=0.5)
    ax.tick_params(axis='y', length=0)
np.atleast_1d(axs)[0].set_yticks(range(len(targets)))
np.atleast_1d(axs)[0].set_yticklabels([f"{t}  ({bench.target_dim(t)}D)" for t in targets], fontsize=9, color=INK)
fig.subplots_adjust(top=0.9, bottom=0.075, wspace=0.12)
fig.supxlabel("Prediction error with neighbours relative to the current tree (one dot per run; left of the line is better)",
              fontsize=9.5, color=INK2, y=0.015)
all_r = np.array([v[0] for v in r.values()])
fig.suptitle(f"20 000-point streams: {gmean(all_r):.2f} of the current tree's error over {len(all_r)} runs, "
             f"{int(np.sum(all_r > 1.02))} more than 2% worse", fontsize=12, fontweight='bold', x=0.01, ha='left', y=0.975)
save(fig, "2_neighbours_20k_by_target.png")

# 3. the error along a stream, two runs
A, B = report.load('nbf', 20000), report.load('base', 20000)
picks = [k for k in (('gaussian_peaks', 'walker', 1), ('resonance_xsec', 'uniform', 1), ('eft_signal_strength', 'sweeping', 1),
                     ('relic_density', 'uniform', 1), ('rotated_rosenbrock', 'uniform', 1), ('diagonal_ridge', 'sweeping', 1),
                     ('chirp', 'focusing', 1)) if k in A and k in B][:3]
fig, axs = plt.subplots(1, len(picks), figsize=(4.0 * len(picks), 3.4))
for ax, key in zip(np.atleast_1d(axs), picks):
    for rec, colour, name in ((B[key], ORANGE, 'current tree'), (A[key], BLUE, 'with neighbours')):
        err2 = (rec['pred'] - rec['y']) ** 2
        window = 2000
        roll = np.sqrt(np.convolve(err2, np.ones(window) / window, mode='valid')) / np.ptp(rec['y'])
        x = np.arange(window, len(err2) + 1)
        ax.plot(x, roll, color=colour, linewidth=2.0)
        ax.text(x[-1] + 300, roll[-1], name, color=INK, fontsize=9, va='center')
    ax.set_yscale('log')
    ax.set_xlim(0, 27500)
    ax.set_xticks([0, 10000, 20000])
    ax.set_title(f"{key[0]}, {key[1]} stream", fontsize=10, fontweight='bold')
    ax.set_xlabel("points processed")
np.atleast_1d(axs)[0].set_ylabel("NRMSE of the last 2000 predictions")
fig.suptitle("The gain holds along the stream (20 000 points)", fontsize=12, fontweight='bold', x=0.01, ha='left', y=1.03)
fig.tight_layout()
save(fig, "3_error_along_the_stream.png")

# 4. depth of the tree of cuts by stream type (measured on the current tree)
fig, ax = plt.subplots(figsize=(7.4, 4.2))
series = [
    ("drifting cluster", [2000, 4000, 6000, 8000, 10000, 12000], [25, 51, 76, 95, 113, 131], RED),
    ("sweeping", [1500, 3000, 6000, 12000], [7, 12, 21, 23], ORANGE),
    ("walker", [1500, 3000, 6000, 12000], [7, 10, 11, 12], AQUA),
    ("uniform", [1500, 3000, 6000, 12000], [5, 6, 8, 9], BLUE),
]
for name, n, depth, colour in series:
    ax.plot(n, depth, '-o', color=colour, linewidth=2.0, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.0)
    ax.text(n[-1] + 250, depth[-1], name, color=INK, fontsize=9.5, va='center')
ax.plot([1500, 3000, 6000, 12000], [4.4, 5.5, 6.5, 7.5], '--', color=MUTED, linewidth=1.2)
ax.text(12250, 5.6, "balanced tree", color=MUTED, fontsize=9, va='center')
ax.set_yscale('log')
ax.set_yticks([5, 10, 20, 50, 100])
ax.set_yticklabels(['5', '10', '20', '50', '100'])
ax.minorticks_off()
ax.set_xlim(0, 15200)
ax.set_xlabel("points processed")
ax.set_ylabel("levels down to the deepest leaf")
ax.set_title("On a stream that moves on, the tree of cuts gains a level per split\n", pad=10)
ax.text(0.0, 1.03, "Nbar = 100; the new leaf index stays 1 to 2 levels deep in all four cases, and predictions no longer follow the cuts",
        transform=ax.transAxes, fontsize=8.5, color=INK2)
save(fig, "4_depth_of_the_cut_tree.png")
