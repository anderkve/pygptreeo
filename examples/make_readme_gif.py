"""Generate the animated GIF shown at the top of the pyGPTreeO README.

The animation tells the core pyGPTreeO story at a glance: an unknown target
function is learned *online* and *locally*, from a stream of input points, in
exactly those regions of input space where data arrives.

To make the "local" aspect obvious, the input points do not cover the domain
uniformly. Instead they mimic an adaptive sampler: a cluster of points that
starts out broad (exploring much of the input space) and gradually shrinks while
its centre sweeps a curved diagonal path, finally settling in -- and densely
exploring -- a converged region of interest. As the stream proceeds you can
watch pyGPTreeO's prediction "fill in" the true function where the data goes,
while regions that never receive points stay unlearned and uncertain.

Four panels, sharing the same x_1, x_2 axes:

  1. Unknown target function (fixed reference heat map) with the incoming data
     stream drawn on top: small white dots = points seen earlier, red = the
     points that just arrived.
  2. pyGPTreeO's current prediction (full opacity), coloured on the same scale as
     panel 1. A cyan contour encloses the region where the *relative* predictive
     uncertainty (std / |prediction|) is below 1%, i.e. where the model is sharply
     confident. The thin white rectangles are the leaves of the GP tree -- the
     local GP models that the tree has grown, denser where more data has arrived.
  3. pyGPTreeO's *relative* predictive uncertainty, std / |prediction|, on a
     logarithmic colour scale from 0.1% to 100%: low (dark) where data has been
     seen, high (bright) where it has not.
  4. The *relative* prediction error, |prediction - truth| / |truth|, on the same
     0.1%..100% log scale: low (dark) where data has been seen, confirming that the
     model is actually accurate -- not just confident -- in the sampled regions.

  The same cyan < 1% relative-uncertainty contour is overlaid on panels 2, 3 and 4.

Usage:
    python make_readme_gif.py                     # full-quality render (Himmelblau)
    python make_readme_gif.py --quick             # small/fast preview
    python make_readme_gif.py --target eggholder  # the harder Eggholder target
    python make_readme_gif.py --alt               # Nbar=100 variant

Output: examples/example_plots/animation/<basename>.gif (+ <basename>_final.png),
where <basename> reflects the chosen --target and --alt options.
"""

import argparse
import contextlib
import io
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm, colors
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pygptreeo import GPTree, Default_GPR, AdditiveMaternKernel
from target_functions import Himmelblau, Eggholder

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true",
                    help="Small/fast preview render for iterating on the design.")
parser.add_argument("--alt", action="store_true",
                    help="Alternative version: Nbar=100, saved to a separate file.")
parser.add_argument("--target", choices=["himmelblau", "eggholder"],
                    default="himmelblau", help="Target function to learn.")
parser.add_argument("--final-only", action="store_true",
                    help="Run the full stream but render only the final frame "
                         "(saved as the PNG, no GIF) -- for previewing the end state.")
parser.add_argument("--seed", type=int, default=4)
args = parser.parse_args()

# Target function and the matching function-value colour range. Both targets sit
# well away from 0 so the GP's default zero-mean prior is a poor guess everywhere
# -- the model only does well where it has actually seen data, not by luck near a
# minimum. Himmelblau (raw 0..7) is shifted by +1 and scaled by 10 to lift it off
# zero; the rugged Eggholder (the harder target) already spans ~18..2009.
if args.target == "eggholder":
    def TARGET(x):
        return Eggholder(x)
    TARGET_NAME = "Eggholder"
    FN_VMIN, FN_VMAX = 0.0, 2000.0
    FN_TICKS = [0, 500, 1000, 1500, 2000]
    # A different sweep for this target: upper-left -> lower-right, more bowed,
    # with a somewhat broader cluster.
    START_CENTER = np.array([0.18, 0.82])
    FINAL_CENTER = np.array([0.78, 0.25])
    BOW_AMP = 0.20
    CLUSTER_SIGMA_MAX, CLUSTER_SIGMA_MIN = 0.26, 0.07
else:
    def TARGET(x):
        return 10.0 * (Himmelblau(x) + 1.0)
    TARGET_NAME = "Himmelblau"
    FN_VMIN, FN_VMAX = 10.0, 80.0
    FN_TICKS = list(range(10, 81, 10))
    # Sweep from lower-left to the upper-right minimum, gently bowed.
    START_CENTER = np.array([0.20, 0.20])
    FINAL_CENTER = np.array([0.80, 0.70])
    BOW_AMP = 0.12
    CLUSTER_SIGMA_MAX, CLUSTER_SIGMA_MIN = 0.20, 0.04

if args.quick:
    N_PTS = 260
    PTS_PER_FRAME = 20
    GRID = 44
    HOLD_FRAMES = 4
else:
    N_PTS = 1500
    PTS_PER_FRAME = 27
    GRID = 70
    HOLD_FRAMES = 12

NBAR = 100 if args.alt else 50
FPS = 12

# Sampling schedule. The cluster centre sweeps from START_CENTER to FINAL_CENTER
# (both set per target above) and then settles there for the last HOLD_FRAC of the
# stream, while the cluster width shrinks from CLUSTER_SIGMA_MAX to
# CLUSTER_SIGMA_MIN. This mimics an optimiser/adaptive sampler that explores
# broadly at first and converges onto a region, which it then samples in detail.
HOLD_FRAC = 0.45

# Function-value (viridis) colour scale: FN_VMIN/FN_VMAX/FN_TICKS are set per
# target above; discretised into 21 levels.
FN_LEVELS = 21

# Panels 3 and 4 show the *relative* predictive uncertainty (std / |prediction|)
# and the *relative* error (|prediction - truth| / |truth|) on a logarithmic
# colour scale spanning 0.001 to 1.0, i.e. from 0.1% to 100%.
UNC_LEVELS = 20  # discrete colour levels for the relative-uncertainty (magma) map
ERR_LEVELS = 20  # discrete colour levels for the relative-error (inferno) map
REL_VMIN = 1e-3
REL_VMAX = 1.0

# Contour (drawn on the prediction, uncertainty and error panels) bounding the
# region where the relative predictive uncertainty (std / |prediction|) < 1%.
REL_UNC_LEVEL = 0.01
CONTOUR_COLOR = "cyan"

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "example_plots", "animation")
os.makedirs(OUT_DIR, exist_ok=True)
BASENAME = "pygptreeo_local_learning"
if args.target == "eggholder":
    BASENAME += "_eggholder"
if args.alt:
    BASENAME += "_nbar100"
GIF_PATH = os.path.join(OUT_DIR, BASENAME + ".gif")
PNG_PATH = os.path.join(OUT_DIR, BASENAME + "_final.png")

np.random.seed(args.seed)


# --------------------------------------------------------------------------
# Build the input stream: a concentrated cluster sweeping a curved diagonal band
# --------------------------------------------------------------------------

def path_center(t):
    """Cluster centre at stream fraction t in [0, 1].

    The centre sweeps from START_CENTER to FINAL_CENTER along a gently bowed arc,
    reaching FINAL_CENTER at the start of the final HOLD_FRAC of the stream and
    then staying there, so the converged minimum is sampled for an extended time.
    """
    p = min(t / (1.0 - HOLD_FRAC), 1.0)          # path progress, then held at 1
    s = p * p * (3.0 - 2.0 * p)                   # smoothstep ease in/out
    base = START_CENTER + (FINAL_CENTER - START_CENTER) * s
    direction = FINAL_CENTER - START_CENTER
    normal = np.array([-direction[1], direction[0]])
    normal = normal / np.linalg.norm(normal)
    bow = BOW_AMP * np.sin(np.pi * p)            # single arc, zero at both ends
    c = base + bow * normal
    return float(np.clip(c[0], 0.03, 0.97)), float(np.clip(c[1], 0.03, 0.97))

def cluster_sigma(t):
    """Cluster width at stream fraction t: shrinks from MAX to MIN as t -> 1."""
    return CLUSTER_SIGMA_MIN + (CLUSTER_SIGMA_MAX - CLUSTER_SIGMA_MIN) * (1.0 - t) ** 1.5

ts = (np.arange(N_PTS) + 0.5) / N_PTS
centers = np.array([path_center(t) for t in ts])
sigmas = np.array([cluster_sigma(t) for t in ts])
X_input = np.clip(centers + np.random.normal(0.0, 1.0, (N_PTS, 2)) * sigmas[:, None],
                  0.0, 1.0)
y_input = np.array([TARGET(xi) for xi in X_input]).reshape(-1, 1)


# --------------------------------------------------------------------------
# True target function on a grid (fixed reference)
# --------------------------------------------------------------------------

gx = np.linspace(0.0, 1.0, GRID)
GX, GY = np.meshgrid(gx, gx)
grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
Z_true = np.array([TARGET(p) for p in grid_pts]).reshape(GRID, GRID)
vmin, vmax = FN_VMIN, FN_VMAX


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def leaf_boxes(root):
    """Return [(lo, hi), ...] bounding boxes for every leaf of the GP tree."""
    boxes = []
    def rec(node, lo, hi):
        if node.is_leaf:
            boxes.append((lo.copy(), hi.copy()))
            return
        j, s = node.split_index, node.split_position
        hi_l = hi.copy(); hi_l[j] = s
        lo_r = lo.copy(); lo_r[j] = s
        rec(node.children[0], lo, hi_l)
        rec(node.children[1], lo_r, hi)
    rec(root, np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    return boxes


def update_quiet(gpt, x, y):
    """Feed one point to the tree, suppressing the library's stdout chatter."""
    with contextlib.redirect_stdout(io.StringIO()):
        gpt.update_tree(x.reshape(1, -1), y.reshape(1, 1), 0.001 * np.abs(y).reshape(1, 1))


# --------------------------------------------------------------------------
# Figure scaffolding
# --------------------------------------------------------------------------

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14})
fig = plt.figure(figsize=(20.0, 5.2))

cmap_fn = plt.get_cmap("viridis", FN_LEVELS)   # 21 discrete colours over [FN_VMIN, FN_VMAX]
cmap_unc = plt.get_cmap("magma", UNC_LEVELS)
cmap_err = plt.get_cmap("inferno", ERR_LEVELS)
norm_fn = colors.Normalize(vmin=vmin, vmax=vmax)

extent = [0, 1, 0, 1]

# Manual layout: four image panels in a row, each followed (where needed) by a
# slim colour bar with room for its tick labels.
PB, PH, PW, CW = 0.10, 0.70, 0.190, 0.008      # bottom, height, panel/cbar widths
x = 0.028
axT = fig.add_axes([x, PB, PW, PH]);    x += PW + 0.010
axP = fig.add_axes([x, PB, PW, PH]);    x += PW + 0.006
cax_fn = fig.add_axes([x, PB, CW, PH]); x += CW + 0.034
axU = fig.add_axes([x, PB, PW, PH]);    x += PW + 0.006
cax_un = fig.add_axes([x, PB, CW, PH]); x += CW + 0.034
axE = fig.add_axes([x, PB, PW, PH]);    x += PW + 0.006
cax_er = fig.add_axes([x, PB, CW, PH])

# Panel 1: true function + data stream
axT.imshow(Z_true, origin="lower", extent=extent, cmap=cmap_fn,
           norm=norm_fn, aspect="auto")
old_scat = axT.scatter([], [], s=3, c="white", edgecolors="none", alpha=0.85,
                       zorder=3)
new_scat = axT.scatter([], [], s=12, c="red", edgecolors="white",
                       linewidths=0.3, zorder=4)
axT.set_title("Unknown target function\n+ stream of input points")

# Panel 2: prediction (full opacity); a contour marks relative uncertainty < 1%
pred_im = axP.imshow(np.zeros((GRID, GRID)), origin="lower", extent=extent,
                     cmap=cmap_fn, norm=norm_fn, aspect="auto", zorder=2)
axP.set_title("pyGPTreeO prediction\n(cyan contour: relative uncertainty < 1%)")

# Panel 3: relative predictive uncertainty (log scale 0.1% .. 100%)
norm_rel = colors.LogNorm(vmin=REL_VMIN, vmax=REL_VMAX)
unc_im = axU.imshow(np.full((GRID, GRID), REL_VMIN), origin="lower", extent=extent,
                    cmap=cmap_unc, norm=norm_rel, aspect="auto")
axU.set_title("pyGPTreeO relative uncertainty\n(std / |prediction|)")

# Panel 4: relative prediction error (log scale 0.1% .. 100%)
err_im = axE.imshow(np.full((GRID, GRID), REL_VMIN), origin="lower", extent=extent,
                    cmap=cmap_err, norm=norm_rel, aspect="auto")
axE.set_title("pyGPTreeO relative error\n(|prediction $-$ truth| / |truth|)")

for ax in (axT, axP, axU, axE):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(r"$x_1$")
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
axT.set_ylabel(r"$x_2$")
for ax in (axP, axU, axE):       # all panels share the same y-axis -> label once
    ax.set_yticklabels([])

# Colour bars
cb_fn = fig.colorbar(cm.ScalarMappable(norm=norm_fn, cmap=cmap_fn), cax=cax_fn, label="y")
cb_fn.set_ticks(FN_TICKS)
cb_un = fig.colorbar(cm.ScalarMappable(norm=norm_rel, cmap=cmap_unc),
                     cax=cax_un, label="relative uncertainty")
cb_er = fig.colorbar(cm.ScalarMappable(norm=norm_rel, cmap=cmap_err),
                     cax=cax_er, label="relative error")
# Log decade ticks labelled as percentages (0.1% .. 100%).
rel_ticks = [1e-3, 1e-2, 1e-1, 1e0]
rel_labels = ["0.1%", "1%", "10%", "100%"]
cb_un.set_ticks(rel_ticks); cb_un.set_ticklabels(rel_labels)
cb_er.set_ticks(rel_ticks); cb_er.set_ticklabels(rel_labels)

suptitle = fig.suptitle("", fontsize=18, x=0.5, y=0.955)
leaf_patches = []
contours = []


# --------------------------------------------------------------------------
# Frame rendering
# --------------------------------------------------------------------------

def render_frame(n_seen):
    global leaf_patches, contours

    mu, std = gpt.predict(grid_pts, show_progress=False)
    mu = mu.reshape(GRID, GRID)
    std = std.reshape(GRID, GRID)

    # Panel 2: prediction at full opacity, coloured on the function scale
    pred_im.set_data(mu)

    # Leaf rectangles
    for p in leaf_patches:
        p.remove()
    leaf_patches = []
    for lo, hi in leaf_boxes(gpt.root):
        r = Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
                      fill=False, edgecolor="white", linewidth=0.6,
                      alpha=0.55, zorder=3)
        axP.add_patch(r)
        leaf_patches.append(r)

    # Panels 3 & 4: relative uncertainty and relative error (clipped to the log range)
    rel_unc = std / np.maximum(np.abs(mu), 1e-9)
    rel_err = np.abs(mu - Z_true) / np.abs(Z_true)
    unc_im.set_data(np.clip(rel_unc, REL_VMIN, REL_VMAX))
    err_im.set_data(np.clip(rel_err, REL_VMIN, REL_VMAX))

    # Contour bounding relative uncertainty < 1%, on panels 2, 3 and 4
    for c in contours:
        c.remove()
    contours = []
    for ax in (axP, axU, axE):
        contours.append(ax.contour(GX, GY, rel_unc, levels=[REL_UNC_LEVEL],
                                   colors=CONTOUR_COLOR, linewidths=1.8, zorder=4))

    # Panel 1: data stream (faded older, bright newest batch)
    new_lo = max(0, n_seen - PTS_PER_FRAME)
    old_scat.set_offsets(X_input[:new_lo] if new_lo > 0 else np.empty((0, 2)))
    new_scat.set_offsets(X_input[new_lo:n_seen])

    suptitle.set_text(
        f"pyGPTreeO learning the {TARGET_NAME} function online   "
        f"|   {n_seen} input points seen   |   {len(gpt.root.leaves)} local GPs")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(buf[..., :3].copy())


# --------------------------------------------------------------------------
# Run the stream, collecting frames
# --------------------------------------------------------------------------

kernel = AdditiveMaternKernel(d=2, order=2, nu=1.5)
gpt = GPTree(GPR=Default_GPR(kernel=kernel, alpha=1e-6), Nbar=NBAR, theta=0.05,
             splitting_strategy="gradual", split_position_method="median",
             retrain_every_n_points=2, use_calibrated_sigma=True)

frames = []
n_seen = 0
for i in range(N_PTS):
    update_quiet(gpt, X_input[i], y_input[i])
    n_seen += 1
    if args.final_only:
        if n_seen == N_PTS:
            frames.append(render_frame(n_seen))
    elif n_seen % PTS_PER_FRAME == 0 or n_seen == N_PTS:
        frames.append(render_frame(n_seen))
        print(f"frame {len(frames):3d}  ({n_seen}/{N_PTS} pts, "
              f"{len(gpt.root.leaves)} leaves)", flush=True)

frames[-1].save(PNG_PATH)

if args.final_only:
    print(f"\nWrote {PNG_PATH} (final frame only; no GIF)")
else:
    # Hold on the final frame so viewers can read the end state
    frames.extend([frames[-1]] * HOLD_FRAMES)
    # Save with a full per-frame palette (no shared/lossy quantization). Because
    # the colormaps are discretised into a fixed set of colours, unchanged regions
    # render as the same colour frame to frame, so the animation stays stable.
    frames[0].save(GIF_PATH, save_all=True, append_images=frames[1:],
                   duration=int(1000 / FPS), loop=0, optimize=True)
    print(f"\nWrote {GIF_PATH}  ({len(frames)} frames, "
          f"{os.path.getsize(GIF_PATH) / 1e6:.1f} MB)")
    print(f"Wrote {PNG_PATH}")
