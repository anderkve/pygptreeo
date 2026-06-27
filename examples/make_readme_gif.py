"""Generate the animated GIF shown at the top of the pyGPTreeO README.

The animation tells the core pyGPTreeO story at a glance: an unknown target
function is learned *online* and *locally*, from a stream of input points, in
exactly those regions of input space where data arrives.

To make the "local" aspect obvious, the input points do not cover the domain
uniformly. Instead they arrive as a moving, concentrated cluster that sweeps a
curved diagonal band across the 2D input space. As the stream proceeds you can
watch pyGPTreeO's prediction "fill in" the true function along the path of the
data, while the two off-diagonal corners -- which never receive any points --
stay unlearned and uncertain.

Three panels, sharing the same x_1, x_2 axes:

  1. Unknown target function (fixed reference heat map) with the incoming data
     stream drawn on top: faded grey = points seen earlier, bright red = the
     points that just arrived.
  2. pyGPTreeO's current prediction. Each pixel is shown only as strongly as the
     model is confident there (low GP std -> opaque, high GP std -> transparent),
     so the learned function literally appears only where data has been seen. The
     thin white rectangles are the leaves of the GP tree -- the local GP models
     that the tree has grown, denser where more data has arrived.
  3. pyGPTreeO's predictive uncertainty (GP standard deviation): low (dark) along
     the visited band, high (bright) in the unvisited corners.

Usage:
    python make_readme_gif.py                 # full-quality render
    python make_readme_gif.py --quick         # small/fast preview

Output: examples/example_plots/animation/pygptreeo_local_learning.gif
        examples/example_plots/animation/pygptreeo_local_learning_final.png
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
from target_functions import Himmelblau

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true",
                    help="Small/fast preview render for iterating on the design.")
parser.add_argument("--seed", type=int, default=4)
args = parser.parse_args()

TARGET = Himmelblau
TARGET_NAME = "Himmelblau"

if args.quick:
    N_PTS = 260
    PTS_PER_FRAME = 20
    GRID = 44
    HOLD_FRAMES = 4
else:
    N_PTS = 760
    PTS_PER_FRAME = 14
    GRID = 70
    HOLD_FRAMES = 12

NBAR = 100
CLUSTER_SIGMA = 0.06
FPS = 12

# Function-value (viridis) colour scale: 0..7 in 21 discrete levels, i.e. exactly
# 3 colours per unit of target-function value.
FN_VMIN = 0.0
FN_VMAX = 7.0
FN_LEVELS = 21
UNC_LEVELS = 20  # discrete colour levels for the uncertainty (magma) colormap

# Predictive-std scale (calibrated for this target/setup): well-learned regions
# sit near STD_LO, never-visited regions near/above STD_HI.
STD_LO = 0.03
STD_HI = 0.30
UNC_VMAX = 0.40

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "example_plots", "animation")
os.makedirs(OUT_DIR, exist_ok=True)
GIF_PATH = os.path.join(OUT_DIR, "pygptreeo_local_learning.gif")
PNG_PATH = os.path.join(OUT_DIR, "pygptreeo_local_learning_final.png")

np.random.seed(args.seed)


# --------------------------------------------------------------------------
# Build the input stream: a concentrated cluster sweeping a curved diagonal band
# --------------------------------------------------------------------------

def path_center(t):
    """Cluster centre at stream fraction t in [0, 1]: a gently S-bowed diagonal."""
    d = 0.15 + 0.70 * t
    perp = 0.12 * np.sin(2.0 * np.pi * t)        # bow off the main diagonal
    cx = np.clip(d - perp / np.sqrt(2.0), 0.05, 0.95)
    cy = np.clip(d + perp / np.sqrt(2.0), 0.05, 0.95)
    return cx, cy

ts = (np.arange(N_PTS) + 0.5) / N_PTS
centers = np.array([path_center(t) for t in ts])
X_input = np.clip(centers + np.random.normal(0.0, CLUSTER_SIGMA, (N_PTS, 2)), 0.0, 1.0)
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

plt.rcParams.update({"font.size": 13, "axes.titlesize": 15})
fig, (axT, axP, axU) = plt.subplots(1, 3, figsize=(15.0, 5.0))
fig.subplots_adjust(left=0.035, right=0.895, bottom=0.10, top=0.80, wspace=0.30)

cmap_fn = plt.get_cmap("viridis", FN_LEVELS)   # 21 colours over y in [0, 7]
cmap_unc = plt.get_cmap("magma", UNC_LEVELS)
norm_fn = colors.Normalize(vmin=vmin, vmax=vmax)

extent = [0, 1, 0, 1]

# Panel 1: true function + data stream
axT.imshow(Z_true, origin="lower", extent=extent, cmap=cmap_fn,
           norm=norm_fn, aspect="auto")
old_scat = axT.scatter([], [], s=10, c="0.7", edgecolors="none", zorder=3)
new_scat = axT.scatter([], [], s=26, c="red", edgecolors="white",
                       linewidths=0.4, zorder=4)
axT.set_title("Unknown target function\n+ stream of input points")

# Panel 2: confidence-masked prediction (filled in update())
pred_im = axP.imshow(np.zeros((GRID, GRID, 4)), origin="lower", extent=extent,
                     aspect="auto", zorder=2)
axP.set_title("pyGPTreeO prediction\n(opacity = model confidence)")

# Panel 3: uncertainty
unc_im = axU.imshow(np.zeros((GRID, GRID)), origin="lower", extent=extent,
                    cmap=cmap_unc, aspect="auto", vmin=0.0, vmax=UNC_VMAX)
axU.set_title("pyGPTreeO uncertainty\n(GP standard deviation)")

for ax in (axT, axP, axU):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(r"$x_1$")
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
axT.set_ylabel(r"$x_2$")
axP.set_facecolor("0.85")  # neutral background showing through where unconfident

# Shared colour bars
cax_fn = fig.add_axes([0.602, 0.10, 0.010, 0.70])
cb_fn = fig.colorbar(cm.ScalarMappable(norm=norm_fn, cmap=cmap_fn), cax=cax_fn, label="y")
cb_fn.set_ticks(list(range(int(FN_VMIN), int(FN_VMAX) + 1)))
cax_un = fig.add_axes([0.905, 0.10, 0.010, 0.70])
cb_un = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(0, UNC_VMAX), cmap=cmap_unc),
                     cax=cax_un, label="GP std")
cb_un.set_ticks([0, UNC_VMAX])
cb_un.set_ticklabels(["0\n(learned)", "high\n(unknown)"])

suptitle = fig.suptitle("", fontsize=18, y=0.955)
leaf_patches = []


# --------------------------------------------------------------------------
# Frame rendering
# --------------------------------------------------------------------------

def render_frame(n_seen):
    global leaf_patches

    mu, std = gpt.predict(grid_pts, show_progress=False)
    mu = mu.reshape(GRID, GRID)
    std = std.reshape(GRID, GRID)

    # Panel 2: colour by prediction, opacity by confidence
    conf = np.clip((STD_HI - std) / (STD_HI - STD_LO), 0.0, 1.0)
    rgba = cmap_fn(norm_fn(mu))
    rgba[..., 3] = conf
    pred_im.set_data(rgba)

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

    # Panel 3: uncertainty
    unc_im.set_data(std)

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
gpt = GPTree(GPR=Default_GPR(kernel=kernel, alpha=1e-6), Nbar=NBAR, theta=0.01,
             split_position_method="median", retrain_every_n_points=2,
             use_calibrated_sigma=True)

frames = []
n_seen = 0
for i in range(N_PTS):
    update_quiet(gpt, X_input[i], y_input[i])
    n_seen += 1
    if n_seen % PTS_PER_FRAME == 0 or n_seen == N_PTS:
        frames.append(render_frame(n_seen))
        print(f"frame {len(frames):3d}  ({n_seen}/{N_PTS} pts, "
              f"{len(gpt.root.leaves)} leaves)", flush=True)

# Hold on the final frame so viewers can read the end state
frames.extend([frames[-1]] * HOLD_FRAMES)

frames[-1].save(PNG_PATH)

# Save with a full per-frame palette (no shared/lossy quantization). Because the
# colormaps are discretised into a fixed set of colours, unchanged regions render
# as exactly the same colour from frame to frame, so the animation stays stable.
frames[0].save(GIF_PATH, save_all=True, append_images=frames[1:],
               duration=int(1000 / FPS), loop=0, optimize=True)
print(f"\nWrote {GIF_PATH}  ({len(frames)} frames, "
      f"{os.path.getsize(GIF_PATH) / 1e6:.1f} MB)")
print(f"Wrote {PNG_PATH}")
