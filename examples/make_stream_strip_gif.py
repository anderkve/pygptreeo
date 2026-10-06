"""Generate the "stream strip chart" GIF: pyGPTreeO on a target of many inputs over a long stream.

The README's first animation shows a 2D target. This one shows a target of ten
inputs learned from tens of thousands of points, where there is no picture of the
prediction to draw. Instead the stream itself is the x axis (logarithmic, from
100 points to the end) and three strips are stacked on it:

  1. The input stream: one row per input coordinate, each column the points of
     one bin of the stream (a single point at the left, hundreds at the right).
     On the walker stream the coordinates drift as the Metropolis walker explores
     the target.
  2. The plain GP tree's relative error, |prediction - truth| / |truth|, of every
     point predicted *before* it is given to the tree: per column, the distribution
     of the errors of the points in that bin on a log scale from 0.01% to 100%
     (each column scaled to its peak), with the running median as a line.
  3. The same for the hybrid, GPTree(global_mean='net'): GP leaves on the
     residual of a tree-wide feature network, with the network's refits marked.

Usage:
    OMP_NUM_THREADS=1 python make_stream_strip_gif.py --run tree     # stream the points through the plain tree
    OMP_NUM_THREADS=1 python make_stream_strip_gif.py --run hybrid   # ... and through the hybrid (needs PyTorch)
    python make_stream_strip_gif.py --plot                            # render the GIF from the two records
    python make_stream_strip_gif.py --quick                           # small end-to-end preview
    python make_stream_strip_gif.py                                   # run whatever record is missing, then plot

Output: examples/example_plots/animation/pygptreeo_stream_strip_<target>_d<d>_<stream>.gif
(+ <same>_final.png); the per-point records go to examples/results/stream_strip/.
"""

import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

TARGET_LABELS = {'active_peaks': 'ActiveSubspacePeaks', 'gaussian_peaks': 'GaussianPeaks',
                 'rotated_rosenbrock': 'RotatedRosenbrock', 'step_ridge': 'StepRidge',
                 'michalewicz': 'Michalewicz', 'ackley': 'Ackley', 'griewank': 'Griewank',
                 'chirp': 'Chirp', 'eggholder': 'Eggholder'}
STREAM_LABELS = {'walker': 'Metropolis-walker stream', 'uniform': 'uniform stream',
                 'focusing': 'focusing stream', 'sweeping': 'sweeping stream'}

parser = argparse.ArgumentParser()
parser.add_argument('--run', choices=['tree', 'hybrid', 'both'],
                    help="Stream the points through this model and store the per-point record.")
parser.add_argument('--plot', action='store_true', help="Render the GIF from the stored records.")
parser.add_argument('--quick', action='store_true',
                    help="Small end-to-end preview: 3000 points, fewer frames, low resolution.")
parser.add_argument('--final-only', action='store_true',
                    help="Render only the final frame (the PNG, no GIF).")
parser.add_argument('--target', default='active_peaks', choices=sorted(TARGET_LABELS))
parser.add_argument('--d', type=int, default=10)
parser.add_argument('--stream', default='walker', choices=sorted(STREAM_LABELS))
parser.add_argument('--N', type=int, default=40000)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()
if args.quick:
    args.N = min(args.N, 3000)

RESULTS_DIR = os.path.join(HERE, 'results', 'stream_strip')
OUT_DIR = os.path.join(HERE, 'example_plots', 'animation')
TAG = f"{args.target}_d{args.d}_{args.stream}"
BASENAME = f"pygptreeo_stream_strip_{TAG}" + ("_quick" if args.quick else "")


def record_path(config):
    return os.path.join(RESULTS_DIR, f"{TAG}_N{args.N}_seed{args.seed}_{config}.npz")


# --------------------------------------------------------------------------
# Run phase: every point is predicted before it is given to the tree
# --------------------------------------------------------------------------

def make_data():
    import target_functions as tf
    from benchmark_global_mean_streams import make_stream
    targets = {'active_peaks': tf.ActiveSubspacePeaks, 'gaussian_peaks': tf.GaussianPeaks,
               'rotated_rosenbrock': tf.RotatedRosenbrock, 'step_ridge': tf.StepRidge,
               'michalewicz': tf.Michalewicz, 'ackley': tf.Ackley, 'griewank': tf.Griewank,
               'chirp': tf.Chirp, 'eggholder': tf.Eggholder}
    target = targets[args.target]
    rng = np.random.RandomState(args.seed)
    X, _ = make_stream(args.stream, target, args.d, args.N, rng)
    y = target(X.T)
    sigma = np.maximum(1e-3 * np.abs(y), 1e-6)     # observation-noise std passed with each point
    return X, y, sigma


def run(config):
    """Stream the points through the plain tree or the hybrid; the benchmark settings."""
    from pygptreeo import GPTree, Default_GPR
    X, y, sigma = make_data()
    np.random.seed(args.seed)
    common = dict(Nbar=100, theta=1e-4, retrain_every_n_points=25, splitting_strategy='gradual',
                  use_calibrated_sigma=True)
    if config == 'tree':
        gpt = GPTree(GPR=Default_GPR(n_restarts_optimizer=1), **common)
    else:
        from pygptreeo import NetGlobalMean
        gpt = GPTree(GPR=Default_GPR(n_restarts_optimizer=1),
                     global_mean=NetGlobalMean(steps_per_update=1, random_state=args.seed), **common)
    N = len(y)
    mu, sd = np.empty(N), np.empty(N)
    t_pred, t_upd = np.empty(N), np.empty(N)
    n_leaves, version = np.empty(N, dtype=int), np.zeros(N, dtype=int)
    t_start = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(N):
            xi = X[i:i + 1]
            t0 = time.perf_counter()
            m, s = gpt.predict(xi)
            t1 = time.perf_counter()
            gpt.update_tree(xi, np.array([[y[i]]]), np.array([[sigma[i]]]))
            t2 = time.perf_counter()
            mu[i], sd[i], t_pred[i], t_upd[i] = m[0, 0], s[0, 0], t1 - t0, t2 - t1
            n_leaves[i] = len(gpt.root.leaves)
            if config == 'hybrid':
                version[i] = gpt.global_mean.learner.version
            if (i + 1) % 2000 == 0:
                print(f"{config}: {i + 1}/{N} points, {time.time() - t_start:.0f} s, "
                      f"{n_leaves[i]} leaves", file=sys.__stdout__, flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez_compressed(record_path(config), X=X.astype(np.float32), y=y, mu=mu, sd=sd,
                        t_pred=t_pred, t_upd=t_upd, n_leaves=n_leaves, version=version)
    print(f"wrote {record_path(config)} in {time.time() - t_start:.0f} s")


# --------------------------------------------------------------------------
# Plot phase
# --------------------------------------------------------------------------

N_START = 100            # the x axis (points seen) runs from here to N, logarithmically
N_COLS = 600             # columns of the strips (log-spaced bins of the stream; fewer at the left)
REL_LOG_MIN, REL_LOG_MAX = -4.0, 0.0     # relative error from 0.01% to 100%
ROWS_PER_DECADE = 10
WINDOW = 1.25            # the running median covers the points in (n / WINDOW, n]
FPS = 12
END_PAUSE_MS = 2500
N_FRAMES = 24 if args.quick else 64
DPI = 50 if args.quick else 100

# One sequential ramp per strip; each error strip's median line in a mid step of its own hue.
CMAP_X, CMAP_TREE, CMAP_HYBRID = 'Greys', 'Blues', 'Oranges'
COL_TREE, COL_HYBRID = '#2171b5', '#d94801'
INK, INK_MUTED = '#222222', '#777777'


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib import patheffects
    from PIL import Image

    rec = {c: np.load(record_path(c)) for c in ('tree', 'hybrid')}
    X, y = rec['tree']['X'], rec['tree']['y']
    N, d = X.shape
    n = np.arange(1, N + 1)                                   # points seen when point i arrives
    rel = {c: np.abs(rec[c]['mu'] - y) / np.abs(y) for c in rec}
    v = rec['hybrid']['version']
    refits = n[np.flatnonzero(np.diff(v) > 0) + 1]            # points seen when a network version was published

    # Column edges: log-spaced in n, rounded to whole points, so the leftmost columns hold one point each.
    edges = np.unique(np.round(np.geomspace(N_START, N, N_COLS + 1)).astype(int))
    n_cols = len(edges) - 1
    col = np.clip(np.searchsorted(edges, n, side='right') - 1, -1, n_cols - 1)
    col[n < N_START] = -1
    n_rows = int(ROWS_PER_DECADE * (REL_LOG_MAX - REL_LOG_MIN))
    row_edges = 10.0 ** np.linspace(REL_LOG_MIN, REL_LOG_MAX, n_rows + 1)
    row = {c: np.clip(((np.log10(np.maximum(rel[c], 1e-300)) - REL_LOG_MIN) * ROWS_PER_DECADE).astype(int),
                      0, n_rows - 1) for c in rel}

    def coordinate_strip(n_seen):
        m = (col >= 0) & (n <= n_seen)
        counts = np.bincount(col[m], minlength=n_cols).astype(float)
        C = np.array([np.bincount(col[m], weights=X[m, k], minlength=n_cols) for k in range(d)])
        with np.errstate(invalid='ignore', divide='ignore'):
            C = C / counts
        return np.ma.masked_invalid(C)

    def error_density(c, n_seen):
        m = (col >= 0) & (n <= n_seen)
        D = np.bincount(row[c][m] * n_cols + col[m], minlength=n_rows * n_cols).reshape(n_rows, n_cols).astype(float)
        peak = D.max(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            D = D / peak
        return np.ma.masked_invalid(D)

    def running_median(c, n_seen):
        xs = np.minimum(edges[1:], n_seen)
        xs = np.unique(xs[edges[:-1] <= n_seen])
        med = [np.median(rel[c][(n > x / WINDOW) & (n <= x)]) for x in xs]
        return xs, np.array(med)

    # ---- figure scaffolding ----
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14})
    fig = plt.figure(figsize=(16.0, 8.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[10, 14, 14], hspace=0.2,
                          left=0.07, right=0.945, top=0.84, bottom=0.085)
    axX = fig.add_subplot(gs[0])
    axT = fig.add_subplot(gs[1], sharex=axX)
    axH = fig.add_subplot(gs[2], sharex=axX)

    for ax in (axX, axT, axH):
        ax.set_xscale('log')
        ax.set_xlim(N_START, N)
        for s in ax.spines.values():
            s.set_color(INK_MUTED); s.set_linewidth(0.8)
        ax.tick_params(colors=INK, length=3)
    ticks = [t for t in (100, 1000, 10000, 100000) if N_START <= t <= N]
    if N not in ticks:
        ticks.append(N)
    axH.set_xticks(ticks)
    axH.set_xticklabels([f"{t:,}".replace(",", "\u2009") for t in ticks])
    axH.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axH.set_xlabel("points seen (log scale)")
    for ax in (axX, axT):
        plt.setp(ax.get_xticklabels(), visible=False)

    # Strip 1: the input stream, one row per coordinate, x_1 on top
    cmap_x = plt.get_cmap(CMAP_X, 16)
    axX.set_ylim(d + 0.5, 0.5)
    axX.set_yticks(np.arange(1, d + 1))
    axX.set_yticklabels([rf"$x_{{{k}}}$" for k in range(1, d + 1)], fontsize=10)
    axX.set_title(f"Input stream: the {d} coordinates of each point", loc='left')
    cax = fig.add_axes([0.953, axX.get_position().y0, 0.009, axX.get_position().height])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap=cmap_x), cax=cax)
    cb.set_ticks([0, 0.5, 1]); cb.ax.tick_params(labelsize=10)

    # Strips 2 and 3: the relative error of each point, predicted before it was added
    cmaps = {'tree': plt.get_cmap(CMAP_TREE, 16), 'hybrid': plt.get_cmap(CMAP_HYBRID, 16)}
    line_col = {'tree': COL_TREE, 'hybrid': COL_HYBRID}
    axes = {'tree': axT, 'hybrid': axH}
    titles = {'tree': "Plain GP tree: relative error of each point, predicted before the point is added"
                      "   (line: running median)",
              'hybrid': "Hybrid, GP tree + feature network: relative error of the same points"
                        "   (line: running median, vertical lines: network refits)"}
    for c, ax in axes.items():
        ax.set_yscale('log')
        ax.set_ylim(10 ** REL_LOG_MIN, 10 ** REL_LOG_MAX)
        ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
        ax.set_yticklabels(["0.01%", "0.1%", "1%", "10%", "100%"])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(axis='y', which='major', color=INK_MUTED, alpha=0.25, linewidth=0.6)
        ax.set_title(titles[c], loc='left')
    for ax, label in ((axT, "relative error"), (axH, "relative error")):
        ax.set_ylabel(label)

    suptitle = fig.suptitle("", fontsize=16, x=0.5, y=0.99, va="top", linespacing=1.3)
    target_label, stream_label = TARGET_LABELS[args.target], STREAM_LABELS[args.stream]

    artists = []
    halo = [patheffects.withStroke(linewidth=3.5, foreground='white')]

    def render_frame(n_seen):
        for a in artists:
            a.remove()
        artists.clear()
        artists.append(axX.pcolormesh(edges, np.arange(d + 1) + 0.5, coordinate_strip(n_seen),
                                      cmap=cmap_x, vmin=0, vmax=1, shading='flat', rasterized=True))
        for c, ax in axes.items():
            artists.append(ax.pcolormesh(edges, row_edges, error_density(c, n_seen), cmap=cmaps[c],
                                         vmin=0, vmax=1, shading='flat', rasterized=True))
            xs, med = running_median(c, n_seen)
            artists.extend(ax.plot(xs, med, color=line_col[c], lw=2.0, zorder=4, path_effects=halo))
            artists.append(ax.annotate(f"{100 * med[-1]:.2g}%", (xs[-1], med[-1]), xytext=(6, 0),
                                       textcoords='offset points', va='center', fontsize=11,
                                       color=INK, zorder=5, path_effects=halo))
        for r in refits[refits <= n_seen]:
            artists.append(axH.axvline(r, color=INK_MUTED, lw=1.0, zorder=3))
        i = n_seen - 1
        leaves_t, leaves_h = rec['tree']['n_leaves'][i], rec['hybrid']['n_leaves'][i]
        n_txt = f"{n_seen:,}".replace(",", "\u2009")
        suptitle.set_text(
            f"pyGPTreeO learning a {d}-input target ({target_label}) from a {stream_label}\n"
            f"{n_txt} points seen   |   {leaves_t} / {leaves_h} local GPs (tree / hybrid)   |   "
            f"{int(np.sum(refits <= n_seen))} network refits")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        return Image.fromarray(buf[..., :3].copy())

    os.makedirs(OUT_DIR, exist_ok=True)
    gif_path = os.path.join(OUT_DIR, BASENAME + ".gif")
    png_path = os.path.join(OUT_DIR, BASENAME + "_final.png")
    fig.set_dpi(DPI)
    if args.final_only:
        render_frame(N).save(png_path)
        print(f"wrote {png_path} (final frame only; no GIF)")
        return
    frame_ns = np.unique(np.round(np.geomspace(N_START, N, N_FRAMES)).astype(int))
    frames = []
    for k, n_seen in enumerate(frame_ns):
        frames.append(render_frame(int(n_seen)))
        print(f"frame {k + 1:3d}/{len(frame_ns)}  ({n_seen} points)", flush=True)
    frames[-1].save(png_path)
    durations = [int(1000 / FPS)] * len(frames)
    durations[-1] = END_PAUSE_MS
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"wrote {gif_path}  ({len(frames)} frames, {os.path.getsize(gif_path) / 1e6:.1f} MB)")
    print(f"wrote {png_path}")


if __name__ == '__main__':
    configs = {'tree': ['tree'], 'hybrid': ['hybrid'], 'both': ['tree', 'hybrid'], None: []}[args.run]
    if args.run is None and not args.plot:          # default: run what is missing, then plot
        configs = [c for c in ('tree', 'hybrid') if not os.path.exists(record_path(c))]
        args.plot = True
    for c in configs:
        run(c)
    if args.plot:
        plot()
