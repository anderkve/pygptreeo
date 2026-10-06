"""Generate the "stream strip chart" GIF: pyGPTreeO on a target of many inputs over a long stream.

The README's first animation shows a 2D target. This one shows a target of ten
inputs learned from tens of thousands of points, where there is no picture of the
prediction to draw. Instead the stream itself is the x axis (logarithmic, from
100 points to the end) and three panels are stacked on it:

  1. The input stream: one curve per input coordinate (a running mean over at
     least ten points). On the walker stream the coordinates drift as the
     Metropolis walker explores the target.
  2. The function value at each point, and the model's prediction of it made
     *before* the point is given to the tree, as markers; a subsample of the
     points, spread evenly along the log axis, so the markers stay readable.
  3. The relative error, |prediction - truth| / |truth|, of every point predicted
     before it is added: per column, the distribution of the errors of the points
     in that bin of the stream on a log scale from 0.01% to 100% (each column
     scaled to its peak), with the running median as a line and, for the hybrid,
     the network's refits as vertical lines.

The model is the hybrid, GPTree(global_mean='net') (GP leaves on the residual of a
tree-wide feature network), or with ``--config tree`` the plain GP tree.

Usage:
    OMP_NUM_THREADS=1 python make_stream_strip_gif.py --run      # stream the points through the model (the hybrid needs PyTorch)
    python make_stream_strip_gif.py --plot                        # render the GIF from the stored record
    python make_stream_strip_gif.py --quick                       # small end-to-end preview
    python make_stream_strip_gif.py                               # run if the record is missing, then plot

Output: examples/example_plots/animation/pygptreeo_stream_strip_<target>_d<d>_<stream>[_tree].gif
(+ <same>_final.png); the per-point record goes to examples/results/stream_strip/.
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
MODEL_LABELS = {'hybrid': 'hybrid, GP tree + feature network', 'tree': 'plain GP tree'}

parser = argparse.ArgumentParser()
parser.add_argument('--run', action='store_true',
                    help="Stream the points through the model and store the per-point record.")
parser.add_argument('--plot', action='store_true', help="Render the GIF from the stored record.")
parser.add_argument('--quick', action='store_true',
                    help="Small end-to-end preview: 3000 points, fewer frames, low resolution.")
parser.add_argument('--final-only', action='store_true',
                    help="Render only the final frame (the PNG, no GIF).")
parser.add_argument('--config', default='hybrid', choices=sorted(MODEL_LABELS),
                    help="The model: the hybrid (default) or the plain GP tree.")
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
RECORD_PATH = os.path.join(RESULTS_DIR, f"{TAG}_N{args.N}_seed{args.seed}_{args.config}.npz")
BASENAME = (f"pygptreeo_stream_strip_{TAG}" + ("" if args.config == 'hybrid' else f"_{args.config}")
            + ("_quick" if args.quick else ""))


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


def run():
    """Stream the points through the model with the benchmark settings; store the per-point record."""
    from pygptreeo import GPTree, Default_GPR
    X, y, sigma = make_data()
    np.random.seed(args.seed)
    common = dict(Nbar=100, theta=1e-4, retrain_every_n_points=25, splitting_strategy='gradual',
                  use_calibrated_sigma=True)
    if args.config == 'tree':
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
            if args.config == 'hybrid':
                version[i] = gpt.global_mean.learner.version
            if (i + 1) % 2000 == 0:
                print(f"{args.config}: {i + 1}/{N} points, {time.time() - t_start:.0f} s, "
                      f"{n_leaves[i]} leaves", file=sys.__stdout__, flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez_compressed(RECORD_PATH, X=X.astype(np.float32), y=y, mu=mu, sd=sd,
                        t_pred=t_pred, t_upd=t_upd, n_leaves=n_leaves, version=version)
    print(f"wrote {RECORD_PATH} in {time.time() - t_start:.0f} s")


# --------------------------------------------------------------------------
# Plot phase
# --------------------------------------------------------------------------

N_START = 100            # the x axis (points seen) runs from here to N, logarithmically
N_COLS = 600             # columns of the error strip (log-spaced bins of the stream; fewer at the left)
MIN_WINDOW = 10          # a coordinate curve averages at least this many points
MARKERS_PER_COL = 3      # the function-value panel shows at most this many points per column
REL_LOG_MIN, REL_LOG_MAX = -4.0, 0.0     # relative error from 0.01% to 100%
ROWS_PER_DECADE = 10
WINDOW = 1.25            # the running median covers the points in (n / WINDOW, n]
FPS = 12
END_PAUSE_MS = 2500
N_FRAMES = 24 if args.quick else 64
DPI = 50 if args.quick else 100

# The model's hue: a sequential ramp for the error strip, a mid step of it for its lines and markers.
CMAP = {'hybrid': 'Oranges', 'tree': 'Blues'}[args.config]
COL_MODEL = {'hybrid': '#d94801', 'tree': '#2171b5'}[args.config]
INK, INK_MUTED = '#222222', '#777777'


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from PIL import Image

    rec = np.load(RECORD_PATH)
    X, y, mu = rec['X'].astype(float), rec['y'], rec['mu']
    N, d = X.shape
    n = np.arange(1, N + 1)                                   # points seen when point i arrives
    rel = np.abs(mu - y) / np.abs(y)
    v = rec['version']
    refits = n[np.flatnonzero(np.diff(v) > 0) + 1]            # points seen when a network version was published

    # Column edges: log-spaced in n, rounded to whole points, so the leftmost columns hold one point each.
    edges = np.unique(np.round(np.geomspace(N_START, N, N_COLS + 1)).astype(int))
    n_cols = len(edges) - 1
    col = np.clip(np.searchsorted(edges, n, side='right') - 1, -1, n_cols - 1)
    col[n < N_START] = -1
    n_rows = int(ROWS_PER_DECADE * (REL_LOG_MAX - REL_LOG_MIN))
    row_edges = 10.0 ** np.linspace(REL_LOG_MIN, REL_LOG_MAX, n_rows + 1)
    row = np.clip(((np.log10(np.maximum(rel, 1e-300)) - REL_LOG_MIN) * ROWS_PER_DECADE).astype(int),
                  0, n_rows - 1)
    cum_X = np.vstack([np.zeros((1, d)), np.cumsum(X, axis=0)])     # cum_X[b] - cum_X[a] sums points a..b-1

    # The subsample shown in the function-value panel: up to MARKERS_PER_COL points per column, evenly spaced.
    shown = np.zeros(N, dtype=bool)
    for j in range(n_cols):
        idx = np.flatnonzero(col == j)
        if len(idx):
            shown[idx[np.linspace(0, len(idx) - 1, min(len(idx), MARKERS_PER_COL)).round().astype(int)]] = True

    def coordinate_curves(n_seen):
        """Per column: the mean coordinates of the last max(MIN_WINDOW, column width) points up to its right edge."""
        b = np.minimum(edges[1:], n_seen)
        keep = edges[:-1] <= n_seen
        b = b[keep]
        a = np.maximum(np.minimum(edges[:-1][keep], b - MIN_WINDOW), 0)
        return b, (cum_X[b] - cum_X[a]) / (b - a)[:, None]

    def error_density(n_seen):
        m = (col >= 0) & (n <= n_seen)
        D = np.bincount(row[m] * n_cols + col[m], minlength=n_rows * n_cols).reshape(n_rows, n_cols).astype(float)
        with np.errstate(invalid='ignore', divide='ignore'):
            D = D / D.max(axis=0)
        return np.ma.masked_invalid(D)

    def running_median(n_seen):
        xs = np.unique(np.minimum(edges[1:], n_seen)[edges[:-1] <= n_seen])
        return xs, np.array([np.median(rel[(n > x / WINDOW) & (n <= x)]) for x in xs])

    # ---- figure scaffolding ----
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14})
    fig = plt.figure(figsize=(16.0, 8.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[10, 12, 12], hspace=0.24,
                          left=0.07, right=0.95, top=0.85, bottom=0.08)
    axX = fig.add_subplot(gs[0])
    axF = fig.add_subplot(gs[1], sharex=axX)
    axE = fig.add_subplot(gs[2], sharex=axX)

    for ax in (axX, axF, axE):
        ax.set_xscale('log')
        ax.set_xlim(N_START, N)
        for s in ax.spines.values():
            s.set_color(INK_MUTED); s.set_linewidth(0.8)
        ax.tick_params(colors=INK, length=3)
    ticks = [t for t in (100, 1000, 10000, 100000) if N_START <= t <= N]
    if N not in ticks:
        ticks.append(N)
    axE.set_xticks(ticks)
    axE.set_xticklabels([f"{t:,}".replace(",", " ") for t in ticks])
    axE.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axE.set_xlabel("points seen (log scale)")
    for ax in (axX, axF):
        plt.setp(ax.get_xticklabels(), visible=False)
    legend_kw = dict(loc='lower right', bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=10.5,
                     borderaxespad=0.15, handlelength=1.6, columnspacing=1.0, handletextpad=0.5)

    # Panel 1: one curve per coordinate
    coord_colors = [plt.get_cmap('tab10')(k % 10) for k in range(d)]
    axX.set_ylim(0, 1)
    axX.set_yticks([0, 0.5, 1])
    axX.set_ylabel("coordinate")
    axX.grid(axis='y', which='major', color=INK_MUTED, alpha=0.25, linewidth=0.6)
    axX.set_title(f"Input stream: the {d} coordinates of each point", loc='left')
    axX.legend(handles=[matplotlib.lines.Line2D([], [], color=coord_colors[k], lw=1.5, label=rf"$x_{{{k + 1}}}$")
                        for k in range(d)], ncol=d, **legend_kw)

    # Panel 2: the function value at each point and the prediction made before the point was added
    pad = 0.03 * (y.max() - y.min())
    axF.set_ylim(y.min() - pad, y.max() + pad)
    axF.set_ylabel("function value")
    axF.grid(axis='y', which='major', color=INK_MUTED, alpha=0.25, linewidth=0.6)
    axF.set_title("Function value at each point: the truth, and the prediction made before the point is added"
                  "   (a subsample of the points)", loc='left')
    axF.legend(handles=[matplotlib.lines.Line2D([], [], color=INK, marker='o', ms=5, ls='', label="true value"),
                        matplotlib.lines.Line2D([], [], color=COL_MODEL, marker='o', ms=7, mfc='none', mew=1.2,
                                                ls='', label="prediction")], ncol=2, **legend_kw)

    # Panel 3: the relative error of each point, predicted before it was added
    cmap_err = plt.get_cmap(CMAP, 16)
    axE.set_yscale('log')
    axE.set_ylim(10 ** REL_LOG_MIN, 10 ** REL_LOG_MAX)
    axE.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    axE.set_yticklabels(["0.01%", "0.1%", "1%", "10%", "100%"])
    axE.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axE.set_ylabel("relative error")
    axE.grid(axis='y', which='major', color=INK_MUTED, alpha=0.25, linewidth=0.6)
    axE.set_title("Relative error of each point, predicted before the point is added   (line: running median"
                  + (", vertical lines: network refits)" if args.config == 'hybrid' else ")"), loc='left')

    suptitle = fig.suptitle("", fontsize=16, x=0.5, y=0.99, va="top", linespacing=1.3)
    target_label, stream_label = TARGET_LABELS[args.target], STREAM_LABELS[args.stream]
    model_label = MODEL_LABELS[args.config]
    halo = [patheffects.withStroke(linewidth=3.5, foreground='white')]
    artists = []

    # The curves and markers are drawn without anti-aliasing: it keeps the GIF less than half the size.
    def render_frame(n_seen):
        for a in artists:
            a.remove()
        artists.clear()
        xs, C = coordinate_curves(n_seen)
        for k in range(d):
            artists.extend(axX.plot(xs, C[:, k], color=coord_colors[k], lw=1.2, zorder=3, antialiased=False))
        m = shown & (n <= n_seen)
        artists.append(axF.scatter(n[m], y[m], s=14, color=INK, linewidths=0, zorder=3, antialiased=False))
        artists.append(axF.scatter(n[m], mu[m], s=34, facecolors='none', edgecolors=COL_MODEL, linewidths=1.0,
                                   zorder=4, antialiased=False))
        artists.append(axE.pcolormesh(edges, row_edges, error_density(n_seen), cmap=cmap_err,
                                      vmin=0, vmax=1, shading='flat', rasterized=True))
        xs, med = running_median(n_seen)
        artists.extend(axE.plot(xs, med, color=COL_MODEL, lw=2.0, zorder=4, path_effects=halo))
        artists.append(axE.annotate(f"{100 * med[-1]:.2g}%", (xs[-1], med[-1]), xytext=(6, 0),
                                    textcoords='offset points', va='center', fontsize=11,
                                    color=INK, zorder=5, path_effects=halo))
        for r in refits[refits <= n_seen]:
            artists.append(axE.axvline(r, color=INK_MUTED, lw=1.0, zorder=3))
        n_txt = f"{n_seen:,}".replace(",", " ")
        status = f"{n_txt} points seen   |   {rec['n_leaves'][n_seen - 1]} local GPs"
        if args.config == 'hybrid':
            status += f"   |   {int(np.sum(refits <= n_seen))} network refits"
        suptitle.set_text(f"pyGPTreeO ({model_label}) learning a {d}-input target ({target_label}) "
                          f"from a {stream_label}\n{status}")
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
    if not args.run and not args.plot:              # default: run if the record is missing, then plot
        args.run, args.plot = not os.path.exists(RECORD_PATH), True
    if args.run:
        run()
    if args.plot:
        plot()
