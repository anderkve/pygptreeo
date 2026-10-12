"""pyGPTreeO on the gg -> HH two-loop virtual amplitude grid.

The NLO gg -> HH calculation with full top-mass dependence (Heinrich et al.,
arXiv:1703.09252) evaluates the two-loop virtual amplitude numerically at 6320
phase-space points, each with its own quasi-Monte-Carlo integration error, and
the POWHEG-BOX ggHH code interpolates them: the points are mapped to a
uniformising variable x = cdf(beta), averaged onto a regular 101 x 31 grid by
weighted local linear fits, and interpolated with Clough-Tocher
(github.com/mppmu/hhgrid, share/creategrid.py). This script runs pyGPTreeO on the
same public points, learning log V with the quoted error of each point as its
observation noise, against a faithful port of that grid pipeline:

1. closure test: learn from a random 80 %, predict the held-out 20 %;
2. accuracy against the number of points, grid and tree built on the same
   growing subset, the tree streamed in the order of the phase-space sample;
3. acquisition: the tree chooses which of the remaining points to "buy" from
   its own sigma, against the random order;
4. maps of the tree's prediction and sigma, with the ttbar threshold.

Usage: python hhgrid_amplitude_demo.py [--hybrid] [--seeds 1,2,3] [--quick]
The grid files are downloaded once to examples/data/hhgrid/.
"""
import argparse
import io
import contextlib
import os
import sys
import time
import urllib.request
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from scipy.interpolate import CloughTocher2DInterpolator, splev, splrep

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pygptreeo import GPTree

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "hhgrid")
OUT_DIR = os.path.join(HERE, "results", "hhgrid_amplitude")
RAW = "https://raw.githubusercontent.com/mppmu/hhgrid/master/share/grids_sm/"
FILES = ("Virt_full.grid", "events.cdf")
MT, MH = 173.0, 125.0
BETA_THRESHOLD = np.sqrt(1.0 - (MH / MT) ** 2)   # s = 4 mt^2
# Categorical colours (fixed order): tree, POWHEG grid, tree with acquisition, raw Clough-Tocher
COLORS = {"tree": "#2a78d6", "grid": "#eb6834", "acq": "#1baf7a", "ct": "#eda100", "hybrid": "#6250d6"}


def fetch_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    for f in FILES:
        path = os.path.join(DATA_DIR, f)
        if not os.path.exists(path):
            print(f"downloading {f}")
            urllib.request.urlretrieve(RAW + f, path)
    b, cth, V, E = np.loadtxt(os.path.join(DATA_DIR, "Virt_full.grid"), unpack=True)
    cdf = np.loadtxt(os.path.join(DATA_DIR, "events.cdf"))
    return b, cth, V, E, cdf


class PowhegGrid:
    """Port of the ggHH grid of share/creategrid.py: cdf transform of beta, 101 x 31
    nodes filled by weighted linear fits of the 9 nearest points plus the points in
    the node's cell, Clough-Tocher interpolation of the node values."""

    NBIN = (100, 30)
    NNEIGHBOUR = 9

    def __init__(self, b, cth, V, E, cdf):
        self.spl = splrep(cdf[:, 0], cdf[:, 1], k=1)
        x = splev(b, self.spl)
        pts = np.column_stack([x, cth])
        nb = np.array(self.NBIN)
        dx = 1.0 / nb
        nodes_k = np.array(list(np.ndindex(nb[0] + 1, nb[1] + 1)))
        nodes = nodes_k * dx
        # Weights as in creategrid.py: curve_fit is given sigma = e^2, so each point
        # enters the linear fit with weight 1 / e^4.
        w = 1.0 / np.clip(E ** 2, 1e-20, 1e20) ** 2
        values = np.empty(len(nodes))
        for n, (k, x0) in enumerate(zip(nodes_k, nodes)):
            d2 = np.sum((pts - x0) ** 2, axis=1)
            order = np.argsort(d2)
            near = order[: self.NNEIGHBOUR]
            rest = order[self.NNEIGHBOUR:]
            shift = (k == nb).astype(float)        # last node along an axis: the cell below it
            lo = x0 - shift * dx
            hi = x0 + (1 - shift) * dx
            inside = np.all((pts[rest] > lo) & (pts[rest] < hi), axis=1)
            sel = np.concatenate([near, rest[inside]])
            A = np.column_stack([np.ones(len(sel)), pts[sel] - x0])
            sw = np.sqrt(w[sel])
            coef, *_ = np.linalg.lstsq(A * sw[:, None], V[sel] * sw, rcond=None)
            values[n] = coef[0]
        self.interp = CloughTocher2DInterpolator(nodes, values, fill_value=0.0)

    def __call__(self, b, cth):
        return self.interp(np.column_stack([splev(b, self.spl), cth]))


def metrics(pred_log, y_log, rel_err, sigma=None):
    """Accuracy of log-amplitude predictions against noisy held-out values."""
    dev = 100.0 * np.abs(np.exp(pred_log - y_log) - 1.0)
    r = pred_log - y_log
    # chi2: mean squared deviation in units of the QMC error; one for an exact surrogate
    out = dict(median_dev=np.median(dev), frac_gt5=100.0 * np.mean(dev > 5.0),
               within1=100.0 * np.mean(np.abs(r) < rel_err), within2=100.0 * np.mean(np.abs(r) < 2 * rel_err),
               chi2=np.mean((r / rel_err) ** 2))
    if sigma is not None:
        pull = r / np.sqrt(sigma ** 2 + rel_err ** 2)
        out.update(cov68=np.mean(np.abs(pull) < 1), cov95=np.mean(np.abs(pull) < 2),
                   median_sigma=100.0 * np.median(sigma))
    return out


def make_tree(hybrid, seed):
    with contextlib.redirect_stdout(io.StringIO()):
        if hybrid:
            return GPTree(Nbar=100, global_mean='net', global_mean_kwargs=dict(random_state=seed))
        return GPTree(Nbar=100)


def stream(gpt, X, y, sig, idx):
    with contextlib.redirect_stdout(io.StringIO()):
        for i in idx:
            gpt.update_tree(X[i:i + 1], y[i:i + 1].reshape(1, 1), sig[i])


def predict(gpt, X):
    with contextlib.redirect_stdout(io.StringIO()):
        mu, sd = gpt.predict(X, mode='loop')
    return mu.ravel(), sd.ravel()


def fmt_row(name, m, t=None):
    s = (f"| {name} | {m['median_dev']:.2f} % | {m['frac_gt5']:.1f} % | {m['within1']:.0f} % / {m['within2']:.0f} % | "
         f"{m['chi2']:.2f} |")
    if 'cov68' in m:
        s += f" {m['median_sigma']:.2f} % | {m['cov68']:.2f} / {m['cov95']:.2f} |"
    else:
        s += " – | – |"
    return s + (f" {t:.1f} s |" if t is not None else " – |")


def closure(X, y, sig, V, E, b, cth, cdf, seeds, hybrid):
    print("\n## Closure test: 80 % learned, 20 % held out\n")
    print("| method | median |dev| | off by > 5 % | within 1 / 2 QMC errors | chi2 | median sigma | coverage 68 / 95 | time |")
    print("|---|---|---|---|---|---|---|---|")
    acc = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(V)); ntr = int(0.8 * len(V)); tr, te = perm[:ntr], perm[ntr:]
        rel = E[te] / V[te]
        rows = []
        t0 = time.time(); grid = PowhegGrid(b[tr], cth[tr], V[tr], E[tr], cdf)
        g = grid(b[te], cth[te]); ok = g > 0
        rows.append(("POWHEG grid (cdf, 101x31 nodes, Clough-Tocher)",
                     metrics(np.log(np.where(ok, g, 1e-300)), y[te], rel), time.time() - t0))
        t0 = time.time(); ct = CloughTocher2DInterpolator(X[tr], y[tr])
        p = ct(X[te]); okc = np.isfinite(p)
        rows.append(("Clough-Tocher on the raw points", metrics(p[okc], y[te][okc], rel[okc]), time.time() - t0))
        t0 = time.time(); gpt = make_tree(False, seed); stream(gpt, X, y, sig, tr); t_fit = time.time() - t0
        mu, sd = predict(gpt, X[te])
        rows.append(("GPTree, Nbar = 100", metrics(mu, y[te], rel, sd), t_fit))
        if hybrid:
            t0 = time.time(); gpt = make_tree(True, seed); stream(gpt, X, y, sig, tr); t_fit = time.time() - t0
            mu, sd = predict(gpt, X[te])
            rows.append(("hybrid, global_mean='net'", metrics(mu, y[te], rel, sd), t_fit))
        for name, m, t in rows:
            acc.setdefault(name, []).append((m, t))
    for name, lst in acc.items():
        m = {k: np.mean([d[k] for d, _ in lst]) for k in lst[0][0]}
        print(fmt_row(name, m, np.mean([t for _, t in lst])))
    print(f"\n{len(seeds)} seed(s), {ntr} points learned, {len(V) - ntr} held out; "
          f"held-out QMC error median {100 * np.median(E / V):.1f} %.")


def accuracy_vs_points(X, y, sig, V, E, b, cth, cdf, seed, hybrid, checkpoints):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(V)); ntr = int(0.8 * len(V)); tr, te = perm[:ntr], perm[ntr:]
    rel = E[te] / V[te]
    res = {"N": [], "grid": [], "tree": [], "acq": []}
    # Random order (the phase-space sample's own order): tree streamed, grid rebuilt
    gpt = make_tree(hybrid, seed)
    done = 0
    for N in checkpoints:
        stream(gpt, X, y, sig, tr[done:N]); done = N
        mu, sd = predict(gpt, X[te])
        res["N"].append(N); res["tree"].append(metrics(mu, y[te], rel, sd))
        g = PowhegGrid(b[tr[:N]], cth[tr[:N]], V[tr[:N]], E[tr[:N]], cdf)(b[te], cth[te])
        res["grid"].append(metrics(np.log(np.where(g > 0, g, 1e-300)), y[te], rel))
    # Acquisition: the tree buys the pool points with the largest sigma, in batches
    gpt = make_tree(hybrid, seed)
    pool = list(tr[checkpoints[0]:]); bought = list(tr[:checkpoints[0]])
    stream(gpt, X, y, sig, bought)
    batch = 50
    for N in checkpoints:
        while len(bought) < N and pool:
            _, sd_pool = predict(gpt, X[pool])
            take = np.argsort(sd_pool)[::-1][:min(batch, N - len(bought))]
            chosen = [pool[i] for i in take]
            stream(gpt, X, y, sig, chosen)
            bought += chosen
            pool = [p for p in pool if p not in set(chosen)]
        mu, sd = predict(gpt, X[te])
        res["acq"].append(metrics(mu, y[te], rel, sd))
    return res, (tr, te)


def plot_curves(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    series = [("tree", "GPTree, random order", COLORS["tree"]), ("acq", "GPTree, sigma-driven acquisition", COLORS["acq"]),
              ("grid", "POWHEG grid", COLORS["grid"])]
    for ax, key, label in zip(axes, ["median_dev", "frac_gt5", "chi2"],
                              ["median |dev| of held-out points [%]", "held-out points off by > 5 % [%]",
                               "mean squared deviation / QMC error$^2$"]):
        for k, name, c in series:
            ax.plot(res["N"], [m[key] for m in res[k]], "-o", color=c, lw=2, ms=5, label=name)
        ax.set_xscale("log"); ax.set_xlabel("number of two-loop points used"); ax.set_ylabel(label)
        ax.set_xticks(res["N"]); ax.set_xticklabels([str(n) for n in res["N"]], fontsize=8); ax.minorticks_off()
        ax.grid(alpha=0.25)
    axes[0].set_yscale("log"); axes[1].set_yscale("log"); axes[2].set_yscale("log")
    axes[2].axhline(1.0, color="#52514e", lw=1, ls=":")
    axes[2].text(res["N"][0], 1.05, "exact surrogate", color="#52514e", fontsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_maps(gpt, b, cth, V, E, te, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bb, cc = np.meshgrid(np.linspace(b.min(), b.max(), 220), np.linspace(0, 1, 110))
    G = np.column_stack([bb.ravel(), cc.ravel()])
    mu, sd = predict(gpt, G)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    im0 = axes[0].scatter(b, cth, c=100 * E / V, s=3, cmap="Blues", norm=matplotlib.colors.LogNorm(0.1, 20))
    axes[0].set_title("the 6320 two-loop points, colour = QMC error [%]", fontsize=9)
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].pcolormesh(bb, cc, mu.reshape(bb.shape), cmap="Blues", shading="auto")
    axes[1].set_title("GPTree prediction of log V", fontsize=9); fig.colorbar(im1, ax=axes[1])
    im2 = axes[2].pcolormesh(bb, cc, 100 * sd.reshape(bb.shape), cmap="Blues", shading="auto",
                             norm=matplotlib.colors.LogNorm(0.1, 20))
    axes[2].set_title("GPTree calibrated sigma [% of V]", fontsize=9); fig.colorbar(im2, ax=axes[2])
    for ax in axes:
        ax.axvline(BETA_THRESHOLD, color="#52514e", lw=1, ls="--")
        ax.set_xlabel(r"$\beta_s$"); ax.set_ylabel(r"$|\cos\theta|$")
    axes[0].text(BETA_THRESHOLD + 0.01, 0.03, r"$\hat s = 4m_t^2$", color="#52514e", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_pulls(gpt, X, y, V, E, te, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mu, sd = predict(gpt, X[te]); rel = E[te] / V[te]
    pull = (mu - y[te]) / np.sqrt(sd ** 2 + rel ** 2)
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    ax.hist(np.clip(pull, -5, 5), bins=40, range=(-5, 5), density=True, color=COLORS["tree"], alpha=0.85,
            label="held-out pulls")
    z = np.linspace(-5, 5, 200)
    ax.plot(z, np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi), color="#52514e", lw=1.5, label="standard normal")
    ax.set_xlabel(r"(prediction $-$ QMC value) / $\sqrt{\sigma_{\rm tree}^2 + \sigma_{\rm QMC}^2}$")
    ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return pull


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid", action="store_true", help="also run GPTree(global_mean='net') (needs PyTorch)")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--quick", action="store_true", help="one seed, no acquisition")
    args = ap.parse_args()
    seeds = [1] if args.quick else [int(s) for s in args.seeds.split(",")]
    os.makedirs(OUT_DIR, exist_ok=True)

    b, cth, V, E, cdf = fetch_data()
    X = np.column_stack([b, cth]); y = np.log(V); sig = E / V
    print(f"# gg -> HH two-loop virtual grid: {len(V)} points, beta in [{b.min():.3f}, {b.max():.3f}], "
          f"QMC error median {100 * np.median(sig):.2f} % (10-90 %: {100 * np.quantile(sig, .1):.2f}-{100 * np.quantile(sig, .9):.2f} %)")

    closure(X, y, sig, V, E, b, cth, cdf, seeds, args.hybrid)

    checkpoints = [250, 500, 1000, 2000, 4000, int(0.8 * len(V))]
    if not args.quick:
        print("\n## Accuracy against the number of two-loop points (seed 1, held-out 20 %)\n")
        res, (tr, te) = accuracy_vs_points(X, y, sig, V, E, b, cth, cdf, seeds[0], False, checkpoints)
        print("| points | grid: median |dev| / > 5 % / chi2 | tree, random order | tree, acquisition |")
        print("|---|---|---|---|")
        for i, N in enumerate(res["N"]):
            cells = [f"{res[k][i]['median_dev']:.2f} % / {res[k][i]['frac_gt5']:.1f} % / {res[k][i]['chi2']:.2f}"
                     for k in ("grid", "tree", "acq")]
            print(f"| {N} | " + " | ".join(cells) + " |")
        plot_curves(res, os.path.join(OUT_DIR, "accuracy_vs_points.png"))
        np.save(os.path.join(OUT_DIR, "accuracy_vs_points.npy"), res, allow_pickle=True)

    # Maps and pulls from a tree on the 80 % of seed 1
    rng = np.random.RandomState(seeds[0]); perm = rng.permutation(len(V)); ntr = int(0.8 * len(V))
    tr, te = perm[:ntr], perm[ntr:]
    gpt = make_tree(False, seeds[0]); stream(gpt, X, y, sig, tr)
    plot_maps(gpt, b, cth, V, E, te, os.path.join(OUT_DIR, "maps.png"))
    pull = plot_pulls(gpt, X, y, V, E, te, os.path.join(OUT_DIR, "pulls.png"))
    print(f"\nPulls of the held-out points: mean {pull.mean():.2f}, std {pull.std():.2f}, "
          f"|pull| < 1: {np.mean(np.abs(pull) < 1):.2f}, < 2: {np.mean(np.abs(pull) < 2):.2f}; "
          f"leaves {len(gpt.root.leaves)}.")
    print(f"Figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
