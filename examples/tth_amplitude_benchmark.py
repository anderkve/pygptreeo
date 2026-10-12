"""pyGPTreeO on 5D tree-level ttH amplitudes: the test functions f1 and f3 of
"Interpolating amplitudes" (Bresó-Pla, Heinrich, Magerya, Olsson, arXiv:2412.09534).

The amplitudes are |M|^2 for q qbar -> t tbar H (f1) and g g -> t tbar H (f3) on
the paper's five phase-space variables (beta^2, f_ttbar, theta_H, theta_t, phi_t),
folded by the amplitude's symmetries onto the unit cube, computed with MadGraph
(examples/tth_amplitude/README.md has the recipe). The paper's error is the
relative L1 distance eps = sum |f~ - f| / sum f of the weighted amplitude
f = w |M|^2 over uniform test points, and on LO-unweighted points the mean
relative error.

The plain tree and the hybrid (GP leaves on the residual of a tree-wide feature
network) are streamed over Sobol points, evaluated at checkpoints. Optionally the
training values carry a relative Gaussian noise (the NoisyEvaluator of the
paper's code uses 3e-3), passed to the tree as the per-point sigma.

Usage: python tth_amplitude_benchmark.py <data_dir> [--channels gg,uu] [--configs tree,hybrid,acq]
           [--nmax 131072] [--noise 0] [--seed 1]
``acq`` is the plain tree choosing which pool points to learn from (the largest
expected error contribution w |M|^2 sigma) instead of taking them in Sobol order.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pygptreeo import GPTree

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "tth_amplitude")
COLORS = {"tree": "#2a78d6", "hybrid": "#eb6834", "acq": "#1baf7a"}
CHANNEL_NAMES = {"uu": "f1: q qbar -> t tbar H", "gg": "f3: g g -> t tbar H"}


def load(data_dir, ch):
    tr = np.load(os.path.join(data_dir, f"{ch}_train_sobol.npz"))
    te = np.load(os.path.join(data_dir, f"{ch}_test_uniform.npz"))
    path_u = os.path.join(data_dir, f"{ch}_test_unweighted.npz")
    tu = np.load(path_u) if os.path.exists(path_u) else None
    return tr, te, tu


def make(config, seed):
    with contextlib.redirect_stdout(io.StringIO()):
        if config == "hybrid":
            return GPTree(Nbar=100, global_mean='net', global_mean_kwargs=dict(steps_per_update=8, random_state=seed))
        return GPTree(Nbar=100)


def evaluate(gpt, te, tu):
    with contextlib.redirect_stdout(io.StringIO()):
        mu, sd = gpt.predict(te["X"], mode='loop')
    mu, sd = mu.ravel(), sd.ravel()
    a_true = te["me"]; a_pred = np.exp(mu)
    f_true = te["w"] * a_true; f_pred = te["w"] * a_pred
    rel = np.abs(a_pred / a_true - 1.0)
    pull = (mu - np.log(a_true)) / sd
    out = dict(eps_uniform=float(np.sum(np.abs(f_pred - f_true)) / np.sum(f_true)),
               median_rel=float(np.median(rel)), within1pct=float(np.mean(rel < 0.01)),
               cov68=float(np.mean(np.abs(pull) < 1)), median_sigma=float(np.median(sd)))
    if tu is not None:
        with contextlib.redirect_stdout(io.StringIO()):
            mu_u, _ = gpt.predict(tu["X"], mode='loop')
        out["eps_unweighted"] = float(np.mean(np.abs(np.exp(mu_u.ravel()) / tu["me"] - 1.0)))
    return out


def run(ch, config, data_dir, nmax, noise, seed, checkpoints):
    tr, te, tu = load(data_dir, ch)
    X = tr["X"][:nmax]; a = tr["me"][:nmax]
    rng = np.random.RandomState(seed)
    y = np.log(a) + (rng.normal(0.0, noise, len(a)) if noise > 0 else 0.0)
    sig = np.full(len(a), noise if noise > 0 else 1e-6)
    gpt = make(config, seed)
    rows, done, t0 = [], 0, time.time()
    for N in checkpoints:
        if N > nmax:
            break
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(done, N):
                gpt.update_tree(X[i:i + 1], y[i:i + 1].reshape(1, 1), sig[i])
        done = N
        t_train = time.time() - t0
        m = evaluate(gpt, te, tu); m.update(N=N, train_s=t_train, ms_per_point=1e3 * t_train / N, leaves=len(gpt.root.leaves))
        rows.append(m)
        print(f"{ch} {config:6s} N={N:6d}  eps_uniform {m['eps_uniform']:.2e}  eps_unweighted {m.get('eps_unweighted', float('nan')):.2e}  "
              f"median rel {m['median_rel']:.2e}  within 1 %: {100 * m['within1pct']:.0f} %  cov68 {m['cov68']:.2f}  "
              f"{m['ms_per_point']:.1f} ms/pt  leaves {m['leaves']}", flush=True)
    return rows


def run_acquisition(ch, data_dir, nmax, seed, checkpoints, batch=500, candidates=20000):
    """The tree buys, from the Sobol pool, the points with the largest expected error
    contribution w * |M|^2 * sigma; the phase-space weight w is analytic, the amplitude
    the tree's own prediction."""
    tr, te, tu = load(data_dir, ch)
    X = tr["X"]; a = tr["me"]; w = tr["w"]
    y = np.log(a); sig = 1e-6
    rng = np.random.RandomState(seed)
    gpt = make("tree", seed)
    bought = list(range(checkpoints[0]))
    pool = np.arange(checkpoints[0], len(a))
    with contextlib.redirect_stdout(io.StringIO()):
        for i in bought:
            gpt.update_tree(X[i:i + 1], y[i:i + 1].reshape(1, 1), sig)
    rows, t0 = [], time.time()
    for N in checkpoints:
        if N > nmax:
            break
        while len(bought) < N and len(pool):
            cand = rng.choice(pool, size=min(candidates, len(pool)), replace=False)
            with contextlib.redirect_stdout(io.StringIO()):
                mu, sd = gpt.predict(X[cand], mode='loop')
            score = w[cand] * np.exp(mu.ravel()) * sd.ravel()
            take = cand[np.argsort(score)[::-1][:min(batch, N - len(bought))]]
            with contextlib.redirect_stdout(io.StringIO()):
                for i in take:
                    gpt.update_tree(X[i:i + 1], y[i:i + 1].reshape(1, 1), sig)
            bought += list(take)
            pool = np.setdiff1d(pool, take, assume_unique=True)
        m = evaluate(gpt, te, tu); t_train = time.time() - t0
        m.update(N=N, train_s=t_train, ms_per_point=1e3 * t_train / N, leaves=len(gpt.root.leaves))
        rows.append(m)
        print(f"{ch} acq    N={N:6d}  eps_uniform {m['eps_uniform']:.2e}  eps_unweighted {m.get('eps_unweighted', float('nan')):.2e}  "
              f"median rel {m['median_rel']:.2e}  within 1 %: {100 * m['within1pct']:.0f} %  cov68 {m['cov68']:.2f}  "
              f"{m['ms_per_point']:.1f} ms/pt  leaves {m['leaves']}", flush=True)
    return rows


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chans = sorted({k[0] for k in results})
    fig, axes = plt.subplots(1, len(chans), figsize=(5.2 * len(chans), 3.8), squeeze=False)
    for ax, ch in zip(axes[0], chans):
        for (c, cfg), rows in results.items():
            if c != ch:
                continue
            ax.plot([r["N"] for r in rows], [r["eps_uniform"] for r in rows], "-o", color=COLORS[cfg], lw=2, ms=5,
                    label=f"{cfg}, uniform test")
            if "eps_unweighted" in rows[0]:
                ax.plot([r["N"] for r in rows], [r["eps_unweighted"] for r in rows], "--s", color=COLORS[cfg], lw=1.5, ms=4,
                        label=f"{cfg}, unweighted test")
        ax.axhline(0.01, color="#52514e", lw=1, ls=":")
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 1000, 0.0107, "1 % target", color="#52514e", fontsize=8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("training points"); ax.set_ylabel(r"$\epsilon$ (relative L1 error)")
        ax.set_title(CHANNEL_NAMES[ch], fontsize=10); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--channels", default="gg,uu")
    ap.add_argument("--configs", default="tree,hybrid")
    ap.add_argument("--nmax", type=int, default=131072)
    ap.add_argument("--noise", type=float, default=0.0, help="relative Gaussian noise added to the training values")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    checkpoints = [1000, 2000, 4000, 8000, 16000, 32768, 65536, 131072]
    results = {}
    for ch in args.channels.split(","):
        for cfg in args.configs.split(","):
            if cfg == "acq":
                results[(ch, cfg)] = run_acquisition(ch, args.data_dir, args.nmax, args.seed, checkpoints)
            else:
                results[(ch, cfg)] = run(ch, cfg, args.data_dir, args.nmax, args.noise, args.seed, checkpoints)
    tag = args.tag or (f"noise{args.noise:g}" if args.noise > 0 else "exact")
    with open(os.path.join(OUT_DIR, f"results_{tag}.json"), "w") as f:
        json.dump({f"{ch}_{cfg}": rows for (ch, cfg), rows in results.items()}, f, indent=1)
    plot(results, os.path.join(OUT_DIR, f"eps_vs_points_{tag}.png"))
    print("\n| channel | config | N | eps uniform | eps unweighted | median rel | within 1 % | cov68 | ms/pt |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (ch, cfg), rows in results.items():
        for r in rows:
            print(f"| {ch} | {cfg} | {r['N']} | {r['eps_uniform']:.2e} | {r.get('eps_unweighted', float('nan')):.2e} | "
                  f"{r['median_rel']:.2e} | {100 * r['within1pct']:.0f} % | {r['cov68']:.2f} | {r['ms_per_point']:.1f} |")


if __name__ == "__main__":
    main()
