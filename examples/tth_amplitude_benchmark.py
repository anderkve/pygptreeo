"""pyGPTreeO on 5D tree-level ttH amplitudes: the test functions f1 and f3 of
"Interpolating amplitudes" (Bresó-Pla, Heinrich, Magerya, Olsson, arXiv:2412.09534).

The amplitudes are |M|^2 for q qbar -> t tbar H (f1) and g g -> t tbar H (f3) on
the paper's five phase-space variables (beta^2, f_ttbar, theta_H, theta_t, phi_t),
folded by the amplitude's symmetries onto the unit cube (``--domain full`` for the
full polar-angle domain the paper's networks use), computed with MadGraph
(examples/tth_amplitude/README.md has the recipe). The paper's error is the
relative L1 distance eps = sum |f~ - f| / sum f of the weighted amplitude
f = w |M|^2 over uniform test points, where w is the phase-space measure times
the paper's flux and pseudo-PDF factor rho(beta^2) (``--no-rho`` drops the latter),
and on LO-unweighted points the mean relative error.

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
    def arrays(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    tr = arrays(os.path.join(data_dir, f"{ch}_train_sobol.npz"))
    unw = "unweighted_rho" if OPTS["rho"] else "unweighted"     # points drawn with probability ~ f
    if OPTS["train"] in ("unweighted", "mixed"):
        tw = arrays(os.path.join(data_dir, f"{ch}_train_{unw}.npz"))
        if OPTS["train"] == "unweighted":
            tr = tw
        else:   # alternate Sobol and f-weighted points
            n = min(len(tr["me"]), len(tw["me"]))
            tr = {k: np.empty((2 * n,) + tr[k].shape[1:]) for k in tr}
            for k in tr:
                tr[k][0::2] = arrays(os.path.join(data_dir, f"{ch}_train_sobol.npz"))[k][:n]; tr[k][1::2] = tw[k][:n]
    te = arrays(os.path.join(data_dir, f"{ch}_test_uniform.npz"))
    path_u = os.path.join(data_dir, f"{ch}_test_{unw}.npz")
    tu = arrays(path_u) if os.path.exists(path_u) else None
    for d in (tr, te, tu):
        if d is not None:
            if OPTS["rho"]:
                d["w"] = d["w"] * rho(ch, d["X"])
            d["X"] = tree_inputs(d["X"])
    return tr, te, tu


NET_KWARGS = {}   # overrides for the hybrid's FeatureNetLearner, from --net
OPTS = dict(kernel="matern32", nbar=100, train="sobol", inputs="cube", features="cube", split="min_lengthscale",
            retrain=25, strategy="gradual", target="log", glob="none", rho=True)
GLUON_TOP_PAIRS = [1, 2, 4, 5]   # columns of invariants(): (0,2), (0,3), (1,2), (1,3)


def target_shift(X_tree):
    """What is added to log|M|^2 before learning: zero, or p * sum of the gluon-top
    log invariants (``--target pole1``, ``pole2``), which flattens the t-channel
    top propagators 1/(p_g . p_t)^p."""
    if OPTS["target"] == "log":
        return 0.0
    p = {"pole1": 1.0, "pole2": 2.0}[OPTS["target"]]
    inv = X_tree if OPTS["inputs"] == "invariants" else (X_tree[:, 5:] if OPTS["inputs"] == "both" else invariants(X_tree))
    return p * inv[:, GLUON_TOP_PAIRS].sum(axis=1)
MT, MH = 1.0, np.sqrt(12.0 / 23.0)
RHO_C = {"uu": (1.0132, 0.9943, 0.3506), "gg": (1.0134, 0.7344, 0.0987)}   # eq. (10) of arXiv:2412.09534
THETA_MAX = 0.5 * np.pi   # theta_H and theta_t range: pi/2 on the folded domain, pi on the full one


def rho(ch, X):
    """The paper's flux times pseudo-PDF factor (1/2s) d rho / d beta^2, which multiplies
    the phase-space measure the driver returns; a function of the unit-cube coordinates."""
    c1, c2, c3 = RHO_C[ch]; b2 = 0.1 + 0.86 * X[:, 0]
    return (1 - c1 * b2) ** 2 / ((1 - c2 * b2) * (1 - c3 * b2))


def momenta(X):
    """Four-momenta (n, 5, 4) of g/q, g/qbar, t, tbar, H from the unit-cube variables,
    as tth_driver.cpp builds them (folded domain, m_t = 1)."""
    X = np.atleast_2d(X)
    b2 = 0.1 + 0.86 * X[:, 0]; frac = X[:, 1]; th = THETA_MAX * X[:, 2]; tt = THETA_MAX * X[:, 3]; ph = np.pi * X[:, 4]
    s = (2 * MT + MH) ** 2 / (1 - b2); rs = np.sqrt(s)
    stt = 4 * MT ** 2 + frac * ((rs - MH) ** 2 - 4 * MT ** 2); rstt = np.sqrt(stt)
    pt = np.sqrt(stt - 4 * MT ** 2) / 2
    t1 = np.column_stack([rstt / 2, pt * np.sin(tt) * np.cos(ph), pt * np.sin(tt) * np.sin(ph), pt * np.cos(tt)])
    t2 = np.column_stack([t1[:, 0], -t1[:, 1], -t1[:, 2], -t1[:, 3]])
    kl = (stt - (rs + MH) ** 2) * (stt - (rs - MH) ** 2); qh2 = kl / (4 * s); qh = np.sqrt(qh2)
    g = np.sqrt(1 + qh2 / stt); gb = qh / rstt
    def boost_rot(p):
        E = g * p[:, 0] + gb * p[:, 3]; pz = gb * p[:, 0] + g * p[:, 3]; px, py = p[:, 1], p[:, 2]
        return np.column_stack([E, -np.cos(th) * px - np.sin(th) * pz, py, np.sin(th) * px - np.cos(th) * pz])
    q1 = np.column_stack([rs / 2, 0 * rs, 0 * rs, rs / 2]); q2 = np.column_stack([rs / 2, 0 * rs, 0 * rs, -rs / 2])
    qH = np.column_stack([np.sqrt(qh2 + MH ** 2), qh * np.sin(th), 0 * qh, qh * np.cos(th)])
    return np.stack([q1, q2, boost_rot(t1), boost_rot(t2), qH], axis=1)


def invariants(X):
    """log(p_i . p_j) for the ten pairs of external momenta (the inputs of the
    networks of arXiv:2412.09534)."""
    P = momenta(X)
    metric = np.array([1.0, -1.0, -1.0, -1.0])
    out = []
    for i in range(5):
        for j in range(i + 1, 5):
            out.append(np.log(np.sum(P[:, i] * P[:, j] * metric, axis=1)))
    return np.column_stack(out)


def tree_inputs(X):
    if OPTS["inputs"] == "invariants":
        return invariants(X)
    if OPTS["inputs"] == "both":
        return np.hstack([X, invariants(X)])
    return X


def make(config, seed):
    from pygptreeo import Default_GPR, AdditiveMaternKernel
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
    d = {"cube": 5, "invariants": 10, "both": 15}[OPTS["inputs"]]
    kernels = {"matern32": None,
               "matern52": ConstantKernel() * Matern(nu=2.5, length_scale=np.ones(d)),
               "rbf": ConstantKernel() * RBF(length_scale=np.ones(d)),
               "additive": AdditiveMaternKernel(d=d, order=2, nu=1.5)}
    gpr = Default_GPR(kernel=kernels[OPTS["kernel"]])
    tree_kw = dict(Nbar=OPTS["nbar"], split_dimension_criteria=OPTS["split"], splitting_strategy=OPTS["strategy"],
                   retrain_every_n_points=OPTS["retrain"])
    if OPTS["glob"] == "additive":
        tree_kw["global_mean"] = "additive_gp"
    with contextlib.redirect_stdout(io.StringIO()):
        if config == "hybrid":
            kw = dict(steps_per_update=8, random_state=seed); kw.update(NET_KWARGS)
            if OPTS["features"] == "invariants":
                kw["feature_map"] = invariants if OPTS["inputs"] == "cube" else None
            tree_kw.pop("global_mean", None)
            return GPTree(GPR=gpr, global_mean='net', global_mean_kwargs=kw, **tree_kw)
        return GPTree(GPR=gpr, **tree_kw)


def evaluate(gpt, te, tu):
    with contextlib.redirect_stdout(io.StringIO()):
        mu, sd = gpt.predict(te["X"], mode='loop')
    mu, sd = mu.ravel() - target_shift(te["X"]), sd.ravel()
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
        out["eps_unweighted"] = float(np.mean(np.abs(np.exp(mu_u.ravel() - target_shift(tu["X"])) / tu["me"] - 1.0)))
    return out


def run(ch, config, data_dir, nmax, noise, seed, checkpoints):
    tr, te, tu = load(data_dir, ch)
    X = tr["X"][:nmax]; a = tr["me"][:nmax]
    rng = np.random.RandomState(seed)
    y = np.log(a) + target_shift(X) + (rng.normal(0.0, noise, len(a)) if noise > 0 else 0.0)
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
    y = np.log(a) + target_shift(X); sig = 1e-6
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
    ap.add_argument("--net", default="", help="hybrid network settings, e.g. hidden=256,depth=4,steps=600,polish_steps=10000")
    ap.add_argument("--kernel", default="matern32", choices=["matern32", "matern52", "rbf", "additive"], help="leaf kernel")
    ap.add_argument("--nbar", type=int, default=100)
    ap.add_argument("--train", default="sobol", choices=["sobol", "unweighted", "mixed"],
                    help="training points: Sobol, drawn with probability proportional to w |M|^2, or alternating")
    ap.add_argument("--inputs", default="cube", choices=["cube", "invariants", "both"], help="the tree's inputs")
    ap.add_argument("--features", default="cube", choices=["cube", "invariants"], help="the hybrid network's inputs")
    ap.add_argument("--split", default="min_lengthscale", help="GPTree split_dimension_criteria")
    ap.add_argument("--retrain", type=int, default=25,
                    help="retrain a leaf's GP every this many new points; a value above Nbar retrains only when the leaf splits")
    ap.add_argument("--strategy", default="gradual", choices=["gradual", "standard"], help="GPTree splitting_strategy")
    ap.add_argument("--target", default="log", choices=["log", "pole1", "pole2"],
                    help="learn log|M|^2, or log|M|^2 with the gluon-top propagator poles 1/(p_g.p_t)^p factored out")
    ap.add_argument("--global", dest="glob", default="none", choices=["none", "additive"],
                    help="tree-wide global model whose residual the leaves fit (plain tree only)")
    ap.add_argument("--no-rho", dest="rho", action="store_false", help="weights without the flux and pseudo-PDF factor")
    ap.add_argument("--domain", default="folded", choices=["folded", "full"], help="polar angles in [0, pi/2] or [0, pi]")
    args = ap.parse_args()
    globals()["THETA_MAX"] = np.pi if args.domain == "full" else 0.5 * np.pi
    for item in filter(None, args.net.split(",")):
        k, v = item.split("="); NET_KWARGS[k] = None if v == "none" else int(v)
    OPTS.update(rho=args.rho, kernel=args.kernel, nbar=args.nbar, train=args.train, inputs=args.inputs, features=args.features,
                split=args.split, retrain=args.retrain, strategy=args.strategy, target=args.target, glob=args.glob)
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
