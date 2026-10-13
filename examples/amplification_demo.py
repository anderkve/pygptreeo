"""Statistical amplification of a 5D ttH surrogate: how many exact events is the tree worth?

Follows arXiv:2608.06450. The cross-section fraction in a kinematic bin V,
I_V = sum_{i in V} w_i f_i / sum_i w_i f_i over a large uniform sample of exact values
f_i = w |M|^2, is compared with the same fraction computed from the tree's prediction.
The deviation is the surrogate's systematic error on that bin; n_equiv = I (1 - I) / dev^2
is the number of unweighted exact events whose statistical error on I_V equals it, and
divided by the number of training points in V it is the amplification factor. The
deviation can only be resolved down to the uniform sample's own Monte Carlo error, so
below that n_equiv is a lower bound. The tree's calibrated uncertainty propagated into
I_V (per-point errors taken as independent, Eq. 16 of the paper) is printed with it, and
checked against the measured deviation. A targeted sample, uniform points kept only in
the tail p_T(H) > pt_min, resolves the tail bins further (Sec. 2.3 of the paper).

Usage: python amplification_demo.py <data_dir> <channel> --uniform <npz of ~1e6 uniform points>
           [--targeted <npz of uniform points in the tail> --pt-min 3]
           [--nmax 65536] [--nbar 300] [--train mixed] [--retrain 25] [--tag name]
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tth_amplitude_benchmark as bench

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "tth_amplitude")
PT_EDGES = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 20.0])   # p_T(H) / m_t


def raw_stream(data_dir, ch, train):
    """Unit-cube coordinates of the training stream, in the order the benchmark feeds them."""
    sob = np.load(os.path.join(data_dir, f"{ch}_train_sobol.npz"))["X"]
    if train == "sobol":
        return sob
    unw = np.load(os.path.join(data_dir, f"{ch}_train_{'unweighted_rho' if bench.OPTS['rho'] else 'unweighted'}.npz"))["X"]
    if train == "unweighted":
        return unw
    n = min(len(sob), len(unw)); X = np.empty((2 * n, 5)); X[0::2] = sob[:n]; X[1::2] = unw[:n]
    return X


def pt_higgs(X):
    P = bench.momenta(X)[:, 4]
    return np.sqrt(P[:, 1] ** 2 + P[:, 2] ** 2)


def fractions(mu, sd, g_true, bins, which):
    """Fraction of sum(w f) in each bin of `which`, for the truth and the tree, with the MC error of
    the former and the tree's calibrated sigma propagated into the latter (Eq. 16 of the paper)."""
    g = g_true * np.exp(mu); sg = g * sd
    G_true, G_tree = g_true.sum(), g.sum()
    out = {}
    for b in which:
        m = bins == b
        if m.sum() < 100:
            continue
        I_true = g_true[m].sum() / G_true; I = g[m].sum() / G_tree
        s_stat = np.sqrt(np.sum(((m - I_true) / G_true) ** 2 * g_true ** 2))
        s_model = np.sqrt(np.sum(((m - I) / G_tree) ** 2 * sg ** 2))
        out[b] = dict(n=int(m.sum()), I_true=I_true, I=I, s_stat=s_stat, s_model=s_model)
    return out


def averaging(frac, n_train_bins, P=None):
    """Per-bin averaging amplification. With P = (P_true, P_tree, var_P) the fractions are
    conditional on the targeted region and are scaled by its probability (Eq. 32-34)."""
    rows = []
    for b, f in frac.items():
        I_true, I, s_stat, s_model = f["I_true"], f["I"], f["s_stat"], f["s_model"]
        if P is not None:
            P_true, P_tree, var_P = P
            s_stat = np.sqrt(P_true ** 2 * s_stat ** 2 + I_true ** 2 * var_P); s_model = P_tree * s_model
            I_true, I = P_true * I_true, P_tree * I
        dev = I - I_true
        resolved = dev ** 2 > 4 * s_stat ** 2
        n_eq = I * (1 - I) / (dev ** 2 if resolved else s_stat ** 2)
        rows.append(dict(bin=b, pt_lo=float(PT_EDGES[b]), pt_hi=float(PT_EDGES[b + 1]), n_uniform=f["n"], targeted=P is not None,
                         n_train=int(n_train_bins[b]), I_true=float(I_true), I_tree=float(I), dev=float(dev),
                         sigma_stat=float(s_stat), sigma_model=float(s_model), pull=float(dev / np.sqrt(s_model ** 2 + s_stat ** 2)),
                         resolved=bool(resolved), n_equiv=float(n_eq), G=float(n_eq / max(n_train_bins[b], 1)),
                         n_equiv_prop=float(I * (1 - I) / s_model ** 2)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir"); ap.add_argument("channel")
    ap.add_argument("--uniform", required=True, help="npz of uniform points with exact w and |M|^2")
    ap.add_argument("--targeted", default="", help="npz of uniform points kept only in the tail p_T(H) > pt_min")
    ap.add_argument("--pt-min", type=float, default=3.0, help="start of the targeted tail, in units of m_t")
    ap.add_argument("--nmax", type=int, default=65536)
    ap.add_argument("--checkpoints", default="1000,4000,16000,65536,131072")
    ap.add_argument("--nbar", type=int, default=300); ap.add_argument("--train", default="mixed")
    ap.add_argument("--retrain", type=int, default=25); ap.add_argument("--kernel", default="rbf")
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--tag", default="")
    args = ap.parse_args()
    bench.OPTS.update(kernel=args.kernel, nbar=args.nbar, train=args.train, inputs="invariants", retrain=args.retrain)
    os.makedirs(OUT_DIR, exist_ok=True)
    ch = args.channel
    tr, _, _ = bench.load(args.data_dir, ch)
    bins_train = np.digitize(pt_higgs(raw_stream(args.data_dir, ch, args.train)), PT_EDGES) - 1
    zu = np.load(args.uniform); Xu = bench.tree_inputs(zu["X"]); gu = zu["w"] * zu["me"]
    weight = (lambda X: bench.rho(ch, X)) if bench.OPTS["rho"] else (lambda X: 1.0)   # the samples carry the measure only
    gu = gu * weight(zu["X"])
    bins_u = np.digitize(pt_higgs(zu["X"]), PT_EDGES) - 1
    if args.targeted:
        zt = np.load(args.targeted); Xt = bench.tree_inputs(zt["X"]); gt = zt["w"] * zt["me"] * weight(zt["X"])
        bins_t = np.digitize(pt_higgs(zt["X"]), PT_EDGES) - 1
        tail = np.flatnonzero(PT_EDGES[:-1] >= args.pt_min)
        in_tail = bins_u >= tail[0]
    X = tr["X"]; y = np.log(tr["me"]); sig = 1e-6
    gpt = bench.make("tree", args.seed)
    checkpoints = [c for c in map(int, args.checkpoints.split(",")) if c <= args.nmax]
    done, out, t0 = 0, {}, time.time()
    for N in checkpoints:
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(done, N):
                gpt.update_tree(X[i:i + 1], y[i:i + 1].reshape(1, 1), sig)
            mu_u, sd_u = gpt.predict(Xu, mode='loop')
            if args.targeted:
                mu_t, sd_t = gpt.predict(Xt, mode='loop')
        done = N
        mu_u, sd_u = mu_u.ravel() - np.log(zu["me"]), sd_u.ravel()
        n_train_bins = np.bincount(bins_train[:N], minlength=len(PT_EDGES) - 1)
        frac = fractions(mu_u, sd_u, gu, bins_u, range(len(PT_EDGES) - 1))
        rows = averaging(frac, n_train_bins)
        if args.targeted:
            # the tail's probability from the uniform sample, its inner fractions from the targeted one
            mu_t, sd_t = mu_t.ravel() - np.log(zt["me"]), sd_t.ravel()
            P_true = gu[in_tail].sum() / gu.sum(); P_tree = (gu * np.exp(mu_u))[in_tail].sum() / (gu * np.exp(mu_u)).sum()
            var_P = np.sum(((in_tail - P_true) / gu.sum()) ** 2 * gu ** 2)
            rows += averaging(fractions(mu_t, sd_t, gt, bins_t, tail), n_train_bins, (P_true, P_tree, var_P))
        pull = mu_u / sd_u; rel = np.abs(np.exp(mu_u) - 1.0)
        eps = float(np.sum(np.abs(gu * np.exp(mu_u) - gu)) / gu.sum())
        summary = dict(N=N, elapsed=time.time() - t0, eps_uniform=eps, mean_rel=float(rel.mean()), median_rel=float(np.median(rel)),
                       cov68=float(np.mean(np.abs(pull) < 1)), cov95=float(np.mean(np.abs(pull) < 2)), bins=rows)
        out[N] = summary
        print(f"{ch} N={N:6d}  eps {eps:.2e}  mean |Delta| {rel.mean():.2e}  median {np.median(rel):.2e}  "
              f"coverage 1 sigma {summary['cov68']:.2f}  2 sigma {summary['cov95']:.2f}  {time.time() - t0:.0f} s", flush=True)
        print("  p_T(H)/m_t    n_train   I_true    I_tree    dev       sigma_stat  sigma_model  pull   n_equiv       G")
        for r in rows:
            ge = " " if r["resolved"] else ">"
            print(f"  [{r['pt_lo']:4.1f}, {r['pt_hi']:4.1f}){'T' if r['targeted'] else ' '} {r['n_train']:7d}  {r['I_true']:.5f}  {r['I_tree']:.5f}  "
                  f"{r['dev']:+.1e}  {r['sigma_stat']:.1e}     {r['sigma_model']:.1e}    {r['pull']:+5.2f}  {ge}{r['n_equiv']:8.2g}  {ge}{r['G']:8.2g}", flush=True)
    tag = args.tag or f"{ch}_{args.train}_nbar{args.nbar}"
    with open(os.path.join(OUT_DIR, f"amplification_{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
