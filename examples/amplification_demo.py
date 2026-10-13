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
checked against the measured deviation. The "differential" amplification compares the
distribution of log(A_tree / A_true) over V between a weighted LO-unweighted sample and
an independent holdout with a Kolmogorov-Smirnov distance.

Usage: python amplification_demo.py <data_dir> <channel> --uniform <npz of ~1e6 uniform points>
           --unweighted <npz of LO-unweighted points not used in training>
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
K_MEAN = np.sqrt(0.5 * np.pi * np.log(2.0))      # <K> of the Kolmogorov distribution
PT_EDGES = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 20.0])   # p_T(H) / m_t


def raw_stream(data_dir, ch, train):
    """Unit-cube coordinates of the training stream, in the order the benchmark feeds them."""
    sob = np.load(os.path.join(data_dir, f"{ch}_train_sobol.npz"))["X"]
    if train == "sobol":
        return sob
    unw = np.load(os.path.join(data_dir, f"{ch}_train_unweighted.npz"))["X"]
    if train == "unweighted":
        return unw
    n = min(len(sob), len(unw)); X = np.empty((2 * n, 5)); X[0::2] = sob[:n]; X[1::2] = unw[:n]
    return X


def pt_higgs(X):
    P = bench.momenta(X)[:, 4]
    return np.sqrt(P[:, 1] ** 2 + P[:, 2] ** 2)


def ks_weighted(t_a, w_a, t_b):
    """sup_t |F_A(t) - F_B(t)| between the w-weighted sample A and the unweighted sample B."""
    grid = np.sort(np.concatenate([t_a, t_b]))
    ia = np.argsort(t_a); ca = np.cumsum(w_a[ia]) / w_a.sum()
    Fa = np.concatenate([[0.0], ca])[np.searchsorted(t_a[ia], grid, side="right")]
    Fb = np.searchsorted(np.sort(t_b), grid, side="right") / len(t_b)
    return float(np.max(np.abs(Fa - Fb)))


def averaging(mu, sd, g_true, bins, n_train_bins):
    """Per-bin averaging amplification from the uniform sample: g_true = w f, the tree's g = w f_tree."""
    g = g_true * np.exp(mu); sg = g * sd
    G_true, G_tree = g_true.sum(), g.sum()
    rows = []
    for b in range(len(PT_EDGES) - 1):
        m = bins == b
        if m.sum() < 100:
            continue
        I_true = g_true[m].sum() / G_true; I = g[m].sum() / G_tree
        s_stat = np.sqrt(np.sum(((m - I_true) / G_true) ** 2 * g_true ** 2))      # MC error of the truth fraction
        s_model = np.sqrt(np.sum(((m - I) / G_tree) ** 2 * sg ** 2))               # propagated tree sigma, Eq. 16
        dev = I - I_true
        resolved = dev ** 2 > 4 * s_stat ** 2
        n_eq = I * (1 - I) / (dev ** 2 if resolved else s_stat ** 2)
        rows.append(dict(bin=b, pt_lo=float(PT_EDGES[b]), pt_hi=float(PT_EDGES[b + 1]), n_uniform=int(m.sum()),
                         n_train=int(n_train_bins[b]), I_true=float(I_true), I_tree=float(I), dev=float(dev),
                         sigma_stat=float(s_stat), sigma_model=float(s_model), pull=float(dev / np.sqrt(s_model ** 2 + s_stat ** 2)),
                         resolved=bool(resolved), n_equiv=float(n_eq), G=float(n_eq / max(n_train_bins[b], 1)),
                         n_equiv_prop=float(I * (1 - I) / s_model ** 2)))
    return rows


def differential(mu_a, mu_b, me_a, me_b, bins_a, bins_b, n_train_bins):
    """Per-bin KS amplification: A is the weighted LO-unweighted sample, B the holdout."""
    w = np.exp(mu_a) / me_a
    t_a = mu_a - np.log(me_a); t_b = mu_b - np.log(me_b)
    out = {}
    for b in range(len(PT_EDGES) - 1):
        in_a = bins_a == b; in_b = bins_b == b
        if in_a.sum() < 20 or in_b.sum() < 20:
            continue
        m_ks = ks_weighted(t_a[in_a], w[in_a], t_b[in_b])
        den = in_b.sum() * (m_ks / K_MEAN) ** 2 - 1.0
        n_eq = in_b.sum() / den if den > 0 else float("inf")
        out[b] = dict(ks=float(m_ks), n_holdout=int(in_b.sum()), n_equiv_ks=float(n_eq), G_ks=float(n_eq / max(n_train_bins[b], 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir"); ap.add_argument("channel")
    ap.add_argument("--uniform", required=True, help="npz of uniform points with exact w and |M|^2")
    ap.add_argument("--unweighted", required=True, help="npz of LO-unweighted points not used in training")
    ap.add_argument("--nmax", type=int, default=65536)
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
    bins_u = np.digitize(pt_higgs(zu["X"]), PT_EDGES) - 1
    zw = np.load(args.unweighted); Xw = bench.tree_inputs(zw["X"]); mew = zw["me"]; half = len(mew) // 2
    bins_w = np.digitize(pt_higgs(zw["X"]), PT_EDGES) - 1
    X = tr["X"]; y = np.log(tr["me"]); sig = 1e-6
    gpt = bench.make("tree", args.seed)
    checkpoints = [c for c in (1000, 4000, 16000, 65536, 131072) if c <= args.nmax]
    done, out, t0 = 0, {}, time.time()
    for N in checkpoints:
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(done, N):
                gpt.update_tree(X[i:i + 1], y[i:i + 1].reshape(1, 1), sig)
            mu_u, sd_u = gpt.predict(Xu, mode='loop')
            mu_w, sd_w = gpt.predict(Xw, mode='loop')
        done = N
        mu_u, sd_u, mu_w = mu_u.ravel() - np.log(zu["me"]), sd_u.ravel(), mu_w.ravel()
        n_train_bins = np.bincount(bins_train[:N], minlength=len(PT_EDGES) - 1)
        rows = averaging(mu_u, sd_u, gu, bins_u, n_train_bins)
        ks = differential(mu_w[:half], mu_w[half:], mew[:half], mew[half:], bins_w[:half], bins_w[half:], n_train_bins)
        for r in rows:
            r.update(ks.get(r["bin"], {}))
        pull = mu_u / sd_u; rel = np.abs(np.exp(mu_u) - 1.0)
        eps = float(np.sum(np.abs(gu * np.exp(mu_u) - gu)) / gu.sum())
        summary = dict(N=N, elapsed=time.time() - t0, eps_uniform=eps, mean_rel=float(rel.mean()), median_rel=float(np.median(rel)),
                       cov68=float(np.mean(np.abs(pull) < 1)), cov95=float(np.mean(np.abs(pull) < 2)), bins=rows)
        out[N] = summary
        print(f"{ch} N={N:6d}  eps {eps:.2e}  mean |Delta| {rel.mean():.2e}  median {np.median(rel):.2e}  "
              f"coverage 1 sigma {summary['cov68']:.2f}  2 sigma {summary['cov95']:.2f}  {time.time() - t0:.0f} s", flush=True)
        print("  p_T(H)/m_t    n_train   I_true    I_tree    dev       sigma_stat  sigma_model  pull   n_equiv       G       n_equiv_KS   G_KS")
        for r in rows:
            ge = " " if r["resolved"] else ">"
            print(f"  [{r['pt_lo']:4.1f}, {r['pt_hi']:4.1f})  {r['n_train']:7d}  {r['I_true']:.5f}  {r['I_tree']:.5f}  {r['dev']:+.1e}  "
                  f"{r['sigma_stat']:.1e}     {r['sigma_model']:.1e}    {r['pull']:+5.2f}  {ge}{r['n_equiv']:8.2g}  {ge}{r['G']:8.2g}  "
                  f"{r.get('n_equiv_ks', float('nan')):9.3g}  {r.get('G_ks', float('nan')):8.3g}", flush=True)
    tag = args.tag or f"{ch}_{args.train}_nbar{args.nbar}"
    with open(os.path.join(OUT_DIR, f"amplification_{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
