"""Where does the weighted error of a 5D ttH surrogate come from?

Trains the tree (options as in tth_amplitude_benchmark.py) on the first N points and
breaks the error sum  Σ|f~ - f|  over the uniform test set down by the size of f, by
the smallest log invariant (collinearity) and by each phase-space variable.

Usage: python tth_error_diagnostic.py <data_dir> <channel> [--nmax 16384] [benchmark options]
"""
import argparse
import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tth_amplitude_benchmark as bench

ap = argparse.ArgumentParser()
ap.add_argument("data_dir"); ap.add_argument("channel")
ap.add_argument("--nmax", type=int, default=16384)
ap.add_argument("--kernel", default="rbf"); ap.add_argument("--nbar", type=int, default=300)
ap.add_argument("--train", default="mixed"); ap.add_argument("--inputs", default="invariants")
ap.add_argument("--retrain", type=int, default=25)
args = ap.parse_args()
bench.OPTS.update(kernel=args.kernel, nbar=args.nbar, train=args.train, inputs=args.inputs, retrain=args.retrain)

tr, te, tu = bench.load(args.data_dir, args.channel)
X = tr["X"][:args.nmax]; y = np.log(tr["me"][:args.nmax])
gpt = bench.make("tree", 1)
with contextlib.redirect_stdout(io.StringIO()):
    for i in range(args.nmax):
        gpt.update_tree(X[i:i + 1], y[i:i + 1].reshape(1, 1), 1e-6)
    mu, sd = gpt.predict(te["X"], mode='loop')
mu, sd = mu.ravel(), sd.ravel()
f = te["w"] * te["me"]; ft = te["w"] * np.exp(mu)
err = np.abs(ft - f); eps = err.sum() / f.sum()
rel = np.abs(np.exp(mu) / te["me"] - 1.0)
print(f"{args.channel}: N={args.nmax}, eps_uniform {eps:.3e}, median rel {np.median(rel):.2e}, 99 % rel {np.quantile(rel, .99):.2e}, "
      f"max rel {rel.max():.2e}; pull coverage |r|<sd {np.mean(np.abs(mu - np.log(te['me'])) < sd):.2f}")

def breakdown(name, v, bins):
    idx = np.digitize(v, bins)
    print(f"\n{name}: share of the error sum / share of the integral / mean relative error / pull std, per bin")
    for b in range(1, len(bins)):
        m = idx == b
        if not m.any():
            continue
        pull = (mu[m] - np.log(te["me"][m])) / sd[m]
        print(f"  [{bins[b-1]:8.3g}, {bins[b]:8.3g}): {100 * err[m].sum() / err.sum():5.1f} % / {100 * f[m].sum() / f.sum():5.1f} % / "
              f"{np.mean(rel[m]):.2e} / {pull.std():.2f}   (n={m.sum()})")

# By the size of f: quantiles of f over the test set
q = np.quantile(f, [0, .5, .9, .99, .999, 1.0]); q[-1] *= 1.0001
breakdown("f = w|M|^2 by quantile (50 / 90 / 99 / 99.9 %)", f, q)
# By the smallest invariant (collinearity of a top with a beam, or a soft Higgs)
P = bench.momenta(te["X"] if args.inputs == "cube" else None) if args.inputs == "cube" else None
Xc = np.load(os.path.join(args.data_dir, f"{args.channel}_test_uniform.npz"))["X"]
inv = bench.invariants(Xc)
smin = inv.min(axis=1)
breakdown("smallest log(p_i.p_j)", smin, np.quantile(smin, [0, .01, .05, .2, .5, 1.0]) * np.array([1, 1, 1, 1, 1, 1.0001]))
names = ["beta^2 (x1)", "f_ttbar (x2)", "theta_H (x3)", "theta_t (x4)", "phi_t (x5)"]
for k, nm in enumerate(names):
    breakdown(nm + " in the unit cube", Xc[:, k], np.linspace(0, 1.0001, 6))
# Which pair is the smallest invariant where the error is largest
pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
top = np.argsort(err)[::-1][:int(0.01 * len(err))]
am = inv[top].argmin(axis=1)
print("\nfor the 1 % of test points with the largest error contribution, the smallest invariant is p_i.p_j with (i,j) =",
      {pairs[k]: int(c) for k, c in zip(*np.unique(am, return_counts=True))}, "(0,1 = incoming; 2,3 = t, tbar; 4 = H)")
print("share of the error sum from that 1 %:", f"{100 * err[top].sum() / err.sum():.1f} %")
