"""Where does the error of a one-loop t tbar H surrogate come from?

Trains the tree on the first N points of the one-loop data (options as in
tth_amplitude_benchmark.py; --ratio learns a / a_LO, otherwise the amplitude itself)
and breaks the error sum of eq. (30) of arXiv:2412.09534, sum |r~ - r| with r = a / a_LO
on the LO-unweighted test points, down by the energy log(p1.p2), the threshold velocity
beta_ttbar, the size of |r| and the Born.

Usage: python tth_loop_error_diagnostic.py <data_dir> <channel> [--nmax 16384] [--ratio] [--target lin] [benchmark options]
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
ap.add_argument("--kernel", default="rbf"); ap.add_argument("--nbar", type=int, default=1000)
ap.add_argument("--train", default="mixed"); ap.add_argument("--inputs", default="invbeta")
ap.add_argument("--retrain", type=int, default=1000000000)
ap.add_argument("--target", default="lin"); ap.add_argument("--ratio", action="store_true")
args = ap.parse_args()
bench.OPTS.update(kernel=args.kernel, nbar=args.nbar, train=args.train, inputs=args.inputs, retrain=args.retrain,
                  target=args.target, ratio=args.ratio)

tr, te, tu = bench.load(args.data_dir, args.channel)
X = tr["X"][:args.nmax]; a = tr["me"][:args.nmax]
bench.set_scale(a, X)
y = bench.to_target(a, X)
sig = 1e-6 * np.abs(y) if bench.linear_target() else np.full(len(y), 1e-6)
gpt = bench.make("tree", 1)
with contextlib.redirect_stdout(io.StringIO()):
    for i in range(args.nmax):
        gpt.update_tree(X[i:i + 1], y[i:i + 1].reshape(1, 1), sig[i])
    mu, sd = gpt.predict(tu["X"], mode='loop')
mu, sd = mu.ravel(), sd.ravel()
pred, psd = bench.from_target(mu, sd, tu["X"])
born = 1.0 if args.ratio else tu["tree"]
r = tu["me"] / born; rt = pred / born
err = np.abs(rt - r); eps = err.sum() / np.abs(r).sum()
pull = (mu - bench.to_target(tu["me"], tu["X"])) / sd
print(f"{args.channel}: N={args.nmax}, {'ratio' if args.ratio else 'amplitude'} target, eps (eq. 30) {eps:.3e}, "
      f"median |r~ - r| / |r| {np.median(err / np.abs(r)):.2e}, 99 % {np.quantile(err / np.abs(r), .99):.2e}, "
      f"pull coverage {np.mean(np.abs(pull) < 1):.2f}, leaves {len(gpt.root.leaves)}")
top = np.argsort(err)[::-1][:int(0.01 * len(err))]
print(f"share of the error sum from the 1 % of test points with the largest |r~ - r|: {100 * err[top].sum() / err.sum():.1f} %")


def breakdown(name, v, bins):
    idx = np.digitize(v, bins)
    print(f"\n{name}: share of the error sum / share of sum |r| / mean |r~ - r| / median |r| / pull std, per bin")
    for b in range(1, len(bins)):
        m = idx == b
        if not m.any():
            continue
        print(f"  [{bins[b-1]:8.3g}, {bins[b]:8.3g}): {100 * err[m].sum() / err.sum():5.1f} % / {100 * np.abs(r[m]).sum() / np.abs(r).sum():5.1f} % / "
              f"{err[m].mean():.2e} / {np.median(np.abs(r[m])):6.2f} / {pull[m].std():.2f}   (n={m.sum()})")


# the loaded X are the tree's inputs; the breakdown variables come from the cube coordinates of the test file
d = np.load(os.path.join(args.data_dir, f"{args.channel}_test_unweighted_rho.npz"))
Xc = d["X"][np.isfinite(d["me"]) & (d["acc"] < 1e-5)]
assert len(Xc) == len(r)
inv = bench.invariants(Xc); beta = bench.beta_tt(Xc)
breakdown("log(p1.p2), the energy", inv[:, 0], np.quantile(inv[:, 0], [0, .5, .8, .95, .99, 1.0]) * np.array([1, 1, 1, 1, 1, 1.0001]))
breakdown("beta_ttbar", beta, np.array([0, 0.3, 0.6, 0.8, 0.9, 0.97, 1.0001]))
q = np.quantile(np.abs(r), [0, .5, .9, .99, 1.0]); q[-1] *= 1.0001
breakdown("|r| = |a / a_LO| by quantile (50 / 90 / 99 %)", np.abs(r), q)
q = np.quantile(tu["tree"], [0, .01, .1, .5, 1.0]); q[-1] *= 1.0001
breakdown("the Born a_LO by quantile (1 / 10 / 50 %)", tu["tree"], q)
smin = inv.min(axis=1)
breakdown("smallest log(p_i.p_j)", smin, np.quantile(smin, [0, .01, .05, .2, .5, 1.0]) * np.array([1, 1, 1, 1, 1, 1.0001]))
