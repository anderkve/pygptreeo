"""A training stream with 20 % of its points aimed at the surrogate's low-accuracy regions,
the adaptive sampling of arXiv:2601.00950 (Sec. 2.4): a tree trained on the first points
of the stream is tested on the uniform test set, and a Gaussian kernel density estimate
weighted by the relative errors |Delta_i|, with Scott's bandwidth on the effective sample
size, is sampled for the extra points, which the exact amplitude then evaluates.

Writes <out_dir>/<channel>_train_unweighted.npz, a stream the benchmark reads with
--train unweighted: "interleaved" puts one KDE point after every four stream points,
"twostep" puts all KDE points after the first n_first points (the paper's two-step procedure).

Usage: python tth_kde_stream.py <data_dir> <channel> <driver> <out_dir> [--n-first 16384]
           [--n-kde 16384] [--order interleaved|twostep] [--nbar 300] [--train unweighted]
"""
import argparse
import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tth_amplitude_benchmark as bench
from gated_integration_demo import oracle

ap = argparse.ArgumentParser()
ap.add_argument("data_dir"); ap.add_argument("channel"); ap.add_argument("driver"); ap.add_argument("out_dir")
ap.add_argument("--n-first", type=int, default=16384, help="stream points the first tree is trained on")
ap.add_argument("--n-kde", type=int, default=16384)
ap.add_argument("--order", default="interleaved", choices=["interleaved", "twostep"])
ap.add_argument("--nbar", type=int, default=300); ap.add_argument("--train", default="unweighted")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()
bench.OPTS.update(kernel="rbf", nbar=args.nbar, train=args.train, inputs="invariants")
ch = args.channel
tr, te, _ = bench.load(args.data_dir, ch)
stream = {k: np.load(os.path.join(args.data_dir, f"{ch}_train_{'sobol' if args.train == 'sobol' else 'unweighted'}.npz"))[k]
          for k in ("X", "P", "w", "me")}
Xc_test = np.load(os.path.join(args.data_dir, f"{ch}_test_uniform.npz"))["X"]

# 1. The first tree and its relative errors on the uniform test set
gpt = bench.make("tree", args.seed)
X, y = tr["X"], np.log(tr["me"])
with contextlib.redirect_stdout(io.StringIO()):
    for i in range(args.n_first):
        gpt.update_tree(X[i:i + 1], y[i:i + 1].reshape(1, 1), 1e-6)
    mu, _ = gpt.predict(te["X"], mode='loop')
delta = np.abs(np.exp(mu.ravel()) / te["me"] - 1.0)
print(f"{ch}: tree on {args.n_first} points, mean |Delta| {delta.mean():.2e}, median {np.median(delta):.2e}")

# 2. |Delta|-weighted Gaussian KDE in the unit cube, Scott's bandwidth on the effective sample size
rng = np.random.RandomState(args.seed + 100)
wts = delta / delta.sum()
n_eff = 1.0 / np.sum(wts ** 2); d = Xc_test.shape[1]
h = n_eff ** (-1.0 / (d + 4))
L = np.linalg.cholesky(np.cov(Xc_test.T))
print(f"n_eff {n_eff:.0f}, bandwidth h {h:.3f}")
Xk = np.empty((0, d))
while len(Xk) < args.n_kde:
    idx = rng.choice(len(Xc_test), size=args.n_kde, p=wts)
    cand = Xc_test[idx] + h * rng.normal(size=(args.n_kde, d)) @ L.T
    ok = np.all((cand > 0) & (cand < 1), axis=1)      # outside the cube: draw again
    Xk = np.vstack([Xk, cand[ok]])
Xk = Xk[:args.n_kde]
print(f"KDE points: beta^2 > 0.8 for {100 * np.mean(Xk[:, 0] > 0.8):.0f} %, phi_t > 0.8 pi for {100 * np.mean(Xk[:, 4] > 0.8):.0f} % "
      f"(stream: {100 * np.mean(stream['X'][:, 0] > 0.8):.0f} %, {100 * np.mean(stream['X'][:, 4] > 0.8):.0f} %)")
wk, ak = oracle(args.driver, Xk)
Pk = np.column_stack([0.1 + 0.86 * Xk[:, 0], Xk[:, 1], 0.5 * np.pi * Xk[:, 2], 0.5 * np.pi * Xk[:, 3], np.pi * Xk[:, 4]])
kde = dict(X=Xk, P=Pk, w=wk, me=ak)

# 3. The stream
n_s = len(stream["me"])
if args.order == "interleaved":
    n_blocks = min(n_s // 4, args.n_kde)
    out = {k: np.empty((5 * n_blocks,) + stream[k].shape[1:]) for k in stream}
    for k in out:
        v = out[k].reshape((n_blocks, 5) + stream[k].shape[1:])
        v[:, :4] = stream[k][:4 * n_blocks].reshape((n_blocks, 4) + stream[k].shape[1:]); v[:, 4] = kde[k][:n_blocks]
else:
    out = {k: np.concatenate([stream[k][:args.n_first], kde[k], stream[k][args.n_first:]]) for k in stream}
os.makedirs(args.out_dir, exist_ok=True)
for f in os.listdir(args.data_dir):
    if f.endswith(".npz") and f != f"{ch}_train_unweighted.npz" and not os.path.exists(os.path.join(args.out_dir, f)):
        os.symlink(os.path.join(args.data_dir, f), os.path.join(args.out_dir, f))
np.savez(os.path.join(args.out_dir, f"{ch}_train_unweighted.npz"), **out)
np.savez(os.path.join(args.out_dir, f"{ch}_kde_points.npz"), **kde)
print(f"wrote {args.out_dir}/{ch}_train_unweighted.npz: {len(out['me'])} points ({args.order})")
