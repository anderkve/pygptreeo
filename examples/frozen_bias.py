"""Where the integral shift of a frozen, ungated tree comes from.

Pretrains a tree on N uniform points as gated_integration_demo.py does, predicts the
same 200 000 uniform Monte Carlo points, and splits the signed error of the weighted
integrand by decile of the exact integrand, by the largest points and by leaf. For
the log target the shift is also split into the mean log-space error and the
convexity term exp(d) - 1 - d. Usage:
    python frozen_bias.py <tth_driver> <log|lin> <nbar> <retrain> <npretrain> <tag>
"""
import contextlib
import io
import json
import os
import sys
import time

import numpy as np

import gated_integration_demo as demo
import tth_amplitude_benchmark as bench

driver, target, nbar, retrain, npre, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
seed, npoints = 1, 200000
ch = "gg" if "gg_tth" in driver else "uu"
gpt = demo.make("tree", seed, "invariants", "rbf", nbar, retrain, target)
t0 = time.time()
Xp = np.random.RandomState(seed + 1000).uniform(size=(npre, 5))
_, a_p = bench.oracle(driver, Xp)
Xp = bench.invariants(Xp)
bench.set_scale(a_p, Xp)
y = bench.to_target(a_p, Xp)
sig = 1e-6 * (np.abs(y) if bench.linear_target() else np.ones(len(y)))
with contextlib.redirect_stdout(io.StringIO()):
    for i in range(len(a_p)):
        gpt.update_tree(Xp[i:i + 1], y[i:i + 1].reshape(1, 1), sig[i])
t_pre = time.time() - t0

X = np.random.RandomState(seed).uniform(size=(npoints, 5))   # the demo's Monte Carlo points
w, a = bench.oracle(driver, X)
w = w * bench.rho(ch, X)
Xt = bench.invariants(X)
with contextlib.redirect_stdout(io.StringIO()):
    mu, sd = gpt.predict(Xt, mode="loop")
mu, sd = mu.ravel(), sd.ravel()
a_pred, _ = bench.from_target(mu, sd, Xt)
f, d = w * a, w * (a_pred - a)
shift, mc = d.mean() / f.mean(), f.std() / np.sqrt(npoints) / f.mean()
rel = a_pred / a - 1.0


def part(m):
    """Share of the integral, contribution to the shift and relative errors of the points in m."""
    r = dict(n_pts=int(m.sum()), share_of_I=float(f[m].sum() / f.sum()), shift_contrib=float(d[m].sum() / f.sum()),
             mean_rel=float(rel[m].mean()), mean_abs_rel=float(np.abs(rel[m]).mean()))
    if target == "log":
        lr = np.log(a_pred[m]) - np.log(a[m])
        r.update(rms_log_err=float(np.sqrt(np.mean(lr ** 2))), rms_sd=float(np.sqrt(np.mean(sd[m] ** 2))))
    return r


q = np.quantile(f, np.linspace(0, 1, 11))
dec = np.clip(np.searchsorted(q, f, side="right") - 1, 0, 9)
by_decile = [dict(decile=k, **part(dec == k)) for k in range(10)]
top = {str(p): part(f >= np.quantile(f, p)) for p in (0.99, 0.999)}
sym = dict(frac_pos=float((rel > 0).mean()), wmean_rel=float((f * rel).sum() / f.sum()),
           wmean_abs_rel=float((f * np.abs(rel)).sum() / f.sum()), eps_used=float(np.abs(d).sum() / f.sum()))
if target == "log":
    # the shift with the Jensen correction exp(mu + sd^2/2), the mean of exp(f) under the GP posterior
    lr = np.log(a_pred) - np.log(a)
    sym.update(log_wmean=float((f * lr).sum() / f.sum()), convexity_wmean=float((f * (np.exp(lr) - 1 - lr)).sum() / f.sum()),
               shift_jensen=float((w * (a_pred * np.exp(sd ** 2 / 2) - a)).sum() / f.sum()),
               rms_log_err=float(np.sqrt(np.mean(lr ** 2))), rms_sd=float(np.sqrt(np.mean(sd ** 2))))

leaves = gpt.root.leaves
lab = np.empty(npoints, dtype=int)


def route(node, idx):
    """Hard assignment of the points idx to the leaves below node."""
    if not node.children:
        lab[idx] = leaves.index(node)
        return
    left = Xt[idx, node.split_index] <= node.split_position
    route(node.left, idx[left])
    route(node.right, idx[~left])


route(gpt.root, np.arange(npoints))
by_leaf = sorted([dict(leaf=j, n_train=int(leaves[j].n_points), **part(lab == j)) for j in range(len(leaves)) if (lab == j).any()],
                 key=lambda r: -abs(r["shift_contrib"]))
out = dict(tag=tag, channel=ch, target=target, nbar=nbar, retrain=retrain, npretrain=npre, t_pretrain=t_pre, n_leaves=len(leaves),
           shift=float(shift), mc_rel_err=float(mc), shift_over_mc=float(shift / mc), sym=sym, by_decile=by_decile, top=top, by_leaf=by_leaf[:12])
os.makedirs(demo.OUT_DIR, exist_ok=True)
with open(os.path.join(demo.OUT_DIR, f"frozen_bias_{tag}.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print(f"{tag}: target {target}, nbar {nbar}, retrain {retrain}, pretrained on {npre}: {len(leaves)} leaves, "
      f"shift {100 * shift:+.3f} % = {shift / mc:+.2f} MC errors, eps {sym['eps_used']:.1e}, pretraining {t_pre:.0f} s"
      + (f"; with the Jensen correction {100 * sym['shift_jensen']:+.3f} %" if target == "log" else ""))
