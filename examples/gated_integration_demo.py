"""Uncertainty-gated surrogate integration of a 5D ttH amplitude.

A Monte Carlo integration of f = w |M|^2 over the unit cube (the phase space of
examples/tth_amplitude) in which the exact amplitude is evaluated only where the
tree's calibrated relative uncertainty exceeds a tolerance; everywhere else the
tree's prediction is used, and every exact evaluation is fed back to the tree.
The exact value of every point is also computed (the oracle is cheap) so the
error the surrogate introduces into the integral can be compared with the
Monte Carlo error of the same points.

Usage: python gated_integration_demo.py <path to tth_driver> [--config tree|hybrid]
           [--tol 0.01] [--npoints 100000] [--chunk 500] [--seed 1] [--tag name]
"""
import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pygptreeo import GPTree

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "tth_amplitude")


def oracle(driver, X):
    """Phase-space weight and |M|^2 at unit-cube points X (n, 5), folded domain of f1/f3."""
    P = np.column_stack([0.1 + 0.86 * X[:, 0], X[:, 1], 0.5 * np.pi * X[:, 2], 0.5 * np.pi * X[:, 3], np.pi * X[:, 4]])
    inp = "\n".join(" ".join(f"{v:.17g}" for v in row) for row in P) + "\n"
    # The driver reads its parameter card relative to its own directory
    out = subprocess.run([os.path.abspath(driver)], input=inp, capture_output=True, text=True, check=True,
                         cwd=os.path.dirname(os.path.abspath(driver))).stdout
    A = np.array([l.split() for l in out.splitlines() if len(l.split()) == 2], dtype=float)
    return A[:, 0], A[:, 1]


def make(config, seed):
    with contextlib.redirect_stdout(io.StringIO()):
        if config == "hybrid":
            return GPTree(Nbar=100, global_mean='net', global_mean_kwargs=dict(steps_per_update=8, random_state=seed))
        return GPTree(Nbar=100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("driver")
    ap.add_argument("--config", default="tree")
    ap.add_argument("--tol", type=float, default=0.01, help="relative uncertainty above which the exact amplitude is called")
    ap.add_argument("--npoints", type=int, default=100000)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    gpt = make(args.config, args.seed)
    sig_exact = 1e-6
    f_used, f_exact, called = [], [], []
    log = []
    t0 = time.time()
    for start in range(0, args.npoints, args.chunk):
        X = rng.uniform(size=(args.chunk, 5))
        w, a = oracle(args.driver, X)                    # exact values, for the record
        if start == 0:
            gate = np.ones(args.chunk, dtype=bool)        # nothing learned yet
            mu = np.log(a)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                mu, sd = gpt.predict(X, mode='loop')
            mu, sd = mu.ravel(), sd.ravel()
            gate = sd > args.tol
        with contextlib.redirect_stdout(io.StringIO()):
            for i in np.flatnonzero(gate):
                gpt.update_tree(X[i:i + 1], np.log(a[i:i + 1]).reshape(1, 1), sig_exact)
        f_true = w * a
        f_sur = w * np.where(gate, a, np.exp(mu))
        f_used.append(f_sur); f_exact.append(f_true); called.append(gate)
        n_seen = start + args.chunk
        if n_seen in (2000, 5000, 10000, 20000, 50000, 100000, 200000) or n_seen == args.npoints:
            fu, fe, c = np.concatenate(f_used), np.concatenate(f_exact), np.concatenate(called)
            I_ex, I_su = fe.mean(), fu.mean()
            mc_err = fe.std() / np.sqrt(len(fe))
            rec = dict(N=n_seen, frac_called=float(c.mean()), frac_called_last=float(c[-5000:].mean()),
                       I_exact=float(I_ex), I_gated=float(I_su), surrogate_shift=float((I_su - I_ex) / I_ex),
                       mc_rel_err=float(mc_err / I_ex), leaves=len(gpt.root.leaves), elapsed=time.time() - t0)
            log.append(rec)
            print(f"N={n_seen:7d}  exact calls {100 * rec['frac_called']:.1f} % (last 5000: {100 * rec['frac_called_last']:.1f} %)  "
                  f"gated/exact - 1 = {100 * rec['surrogate_shift']:+.3f} %  MC error {100 * rec['mc_rel_err']:.2f} %  "
                  f"leaves {rec['leaves']}  {rec['elapsed']:.0f} s", flush=True)
    tag = args.tag or f"{args.config}_tol{args.tol:g}"
    with open(os.path.join(OUT_DIR, f"gated_{tag}.json"), "w") as f:
        json.dump(log, f, indent=1)


if __name__ == "__main__":
    main()
