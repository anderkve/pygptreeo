"""Uncertainty-gated surrogate integration of a 5D ttH amplitude.

A Monte Carlo integration of f = w |M|^2 over the unit cube (the phase space of
examples/tth_amplitude, w with the paper's flux and pseudo-PDF factor) in which the
exact amplitude is evaluated only where the tree's calibrated relative uncertainty
exceeds a tolerance; everywhere else the tree's prediction is used, and every exact
evaluation is fed back to the tree. The exact value of every point is also computed
(the oracle is cheap) so the error the surrogate introduces into the integral can be
compared with the Monte Carlo error of the same points.

``--pretrain N --frozen`` is the comparison: a tree trained on N exact uniform points
before the run and not updated during it, used with the same gate (or ungated with
``--tol inf``).

Usage: python gated_integration_demo.py <path to tth_driver> [--config tree|hybrid]
           [--tol 0.01] [--npoints 100000] [--chunk 500] [--seed 1] [--tag name]
           [--inputs cube|invariants] [--kernel matern32|rbf|...] [--nbar 100]
           [--channel uu|gg] [--no-rho] [--pretrain 0] [--frozen]
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pygptreeo import GPTree
import tth_amplitude_benchmark as bench

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "tth_amplitude")


oracle = bench.oracle


def make(config, seed, inputs="cube", kernel="matern32", nbar=100):
    """The surrogate; the tree's inputs, leaf kernel and leaf size as in tth_amplitude_benchmark.py."""
    bench.OPTS.update(inputs=inputs, kernel=kernel, nbar=nbar, features=inputs)
    return bench.make(config, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("driver")
    ap.add_argument("--config", default="tree")
    ap.add_argument("--tol", type=float, default=0.01, help="relative uncertainty above which the exact amplitude is called")
    ap.add_argument("--npoints", type=int, default=100000)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--inputs", default="cube", choices=["cube", "invariants"], help="the tree's inputs")
    ap.add_argument("--kernel", default="matern32", choices=["matern32", "matern52", "rbf", "additive"])
    ap.add_argument("--nbar", type=int, default=100)
    ap.add_argument("--channel", default="", choices=["", "uu", "gg"], help="inferred from the driver path if empty")
    ap.add_argument("--no-rho", dest="rho", action="store_false", help="weights without the flux and pseudo-PDF factor")
    ap.add_argument("--pretrain", type=int, default=0, help="exact uniform points the tree learns before the run")
    ap.add_argument("--frozen", action="store_true", help="no learning during the run")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    ch = args.channel or ("gg" if "gg_tth" in args.driver else "uu")
    weight = (lambda Z: bench.rho(ch, Z)) if args.rho else (lambda Z: 1.0)
    rng = np.random.RandomState(args.seed)
    gpt = make(args.config, args.seed, args.inputs, args.kernel, args.nbar)
    to_tree = bench.invariants if args.inputs == "invariants" else (lambda Z: Z)
    sig_exact = 1e-6
    if args.pretrain:
        Xp = np.random.RandomState(args.seed + 1000).uniform(size=(args.pretrain, 5))
        _, a_p = oracle(args.driver, Xp); Xp = to_tree(Xp)
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(args.pretrain):
                gpt.update_tree(Xp[i:i + 1], np.log(a_p[i:i + 1]).reshape(1, 1), sig_exact)
    f_used, f_exact, called = [], [], []
    log, series = [], []
    run = dict(n=0, calls=0, sum_true=0.0, sum_sur=0.0, sum_true2=0.0, sum_abs=0.0)   # running sums for the per-chunk record
    t0 = time.time()
    for start in range(0, args.npoints, args.chunk):
        X = rng.uniform(size=(args.chunk, 5))
        w, a = oracle(args.driver, X)                    # exact values, for the record
        w = w * weight(X)
        Xt = to_tree(X)
        if start == 0 and not args.pretrain:
            gate = np.ones(args.chunk, dtype=bool)        # nothing learned yet
            mu = np.log(a)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                mu, sd = gpt.predict(Xt, mode='loop')
            mu, sd = mu.ravel(), sd.ravel()
            gate = sd > args.tol
        if not args.frozen:
            with contextlib.redirect_stdout(io.StringIO()):
                for i in np.flatnonzero(gate):
                    gpt.update_tree(Xt[i:i + 1], np.log(a[i:i + 1]).reshape(1, 1), sig_exact)
        f_true = w * a
        f_sur = w * np.where(gate, a, np.exp(mu))
        f_used.append(f_sur); f_exact.append(f_true); called.append(gate)
        n_seen = start + args.chunk
        run["n"] = n_seen; run["calls"] += int(gate.sum()); run["sum_true"] += f_true.sum(); run["sum_sur"] += f_sur.sum()
        run["sum_true2"] += (f_true ** 2).sum(); run["sum_abs"] += np.abs(f_sur - f_true).sum()
        I_ex, I_su = run["sum_true"] / n_seen, run["sum_sur"] / n_seen
        mc = np.sqrt(max(run["sum_true2"] / n_seen - I_ex ** 2, 0.0) / n_seen)
        series.append(dict(N=n_seen, calls=int(gate.sum()), n_exact=run["calls"] + args.pretrain, shift=float((I_su - I_ex) / I_ex),
                           shift_over_mc=float((I_su - I_ex) / mc), mc_rel_err=float(mc / I_ex), eps_used=float(run["sum_abs"] / run["sum_true"])))
        if n_seen in (2000, 5000, 10000, 20000, 50000, 100000, 200000) or n_seen == args.npoints:
            fu, fe, c = np.concatenate(f_used), np.concatenate(f_exact), np.concatenate(called)
            I_ex, I_su = fe.mean(), fu.mean()
            mc_err = fe.std() / np.sqrt(len(fe))
            rec = dict(N=n_seen, frac_called=float(c.mean()), frac_called_last=float(c[-5000:].mean()),
                       n_exact=int(c.sum()) + args.pretrain, eps_used=float(np.sum(np.abs(fu - fe)) / fe.sum()),
                       I_exact=float(I_ex), I_gated=float(I_su), surrogate_shift=float((I_su - I_ex) / I_ex),
                       mc_rel_err=float(mc_err / I_ex), leaves=len(gpt.root.leaves), elapsed=time.time() - t0)
            log.append(rec)
            print(f"N={n_seen:7d}  exact calls {100 * rec['frac_called']:.1f} % (last 5000: {100 * rec['frac_called_last']:.1f} %), "
                  f"{rec['n_exact']} in all  eps on the points used {rec['eps_used']:.1e}  "
                  f"gated/exact - 1 = {100 * rec['surrogate_shift']:+.3f} %  MC error {100 * rec['mc_rel_err']:.2f} %  "
                  f"leaves {rec['leaves']}  {rec['elapsed']:.0f} s", flush=True)
    tag = args.tag or (f"{args.config}_{args.inputs}_{args.kernel}_nbar{args.nbar}_tol{args.tol:g}" +
                       (f"_pre{args.pretrain}" if args.pretrain else "") + ("_frozen" if args.frozen else ""))
    with open(os.path.join(OUT_DIR, f"gated_{tag}.json"), "w") as f:
        json.dump(dict(args=vars(args), channel=ch, checkpoints=log, series=series), f, indent=1)


if __name__ == "__main__":
    main()
