#!/usr/bin/env python3
"""The one-loop t tbar H amplitude as a driver for the benchmark's leaf-driven acquisition.

Usage: tth_loop_driver.py <uu|gg> <tth_ml_driver> [--cc <tth_loop_eval_gg> <contract>] [--ratio]

stdin: lines "beta2 frac theta_H theta_t phi", the tree-level driver's input. stdout: "w a" per
line, w the phase-space weight and a = finite + pi^2/beta * cc, the paper's f2/f4 amplitude in
MadLoop's units as gen_loop_ml.py stores it, or a / a_LO with --ratio; nan where MadLoop's
accuracy estimate is worse than 1e-5. The g g colour correlator comes from the GoSam evaluator.
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import tth_amplitude_benchmark as bench

args = [a for a in sys.argv[1:] if a != "--ratio"]
ratio = "--ratio" in sys.argv
ch, binary = args[0], os.path.abspath(args[1])
cc_bin, cc_contract = (os.path.abspath(args[3]), os.path.abspath(args[4])) if len(args) > 2 and args[2] == "--cc" else (None, None)
MT_GEV = 173.0

P = np.array([l.split() for l in sys.stdin.read().splitlines() if len(l.split()) == 5], dtype=float)
if len(P) == 0:
    sys.exit(0)
X = np.column_stack([(P[:, 0] - 0.1) / 0.86, P[:, 1], P[:, 2] / bench.THETA_MAX, P[:, 3] / bench.THETA_MAX, P[:, 4] / np.pi])
mom = MT_GEV * bench.momenta(X)
mg5 = os.path.abspath(os.path.join(os.path.dirname(binary), "..", "..", ".."))
env = dict(os.environ, LD_LIBRARY_PATH=":".join(os.path.join(mg5, "HEPTools", s, "lib") for s in ("", "ninja", "collier")))
inp = "\n".join(" ".join(f"{v:.17g}" for v in m.ravel()) for m in mom) + "\n"
out = subprocess.run([binary], input=inp, capture_output=True, text=True, check=True, cwd=os.path.dirname(binary), env=env).stdout
rows = np.array([l.split()[1:] for l in out.splitlines() if l.startswith("RES ") and len(l.split()) == 7], dtype=float)
assert len(rows) == len(P), (len(rows), len(P))
tree, finite, pole1, pole2, acc, code = rows.T
if cc_bin:
    inp = "\n".join(" ".join(f"{v:.17g}" for v in row) for row in P) + "\n"
    out = subprocess.run([cc_bin, cc_contract, "--cc-only"], input=inp, capture_output=True, text=True, check=True,
                         cwd=os.path.dirname(cc_contract)).stdout
    g = np.array([l.split() for l in out.splitlines() if len(l.split()) == 8], dtype=float)
    assert len(g) == len(P), (len(g), len(P))
    cc = tree * g[:, 5] / g[:, 1]
else:
    cc = tree / 6.0            # q qbar: the t tbar pair is a colour octet
a = finite + np.pi ** 2 / bench.beta_tt(X) * cc
bad = (np.floor(code / 100) == 4) | ~np.isfinite(a) | (np.where(acc < 0, 1e-6, acc) >= 1e-5)
a = np.where(bad, np.nan, a / tree if ratio else a)
w = bench.phase_space_weight(X)
sys.stdout.write("".join(f"{wi:.17g} {ai:.17g}\n" for wi, ai in zip(w, a)))
