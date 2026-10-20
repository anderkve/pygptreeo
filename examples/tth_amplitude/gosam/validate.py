"""Checks of the one-loop evaluator against known structure.

Usage: validate.py <tth_loop_eval binary> <contract file> <MadGraph tth_driver> <qq|gg>

1. GoSam's tree against the MadGraph driver at random points: the ratio must be constant.
2. The 1/eps^2 coefficient: -(sum of Casimirs of the massless legs) x tree in units of
   alpha_s/(2 pi): 2 C_F = 8/3 for q qbar, 2 C_A = 6 for g g.
3. The Coulomb term: along a line frac_stt -> 0 the finite part diverges as 1/beta and
   finite + pi^2/beta * cc stays bounded.
4. Time per point.
"""
import subprocess
import sys
import time

import numpy as np

binary, contract, driver, ch = sys.argv[1:5]


def gosam(P):
    inp = "\n".join(" ".join(f"{v:.17g}" for v in row) for row in P) + "\n"
    r = subprocess.run([binary, contract], input=inp, capture_output=True, text=True, check=True)
    rows = [l.split() for l in r.stdout.splitlines() if len(l.split()) == 8]
    return np.array(rows, dtype=float), r.stderr


def madgraph(P):
    inp = "\n".join(" ".join(f"{v:.17g}" for v in row) for row in P) + "\n"
    import os
    r = subprocess.run([driver], input=inp, capture_output=True, text=True, check=True, cwd=os.path.dirname(driver))
    return np.array([l.split() for l in r.stdout.splitlines() if len(l.split()) == 2], dtype=float)


rng = np.random.RandomState(3)
X = rng.uniform(size=(40, 5))
P = np.column_stack([0.1 + 0.86 * X[:, 0], X[:, 1], 0.5 * np.pi * X[:, 2], 0.5 * np.pi * X[:, 3], np.pi * X[:, 4]])
t0 = time.time()
A, err = gosam(P)
dt = (time.time() - t0) / len(P)
if err.strip():
    print("stderr:", err[-800:])
w, tree, finite, pole1, pole2, cc, beta, acc = A.T
M = madgraph(P)
print(f"time per point {1e3 * dt:.1f} ms")
print(f"weight / MadGraph weight: {np.median(w / M[:, 0]):.6g} (spread {np.std(w / M[:, 0]) / np.mean(w / M[:, 0]):.1e})")
print(f"tree / MadGraph |M|^2:    {np.median(tree / M[:, 1]):.6g} (spread {np.std(tree / M[:, 1]) / np.mean(tree / M[:, 1]):.1e})")
print(f"pole2 / tree: median {np.median(pole2 / tree):.6f}, spread {np.std(pole2 / tree):.1e}  (expected {-8 / 3 if ch == 'qq' else -6:.6f})")
print(f"pole1 / tree: in [{np.min(pole1 / tree):.4f}, {np.max(pole1 / tree):.4f}]")
print(f"accuracy: median {np.median(acc):.1e}, max {np.max(acc):.1e}; points above 1e-5: {(acc >= 1e-5).sum()}")
print(f"cc / tree: in [{np.min(cc / tree):.4f}, {np.max(cc / tree):.4f}]  (T_t.T_tbar: -4/3 singlet, +1/6 octet)")
a = finite + np.pi ** 2 / beta * cc
print(f"finite / tree: in [{np.min(finite / tree):.3f}, {np.max(finite / tree):.3f}]; a / tree in [{np.min(a / tree):.3f}, {np.max(a / tree):.3f}]")
# the threshold line
fr = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
L = np.column_stack([np.full_like(fr, 0.5), fr, np.full_like(fr, 0.7), np.full_like(fr, 0.9), np.full_like(fr, 1.3)])
B, _ = gosam(L)
w, tree, finite, pole1, pole2, cc, beta, acc = B.T
print("frac_stt    beta      finite/tree   pi^2/beta*cc/tree   subtracted/tree   acc")
for i in range(len(fr)):
    print(f"{fr[i]:8.0e}  {beta[i]:8.2e}  {finite[i] / tree[i]:12.4f}  {np.pi ** 2 / beta[i] * cc[i] / tree[i]:16.4f}  {(finite[i] + np.pi ** 2 / beta[i] * cc[i]) / tree[i]:14.4f}   {acc[i]:.0e}")
