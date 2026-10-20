"""Evaluate the one-loop t tbar H amplitude at the phase-space points of an existing data file.

Usage: gen_loop.py <tth_loop_eval binary> <contract file> <in.npz> <out.npz> [workers=4] [nmax]

Reads P (the five physical variables) from in.npz and writes out.npz with the same X, P and
w, the paper's amplitude a = finite + pi^2/beta * cc as 'me', and the raw pieces tree, finite,
pole1, pole2, cc, beta, acc. Points GoSam could not evaluate to the requested precision are
marked by acc >= 1e-5 (and NaN values from the library are kept as NaN).
"""
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

binary, contract, fin, fout = sys.argv[1:5]
workers = int(sys.argv[5]) if len(sys.argv) > 5 else 4
nmax = int(sys.argv[6]) if len(sys.argv) > 6 else None
d = np.load(fin)
X, P, w = d["X"], d["P"], d["w"]
if nmax:
    X, P, w = X[:nmax], P[:nmax], w[:nmax]
n = len(P)


def run(chunk):
    inp = "\n".join(" ".join(f"{v:.17g}" for v in row) for row in chunk) + "\n"
    res = subprocess.run([binary, contract], input=inp, capture_output=True, text=True, check=True)
    rows = [l.split() for l in res.stdout.splitlines() if len(l.split()) == 8]
    assert len(rows) == len(chunk), (len(rows), len(chunk), res.stderr[-500:])
    return np.array(rows, dtype=float)


t0 = time.time()
chunks = np.array_split(P, max(1, min(workers * 8, n // 50)))
with ThreadPoolExecutor(workers) as ex:
    A = np.concatenate(list(ex.map(run, chunks)))
dt = time.time() - t0
wt, tree, finite, pole1, pole2, cc, beta, acc = A.T
me = finite + np.pi ** 2 / beta * cc
bad = ~(acc < 1e-5)
np.savez(fout, X=X, P=P, w=w, me=me, tree=tree, finite=finite, pole1=pole1, pole2=pole2, cc=cc, beta=beta, acc=acc)
print(f"{fout}: {n} points in {dt:.0f} s ({1e3 * dt * workers / n:.1f} ms/pt/core); {bad.sum()} below precision; "
      f"weight/w ratio {np.median(wt / w):.6g}; a/tree in [{np.nanmin(me / tree):.3g}, {np.nanmax(me / tree):.3g}], median {np.nanmedian(me / tree):.3g}")
