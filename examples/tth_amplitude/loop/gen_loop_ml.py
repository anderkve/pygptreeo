"""Evaluate the one-loop q qbar -> t tbar H amplitude with MadLoop at the points of a data file.

Usage: gen_loop_ml.py <tth_ml_driver binary> <in.npz> <out.npz> [workers=4] [nmax]

Same output layout as gen_loop.py: X, P, w, me = finite + pi^2/beta * cc (the paper's a2, eq. 14),
tree, finite (no alpha_s/(2 pi) factor, 't Hooft-Veltman scheme, mu_R = sqrt(s)/2), pole1, pole2,
cc = <M0|T_t.T_tbar|M0> = tree/6 (the t tbar pair is a colour octet in q qbar -> t tbar H), beta,
acc = MadLoop's estimated relative accuracy (negative when it could not estimate), code = MadLoop's
return code (hundreds digit 4 = exceptional point).
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, "/home/user/pygptreeo/examples")
import tth_amplitude_benchmark as bench

binary, fin, fout = sys.argv[1:4]
workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
nmax = int(sys.argv[5]) if len(sys.argv) > 5 else None
MT_GEV = 173.0
d = np.load(fin)
X, P, w = d["X"], d["P"], d["w"]
if nmax:
    X, P, w = X[:nmax], P[:nmax], w[:nmax]
n = len(X)
mom = MT_GEV * bench.momenta(X)           # (n, 5, 4): q, qbar, t, tbar, H, the process order of MadLoop
beta = bench.beta_tt(X)
env = dict(os.environ, LD_LIBRARY_PATH=":".join(os.path.join(os.environ["MG5"], "HEPTools", s, "lib") for s in ("", "ninja", "collier")))


def run(idx):
    inp = "\n".join(" ".join(f"{v:.17g}" for v in mom[i].ravel()) for i in idx) + "\n"
    res = subprocess.run([binary], input=inp, capture_output=True, text=True, check=True, cwd=os.path.dirname(binary), env=env)
    rows = [l.split()[1:] for l in res.stdout.splitlines() if l.startswith("RES ") and len(l.split()) == 7]
    if len(rows) != len(idx):
        open(f"/tmp/claude-0/-home-user-pygptreeo/c104b40a-aeba-5f56-9265-dbaae8b6a4eb/scratchpad/gosam_tth/ml_bad_{idx[0]}.txt", "w").write(res.stdout)
        raise AssertionError((len(rows), len(idx), res.stderr[-500:]))
    return np.array(rows, dtype=float)


t0 = time.time()
chunks = np.array_split(np.arange(n), max(1, min(workers * 4, n // 100)))
with ThreadPoolExecutor(workers) as ex:
    A = np.concatenate(list(ex.map(run, chunks)))
dt = time.time() - t0
tree, finite, pole1, pole2, acc, code = A.T
cc = tree / 6.0
me = finite + np.pi ** 2 / beta * cc
bad = (np.floor(code / 100) == 4) | ~np.isfinite(me)
acc = np.where(bad, 1.0, np.where(acc < 0, 1e-6, acc))     # the benchmark drops points with acc >= 1e-5
np.savez(fout, X=X, P=P, w=w, me=me, tree=tree, finite=finite, pole1=pole1, pole2=pole2, cc=cc, beta=beta, acc=acc, code=code)
print(f"{fout}: {n} points in {dt:.0f} s ({1e3 * dt * workers / n:.1f} ms/pt/core); {bad.sum()} exceptional; "
      f"pole2/tree median {np.median(pole2 / tree):.6f}; a/tree in [{np.nanmin(me / tree):.3g}, {np.nanmax(me / tree):.3g}], median {np.nanmedian(me / tree):.3g}")
