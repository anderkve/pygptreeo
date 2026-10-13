"""Sample the 5D ttH phase space of arXiv:2412.09534 and evaluate the tree-level |M|^2
with the MadGraph driver. Saves X (unit cube), w (phase-space measure), me.
Usage: gen_tth.py <driver> <out.npz> sobol|uniform <n> <seed> [theta_max: pi/2 (folded, default) or pi]"""
import sys, subprocess, time
import numpy as np
from scipy.stats import qmc

driver, out, kind, n, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
theta_max = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5 * np.pi
if kind == "sobol":
    X = qmc.Sobol(d=5, scramble=True, seed=seed).random(n)
else:
    X = np.random.RandomState(seed).uniform(size=(n, 5))

def to_physical(X):
    # theta_max = pi/2: domain reduced by the symmetries of f1/f3 (resources.py of the paper's code)
    b2 = 0.1 + 0.86 * X[:, 0]
    frac = X[:, 1]
    th = theta_max * X[:, 2]
    tt = theta_max * X[:, 3]
    ph = np.pi * X[:, 4]
    return np.column_stack([b2, frac, th, tt, ph])

P = to_physical(X)
t0 = time.time()
inp = "\n".join(" ".join(f"{v:.17g}" for v in row) for row in P) + "\n"
res = subprocess.run([driver], input=inp, capture_output=True, text=True, check=True)
rows = [l.split() for l in res.stdout.splitlines() if len(l.split()) == 2]
A = np.array(rows, dtype=float)
assert A.shape[0] == n, (A.shape, n)
dt = time.time() - t0
np.savez(out, X=X, P=P, w=A[:, 0], me=A[:, 1])
f = A[:, 0] * A[:, 1]
print(f"{out}: {n} points in {dt:.1f}s ({1e6*dt/n:.0f} us/pt); me in [{A[:,1].min():.3e},{A[:,1].max():.3e}], "
      f"f=w*me in [{f.min():.3e},{f.max():.3e}], mean f {f.mean():.3e}")
