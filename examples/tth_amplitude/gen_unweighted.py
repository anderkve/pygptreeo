"""LO-unweighted test points: uniform points accepted with probability f/fmax, f = w*|M|^2."""
import sys, subprocess
import numpy as np
driver, out, n_want, seed, fmax = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
rng = np.random.RandomState(seed)
keepX, keepP, keepw, keepme = [], [], [], []
tot = 0
while sum(len(k) for k in keepX) < n_want:
    X = rng.uniform(size=(200000, 5))
    P = np.column_stack([0.1 + 0.86*X[:,0], X[:,1], 0.5*np.pi*X[:,2], 0.5*np.pi*X[:,3], np.pi*X[:,4]])
    inp = "\n".join(" ".join(f"{v:.17g}" for v in row) for row in P) + "\n"
    A = np.array([l.split() for l in subprocess.run([driver], input=inp, capture_output=True, text=True, check=True).stdout.splitlines() if len(l.split()) == 2], dtype=float)
    f = A[:,0]*A[:,1]; tot += len(f)
    acc = rng.uniform(size=len(f)) < f / fmax
    keepX.append(X[acc]); keepP.append(P[acc]); keepw.append(A[acc,0]); keepme.append(A[acc,1])
    print(f"pool {tot}: accepted {sum(len(k) for k in keepX)} (max f/fmax {f.max()/fmax:.2f})", flush=True)
X = np.vstack(keepX)[:n_want]; P = np.vstack(keepP)[:n_want]; w = np.concatenate(keepw)[:n_want]; me = np.concatenate(keepme)[:n_want]
np.savez(out, X=X, P=P, w=w, me=me)
print(f"{out}: {len(me)} unweighted points")
