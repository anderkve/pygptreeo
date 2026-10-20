"""<M0|T_t.T_tbar|M0> / Born at the points of a data file, from the GoSam colour correlator (tree level).
Usage: gen_cc.py <tth_loop_eval binary> <contract> <in.npz> <out.npz> [workers]"""
import subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
binary, contract, fin, fout = sys.argv[1:5]
workers = int(sys.argv[5]) if len(sys.argv) > 5 else 4
P = np.load(fin)["P"]
def run(chunk):
    inp = "\n".join(" ".join(f"{v:.17g}" for v in row) for row in chunk) + "\n"
    r = subprocess.run([binary, contract, "--cc-only"], input=inp, capture_output=True, text=True, check=True, cwd=os.path.dirname(contract))
    rows = [l.split() for l in r.stdout.splitlines() if len(l.split()) == 8]
    assert len(rows) == len(chunk)
    return np.array(rows, dtype=float)
with ThreadPoolExecutor(workers) as ex:
    A = np.concatenate(list(ex.map(run, np.array_split(P, max(1, min(workers * 4, len(P) // 50))))))
np.savez(fout, cc_over_tree=A[:, 5] / A[:, 1], tree_gosam=A[:, 1])
print(f"{fout}: {len(P)} points, cc/tree in [{(A[:,5]/A[:,1]).min():.4f}, {(A[:,5]/A[:,1]).max():.4f}]")
