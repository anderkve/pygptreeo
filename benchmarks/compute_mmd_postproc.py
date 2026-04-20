"""Add MMD to the iter-14 npz files without re-running chains."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, "/home/user/pygptreeo")
from benchmarks.mcmc_assisted import mmd_rbf_joint

DATA = "/home/user/pygptreeo/benchmarks/iterations/iteration_14/data"
refs = {}
for f in sorted(glob.glob(os.path.join(DATA, "reference__*.npz"))):
    d = dict(np.load(f, allow_pickle=True))
    cfg = json.loads(str(d["config_json"]))
    refs[(cfg["problem_name"], cfg["seed"])] = d["samples"]

n_updated = 0
for f in sorted(glob.glob(os.path.join(DATA, "assisted__*.npz"))):
    d = dict(np.load(f, allow_pickle=True))
    cfg = json.loads(str(d["config_json"]))
    if "mmd_rbf_joint" in d:
        continue
    ref_samp = refs.get((cfg["problem_name"], cfg["seed"]))
    if ref_samp is None:
        continue
    burn = int(cfg["n_steps"]) // 10
    mmd = mmd_rbf_joint(ref_samp[burn:], d["samples"][burn:])
    d["mmd_rbf_joint"] = np.float64(mmd)
    np.savez(f, **d)
    n_updated += 1
print(f"Updated {n_updated} iter-14 assisted files with mmd_rbf_joint.")
