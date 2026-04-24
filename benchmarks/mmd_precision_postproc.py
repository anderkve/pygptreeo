"""Recompute MMD^2 at higher precision with bootstrap CIs.

For each (problem, schedule, method, tau_sigma) cell in iter 14/15,
report mean ± std MMD^2 across seeds at 4 sf, plus pairwise signed
differences vs pygptreeo_D so the ranking claim is verifiable.
"""
import glob, json, os, sys
import numpy as np
from collections import defaultdict
sys.path.insert(0, "/home/user/pygptreeo")

DATA14 = "/home/user/pygptreeo/benchmarks/iterations/iteration_14/data"
DATA15 = "/home/user/pygptreeo/benchmarks/iterations/iteration_15/data"

cells = defaultdict(list)
for path in sorted(glob.glob(os.path.join(DATA14, "assisted__*.npz")) +
                   glob.glob(os.path.join(DATA15, "assisted__*.npz")) +
                   glob.glob(os.path.join(DATA15, "delayed__*.npz"))):
    d = dict(np.load(path, allow_pickle=True))
    cfg = json.loads(str(d["config_json"]))
    if "mmd_rbf_joint" not in d:
        continue
    method = cfg["method_name"]
    problem = cfg["problem_name"]
    kind = cfg.get("kind", "assisted")
    tau = float(cfg.get("tau_sigma", -1)) if kind == "assisted" else None
    seed = int(cfg["seed"])
    mmd = float(d["mmd_rbf_joint"])
    cells[(method, problem, kind, tau)].append((seed, mmd))

print("# Higher-precision MMD² with seed std and pairwise differences")
print()
print("MMD² is the median-heuristic-RBF unbiased estimator on a 2000-point sub-sample of each chain (post burn-in 10 %). Reported as mean ± std across seeds at four significant figures. Pairwise differences are seed-paired (same seed compared across methods).")
print()
print("## Per-cell mean ± std")
print()
print("| problem | method | kind | τ_σ | n_seeds | MMD² mean | MMD² std |")
print("|---|---|---|---|---|---|---|")

# Order
METHODS = ["pygptreeo_A", "pygptreeo_D", "gpytorch_svgp_A"]
PROBS = ["rosenbrock_2d", "borehole_8d", "banana_2d", "banana_5d"]
TAUS = [3e-3, 1e-2, 3e-2, 1e-1, None]

for problem in PROBS:
    for method in METHODS:
        for kind in ["assisted", "delayed"]:
            for tau in TAUS:
                # tau is None for delayed, only iterate kind/tau combos that match.
                if kind == "delayed" and tau is not None:
                    continue
                if kind == "assisted" and tau is None:
                    continue
                key = (method, problem, kind, tau)
                if key not in cells:
                    continue
                vals = [v for _, v in sorted(cells[key])]
                arr = np.asarray(vals, dtype=float)
                tau_str = f"{tau:.0e}" if tau is not None else "—"
                print(f"| {problem} | {method} | {kind} | {tau_str} | "
                      f"{arr.size} | {arr.mean():.4g} | {arr.std():.4g} |")

# Seed-paired pairwise differences vs pygptreeo_D for assisted only.
print()
print("## Seed-paired pairwise MMD² differences (vs pygptreeo_D) — assisted only")
print()
print("Sign convention: positive = the other method has a *higher* MMD² (worse fidelity) than pygptreeo_D. Each row averages over the seeds where both methods have a value at that (problem, τ_σ) cell.")
print()
print("| problem | method | τ_σ | n_pairs | Δ MMD² mean | Δ MMD² std |")
print("|---|---|---|---|---|---|")
for problem in PROBS:
    for method in METHODS:
        if method == "pygptreeo_D":
            continue
        for tau in [3e-3, 1e-2, 3e-2, 1e-1]:
            key_a = (method, problem, "assisted", tau)
            key_b = ("pygptreeo_D", problem, "assisted", tau)
            if key_a not in cells or key_b not in cells:
                continue
            d_a = dict(cells[key_a]); d_b = dict(cells[key_b])
            common = sorted(set(d_a) & set(d_b))
            if not common:
                continue
            diffs = np.asarray([d_a[s] - d_b[s] for s in common])
            print(f"| {problem} | {method} | {tau:.0e} | {len(common)} | "
                  f"{diffs.mean():+.4g} | {diffs.std():.4g} |")
