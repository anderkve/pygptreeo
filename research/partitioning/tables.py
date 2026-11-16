"""Markdown tables for docs/benchmarks.md from the records.

    python tables.py short      # the 5000-point rounds
    python tables.py long       # the 20 000-point matrix
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # before numpy is loaded
import sys

import numpy as np

import report


def gmean(x):
    x = np.asarray(list(x), dtype=float)
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float('nan')


def ratios(variant, base, N, seeds=(1,), metric='rmse'):
    """{run: error of the variant over that of the base} for the runs both have."""
    A, B = report.load(variant, N), report.load(base, N)
    out = {}
    for key in sorted(set(A) & set(B)):
        if key[2] not in seeds:
            continue
        ma, mb = report.metrics(A[key]), report.metrics(B[key])
        if mb['range'] > 0 and mb[metric] > 0:
            out[key] = (ma[metric] / mb[metric], ma, mb)
    return out


def row(label, variant, base='base', N=5000, seeds=(1,), streams=('uniform', 'walker', 'sweeping')):
    r = ratios(variant, base, N, seeds)
    if not r:
        return f"| {label} | no runs |"
    cells = [f"{gmean(v[0] for k, v in r.items() if k[1] == s):.2f}" for s in streams]
    all_r = np.array([v[0] for v in r.values()])
    cover = np.mean([v[1]['cover'] for v in r.values()])
    return (f"| {label} | " + " | ".join(cells) + f" | {gmean(all_r):.2f} ({all_r.min():.2f} to {all_r.max():.2f}) | "
            f"{int(np.sum(all_r > 1.02))} of {len(all_r)} | {cover:.2f} |")


HEAD = ("| {} | uniform | walker | sweeping | all runs (range) | runs more than 2% worse | coverage |\n"
        "|---|---|---|---|---|---|---|")

if sys.argv[1] == 'short':
    print(HEAD.format('cuts, one GP per leaf'))
    for label, v in [
        ("along the input with the largest spread of the points (`max_spread`)", 'base:split_dimension_criteria=max_spread'),
        ("hyperplane: largest spread of the points", 'oblique:dir=pca'),
        ("hyperplane: between the centroids of two clusters", 'oblique:dir=twomeans'),
        ("hyperplane: random direction", 'oblique:dir=random,scaled=True'),
        ("hyperplane: largest change of the GP mean", 'oblique:dir=grad'),
        ("hyperplane: largest change of the GP mean's gradient", 'oblique:dir=curv'),
        ("the default cut, placed where the gradient changes most", 'oblique:dir=axis,pos=gsplit'),
        ("hyperplane of the gradient's change, placed where it changes most", 'oblique:dir=curv,pos=gsplit'),
        ("the default cut, decorrelated inputs for the GPs", 'oblique:dir=axis,whiten=True'),
        ("the axis of the decorrelated inputs with the shortest length scale", 'oblique:dir=ard,whiten=True'),
        ("hyperplane of the gradient's change, decorrelated inputs", 'oblique:dir=curv,whiten=True'),
        ("the same, placed where the gradient changes most", 'oblique:dir=curv,pos=gsplit,whiten=True'),
        ("a model in every node (multi-resolution)", 'mr'),
        ("`Nbar=200`, a fit every 50 points", 'base:Nbar=200,retrain_every_n_points=50'),
        ("`theta=0.2`", 'base:theta=0.2'),
    ]:
        print(row(label, v))
    print()
    print(HEAD.format('predictions from neighbours'))
    for label, v in [
        ("100 neighbours, no test against the kernel", 'knn:metric=std'),
        ("the same, distances in fitted length scales", 'knn'),
        ("**100 neighbours (the option as it is)**", 'nb'),
        ("150 neighbours", 'nb:n_neighbours=150'),
        ("200 neighbours", 'knn:metric=std,fallback=10,k=200'),
        ("limit of the test 3", 'nb:neighbour_consistency=3'),
        ("limit of the test 30", 'nb:neighbour_consistency=30'),
        ("distances in fitted length scales", 'knn:metric=ard,fallback=10'),
        ("30 of the 100 at growing ranks of distance (no test)", 'knn:far=30'),
        ("4 points per level above the leaf among the 100", 'fisheye:fallback=10'),
        ("`Nbar=50`", 'nb:Nbar=50,n_neighbours=100'),
        ("`Nbar=50`, a fit every 12 points", 'nb:Nbar=50,n_neighbours=100,retrain_every_n_points=12'),
        ("`Nbar=200`, a fit every 50 points", 'nb:Nbar=200,retrain_every_n_points=50,n_neighbours=100'),
        ("a fit every 50 points", 'nb:retrain_every_n_points=50'),
        ("a fit every 100 points", 'knn:metric=std,fallback=10,retrain_every_n_points=100'),
        ("at most 10 leaves searched", 'knn:metric=std,fallback=10,search=tree,max_leaves=10'),
        ("hyperplane cuts along the gradient's change", 'knn:metric=std,fallback=10,dir=curv'),
        ("the same, placed where the gradient changes most", 'knn:metric=std,fallback=10,dir=curv,pos=gsplit'),
    ]:
        print(row(label, v))

if sys.argv[1] == 'long':
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    seeds = tuple(int(s) for s in sys.argv[3].split(',')) if len(sys.argv) > 3 else (1, 2)
    variant = sys.argv[4] if len(sys.argv) > 4 else 'nb'
    base = sys.argv[5] if len(sys.argv) > 5 else 'base'
    r = ratios(variant, base, N, seeds)
    streams = [s for s in ('uniform', 'walker', 'sweeping', 'focusing') if any(k[1] == s for k in r)]
    targets = sorted({k[0] for k in r})
    print(f"{len(r)} runs of {variant} against {base}, N={N}, seeds {seeds}")
    print("| target | d | " + " | ".join(streams) + " |\n|---|---|" + "---|" * len(streams))
    import bench
    for t in targets:
        cells = []
        for s in streams:
            vals = [v[0] for k, v in r.items() if k[0] == t and k[1] == s]
            cells.append(" / ".join(f"{x:.2f}" for x in vals) if vals else "")
        print(f"| `{t}` | {bench.target_dim(t)} | " + " | ".join(cells) + " |")
    print("| geometric mean | | " + " | ".join(f"{gmean(v[0] for k, v in r.items() if k[1] == s):.2f}" for s in streams) + " |")
    all_r = np.array([v[0] for v in r.values()])
    print(f"\nall runs: geometric mean {gmean(all_r):.3f}, range {all_r.min():.2f} to {all_r.max():.2f}, "
          f"{int(np.sum(all_r > 1.02))} of {len(all_r)} more than 2% worse, {int(np.sum(all_r < 0.98))} more than 2% better")
    for name, key in (('MAE', 'mae'), ('median error', 'med'), ('99% quantile', 'q99'), ('largest error', 'max'),
                      ('held-out points of the stream', 'rmse_hist'), ('uniform test points', 'rmse_uni')):
        rr = ratios(variant, base, N, seeds, metric=key)
        print(f"{name}: " + ", ".join(f"{s} {gmean(v[0] for k, v in rr.items() if k[1] == s):.2f}" for s in streams)
              + f", all {gmean(v[0] for v in rr.values()):.2f}")
    for s in streams + ['all']:
        sel = [v for k, v in r.items() if s in (k[1], 'all')]
        print(f"{s:9s} coverage {np.mean([v[1]['cover'] for v in sel]):.3f} (base {np.mean([v[2]['cover'] for v in sel]):.3f})   "
              f"off by more than 5 sigma {100 * np.mean([v[1]['off5'] for v in sel]):.2f}% (base {100 * np.mean([v[2]['off5'] for v in sel]):.2f}%)   "
              f"predict {np.mean([v[1]['t_pred'] for v in sel]):.2f} ms (base {np.mean([v[2]['t_pred'] for v in sel]):.2f})   "
              f"update {np.mean([v[1]['t_upd'] for v in sel]):.2f} ms (base {np.mean([v[2]['t_upd'] for v in sel]):.2f})")
