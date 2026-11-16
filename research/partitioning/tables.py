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

if sys.argv[1] == 'scaling':
    # cost and error along long streams: python tables.py scaling 100000 nbf base
    N = int(sys.argv[2])
    variants = sys.argv[3:5] if len(sys.argv) > 4 else ['nbf', 'base']
    R = {v: {k: dict(r) for k, r in report.load(v, N).items() if k[2] == 1} for v in variants}
    for records in R.values():
        for rec in records.values():        # the first records have no depths or search counts
            rec.setdefault('depth', np.empty((0, 2)))
            rec.setdefault('searched', np.empty((0, 2)))
    common = sorted(set.intersection(*[set(R[v]) for v in variants]))
    common = [k for k in common if report.metrics(R[variants[1]][k])['rmse'] > 0]      # not a constant target
    print(f"{len(common)} runs at N={N}: " + ", ".join(f"{k[0]}/{k[1]}" for k in common))
    edges = [e for e in (0, 1000, 4000, 16000, 64000, 100000) if e <= N]
    print("| points | " + " | ".join(f"{v}: update ms mean / 99th percentile | {v}: predict ms" for v in variants)
          + " | leaves | deepest leaf of the cuts | error of " + variants[0] + " relative to " + variants[1] + " |")
    print("|---|" + "---|" * (2 * len(variants) + 3))
    for lo, hi in zip(edges[:-1], edges[1:]):
        cells = []
        for v in variants:
            upd = np.concatenate([R[v][k]['t_upd'][lo:hi] for k in common])
            pred = np.concatenate([R[v][k]['t_pred'][lo:hi] for k in common])
            cells.append(f"{1e3 * upd.mean():.2f} / {1e3 * np.quantile(upd, 0.99):.0f} | {1e3 * pred.mean():.2f}")
        leaves = np.mean([R[variants[1]][k]['leaves'][hi // 1000 - 1] for k in common])
        depth = [R[variants[1]][k]['depth'][hi // 1000 - 1, 0] for k in common if len(R[variants[1]][k]['depth'])]
        ratio = gmean(np.sqrt(np.mean((R[variants[0]][k]['pred'][lo:hi] - R[variants[0]][k]['y'][lo:hi]) ** 2))
                      / np.sqrt(np.mean((R[variants[1]][k]['pred'][lo:hi] - R[variants[1]][k]['y'][lo:hi]) ** 2)) for k in common)
        print(f"| {lo} to {hi} | " + " | ".join(cells) + f" | {leaves:.0f} | {np.mean(depth):.0f} (at most {np.max(depth):.0f}) | {ratio:.2f} |"
              if depth else f"| {lo} to {hi} | " + " | ".join(cells) + f" | {leaves:.0f} | | {ratio:.2f} |")
    print("\nper run, last quarter:")
    print("| target | d | stream | NRMSE " + variants[1] + " | NRMSE " + variants[0] + " | ratio | coverage | deepest leaf |")
    print("|---|---|---|---|---|---|---|---|")
    import bench
    for k in common:
        ma, mb = report.metrics(R[variants[0]][k]), report.metrics(R[variants[1]][k])
        depth = R[variants[1]][k]['depth']
        print(f"| `{k[0]}` | {bench.target_dim(k[0])} | {k[1]} | {mb['rmse'] / mb['range']:.4f} | {ma['rmse'] / ma['range']:.4f} | "
              f"{ma['rmse'] / mb['rmse']:.2f} | {ma['cover']:.2f} | {int(depth[-1, 0]) if len(depth) else ''} |")
    r = np.array([report.metrics(R[variants[0]][k])['rmse'] / report.metrics(R[variants[1]][k])['rmse'] for k in common])
    print(f"\ngeometric mean {gmean(r):.3f}, range {r.min():.2f} to {r.max():.2f}, {int(np.sum(r > 1.02))} of {len(r)} more than 2% worse")
    searched = [R[variants[0]][k]['searched'] for k in common if len(R[variants[0]][k]['searched'])]
    if searched:
        print("leaves searched per query:", np.round(np.mean([s[-1, 1] / s[-1, 0] for s in searched]), 1))
