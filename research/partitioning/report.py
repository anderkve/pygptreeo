"""Tables from the records of ``run_matrix.py``.

    python report.py --variants "oblique:dir=pca" --base base --N 5000
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # before numpy is loaded
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench                                                    # noqa: E402


def load(variant, N):
    """{(target, stream, seed): record} of a variant at stream length N."""
    out = {}
    for path in glob.glob(os.path.join(bench.RESULTS, variant, f"*__N{N}__s*.npz")):
        target, stream, _, seed = os.path.basename(path)[:-4].split('__')
        out[(target, stream, int(seed[1:]))] = np.load(path)
    return out


def metrics(rec, tail=0.25):
    """Summary of one run over the last ``tail`` of the stream."""
    y, pred, std = rec['y'], rec['pred'], rec['std']
    n = len(y)
    k = int(n * (1 - tail))
    err = np.abs(pred[k:] - y[k:])
    rng = np.ptp(y[k:])
    z = err / np.maximum(std[k:], 1e-300)
    m = dict(rmse=np.sqrt(np.mean(err ** 2)), mae=np.mean(err), med=np.median(err), q99=np.quantile(err, 0.99),
             max=err.max(), range=rng, cover=np.mean(z <= 1.0), off5=np.mean(z > 5.0),
             t_pred=1e3 * rec['t_pred'][k:].mean(), t_upd=1e3 * rec['t_upd'][k:].mean(),
             t_upd_max=1e3 * rec['t_upd'][k:].max(), leaves=int(rec['leaves'][-1]) if len(rec['leaves']) else 0,
             wall=float(rec['wall']))
    for name in ('uni', 'hist'):
        e = np.abs(rec['pred_' + name] - rec['y_' + name])
        zz = e / np.maximum(rec['std_' + name], 1e-300)
        m['rmse_' + name] = np.sqrt(np.mean(e ** 2))
        m['mae_' + name] = np.mean(e)
        m['cover_' + name] = np.mean(zz <= 1.0)
    return m


def gmean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float('nan')


def compare(variant, base, N, keys=('rmse', 'mae', 'med', 'rmse_hist', 'rmse_uni'), per_run=False, streams=None):
    """Ratios variant / base per run, and their geometric means per stream type."""
    A, B = load(variant, N), load(base, N)
    common = sorted(set(A) & set(B))
    if streams:
        common = [c for c in common if c[1] in streams]
    rows = []
    for key in common:
        ma, mb = metrics(A[key]), metrics(B[key])
        if mb['range'] <= 0 or mb['rmse'] <= 0:
            continue
        rows.append((key, ma, mb))
    if not rows:
        print(f"{variant}: no common runs with {base} at N={N}")
        return None
    out = {}
    print(f"\n=== {variant}  vs  {base}   (N={N}, {len(rows)} runs) ===")
    if per_run:
        print(f"{'target':24s} {'stream':9s} s  " + "  ".join(f"{k:>9s}" for k in keys)
              + "   cover  off5%   t_pred  t_upd  leaves")
        for key, ma, mb in rows:
            print(f"{key[0]:24s} {key[1]:9s} {key[2]}  " + "  ".join(f"{ma[k] / mb[k]:9.3f}" for k in keys)
                  + f"   {ma['cover']:.2f}  {100 * ma['off5']:5.1f}  {ma['t_pred']:6.2f} {ma['t_upd']:6.2f}  {ma['leaves']:5d}")
    stream_types = sorted({r[0][1] for r in rows}) + ['all']
    print(f"{'stream':9s} runs  " + "  ".join(f"{k:>9s}" for k in keys)
          + "   worse>2%  better>2%   cover(base)  off5%(base)  t_pred(base)  t_upd(base)")
    for st in stream_types:
        sel = [r for r in rows if st == 'all' or r[0][1] == st]
        ratios = {k: gmean([ma[k] / mb[k] for _, ma, mb in sel]) for k in keys}
        r = np.array([ma['rmse'] / mb['rmse'] for _, ma, mb in sel])
        out[st] = ratios
        print(f"{st:9s} {len(sel):4d}  " + "  ".join(f"{ratios[k]:9.3f}" for k in keys)
              + f"   {np.sum(r > 1.02):3d}       {np.sum(r < 0.98):3d}"
              + f"         {np.mean([ma['cover'] for _, ma, _ in sel]):.2f} ({np.mean([mb['cover'] for _, _, mb in sel]):.2f})"
              + f"  {100 * np.mean([ma['off5'] for _, ma, _ in sel]):4.1f} ({100 * np.mean([mb['off5'] for _, _, mb in sel]):4.1f})"
              + f"   {np.mean([ma['t_pred'] for _, ma, _ in sel]):5.2f} ({np.mean([mb['t_pred'] for _, _, mb in sel]):5.2f})"
              + f"  {np.mean([ma['t_upd'] for _, ma, _ in sel]):5.2f} ({np.mean([mb['t_upd'] for _, _, mb in sel]):5.2f})")
    return out


def absolute(variant, N):
    R = load(variant, N)
    print(f"\n=== {variant} (N={N}) ===")
    print(f"{'target':24s} {'stream':9s} s    nrmse     nmae    cover  off5%  nrmse_hist nrmse_uni  t_pred  t_upd  upd_max leaves  wall")
    for key in sorted(R):
        m = metrics(R[key])
        rng = m['range'] if m['range'] > 0 else np.nan
        ru = np.ptp(R[key]['y_uni'])
        rh = np.ptp(R[key]['y_hist'])
        print(f"{key[0]:24s} {key[1]:9s} {key[2]}  {m['rmse'] / rng:8.5f} {m['mae'] / rng:8.5f}   {m['cover']:.2f}  {100 * m['off5']:5.1f}"
              f"   {m['rmse_hist'] / rh:8.5f}  {m['rmse_uni'] / ru:8.5f}  {m['t_pred']:6.2f} {m['t_upd']:6.2f} {m['t_upd_max']:7.0f}"
              f"  {m['leaves']:5d} {m['wall']:5.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', default=None, help="semicolon-separated; default: all with records")
    ap.add_argument('--base', default='base')
    ap.add_argument('--N', type=int, default=5000)
    ap.add_argument('--per_run', action='store_true')
    ap.add_argument('--absolute', action='store_true')
    ap.add_argument('--streams', default=None)
    a = ap.parse_args()
    variants = a.variants.split(';') if a.variants else sorted(
        v for v in os.listdir(bench.RESULTS) if v != a.base)
    streams = a.streams.split(',') if a.streams else None
    if a.absolute:
        for v in variants:
            absolute(v, a.N)
        return
    summary = {}
    for v in variants:
        out = compare(v, a.base, a.N, per_run=a.per_run, streams=streams)
        if out:
            summary[v] = out
    if len(summary) > 1:
        print(f"\n=== summary: rmse ratio / mae ratio vs {a.base} (N={a.N}) ===")
        sts = ['uniform', 'walker', 'sweeping', 'focusing', 'all']
        print(f"{'variant':44s} " + "  ".join(f"{s:>13s}" for s in sts))
        for v, out in summary.items():
            print(f"{v:44s} " + "  ".join(
                f"{out[s]['rmse']:6.3f}/{out[s]['mae']:6.3f}" if s in out else " " * 13 for s in sts))


if __name__ == '__main__':
    main()
