"""Analyse scan_emulator.py result files: learning curves, calibration, the
chi^2 error of the emulator, a what-if gate analysis for shadow runs, and timings.

Usage: python analyse_scan.py results1.npz [results2.npz ...] [--plots plot_dir]
"""

import argparse
import os

import numpy as np


def rolling(x, w, q=50):
    out = np.full(x.shape, np.nan)
    for i in range(x.size):
        seg = x[max(0, i - w + 1):i + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size >= max(5, w // 4):
            out[i] = np.percentile(seg, q)
    return out


def summarise(path):
    d = np.load(path, allow_pickle=True)
    n = d['value'].size
    label = os.path.basename(path).replace('.npz', '')
    print(f"\n=== {label}: mode={d['mode']}, components={d['components']}, Nbar={int(d['nbar'])}, {n} evaluations, "
          f"wall {float(d['t_wall']):.0f}s")
    chi2_true, chi2_emu, sig = d['chi2_true'], d['chi2_emu'], d['sig_chi2']
    used, ver = d['used_emulator'].astype(bool), d['verified'].astype(bool)
    both = np.isfinite(chi2_true) & np.isfinite(chi2_emu)
    dchi2 = np.abs(chi2_emu - chi2_true)
    t_am3, t_pred, t_upd = d['t_am3'][-1], d['t_pred'][-1], d['t_update'][-1]
    n_am3 = int(np.isfinite(chi2_true).sum())
    print(f"AM3 calls {n_am3} ({t_am3:.0f}s, {t_am3 / max(n_am3, 1):.3f}s each); emulator-only {int((used & ~ver).sum())}; "
          f"tree predict {t_pred:.0f}s ({1e3 * t_pred / max(np.isfinite(chi2_emu).sum(), 1):.1f} ms each), "
          f"tree update {t_upd:.0f}s ({t_upd / max(n_am3, 1):.3f}s per AM3 call); final tree {int(d['n_points'][-1])} pts, "
          f"{int(d['n_leaves'][-1])} leaves, k={int(d['k'][-1])}")
    print(f"best objective {float(d['best_fun']):.3f}, true chi2 of best point {float(d['best_true']):.3f}")
    # Learning curve in blocks of the DE generation size
    gen = 84
    print("block  | evals        | emu err at data pts [dex]: med rms  p90 rms  max   | |dchi2|: med  p90  max   | |err|/sigma med (p90 of z_max) | chi2_true min")
    for b in range(0, n, gen * 5):
        s = slice(b, min(n, b + gen * 5))
        m = both[s]
        if m.sum() < 5:
            print(f"{b // gen:5d}  | {b:5d}-{s.stop:5d} | (emulator not ready)")
            continue
        rd, md, dc = d['rms_data'][s][m], d['max_data'][s][m], dchi2[s][m]
        zm, zx = d['z_med'][s][m], d['z_max'][s][m]
        print(f"{b // gen:5d}  | {b:5d}-{s.stop:5d} | {np.median(rd):8.3f} {np.percentile(rd, 90):8.3f} {md.max():6.2f} | "
              f"{np.median(dc):6.2f} {np.percentile(dc, 90):6.1f} {dc.max():7.1f} | {np.median(zm):5.2f} ({np.percentile(zx, 90):5.1f})"
              f"            | {np.nanmin(chi2_true[s]):8.2f}")
    # Where it matters: points with low true chi2
    for cut in (100, 50):
        m = both & (chi2_true < cut)
        if m.sum() > 0:
            print(f"points with true chi2 < {cut}: {m.sum()}; |dchi2| median {np.median(dchi2[m]):.2f}, p90 {np.percentile(dchi2[m], 90):.2f}, "
                  f"max {dchi2[m].max():.2f}; rms at data median {np.median(d['rms_data'][m]):.3f} dex")
    # Calibration of the propagated chi2 uncertainty
    m = both & np.isfinite(sig) & (sig > 0)
    if m.sum() > 10:
        ratio = dchi2[m] / sig[m]
        print(f"|dchi2| / sig_chi2 (propagated, conservative): median {np.median(ratio):.3f}, p90 {np.percentile(ratio, 90):.3f}, "
              f"p99 {np.percentile(ratio, 99):.3f}, max {ratio.max():.2f}")
    # What-if gate analysis (meaningful for shadow runs: every point has both values).
    # Gates: the propagated bound sig_chi2 < abs + rel * chi2_emu, optionally only for points
    # whose emulated chi2 is above `far` times the best true chi2 seen so far (the emulator
    # then serves the bulk of the exploration and AM3 refines the optimum). Decision quality:
    # among emulated points that truly improved on the running best, those the emulator would
    # have ranked as no improvement ("missed"), and emulated points it would have wrongly
    # ranked as improvements ("false").
    if str(d['mode']) == 'shadow' or both.sum() > 0.9 * n:
        best_so_far = np.minimum.accumulate(np.where(np.isfinite(chi2_true), chi2_true, np.inf))
        prev_best = np.concatenate([[np.inf], best_so_far[:-1]])
        print("what-if gates over evaluations after the emulator was ready "
              f"({both.sum()} points; best chi2 reached {np.nanmin(chi2_true):.1f}):")
        print("   abs   rel   far | emulated | |dchi2| of emulated: p50   p90    max | rel err p90 | improvements: n  missed  false | emulated with chi2_true<60: n, max|dchi2|")
        for a, r, far in [(1.0, 0.02, 0), (5.0, 0.1, 0), (10.0, 0.2, 0), (10.0, 0.3, 0), (20.0, 0.5, 0),
                          (10.0, 0.2, 1.5), (10.0, 0.3, 1.5), (20.0, 0.5, 1.5), (10.0, 0.3, 3.0), (1e9, 1e9, 1.5), (1e9, 1e9, 3.0)]:
            g = both & (sig < a + r * chi2_emu) & (chi2_emu > far * prev_best)
            if g.sum() == 0:
                print(f"  {a:4.0f}  {r:4.2f}  {far:4.1f} | {0:7.1%} | -")
                continue
            dc = dchi2[g]
            rel = dc / chi2_true[g]
            imp = g & (chi2_true < prev_best)
            missed = imp & ~(chi2_emu < prev_best)
            false = g & ~(chi2_true < prev_best) & (chi2_emu < prev_best)
            low = g & (chi2_true < 60)
            print(f"  {a:4.0f}  {r:4.2f}  {far:4.1f} | {g.sum() / both.sum():7.1%} | {np.median(dc):21.1f} {np.percentile(dc, 90):6.1f} {dc.max():7.1f} | "
                  f"{np.percentile(rel, 90):10.3f} | {imp.sum():13d} {missed.sum():6d} {false.sum():6d} | "
                  f"{low.sum():4d}, {dchi2[low].max() if low.sum() else float('nan'):.1f}")
    if used.any():
        m = used & ver
        print(f"active mode: emulator used for {used.sum()} evaluations ({used.mean():.1%}); verified {m.sum()}: "
              f"|dchi2| median {np.median(dchi2[m]) if m.any() else float('nan'):.2f}, max {dchi2[m].max() if m.any() else float('nan'):.2f}")
    return d, label


def plots(runs, plot_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for d, label in runs:
        n = d['value'].size
        i = np.arange(n)
        ax = axes[0, 0]
        ax.plot(i, rolling(d['rms_data'], 84, 50), label=f"{label} median")
        ax.plot(i, rolling(d['rms_data'], 84, 90), ls='--', alpha=0.6, label=f"{label} p90")
        ax = axes[0, 1]
        dchi2 = np.abs(d['chi2_emu'] - d['chi2_true'])
        ax.plot(i, rolling(dchi2, 84, 50), label=f"{label} median")
        ax.plot(i, rolling(dchi2, 84, 90), ls='--', alpha=0.6, label=f"{label} p90")
        ax = axes[1, 0]
        best = np.minimum.accumulate(np.where(np.isfinite(d['chi2_true']), d['chi2_true'], np.inf))
        ax.plot(i, best, label=label)
        ax = axes[1, 1]
        ax.plot(i, d['t_am3'], label=f"{label} AM3")
        ax.plot(i, d['t_update'], ls='--', label=f"{label} tree update")
        ax.plot(i, d['t_pred'], ls=':', label=f"{label} tree predict")
    axes[0, 0].set(yscale='log', xlabel='evaluation', ylabel='rms error at data energies [dex]', title='Emulator error (rolling over 84 evals)')
    axes[0, 1].set(yscale='log', xlabel='evaluation', ylabel='|chi2_emu - chi2_true|', title='Emulator error in chi^2')
    axes[1, 0].set(yscale='log', xlabel='evaluation', ylabel='best true chi2 so far', title='DE convergence')
    axes[1, 1].set(xlabel='evaluation', ylabel='cumulative seconds', title='Time budget')
    for ax in axes.ravel():
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(plot_dir, 'scan_overview.png')
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--plots', default=None)
    args = ap.parse_args()
    runs = [summarise(f) for f in args.files]
    if args.plots:
        plots(runs, args.plots)


if __name__ == '__main__':
    main()
