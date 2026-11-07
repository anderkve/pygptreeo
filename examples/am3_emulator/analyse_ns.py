"""Compare nested-sampling runs made with ns_emulator.py.

Usage: python analyse_ns.py run1.npz [run2.npz ...] [--plots dir] [--reference run.npz]
Prints, per run: calls, AM3 calls, timings, evidence, posterior moments (and
their difference from the reference in units of the reference std), the
validation summary; and the emulator's log L error along the run. Plots the
1-D marginals of every run on top of each other.
"""

import argparse
import os

import numpy as np

NAMES = ['log10_B', 'log10_R', 'log10_Le', 'p', 'log10_gmin', 'log10_gmax', 'doppler']


def load(path):
    d = dict(np.load(path, allow_pickle=True))
    if 'logwt' not in d:                      # UltraNest results carry weights, not log-weights
        d['logwt'] = np.log(np.maximum(d['weights'], 1e-300)); d['logz'] = np.array([float(d['logz'])]); d['logzerr'] = np.array([float(d['logzerr'])])
    w = np.exp(d['logwt'] - d['logwt'].max()); w /= w.sum()
    return d, w


def summarise(path, ref=None):
    d, w = load(path)
    label = os.path.basename(path).replace('.npz', '')
    n_calls, n_am3 = int(d['ncall']), int(d['n_am3'])
    if 'rec_it' not in d:
        d['rec_it'] = np.zeros(n_calls)
    print(f"\n=== {label}: gate {d['gate']}, nlive {int(d['nlive'])}, {int(d['niter'])} iterations, {n_calls} likelihood calls, "
          f"AM3 calls {n_am3} ({n_am3 / n_calls:.1%}); wall {float(d['t_wall']):.0f}s = AM3 {float(d['t_am3']):.0f}s + "
          f"tree predict {float(d['t_pred']):.0f}s + update {float(d['t_update']):.0f}s (+ sampler)")
    print(f"logZ {float(d['logz'][-1]):.2f} +- {float(d['logzerr'][-1]):.2f}; max logL {float(d['logl'].max()):.2f}; "
          f"posterior ESS {1.0 / np.sum(w ** 2):.0f}")
    mean, std = d['post_mean'], d['post_std']
    if ref is not None:
        rm, rs = ref['post_mean'], ref['post_std']
        print("parameter     mean     std    | ref mean  ref std | shift/ref_std  std ratio")
        for n, m, s, a, b in zip(NAMES, mean, std, rm, rs):
            print(f"  {n:12s} {m:8.3f} {s:7.3f} | {a:8.3f} {b:8.3f} | {(m - a) / b:+8.2f} {s / b:9.2f}")
    else:
        print("parameter     mean     std     true")
        for n, m, s, t in zip(NAMES, mean, std, d['theta_true']):
            print(f"  {n:12s} {m:8.3f} {s:7.3f} {t:8.3f}")
    if 'val_ess' in d:
        dl = d['val_logl_true'] - d['val_logl_used']
        print(f"validation at {dl.size} posterior samples: logL_true - logL_used median {np.median(dl):+.3f}, rms {np.sqrt(np.mean(dl**2)):.3f}, "
              f"5-95% [{np.percentile(dl, 5):+.2f}, {np.percentile(dl, 95):+.2f}]; reweighting ESS {float(d['val_ess']):.0f} ({float(d['val_ess']) / dl.size:.0%})")
    # emulator error along the run (where both values exist)
    lt, le, used, ver = d['rec_logl_true'], d['rec_logl_emu'], d['rec_used'].astype(bool), d['rec_verified'].astype(bool)
    both = np.isfinite(lt) & np.isfinite(le)
    n = lt.size
    edges = np.unique(np.concatenate([np.arange(0, n, max(n // 8, 1)), [n]]))
    print("calls          | AM3 in block | emulated | scored: median |dlogL|  p90   | verified: n  median |dlogL|  p90  max | logL_true max in block")
    for a, b in zip(edges[:-1], edges[1:]):
        s = slice(a, b); m = both[s]; mv = m & ver[s]
        d1 = np.abs(lt[s][m] - le[s][m]); d2 = np.abs(lt[s][mv] - le[s][mv])
        print(f"{a:7d}-{b:7d} | {int(np.isfinite(lt[s]).sum()):12d} | {used[s].mean():8.1%} | "
              f"{np.median(d1) if d1.size else np.nan:21.2f} {np.percentile(d1, 90) if d1.size else np.nan:6.2f} | "
              f"{mv.sum():11d} {np.median(d2) if d2.size else np.nan:14.2f} {np.percentile(d2, 90) if d2.size else np.nan:6.2f} {d2.max() if d2.size else np.nan:6.2f} | "
              f"{np.nanmax(lt[s]) if np.isfinite(lt[s]).any() else np.nan:8.2f}")
    return d, w, label


def plots(runs, plot_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, (j, n) in zip(axes.ravel(), enumerate(NAMES)):
        for d, w, label in runs:
            x = d['samples'][:, j]
            ax.hist(x, bins=40, weights=w, histtype='step', density=True, label=label, lw=1.5)
        ax.axvline(runs[0][0]['theta_true'][j], color='k', ls=':', lw=1)
        ax.set_xlabel(n); ax.set_yticks([])
    axes[0, 0].legend(fontsize=7)
    ax = axes[1, 3]
    for d, w, label in runs:
        lt, le = d['rec_logl_true'], d['rec_logl_emu']
        both = np.isfinite(lt) & np.isfinite(le)
        i = np.arange(lt.size)[both]; e = np.abs(lt - le)[both]
        if i.size > 50:
            k = max(i.size // 100, 1)
            med = np.array([np.median(e[max(0, t - 200):t + 1]) for t in range(0, i.size, k)])
            ax.plot(i[::k], med, label=label)
    ax.set(yscale='log', xlabel='likelihood call', ylabel='|dlogL| (rolling median, scored calls)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle('Posterior marginals (dotted: truth) and emulator log L error along the run')
    fig.tight_layout()
    out = os.path.join(plot_dir, 'ns_posteriors.png')
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--plots', default=None)
    ap.add_argument('--reference', default=None)
    args = ap.parse_args()
    ref = np.load(args.reference, allow_pickle=True) if args.reference else None
    runs = [summarise(f, ref) for f in args.files]
    if args.plots:
        plots(runs, args.plots)


if __name__ == '__main__':
    main()
