"""Learning curves from benchmark_bqtree.py result files.

One panel per (target, stream): prequential NRMSE per block of stream points,
mean over seeds, one line per configuration.

Usage:
    python examples/plot_bqtree_curves.py results/bqtree/*.jsonl --out example_plots/bqtree_curves.png
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {  # categorical slots in fixed order
    'tree': '#2a78d6', 'global': '#eb6834', 'bq': '#1baf7a', 'bq:theta=0.3': '#eda100',
    'bq:fit_margin=0.1+theta=0.3': '#e87ba4', 'bq:Nbar=200': '#008300', 'bq:Nbar=200+theta=0.3': '#4a3aa7',
}
LABELS = {'tree': 'GPTree', 'global': 'GPTree + global model', 'bq': 'BQTree',
          'bq:theta=0.3': 'BQTree, theta=0.3', 'bq:fit_margin=0.1+theta=0.3': 'BQTree, margin 0.1, theta=0.3',
          'bq:Nbar=200': 'BQTree, Nbar=200', 'bq:Nbar=200+theta=0.3': 'BQTree, Nbar=200, theta=0.3'}


def load(files):
    rows = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                if line.startswith('RESULT '):
                    rows.append(json.loads(line[7:]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--out', default='examples/example_plots/bqtree_curves.png')
    ap.add_argument('--targets', default=None, help='comma-separated subset')
    ap.add_argument('--configs', default='tree,global,bq,bq:theta=0.3')
    ap.add_argument('--title', default='Prequential NRMSE per block of stream points (mean over seeds)')
    args = ap.parse_args()

    rows = load(args.files)
    configs = args.configs.split(',')
    targets = sorted({r['target'] for r in rows}) if args.targets is None else args.targets.split(',')
    streams = [s for s in ('uniform', 'focusing', 'sweeping', 'walker') if any(r['stream'] == s for r in rows)]
    groups = {}
    for r in rows:
        groups.setdefault((r['target'], r['stream'], r['config']), []).append(r)

    fig, axes = plt.subplots(len(targets), len(streams), figsize=(3.6 * len(streams), 2.9 * len(targets)),
                             squeeze=False, sharex='col')
    fig.patch.set_facecolor('#fcfcfb')
    for i, tgt in enumerate(targets):
        for j, strm in enumerate(streams):
            ax = axes[i, j]
            ax.set_facecolor('#fcfcfb')
            for cfg in configs:
                rs = groups.get((tgt, strm, cfg))
                if not rs:
                    continue
                c = np.mean([r['curve_nrmse'] for r in rs], axis=0)
                b = rs[0]['curve_block']
                x = (np.arange(len(c)) + 1) * b
                ax.plot(x, c, color=COLORS.get(cfg, '#52514e'), lw=2, label=LABELS.get(cfg, cfg))
                ax.plot(x[-1], c[-1], 'o', color=COLORS.get(cfg, '#52514e'), ms=5)
            ax.set_yscale('log')
            ax.grid(True, color='#e6e5e1', lw=0.6)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            ax.spines['left'].set_color('#c3c2b7'); ax.spines['bottom'].set_color('#c3c2b7')
            ax.tick_params(colors='#52514e', labelsize=8)
            if i == 0:
                ax.set_title(strm, fontsize=10, color='#0b0b0b')
            if j == 0:
                ax.set_ylabel(f'{tgt}\nprequential NRMSE', fontsize=8, color='#52514e')
            if i == len(targets) - 1:
                ax.set_xlabel('points seen', fontsize=8, color='#52514e')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        for ax in axes.ravel():
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    fig.legend(handles, labels, loc='lower center', ncol=len(handles), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(args.title, fontsize=11, color='#0b0b0b')
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(args.out, dpi=130, bbox_inches='tight')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
