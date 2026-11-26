"""Figures of the results of the GP-hole prototype.

    python make_plots.py            # figures in plots/, from the logs in results/
    python make_plots.py --collect  # first rerun the searches some figures need

The logs in results/ are the outputs of run_benchmark.py, run_growth.py,
run_series.py and compare_pysr.py. The reruns are kept in results/plot_runs.json.
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

from targets import SERIES, TARGETS, sample

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS, PLOTS = os.path.join(HERE, 'results'), os.path.join(HERE, 'plots')
CACHE = os.path.join(RESULTS, 'plot_runs.json')

INK, INK2, GRID, AXIS = '#191919', '#666666', '#e7e7e7', '#c2c2c2'
BLUE, ORANGE, AQUA = '#257de0', '#f47230', '#00b894'       # GP holes, PySR, plain GP
LIGHT_BLUE, GRAY, BAND = '#9dc8f7', '#c2c2c2', '#f4f4f4'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'text.color': INK,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.75, 'axes.labelcolor': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True,
    'grid.color': GRID, 'grid.linewidth': 0.75, 'xtick.color': AXIS, 'ytick.color': AXIS,
    'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2, 'legend.frameon': False,
    'lines.linewidth': 1.5, 'lines.markersize': 6.5, 'savefig.dpi': 200,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})
RING = dict(markeredgecolor='white', markeredgewidth=1.5)   # a ring in the surface colour

PRETTY = {
    'toy': 'x2·sin(2·x1) + 0.5·x3²', 'I.14.3': 'x1·x2·x3', 'I.12.2': 'x1·x2/(4π·x3²)',
    'I.34.8': 'x1·x2·x3/x4', 'I.13.4': '0.5·x1·(x2² + x3² + x4²)',
    'I.24.6': '0.25·x1·(x2² + x3²)·x4²', 'I.6.2': 'exp(−(x1/x2)²/2)/(√(2π)·x2)',
    'I.12.11': 'x1·(x2 + x3·x4·sin(x5))', 'I.37.4': 'x1 + x2 + 2·√(x1·x2)·cos(x3)',
    'I.40.1': 'x1·exp(−x2·x3/x4)', 'isothermal': 'x1·x2·log(x3/x4)',
    'I.50.26': 'x1·(cos(x2·x3) + x4·cos²(x2·x3))', 'I.10.7': 'x1/√(1 − x2²/x3²)',
    'I.18.4': '(x1·x3 + x2·x4)/(x1 + x2)', 'II.11.28': '1 + x1·x2/(1 − x1·x2/3)',
    'I.30.5': 'arcsin(x1/(x2·x3))', 'I.8.14': '√((x2 − x1)² + (x4 − x3)²)',
    'erf': 'erf(x1)', 'bessel': 'J0(x1)', 'pendulum': 'pendulum period',
}
HARDER = ['I.18.4', 'II.11.28', 'I.30.5', 'I.8.14']
NOISE = 0.01


def parse(name):
    """The runs in a log: one dict per run."""
    runs, target, run = [], None, None
    for line in open(os.path.join(RESULTS, name)):
        m = re.match(r'^(\S+):  ', line)
        if m:
            target = m.group(1)
            continue
        m = re.match(r'^\s+(.*?)seed (\d+): (.*)', line)
        if m:
            head = m.group(1)
            run = dict(target=target, seed=int(m.group(2)), text=m.group(3), n=100, noise=NOISE,
                       library=head.split()[0] if head and not head.startswith('n') else None)
            for key in ('n', 'noise'):
                v = re.search(key + r'\s+(\S+)', head)
                if v:
                    run[key] = float(v.group(1))
            continue
        m = re.search(r'rmse in (\S+) out (\S+?),', line)
        if m and run:
            run.update(rin=float(m.group(1)), rout=float(m.group(2)),
                       seconds=float(re.search(r'(\d+) s$', line).group(1)))
            holes = re.search(r'holes (\d+)', line)
            run['holes'] = int(holes.group(1)) if holes else 0
            runs.append(run)
            run = None
    return runs


def outcome(run):
    if run['holes']:
        return 'hole'
    return 'right' if run['rout'] < run['noise'] else 'other'


def title(fig, text, sub, top=0.97, left=0.02):
    fig.text(left, top, text, fontsize=12, fontweight='bold', va='top')
    fig.text(left, top - 0.26 / fig.get_figheight(), sub, fontsize=9, color=INK2, va='top')


def save(fig, name):
    os.makedirs(PLOTS, exist_ok=True)
    fig.savefig(os.path.join(PLOTS, name))
    plt.close(fig)
    print('wrote', os.path.join('plots', name))


# --- 1. Outcome of each benchmark run, both codes -------------------------------

def fig_benchmark():
    ours, pysr = parse('benchmark_gp_holes.txt'), parse('benchmark_pysr.txt')
    names = [t[0] for t in TARGETS]
    rows = [n for n in names if n not in HARDER] + [None] + HARDER
    fig = plt.figure(figsize=(6.9, 7.4))
    ax = fig.add_axes([0.36, 0.03, 0.66, 0.80])
    ax.set_aspect('equal')
    ax.axis('off')
    order = {'right': 0, 'hole': 1, 'other': 2}
    for r, name in enumerate(rows):
        y = -r
        if name is None:
            continue
        ax.text(-0.6, y, PRETTY[name], ha='right', va='center', fontsize=9)
        for x0, runs, colour in ((0, ours, BLUE), (4.6, pysr, ORANGE)):
            kinds = sorted((outcome(q) for q in runs if q['target'] == name), key=order.get)
            for i, kind in enumerate(kinds):
                face = {'right': colour, 'hole': LIGHT_BLUE, 'other': GRAY}[kind]
                ax.add_patch(FancyBboxPatch((x0 + i + 0.09, y - 0.32), 0.82, 0.64, linewidth=0,
                                            boxstyle='round,pad=0,rounding_size=0.13', facecolor=face))
                if kind == 'hole':
                    ax.text(x0 + i + 0.5, y - 0.02, 'H', ha='center', va='center', fontsize=8,
                            fontweight='bold', color=INK)
    ax.text(1.5, 1.05, 'GP holes', ha='center', fontweight='bold')
    ax.text(6.1, 1.05, 'PySR, 60 s', ha='center', fontweight='bold')
    ax.text(-0.6, 1.05, 'Target', ha='right', color=INK2)
    ax.text(-0.6, -13, 'Harder; the last two cannot be\nwritten with the rewrite rules',
            ha='right', va='center', color=INK2, fontsize=8)
    ax.set_xlim(-0.5, 8.2)
    ax.set_ylim(-len(rows) + 0.4, 1.6)
    fig.legend(handles=[Patch(color=BLUE, label='Right, GP holes'), Patch(color=ORANGE, label='Right, PySR'),
                        Patch(color=LIGHT_BLUE, label='H  A hole is left'),
                        Patch(color=GRAY, label='Other formula')],
               loc='upper left', bbox_to_anchor=(0.01, 0.90), ncol=4, handlelength=1.0, handleheight=1.0,
               columnspacing=1.2)
    title(fig, 'Outcome of each run on the benchmark targets',
          '100 training points per target, 1% noise, 3 seeds. One block per run. Right: no hole,\n'
          'and the error outside the training box is below the noise level.')
    save(fig, '1_benchmark_outcomes.png')


# --- 2. The steps of two searches ------------------------------------------------

def fig_trace(cache):
    fig = plt.figure(figsize=(7.6, 6.2))
    panels = [('toy', [0.43, 0.50, 0.53, 0.33]), ('I.6.2', [0.43, 0.07, 0.53, 0.25])]
    for name, box in panels:
        trace = cache['trace'][name]
        ax = fig.add_axes(box)
        gain = [s - trace[0][0] for s, _, _ in trace]
        ys = np.arange(len(trace))[::-1]
        ax.set_xlim(0, 1.22 * max(gain))
        ax.set_ylim(-0.8, len(trace) - 0.4)
        # Bars with a rounded end: the units of x and y per inch give round corners.
        xpu = 1.22 * max(gain) / (box[2] * fig.get_figwidth())
        ypu = (len(trace) + 0.4) / (box[3] * fig.get_figheight())
        for y, g in zip(ys, gain):
            if g:
                ax.add_patch(FancyBboxPatch((0, y - 0.25), g, 0.5, linewidth=0, facecolor=BLUE,
                                            boxstyle=f"round,pad=0,rounding_size={0.04 * xpu}",
                                            mutation_aspect=ypu / xpu))
                ax.add_patch(plt.Rectangle((0, y - 0.25), 0.05 * xpu, 0.5, linewidth=0, facecolor=BLUE))
        for y, g, (_, key, rule) in zip(ys, gain, trace):
            ax.text(-0.02 * max(gain), y, key.replace('*', '·'), ha='right', va='center', fontsize=8.5)
            ax.text(g + 0.015 * max(gain), y, f"+{g:.0f}" if g else 'start', va='center', fontsize=8.5)
            if g:
                ax.text(g + 0.015 * max(gain), y - 0.36, rule, va='center', fontsize=7.5, color=INK2)
        ax.set_yticks([])
        ax.spines['left'].set_position(('data', 0))
        ax.grid(axis='x')
        ax.set_xlabel('Gain in score over the single hole (log evidence with costs)')
        ax.set_title(f"Target: {PRETTY[name]}", loc='left', fontsize=9.5, x=-0.78, pad=8)
    title(fig, 'A search, step by step',
          'The best model after each step and the rule that produced it. H is a hole, ? a fitted\n'
          'constant. 100 points, 1% noise.')
    save(fig, '2_search_steps.png')


# --- 3. A slice through the toy target, inside and outside the training range ----

def fig_extrapolation(cache):
    c = cache['slice']
    x = np.array(c['x'])
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    fig.subplots_adjust(left=0.08, right=0.80, top=0.78, bottom=0.13)
    ax.axvspan(c['lo'], c['hi'], color=BAND, lw=0)
    ax.text((c['lo'] + c['hi']) / 2, 1.0, 'training range of x1', transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', color=INK2, fontsize=8)
    ax.plot(x, c['target'], color=GRAY, lw=5, solid_capstyle='round', label='Target')
    ax.plot(x, c['gp'], color=AQUA, label='Single hole (plain GP regression)')
    ax.plot(x, c['formula'], color=BLUE, label='Formula found by the search')
    for key, text, dy in (('gp', 'single hole', 0), ('formula', 'formula found', 6), ('target', 'target', -6)):
        ax.annotate(text, (x[-1], c[key][-1]), xytext=(6, dy), textcoords='offset points', va='center',
                    fontsize=8.5)
    ax.grid(axis='y')
    ax.set_xlabel('x1   (x2 = x3 = 2)')
    ax.set_ylabel('y')
    ax.set_xlim(x[0], x[-1])
    fig.legend(loc='upper left', bbox_to_anchor=(0.07, 0.885), ncol=3, columnspacing=1.6)
    title(fig, 'Inside the training range both models fit; outside it only the formula holds',
          f"Target {PRETTY['toy']}, 100 points, 1% noise. Found: {c['text'].replace('*', '·')}")
    save(fig, '3_extrapolation.png')


# --- 4. Fewer points --------------------------------------------------------------

def fig_few_points():
    ours = parse('few_points_gp_holes_noise1.txt') + parse('few_points_gp_holes_noise20.txt')
    for r in ours[:len(parse('few_points_gp_holes_noise1.txt'))]:
        r['noise'] = 0.01
    for r in ours[len(parse('few_points_gp_holes_noise1.txt')):]:
        r['noise'] = 0.2
    pysr = parse('few_points_pysr.txt') + [r for r in parse('benchmark_pysr.txt') if r['target'] == 'toy']
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.90, top=0.74, bottom=0.14, wspace=0.16)
    for ax, noise in zip(axes, (0.01, 0.2)):
        ax.axhline(noise, color=INK2, lw=0.75)
        ax.text(0.98, noise * 1.12, 'noise level', transform=ax.get_yaxis_transform(), color=INK2,
                fontsize=8, ha='right', va='bottom')
        ends = {}
        for runs, colour, label in ((pysr, ORANGE, 'PySR, 60 s'), (ours, BLUE, 'GP holes')):
            sel = [r for r in runs if abs(r['noise'] - noise) < 1e-9]
            ns = sorted({r['n'] for r in sel})
            med = [np.median([r['rout'] for r in sel if r['n'] == n]) for n in ns]
            ends[label] = med[-1]
            ax.plot([r['n'] for r in sel], [r['rout'] for r in sel], 'o', color=colour, ms=3.5, alpha=0.45,
                    markeredgewidth=0)
            ax.plot(ns, med, '-o', color=colour, label=label, **RING)
            ends[label] = (ns[-1], med[-1])
        for label, (n, m) in ends.items():
            above = m >= max(v[1] for v in ends.values())
            ax.annotate(label, (n, m), xytext=(8, 7 if above else -7), textcoords='offset points',
                        va='center', fontsize=8.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ticks = [10, 15, 25, 40, 60, 100, 200]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.set_xticks([], minor=True)
        ax.set_xlim(8.5, 330)
        ax.grid(axis='y')
        ax.set_xlabel('Training points')
        ax.set_title(f"{noise:.0%} noise", loc='left', fontsize=9.5)
    axes[0].set_ylabel('Error outside the training box')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(0.08, 0.86), ncol=2)
    title(fig, 'With few points the GP-hole search still finds the formula',
          f"Target {PRETTY['toy']}. Lines: median of 3 seeds; small dots: the single runs. Errors in units of "
          'the\nstandard deviation of the target.')
    save(fig, '4_few_points.png')


# --- 5. Error inside against outside the training box -----------------------------

def fig_inside_outside():
    ours, pysr = parse('benchmark_gp_holes.txt'), parse('benchmark_pysr.txt')
    lost = sum(not np.isfinite(r['rout']) for r in pysr)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.80, bottom=0.11)
    for v in (NOISE,):
        ax.axhline(v, color=INK2, lw=0.75)
        ax.axvline(v, color=INK2, lw=0.75)
    ax.text(NOISE * 1.08, 2.2e-5, 'noise level', color=INK2, fontsize=8, rotation=90, va='bottom')
    full = [r for r in ours if not r['holes']]
    hole = [r for r in ours if r['holes']]
    ax.plot([r['rin'] for r in full], [r['rout'] for r in full], 'o', color=BLUE, label='GP holes', **RING,
            zorder=3)
    ax.plot([r['rin'] for r in hole], [r['rout'] for r in hole], 'o', color='white', markeredgecolor=BLUE,
            markeredgewidth=1.5, label='GP holes, a hole is left', zorder=3)
    ax.plot([r['rin'] for r in pysr], [r['rout'] for r in pysr], 'o', color=ORANGE, label='PySR, 60 s', **RING,
            zorder=2)
    good = [sum(r['rin'] < NOISE and r['rout'] < NOISE for r in runs) for runs in (ours, pysr)]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1.5e-5, 0.3)
    ax.set_ylim(1.5e-5, 300)
    ax.text(2e-5, 6e-3, f"accurate inside and outside:\nGP holes {good[0]} runs, PySR {good[1]}",
            color=INK2, fontsize=8.5, va='top')
    ax.text(2e-5, 120, 'fits the data, wrong outside', color=INK2, fontsize=8.5, va='top')
    ax.grid(True)
    ax.set_xlabel('Error inside the training box')
    ax.set_ylabel('Error outside the training box')
    fig.legend(loc='upper left', bbox_to_anchor=(0.11, 0.885), ncol=3, columnspacing=1.4, handletextpad=0.3)
    title(fig, 'Every benchmark run: error inside against outside the box',
          f"17 targets, 3 seeds, 100 points, 1% noise. Errors in units of the standard deviation of the\n"
          f"target. {lost} PySR runs with an undefined error outside the box are not shown.")
    save(fig, '5_inside_outside.png')


# --- 6, 7. Targets with a series and no closed form --------------------------------

def reference(target, n_terms):
    """Error and coefficients of the least-squares fit of the first terms of the series."""
    X = sample(target, 20000, np.random.default_rng(0))
    f = target[2](X)
    A = np.column_stack([target[7](X) * X[:, target[4]] ** p for p in target[5][:n_terms]])
    c = np.linalg.lstsq(A, f, rcond=None)[0]
    return np.sqrt(np.mean((A @ c - f) ** 2)) / f.std(), c


def power(p):
    return {0: '1'}.get(p, 'x' + ''.join('⁰¹²³⁴⁵⁶⁷⁸⁹'[int(d)] for d in str(p)) if p > 1 else 'x')


def fig_series_coefficients(cache):
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 4.0), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.70, bottom=0.14, wspace=0.08)
    for ax, target in zip(axes, SERIES):
        powers, taylor = target[5], np.array(target[6])
        ax.axhline(1, color=INK2, lw=0.75)
        runs = [r for r in cache['series'] if r['target'] == target[0] and r['noise'] == 0.001 and r['series']]
        for i, r in enumerate(runs):
            k = r['terms']
            dx = (i - (len(runs) - 1) / 2) * 0.24
            fit = reference(target, k)[1] / taylor[:k]
            found = np.array([r['coef'][p] for p in powers[:k]]) / taylor[:k]
            ax.plot(np.arange(k) + dx, fit, '_', color=INK2, ms=13, markeredgewidth=1.5,
                    label='Least-squares fit of as many terms' if i == 0 else None)
            ax.plot(np.arange(k) + dx, found, 'o', color=BLUE, label='Found by the search' if i == 0 else None,
                    **RING)
        ax.set_xticks(range(len(powers)), [power(p) for p in powers])
        ax.set_xlim(-0.6, len(powers) - 0.4)
        ax.grid(axis='y')
        ax.set_title(PRETTY[target[0]] + ('' if len(runs) == 2 else '   (1 of 2 runs)'), loc='left', fontsize=9.5)
        ax.set_xlabel('Term of the series' + (' in x3' if target[0] == 'pendulum' else ''))
    axes[0].set_ylabel('Coefficient / coefficient of the series')
    axes[0].set_ylim(0, 2.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(0.06, 0.835), ncol=2)
    title(fig, 'At 0.1% noise the search returns the series, with the coefficients of a fit',
          'Library of powers and polynomials, 100 points. One dot per run and term. The last terms differ '
          'from the series:\nthey take up the terms that are left out, as in a least-squares fit of the '
          'noise-free target, and the data hardly fix them.')
    save(fig, '6_series_coefficients.png')


def fig_series_terms(cache):
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.7), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.68, bottom=0.15, wspace=0.08)
    noises = [0.1, 0.01, 0.001]
    for ax, target in zip(axes, SERIES):
        need = [next(k for k in range(1, 6) if reference(target, k)[0] < noise) for noise in noises]
        ax.plot(range(3), need, '_', color=INK2, ms=26, markeredgewidth=1.5,
                label='Terms of the series a least-squares fit needs')
        for j, noise in enumerate(noises):
            runs = [r for r in cache['series'] if r['target'] == target[0] and r['noise'] == noise]
            for i, r in enumerate(runs):
                ax.plot(j + (i - 0.5) * 0.22, r['terms'], 'o', color=BLUE if r['series'] else 'white',
                        markeredgecolor='white' if r['series'] else BLUE, markeredgewidth=1.5)
        ax.set_xticks(range(3), ['10%', '1%', '0.1%'])
        ax.set_xlim(-0.6, 2.6)
        ax.set_yticks(range(1, 7))
        ax.set_ylim(0.4, 6.6)
        ax.grid(axis='y')
        ax.set_title(PRETTY[target[0]], loc='left', fontsize=9.5)
        ax.set_xlabel('Noise')
    axes[0].set_ylabel('Fitted coefficients')
    handles = [Line2D([], [], marker='o', ls='', color=BLUE, label='Search: the series', **RING),
               Line2D([], [], marker='o', ls='', color='white', markeredgecolor=BLUE, markeredgewidth=1.5,
                      label='Search: another polynomial or power'),
               Line2D([], [], marker='_', ls='', color=INK2, ms=14, markeredgewidth=1.5,
                      label='Terms of the series a least-squares fit needs')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.06, 0.82), ncol=3, columnspacing=1.4,
               handletextpad=0.3)
    title(fig, 'Only at 0.1% noise does the search fit as few coefficients as the series needs',
          'Library of powers and polynomials, 100 points, 2 seeds. Targets without a closed form.')
    save(fig, '7_series_terms.png')


# --- 8. Run time ---------------------------------------------------------------------

def fig_runtime():
    ours = parse('benchmark_gp_holes.txt')
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    fig.subplots_adjust(left=0.11, right=0.95, top=0.76, bottom=0.14)
    ax.axhline(60, color=INK2, lw=0.75)
    ax.text(1.62, 64, 'PySR budget (60 s on 4 threads)', color=INK2, fontsize=8, va='bottom')
    rng = np.random.default_rng(1)
    for t in TARGETS:
        s = np.mean([r['seconds'] for r in ours if r['target'] == t[0]])
        ax.plot(len(t[3]) + rng.uniform(-0.12, 0.12), s, 'o', color=BLUE, **RING)
    ax.set_xticks([2, 3, 4, 5])
    ax.set_xlim(1.6, 5.4)
    ax.set_ylim(0, 215)
    ax.grid(axis='y')
    ax.set_xlabel('Inputs of the target')
    ax.set_ylabel('Seconds per run')
    title(fig, 'Run time of the GP-hole search',
          'Mean of 3 seeds per target, 100 points, 2 worker processes on a machine that was\nshared with a '
          'second run.')
    save(fig, '8_run_time.png')


# --- Reruns ----------------------------------------------------------------------------

def collect(workers):
    """Rerun the searches whose models the figures need."""
    import gpholes
    from run_benchmark import data
    cache = dict(trace={}, series=[])
    by_name = {t[0]: t for t in TARGETS}
    for name in ('toy', 'I.6.2'):
        X, y, ystd, tests = data(by_name[name], 100, NOISE, 0)
        model, trace, info = gpholes.search(X, y, seed=0, verbose=False, workers=workers,
                                            max_power=3, library=('pow', 'log', 'exp', 'aff', 'sin'))
        cache['trace'][name] = [(float(s), key, rule) for s, key, rule in trace]
        if name == 'toy':
            lo, hi = by_name[name][3][0]
            x = np.linspace(0.0, hi + 1.3, 300)
            Xs = np.column_stack([x, 2 * np.ones_like(x), 2 * np.ones_like(x)])
            hole = gpholes.Model([gpholes.Term(feats=[gpholes.var(j) for j in range(3)])])
            hole.fit(X, y, restarts=6, rng=np.random.default_rng(0)).condition(X, y)
            cache['slice'] = dict(x=x.tolist(), lo=lo, hi=hi, text=model.text(ystd),
                                  target=by_name[name][2](Xs).tolist(),
                                  gp=(hole.predict(Xs) * ystd).tolist(),
                                  formula=(model.predict(Xs) * ystd).tolist())
    for target in SERIES:
        j, (lo, hi) = target[4], target[3][target[4]]
        for noise in (0.1, 0.01, 0.001):
            for seed in range(2):
                X, y, ystd, tests = data(target, 100, noise, seed)
                model, _, _ = gpholes.search(X, y, seed=seed, verbose=False, workers=workers,
                                             max_power=9, library=('pow', 'poly'))
                # The model along the input of the series, with the factor in front equal to 1.
                x = np.linspace(lo, hi, 400)
                Xs = np.full((400, len(target[3])), 2.0)
                Xs[:, j] = x
                f = model.predict(Xs) * ystd
                p = np.polynomial.Polynomial.fit(x, f, 9)
                coef = p.convert().coef
                exact = np.max(np.abs(p(x) - f)) < 1e-6 * np.abs(f).max() and not model.n_holes()
                off = [k for k in range(10) if k not in target[5]]
                series = bool(exact and all(abs(coef[k]) * hi ** k < 1e-4 * np.abs(f).max() for k in off))
                cache['series'].append(dict(target=target[0], noise=noise, seed=seed, text=model.text(ystd),
                                            coef=coef.tolist(), series=series, terms=int(model.n_par - 2)))
                print(target[0], noise, seed, series, model.text(ystd), flush=True)
    json.dump(cache, open(CACHE, 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--collect', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    if a.collect:
        collect(a.workers)
    cache = json.load(open(CACHE))
    fig_benchmark()
    fig_trace(cache)
    fig_extrapolation(cache)
    fig_few_points()
    fig_inside_outside()
    fig_series_coefficients(cache)
    fig_series_terms(cache)
    fig_runtime()
