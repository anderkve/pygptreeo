"""Rules for queries outside the data, on the records of extrapolation_diag.py.

    python extrapolation_report.py results/extrapolation2

Each rule picks the neighbours' prediction or the leaf's own GP per query; errors
are in units of the target's range, relative to the shipped rule (neighbours
unless their consistency exceeds 10 or the leaf prefers its own GP).
"""
import glob
import sys
import numpy as np

folder = sys.argv[1]
groups = {}
for path in sorted(glob.glob(folder + '/*.npz')):
    parts = path.split('/')[-1][:-4].split('__')
    record = np.load(path)
    rows = record['rows']
    groups.setdefault(tuple(parts[2:]) or ('1e-3', 'default'), []).append(np.column_stack([rows, np.full(len(rows), float(record['range']))]))

RULES = {
    'shipped rule': lambda r: r['cons'] <= 10,
    'own GP always': lambda r: np.zeros(len(r['cons']), bool),
    'limit 5': lambda r: r['cons'] <= 5,
    'limit 3': lambda r: r['cons'] <= 3,
    'nearest > 1 spread: limit 5': lambda r: (r['cons'] <= 10) & ~((r['dnear'] > 1) & (r['cons'] > 5)),
    'nearest > 1 spread: limit 3': lambda r: (r['cons'] <= 10) & ~((r['dnear'] > 1) & (r['cons'] > 3)),
    'nearest > 1 spread: limit 2': lambda r: (r['cons'] <= 10) & ~((r['dnear'] > 1) & (r['cons'] > 2)),
    'nearest > 2 spreads: limit 3': lambda r: (r['cons'] <= 10) & ~((r['dnear'] > 2) & (r['cons'] > 3)),
    'outside centre > 2: limit 5': lambda r: (r['cons'] <= 10) & ~((r['dcen'] > 2) & (r['cons'] > 5)),
    'outside centre > 2: limit 3': lambda r: (r['cons'] <= 10) & ~((r['dcen'] > 2) & (r['cons'] > 3)),
    'outside centre > 3: limit 3': lambda r: (r['cons'] <= 10) & ~((r['dcen'] > 3) & (r['cons'] > 3)),
    'mean within 0.5 range of targets': lambda r: (r['cons'] <= 10) & (np.abs(r['outside']) <= 0.5),
    'mean within 1 range of targets': lambda r: (r['cons'] <= 10) & (np.abs(r['outside']) <= 1.0),
}
NAMES = ('truth', 'nb', 'nbs', 'cons', 'own', 'owns', 'dfit', 'dcen', 'ratio', 'pref', 'scaler', 'dnear', 'shipped', 'shippeds', 'eased', 'easeds', 'tmax', 'outside', 'range')
for key, parts in groups.items():
    A = np.vstack(parts)
    r = dict(zip(NAMES, A.T))
    far = r['dnear'] > 1
    print('\n=== noise %s of the value, %s kernel: %d queries in %d runs; %.0f%% further than 1 leaf spread from their nearest neighbour' % (key[0], key[1], len(A), len(parts), 100 * far.mean()))
    e_own = np.abs(r['own'] - r['truth']) / r['range']
    base = None
    print('%-36s %8s %8s %8s %9s %9s %10s' % ('rule', 'all', 'far', 'near', 'largest', '>3x own', 'own share'))
    rules = dict(RULES)
    rules['eased noise when consistency > 1'] = None
    rules['eased noise when far'] = None
    for name, rule in rules.items():
        if rule is None:
            use = (r['cons'] <= 10) & (r['pref'] == 0)
            mean = np.where(use, r['nb'], r['own'])
            eased = use & (r['cons'] > 1) & (far if 'far' in name else True)
            mean = np.where(eased, r['eased'], mean)
        else:
            use = rule(r) & (r['pref'] == 0)
            mean = np.where(use, r['nb'], r['own'])
        e = np.abs(mean - r['truth']) / r['range']
        rms = [np.sqrt((e[s] ** 2).mean()) for s in (slice(None), far, ~far)]
        if base is None:
            base = rms
        print('%-36s %8.3f %8.3f %8.3f %9.2f %8.2f%% %9.1f%%' % (name, rms[0] / base[0], rms[1] / base[1], rms[2] / base[2], e.max(), 100 * ((e > 3 * e_own) & (e > 0.1)).mean(), 100 * (~use).mean()))
    e = np.abs(r['shipped'] - r['truth']) / r['range']
    print('%-36s %8.3f %8.3f %8.3f %9.2f %8.2f%%   (with kernels mixed at the cuts)' % ('the package', np.sqrt((e ** 2).mean()) / base[0], np.sqrt((e[far] ** 2).mean()) / base[1], np.sqrt((e[~far] ** 2).mean()) / base[2], e.max(), 100 * ((e > 3 * e_own) & (e > 0.1)).mean()))
