"""Write empirical_bench.json: the data of EmpiricalBench, the benchmark of the
PySR paper (Cranmer 2023, arXiv:2305.01582), as its methods were given them.

    python make_empirical_bench.py PATH_TO_CLONE_OF github.com/MilesCranmer/pysr_paper

Original data are read from the clone. Generated data follow src/scripts/benchmark.py
of the clone, with the same random numbers, and equations.yml for the ranges, sizes
and noise. Targets that the paper plots on a log scale are given as their logarithm.
"""
import csv
import json
import sys
from hashlib import sha256
from pathlib import Path

import mpmath as mp
import numpy as np
import yaml

mp.mp.dps = 50
FORMULAS = {      # key: (text of the target as given, function of the inputs)
    'newton': ('log(G*m1*m2/r^2)', lambda r, m1, m2: mp.mpf('6.67408e-11') * m1 * m2 / r ** 2),
    'planck': ('log(2*h*nu^3/c^2/(exp(h*nu/(k*T)) - 1))',
               lambda nu, T: 2 * mp.mpf('6.62607004e-34') * nu ** 3 / mp.mpf('299792458') ** 2
               / (mp.exp(mp.mpf('6.62607004e-34') * nu / (mp.mpf('1.38064852e-23') * T)) - 1)),
    'schechter': ('log(phi/L0*(L/L0)^alpha*exp(-L/L0))',
                  lambda L: mp.mpf('0.002') / mp.mpf('2.5e8') * (L / mp.mpf('2.5e8')) ** mp.mpf('-1.20')
                  * mp.exp(-L / mp.mpf('2.5e8'))),
    'ideal_gas': ('log(n*R*T/V)', lambda n, T, V: n * mp.mpf('8.314') * T / V),
    'rydberg': ('log(1/(R*(1/n_1^2 - 1/n_2^2)))',
                lambda n_1, n_2: 1 / (mp.mpf('1.097e7') * (1 / n_1 ** 2 - 1 / n_2 ** 2))),
}
ORIGINAL = {'hubble': 'v = H0*D', 'kepler': 'P = k*a^1.5', 'leavitt': 'M = alpha*log10(P) + delta',
            'bode': 'log(0.4 + 0.3*2^n)'}


def generated(key, gen):
    rstate = np.random.RandomState(np.frombuffer(sha256(key.encode()).digest(), dtype=np.uint32))
    size, X = int(gen['size']), {}
    if 'ranges' in gen:
        for arg in gen['args']:
            low, high = float(gen['ranges'][arg][0]), float(gen['ranges'][arg][1])
            if len(gen['ranges'][arg]) == 3 and gen['ranges'][arg][2] == 'linear':
                X[arg] = rstate.rand(size) * (high - low) + low
            else:
                X[arg] = np.exp(rstate.rand(size) * (np.log(high) - np.log(low)) + np.log(low))
    else:                                               # Rydberg: two quantum numbers
        X = {'n_1': [], 'n_2': []}
        for _ in range(size):
            n_1 = rstate.randint(1, 7)
            X['n_1'].append(float(n_1))
            X['n_2'].append(float(rstate.randint(n_1 + 1, 8)))
        X = {k: np.array(v) for k, v in X.items()}
    y = np.array([FORMULAS[key][1](**{k: mp.mpf(float(v[i])) for k, v in X.items()}) for i in range(size)])
    scale = 0.1 if str(gen['noise']['scale']) == 'None' else float(gen['noise']['scale'])
    assert gen['noise']['type'] == 'relative'
    y = y + rstate.randn(size) * y * scale
    return X, [float(mp.log(v)) for v in y], FORMULAS[key][0]


def original(key, problem, folder):
    data = problem['data']
    if 'csv' in data:
        rows = list(csv.DictReader(open(folder / data['csv']['filename'])))
        if key == 'hubble':
            return {'D': [float(r['D']) for r in rows]}, [float(r['v']) for r in rows]
        return {'P': [10 ** float(r['logP']) for r in rows]}, [float(r['M']) for r in rows]
    X = {k: [float(v) for v in vals] for k, vals in data['inputs'].items()}
    y = [float(v) for v in data['target']]
    return X, ([float(np.log(v)) for v in y] if key == 'bode' else y)


if __name__ == '__main__':
    folder = Path(sys.argv[1]) / 'src' / 'data' / 'empirical_bench'
    problems = yaml.safe_load(open(folder / 'equations.yml'))['problems']
    out = []
    for name, problem in problems.items():
        key = problem['key']
        if key in ORIGINAL:
            X, y = original(key, problem, folder)
            text, source = ORIGINAL[key], 'original data'
        elif key in FORMULAS:
            X, y, text = generated(key, problem['data_generator'])
            source = 'generated'
        else:
            continue                                    # Tully-Fisher is not in the paper's benchmark
        out.append(dict(key=key, name=name, target=text, source=source, inputs=list(X),
                        X=np.column_stack([X[k] for k in X]).tolist(), y=list(y)))
        print(f"{key:10s} {source:14s} {len(y):4d} points, inputs {list(X)}")
    json.dump(out, open(Path(__file__).parent / 'empirical_bench.json', 'w'), indent=0)
