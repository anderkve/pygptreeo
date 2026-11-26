"""Write a random sample of the Feynman equations of the SRSD benchmark (Matsubara
et al., arXiv:2206.10540), with its ranges of the inputs: srsd_sample.json for the
first sample, srsd_sample2.json for the second, and so on.

    python make_srsd_sample.py PATH_TO_CLONE_OF github.com/omron-sinicx/srsd-benchmark [--sample 2]

A sample is drawn from the equations with at most 5 inputs that were not used to
develop the search (USED) and are not in an earlier sample. The file
datasets/feynman.py of the clone is parsed, not run: the formula of each equation is
kept as an expression of x[0], x[1], ... with numpy functions and the constants of
that file.
"""
import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np

USED = ['i.6.20', 'i.8.14', 'i.10.7', 'i.12.2', 'i.12.11', 'i.13.4', 'i.14.3', 'i.18.4', 'i.24.6',
        'i.30.5', 'i.34.8', 'i.37.4', 'i.40.1', 'i.44.4', 'i.50.26', 'ii.11.28',
        'i.39.22', 'i.41.16']               # the last two are laws of EmpiricalBench
N_SAMPLE = 15
NUMPY = {'sqrt', 'exp', 'log', 'sin', 'cos', 'tan', 'tanh', 'arcsin', 'arccos', 'arctan', 'pi', 'abs'}


def check(node, names):
    """Raise unless the expression only has numbers, arithmetic, x[i], np.f and known names."""
    for n in ast.walk(node):
        ok = isinstance(n, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Load, ast.operator,
                            ast.unaryop, ast.Call, ast.Subscript, ast.Attribute, ast.Name))
        if isinstance(n, ast.Attribute):
            ok = isinstance(n.value, ast.Name) and n.value.id == 'np' and n.attr in NUMPY
        if isinstance(n, ast.Name):
            ok = n.id in names or n.id in ('np', 'x')
        if isinstance(n, ast.Call):
            ok = isinstance(n.func, ast.Attribute) and not n.keywords
        if not ok:
            raise ValueError(ast.dump(n))


def value(node, constants):
    check(node, constants)
    return float(eval(compile(ast.Expression(node), '<range>', 'eval'), {'__builtins__': {}, 'np': np}, constants))


def sample_file(k):
    """The file of sample k."""
    return Path(__file__).parent / ('srsd_sample.json' if k == 1 else f'srsd_sample{k}.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('clone')
    ap.add_argument('--sample', type=int, default=1)
    a = ap.parse_args()
    tree = ast.parse((Path(a.clone) / 'datasets' / 'feynman.py').read_text())
    constants = {n.targets[0].id: n.value.value for n in tree.body
                 if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)}
    equations = []
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        doc = ast.get_docstring(cls)
        name = [n.value.value for n in cls.body if isinstance(n, ast.Assign)][0].replace('feynman-', '')
        init, func = [n for n in cls.body if isinstance(n, ast.FunctionDef)]
        objs = [n for n in ast.walk(init) if isinstance(n, ast.Assign)
                and getattr(n.targets[0], 'id', None) == 'sampling_objs'][0]
        samplers = []
        for call in objs.value.elts:
            kw = {k.arg: k.value.value for k in call.keywords}
            samplers.append(dict(kind=call.func.id.replace('Sampling', '').lower(),
                                 low=value(call.args[0], constants), high=value(call.args[1], constants),
                                 positive=kw.get('uses_positive', True), negative=kw.get('uses_negative', True)))
        assert len(func.body) == 1 and isinstance(func.body[0], ast.Return), name
        check(func.body[0].value, constants)
        equations.append(dict(name=name, formula=re.search(r'Raw: (.*)', doc).group(1),
                              expression=ast.unparse(func.body[0].value), samplers=samplers))
    used = set(USED)
    for k in range(1, a.sample):
        used |= {e['name'] for e in json.load(open(sample_file(k)))['equations']}
    eligible = [e for e in equations if len(e['samplers']) <= 5 and e['name'] not in used]
    pick = np.random.default_rng(a.sample - 1).choice(len(eligible), N_SAMPLE, replace=False)
    sample = [eligible[i] for i in sorted(pick)]
    json.dump(dict(constants=constants, equations=sample), open(sample_file(a.sample), 'w'), indent=1)
    print(len(equations), 'equations,', len(eligible), 'eligible')
    for e in sample:
        print(f"{e['name']:12s} {len(e['samplers'])}  {e['formula'][:100]}")
        print('             ' + '; '.join(f"{s['kind']} [{s['low']:g}, {s['high']:g}]" +
                                           ('' if s['negative'] else ' >0') for s in e['samplers']))
