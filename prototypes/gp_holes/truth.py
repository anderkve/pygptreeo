"""The right formula of each target as a model that the search can write, and
its score on a data set.

    python truth.py [--sets benchmark srsd srsd2 empirical docs]

checks the formulas against the targets without noise.

An entry of FORMS is a formula, a pair (function around the expression,
formula), or a list of such alternatives. A formula has one or more terms:
T(factor, ...) is a product with a weight in front, and H(input, ...,
factors=[...]) a hole. In them:

    x1, x2, ...           the inputs of the target
    pw(u, p)              u^p
    log(u)                log(u), or log|u| for a u that changes sign
    exp(u, c, v=1)        exp(c*u), with u^2 for v=2 and 1/u for v=-1 in place of u
    aff(u, c, p, v=1)     (1 + c*u)^p, the same
    sin(u, b, a=0), cos(u, b)       sin(a + b*u), cos(b*u)
    mul(u, v, p=1), add(u, v, c=1)  u*v^p and u + c*v, as one input

The nodes are those that the library and the pairings give for the data, so a
formula that they cannot write for these inputs is left out. Where the formula
is out of reach of the rules, the entry is the model with a hole that they
should arrive at.
"""
import argparse
import copy

import numpy as np

from gpholes import Model, Term, library, nice, pairings, transforms, var, warp

C = 299792458.0
FORMS = {
    # The benchmark.
    'toy': "T(x2, sin(x1, 2)), T(pw(x3, 2))",
    'I.14.3': "T(x1, x2, x3)",
    'I.12.2': "T(x1, x2, pw(x3, -2))",
    'I.34.8': "T(x1, x2, x3, pw(x4, -1))",
    'I.13.4': "T(x1, pw(x2, 2)), T(x1, pw(x3, 2)), T(x1, pw(x4, 2))",
    'I.24.6': "T(x1, pw(x2, 2), pw(x4, 2)), T(x1, pw(x3, 2), pw(x4, 2))",
    'I.6.2': "T(exp(mul(x1, x2, -1), -0.5, 2), pw(x2, -1))",
    'I.12.11': "T(x1, x2), T(x1, x3, x4, sin(x5, 1))",
    'I.37.4': ["T(x1), T(x2), T(pw(x1, 0.5), pw(x2, 0.5), cos(x3, 1))",
               "T(add(x1, x2)), T(pw(mul(x1, x2), 0.5), cos(x3, 1))"],
    'I.40.1': "T(x1, exp(mul(mul(x2, x3), x4, -1), -1))",
    'isothermal': ["T(x1, x2, log(mul(x3, x4, -1)))", "T(x1, x2, log(x3)), T(x1, x2, log(x4))"],
    # cos^2(u) = (1 + cos(2u))/2
    'I.50.26': "T(x1, cos(mul(x2, x3), 1)), T(x1, x4), T(x1, x4, cos(mul(x2, x3), 2))",
    'I.10.7': "T(x1, aff(mul(x2, x3, -1), -1, -0.5, 2))",
    'I.30.5': "H(mul(mul(x1, x2, -1), x3, -1))",                 # arcsin is not in the library
    'I.8.14': ('square', "T(pw(add(x1, x2, -1), 2)), T(pw(add(x3, x4, -1), 2))"),
    'I.18.4': ["T(x3, aff(mul(x2, x1, -1), 1, -1)), T(x4, aff(mul(x1, x2, -1), 1, -1))",
               "T(x4), T(add(x3, x4, -1), aff(mul(x2, x1, -1), 1, -1))"],
    'II.11.28': "T(aff(mul(x1, x2), -1/3, -1))",                 # 3/(1 - x1*x2/3) - 2
    # The examples of PySR's documentation.
    'simple': "T(cos(x4, 1)), T(pw(x1, 2))",
    'inverse': "T(pw(x1, -1))",
    'selection': "T(pw(x4, 2)), T(pw(x20, 2))",
    'denoising': "T(exp(x1, 1)), T(x2), T(x3)",
    # EmpiricalBench.
    'hubble': "T(x1)",
    'kepler': "T(pw(x1, 1.5))",
    'newton': ["T(log(x1)), T(log(x2)), T(log(x3))", ('exp', "T(pw(x1, -2), x2, x3)")],
    # 3*log(nu) - h*nu/(k*T) - log(1 - exp(-h*nu/(k*T))), the last term as a hole
    'planck': "T(log(x1)), T(mul(x1, x2, -1)), H(log(mul(x1, x2, -1)))",
    'leavitt': "T(log(x1))",
    'schechter': "T(log(x1)), T(x1)",
    'bode': ('exp', "T(exp(x1, 0.693147))"),
    'ideal_gas': ["T(log(x1)), T(log(x2)), T(log(x3))", ('exp', "T(x1, x2, pw(x3, -1))")],
    'rydberg': ('expneg', "T(pw(x1, -2)), T(pw(x2, -2))"),
    # The first sample of SRSD.
    'i.6.20b': [('log', "T(pw(x3, -2), pw(add(x1, x2, -1), 2)), T(log(x3))"),
                "T(exp(mul(add(x1, x2, -1), x3, -1), -0.5, 2), pw(x3, -1))"],
    'i.12.4': "T(x1, pw(x2, -2))",
    'i.14.4': ["T(x1, pw(x2, 2))", ('log', "T(log(x1)), T(log(x2))")],
    'i.27.6': ('inv', "T(pw(x1, -1)), T(x2, pw(x3, -1))"),
    'i.34.27': ["T(x1)", ('log', "T(log(x1))")],
    'i.39.11': "T(x2, x3, pw(x1, -1), aff(x1, -1, -1, -1))",    # 1/(x1 - 1) = (1/x1)/(1 - 1/x1)
    'ii.11.20': "T(x1, pw(x2, 2), x3, pw(x4, -1))",
    'ii.13.34': "T(x1, x2, aff(x2, -1/C**2, -0.5, 2))",
    'ii.27.18': ["T(pw(x1, 2))", ('log', "T(log(x1))")],
    'ii.34.29b': "T(x1, x2, x3)",
    'ii.35.21': "T(x1, pw(x2, 2), x3, pw(x4, -1))",             # tanh(z) = z for the z of the data
    'iii.10.19': "T(x1, pw(add(add(pw(x2, 2), pw(x3, 2)), pw(x4, 2)), 0.5))",
    'bonus.8': ["T(x1)", ('log', "T(log(x1))")],                # the correction is 1e-9 in the data
    'bonus.18': "T(x1, pw(x2, -2)), T(pw(x3, 2))",
    # The second sample of SRSD.
    'i.12.5': "T(x1, x2)",
    'i.25.13': "T(x1, pw(x2, -1))",
    'i.34.14': "T(x1), T(x2)",                                   # the further factor is below 1.005
    'i.43.16': "T(x1, x2, x3, pw(x4, -1))",
    'i.47.23': ["T(pw(x1, 0.5), pw(x2, 0.5), pw(x3, -0.5))", ('log', "T(log(x1)), T(log(x2)), T(log(x3))")],
    'ii.8.7': ["T(pw(x1, 2), pw(x2, -1))", ('log', "T(log(x1)), T(log(x2))")],
    'ii.8.31': ["T(pw(x1, 2))", ('log', "T(log(x1))")],
    'ii.11.3': "T(x1, x2, pw(x3, -1), pw(x4, -2), aff(mul(x5, x4, -1), -1, -1, 2))",
    'ii.38.3': "T(x1, x2, x3, pw(x4, -1))",
    'iii.13.18': "T(x1, pw(x2, 2), x3)",
    # sin^4(u/2) = (3 - 4*cos(u) + cos(2u))/8
    'bonus.1': ('inv', "T(pw(x3, 2), pw(x1, -2), pw(x2, -2)), T(pw(x3, 2), pw(x1, -2), pw(x2, -2), cos(x4, 1)), "
                       "T(pw(x3, 2), pw(x1, -2), pw(x2, -2), cos(x4, 2))"),
    'bonus.7': ('square', "T(x1), T(x2, pw(x3, -2))"),
    # (x5 - 1)/(x5 + 2) = 1 - 1.5/(1 + x5/2)
    'bonus.14': "T(x1, cos(x2, 1), x3), T(x1, cos(x2, 1), pw(x4, 3), pw(x3, -2)), "
                "T(x1, cos(x2, 1), pw(x4, 3), pw(x3, -2), aff(x5, 0.5, -1))",
    'bonus.19': "T(x1, pw(x2, -2)), T(pw(x3, 2)), T(pw(x3, 2), x4)",
    # sin^2(u) = (1 - cos(2u))/2
    'bonus.20': "T(pw(mul(x1, x2, -1), 3)), T(mul(x1, x2, -1)), T(pw(mul(x1, x2, -1), 2)), "
                "T(pw(mul(x1, x2, -1), 2), cos(x3, 2))",
}


class Build:
    """The functions of the formulas for the inputs X. inputs has the number
    that each column of X has in the formula, if not 1, 2, ..."""

    def __init__(self, X, inputs=None):
        self.X = X
        self.names = {f"x{j}": var(k) for k, j in enumerate(inputs or range(1, X.shape[1] + 1))}
        self.names.update({k: getattr(self, k) for k in ('pw', 'log', 'exp', 'aff', 'sin', 'cos', 'mul',
                                                         'add', 'T', 'H')}, C=C)

    def lib(self, u, kinds, v=1):
        """The library's node of one of the kinds on u, u^2 or 1/u."""
        for n in library(u, self.X):
            if n.kind in kinds and (n.kids[0] is u if v == 1 else n.kids[0] is not u and n.kids[0].th[0] == v):
                return n
        raise LookupError

    def pw(self, u, p):
        nodes = [n for n in library(u, self.X) if n.kind == 'pow']
        if nodes[0].free[0]:
            nodes[0].th[0] = p
            return nodes[0]
        return next(n for n in nodes if n.th[0] == p)

    def log(self, u):
        return self.lib(u, ('log', 'logabs'))

    def exp(self, u, c, v=1):
        n = self.lib(u, ('exp',), v)
        n.th[0] = c * n.scale
        return n

    def aff(self, u, c, p, v=1):
        n = self.lib(u, ('aff',), v)
        n.th[:] = c * n.scale, p
        return n

    def sin(self, u, b, a=0.0):
        n = self.lib(u, ('sin',))
        n.th[:] = a, b * n.scale
        return n

    def cos(self, u, b):
        return self.sin(u, b, np.pi / 2)

    def mul(self, u, v, p=1):
        # A pairing is kids[0]*kids[1]^q, with the exponent on the positive one of
        # the two, and only 1 and -1 if neither is.
        for a, b in ((u, v), (v, u)):
            n = next(n for n in pairings(a, b, self.X) if n.kind == 'mono')
            if (n.kids[0] is u or p == 1) and (n.free[0] or p in (1, -1)):
                n.th[0] = p
                return n
        raise LookupError

    def add(self, u, v, c=1):
        n = next(n for n in pairings(u, v, self.X) if n.kind == 'lin')
        n.th[0] = c * n.scale
        return n

    def T(self, *factors):
        return Term([self.pw(f, 1) if f.kind in ('var', 'mono', 'lin') else f for f in factors])

    def H(self, *feats, factors=()):
        return Term(self.T(*factors).factors, feats)


def models(name, X, inputs=None):
    """The alternatives of the formula of a target that can be written for the
    inputs X, as (function around the expression, terms)."""
    forms = FORMS.get(name, [])
    out = []
    for form in forms if isinstance(forms, list) else [forms]:
        outer, text = form if isinstance(form, tuple) else (None, form)
        try:
            terms = eval(text, {'__builtins__': {}}, Build(X, inputs).names)
        except (LookupError, StopIteration, NameError):
            continue
        out.append((outer, [copy.deepcopy(t) for t in (terms if isinstance(terms, tuple) else [terms])]))
    return out


def truth(name, X, y, y_scale, inputs=None):
    """The alternative of the formula of a target with the best score, as an
    entry of the results. None if there is none. The parameters of the formula
    that have a simple value are held there, and the others are fitted from
    their values in the formula."""
    targets, outer = transforms(y, y_scale)
    outer = {o[0]: o[1:] for o in outer}
    best = None
    for name_outer, terms in models(name, X, inputs):
        if name_outer and name_outer not in outer:
            continue
        m = Model(terms)
        m.y_scale = y_scale
        if name_outer:
            m.warp, (m.offset, m.warp_args) = name_outer, outer[name_outer]
        for node in (n for t in m.terms for n in t.nodes()):
            for k in np.flatnonzero(node.free):
                values = nice(node, k)
                node.free[k] = values is None or not np.isclose(values, node.th[k]).any()
        data = (X, *targets[name_outer])
        m.fit(*data, maxiter=300, restarts=3, rng=np.random.default_rng(0)).condition(*data)
        if best is None or m.score > best.score:
            best = m
    if best is None or best.score < -1e9:           # no alternative, or none that could be fitted
        return None
    return dict(score=float(best.score), holes=best.n_holes(), key=best.key(), text=best.text(y_scale))


def residuals(name, X, f, inputs=None):
    """For each alternative, its text and the rms residual of a least-squares
    fit of its weights to the target f without noise, in units of the spread of
    the target: 0 for a formula that is written right. None with a hole, or
    where the function around the expression does not apply."""
    out = []
    for outer, terms in models(name, X, inputs):
        z = f if outer is None else (warp(outer, f) or [None])[0]
        res = None
        if z is not None and all(t.feats is None for t in terms):
            with np.errstate(all='ignore'):
                A = np.column_stack([t.scale(X) for t in terms] + [np.ones(len(X))])
            A = A / np.sqrt(np.mean(A ** 2, 0))
            res = float(np.std(A @ np.linalg.lstsq(A, z, rcond=None)[0] - z) / z.std())
        out.append((f"{outer or ''}({Model(terms).key()})", res))
    return out


if __name__ == '__main__':
    from run_baseline import SEEDS, problems
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', nargs='*', default=list(SEEDS))
    a = ap.parse_args()
    for which in a.sets:
        for name, text, seed, X, y, ystd, tests, inputs in problems(which):
            if seed:
                continue
            Xs, fs = tests.get('in', (X, y))        # without tests: the data, so the residual is the noise
            res = residuals(name, Xs, fs * ystd, inputs)
            print(f"{which:10s} {name:11s} " + ('; '.join(
                f"{text}: " + ('-' if r is None else f"{r:.1e}") for text, r in res) or 'no formula'))
