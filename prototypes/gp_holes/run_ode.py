"""A differential equation for a hole of one input.

    python run_ode.py [--only NAME ...] [--noise 0.1 0.01 0.001] [--seeds 2] [--kernel rbf]

The fit of a hole H(u) gives its posterior mean and the first two derivatives
of the mean at the data. A linear relation between a few of the columns
u^a * d^k H/du^k, with a and k from 0 to 2, and a constant is a differential
equation with polynomial coefficients. Its solutions are a model without a
hole: one term for each basis solution, with a weight, so the evidence is that
of a formula. The equations with the smallest residuals are scored this way,
with their coefficients rounded to simple values where the fit allows.
"""
import argparse
import itertools

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.special import airy, ellipk, erf, j0

from gpholes import CFG, Model, Node, Term, kernel, num, var

# name, formula, function, range of the input, and the equation that holds
TARGETS = [
    ('erf', 'erf(u)', erf, (0.1, 1.5), "H'' + 2*u*H' = 0"),
    ('bessel', 'J0(u)', j0, (0.1, 2.5), "u*H'' + H' + u*H = 0"),
    ('pendulum', 'K(sin(u/2)^2)', lambda u: ellipk(np.sin(u / 2) ** 2), (0.1, 2.0),
     "sin(u)*H'' + cos(u)*H' - sin(u)*H/4 = 0, which has no polynomial coefficients"),
    ('gauss', 'exp(-u^2/2)', lambda u: np.exp(-u ** 2 / 2), (0.1, 2.5), "H' + u*H = 0"),
    ('sine', 'sin(2*u)', lambda u: np.sin(2 * u), (0.5, 3.0), "H'' + 4*H = 0"),
    ('rational', '1/(1 + u)', lambda u: 1 / (1 + u), (0.2, 5.0), "H' + u*H' + H = 0"),
    ('airy', 'Ai(-u)', lambda u: airy(-u)[0], (0.1, 5.0), "H'' + u*H = 0"),
    ('arcsin', 'arcsin(u)', np.arcsin, (0.05, 0.8), "H'' - u*H' - u^2*H'' = 0"),
]
SIMPLE = np.array([1, 2, 3, 4, 0.5, 0.25, 1 / 3])
SIMPLE = np.concatenate([SIMPLE, -SIMPLE])
WIDER = 0.25            # the test outside the data: this much wider on each side, as a fraction of the range


def derivatives(m, X, y):
    """The posterior mean of the hole of a model of one hole of one input at
    the data, with the constant of the model, and its first two derivatives."""
    cache = [dict(free_a=True, free_u=True)]
    _, alpha = m.solve(X, y, (None, y ** 2), cache)
    c, t = cache[0], m.terms[0]
    V, w, A2 = c['V'][:, 0], np.exp(-2 * t.ls[0]), np.exp(2 * t.amp[0])
    d = V[:, None] - V[None]
    k, g = kernel(w * d ** 2)                   # g = -2 dk/d(r2)
    k2 = k / 4 if CFG['kernel'] == 'rbf' else 25 / 12 * np.exp(-np.sqrt(5 * w * d ** 2))    # d2k/d(r2)^2
    s = X[:, 0].std()
    return (A2 * k @ alpha + np.exp(2 * m.hyp[0]) * alpha.sum(), A2 * (-g * w * d) @ alpha / s,
            A2 * (4 * k2 * w ** 2 * d ** 2 - g * w) @ alpha / s ** 2)


def equations(u, h, n_best=3, most=0.3, margin=1.5):
    """Differential equations for the hole, from its mean and derivatives h at
    the points u: (residual, columns, coefficients). A column is (a, k) for
    u^a * d^k H/du^k, or '1'. Returned are the `n_best` equations of two
    columns and of three with the smallest residuals, where each column takes
    part and the equation does not hold as well, within `margin`, with fewer
    of its columns."""
    cols = {(a, k): u ** a * h[k] for a in range(3) for k in range(3)}
    cols['1'] = np.ones(len(u))
    norm = {key: np.linalg.norm(v) for key, v in cols.items()}
    names = list(cols)
    A = np.column_stack([cols[key] / norm[key] for key in names])
    gram, found = A.T @ A, {}
    for size in (2, 3):
        for idx in itertools.combinations(range(len(names)), size):
            keys = [names[i] for i in idx]
            part = [key for key in keys if key != '1']
            # no equation without a derivative, and none that is u times another
            if max(k for a, k in part) == 0 or ('1' not in keys and min(a for a, k in part) > 0):
                continue
            lam, vec = np.linalg.eigh(gram[np.ix_(idx, idx)])
            if np.abs(vec[:, 0]).min() > 0.2 * np.abs(vec[:, 0]).max():
                found[frozenset(keys)] = (np.sqrt(max(lam[0], 0.0)), keys, vec[:, 0] / [norm[key] for key in keys])
    keep = [r for s, r in found.items()
            if r[0] < most and all(r[0] * margin < r2[0] for s2, r2 in found.items() if s2 < s)]
    return [r for size in (2, 3) for r in sorted([r for r in keep if len(r[1]) == size], key=lambda r: r[0])[:n_best]]


def solutions(keys, coef, u, u0):
    """The solutions at the points u of sum_i c_i * u^a_i * d^k_i H/du^k_i =
    delta, where delta is minus the coefficient of the column '1': one column
    for each basis solution of the equation with delta = 0, with value or
    derivative 1 at u0, and with a column '1' the solution of the full
    equation that vanishes at u0 with its derivatives. NaN where the equation
    is singular between u0 and the point."""
    order = max(key[1] for key in keys if key != '1')
    part = [(c, key) for c, key in zip(coef, keys) if key != '1']
    delta = -sum(c for c, key in zip(coef, keys) if key == '1')
    lead = lambda x: sum(c * x ** a for c, (a, k) in part if k == order)
    m = order + ('1' in keys)

    def rhs(x, y):
        y = y.reshape(m, order)
        low = sum(c * x ** a * y[:, k] for c, (a, k) in part if k < order)
        top = (np.where(np.arange(m) == order, delta, 0.0) - low) / lead(x)
        return np.column_stack([y[:, 1:], top]).ravel()

    y0 = np.eye(m, order).ravel()
    out = np.full((len(u), m), np.nan)
    for side in (u >= u0, u < u0):
        if side.any():
            order_u = np.argsort(np.abs(u[side] - u0))
            end = u[side][order_u[-1]]
            if end != u0 and np.ptp(np.sign(lead(np.linspace(u0, end, 200)))) == 0:
                sol = solve_ivp(rhs, (u0, end), y0, t_eval=u[side][order_u], rtol=1e-8, atol=1e-11)
                if sol.status == 0:
                    out[np.flatnonzero(side)[order_u]] = sol.y.reshape(m, order, -1)[:, 0].T
            elif end == u0:
                out[side] = np.eye(m, order)[:, 0]
    return out


class Solution(Node):
    """Column `which` of the solutions of an equation (see solutions) for the feature u."""

    def __init__(self, f, keys, coef, which, u0):
        super().__init__('ode', [f], coef, np.zeros(len(coef), bool))
        self.keys, self.which, self.u0 = keys, which, u0

    def __call__(self, X):
        return solutions(self.keys, self.th, self.kids[0](X), self.u0)[:, self.which]

    def text(self, key=False):
        return f"S{self.which + 1}({self.kids[0].text(key)})"


def as_text(keys, coef):
    """The equation as a string."""
    d = {0: 'H', 1: "H'", 2: "H''"}
    terms = [f"{num(c)}*" + (f"u^{key[0]}*" if key[0] > 1 else 'u*' if key[0] else '') + d[key[1]]
             for c, key in zip(coef, keys) if key != '1']
    rhs = -sum(c for c, key in zip(coef, keys) if key == '1')
    return (' + '.join(terms) + f" = {num(rhs)}").replace('+ -', '- ').replace(' 1*', ' ').removeprefix('1*')


def score(keys, coef, X, y, u0, n_free):
    """The model of the solutions of an equation, fitted: one term for each
    column of solutions. The cost of a symbolic function is counted for each
    column of the equation, and a BIC penalty for each fitted coefficient."""
    n_terms = max(key[1] for key in keys if key != '1') + ('1' in keys)
    m = Model([Term([Solution(var(0), keys, coef, i, u0)]) for i in range(n_terms)])
    with np.errstate(all='ignore'):
        if not all(np.isfinite(t.scale(X)).all() for t in m.terms):
            return None
        m.fit(X, y, maxiter=200, restarts=2, rng=np.random.default_rng(0))
    m.score += CFG['node_cost'] * (n_terms - len(keys)) - 0.5 * np.log(len(y)) * n_free
    return m


def refine(keys, coef, X, y, u0):
    """The coefficients of an equation in units of the first of the highest
    derivative, fitted by least squares of its solutions to the data, and
    each then held at a simple value if the fit allows. Returns them and
    the number that were fitted."""
    order = max(key[1] for key in keys if key != '1')
    lead = next(i for i, key in enumerate(keys) if key != '1' and key[1] == order)
    coef = np.array(coef) / coef[lead]
    free, n = [i for i in range(len(keys)) if i != lead], len(y)

    def rss(p, idx, c):
        c = c.copy()
        c[idx] = p
        with np.errstate(all='ignore'):
            B = np.column_stack([solutions(keys, c, X[:, 0], u0), np.ones(n)])
            if not np.isfinite(B).all():
                return 1e10
            return np.sum((B @ np.linalg.lstsq(B, y, rcond=None)[0] - y) ** 2)

    def fitted(c, idx):
        if not idx:
            return c, rss([], [], c)
        r = minimize(rss, c[idx], args=(idx, c), method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-9, maxiter=120))
        c = c.copy()
        c[idx] = r.x
        return c, r.fun

    coef, best = fitted(coef, free)
    for i in sorted(free, key=lambda i: np.abs(SIMPLE - coef[i]).min() / abs(coef[i])):
        c = coef.copy()
        c[i] = SIMPLE[np.argmin(np.abs(SIMPLE - c[i]))]
        c, r = fitted(c, [j for j in free if j != i])
        if 0.5 * n * np.log(r / best) < 0.5 * np.log(n):        # BIC, with the noise profiled out
            coef, best, free = c, r, [j for j in free if j != i]
    return coef, len(free)


def run(target, n, noise, seed, verbose=True):
    name, text, f, (lo, hi), _ = target
    rng = np.random.default_rng(seed)
    X = rng.uniform(lo, hi, (n, 1))
    ystd = f(X[:, 0]).std()
    y = f(X[:, 0]) / ystd + noise * rng.standard_normal(n)
    wide = WIDER * (hi - lo)
    Xs = np.concatenate([np.linspace(max(lo - wide, 0.5 * lo), lo, 200), np.linspace(hi, hi + wide, 200)])[:, None]
    tests = {'in': np.linspace(lo, hi, 400)[:, None], 'out': Xs}
    hole = Model([Term(feats=[var(0)])]).fit(X, y, restarts=6, rng=rng)
    h = derivatives(hole, X, y)
    u = X[:, 0]
    inner = (u > np.quantile(u, 0.05)) & (u < np.quantile(u, 0.95))     # the derivatives are poor at the ends
    u0 = u[np.argmin(np.abs(u - np.median(u)))]
    results = []
    for res, keys, coef in equations(u[inner], [d[inner] for d in h]):
        coef, n_free = refine(keys, coef, X, y, u0)
        m = score(keys, coef, X, y, u0, n_free)
        if m is not None:
            m.condition(X, y)
            with np.errstate(all='ignore'):
                err = {k: np.sqrt(np.mean((m.predict(Z) - f(Z[:, 0]) / ystd) ** 2)) for k, Z in tests.items()}
            results.append((m.score, res, as_text(keys, coef), n_free, err))
    results.sort(key=lambda r: -r[0])
    hole.condition(X, y)
    err = {k: np.sqrt(np.mean((hole.predict(Z) - f(Z[:, 0]) / ystd) ** 2)) for k, Z in tests.items()}
    if verbose:
        print(f"  noise {noise:g}, seed {seed}: hole {hole.score:.1f}, rmse in {err['in']:.2g} out {err['out']:.2g}")
        for s, res, eq, n_free, e in results[:3]:
            print(f"      {s:8.1f}  {eq}   [residual {res:.3f}, {n_free} fitted, "
                  f"rmse in {e['in']:.2g} out {e['out']:.2g}]")
    return hole.score, results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--noise', nargs='*', type=float, default=[0.1, 0.01, 0.001])
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--kernel', default='rbf')
    a = ap.parse_args()
    CFG['kernel'] = a.kernel
    for target in TARGETS:
        if a.only and target[0] not in a.only:
            continue
        print(f"\n{target[0]}:  {target[1]},  {target[4]}", flush=True)
        for noise in a.noise:
            for seed in range(a.seeds):
                run(target, a.n, noise, seed)
