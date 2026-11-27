"""Symbolic regression with GP holes (prototype).

A model is a sum of terms a(x) * H(u(x)) + w * b(x) + const, where a, b and u
are symbolic and each H ("hole") is an unknown function with a GP prior. Such a
model is itself a GP, so its evidence is closed-form. The search starts from a
single hole H(x1, ..., xd) and applies rewrite rules that replace holes with
more specific structure, keeping the models with the highest evidence.
"""
import copy
import itertools
import time
from multiprocessing import Pool

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

CFG = dict(
    kernel='matern52',  # kernel of the holes: 'rbf' or 'matern52'
    node_cost=3.0,      # cost in log evidence of each symbolic function in a model
    power_cost=10.0,    # ... and of each exponent that is not a simple number
    max_power=3,        # the largest integer power counted as simple
    library=('pow', 'log', 'logabs', 'exp', 'aff', 'sin'),     # functions of one input; also 'poly'
    outer=('log', 'exp', 'expneg', 'inv', 'square'),    # functions around the expression; also 'sqrt'
)
# Values a fitted parameter is rounded to if the evidence allows (see nice).
NICE = {
    'mono': [np.array([-3, -2, -1, -0.5, 0.5, 1, 2, 3])],
    'lin': [np.array([-2, -1, -0.5, 0.5, 1, 2])],
    'sin': [np.array([0, np.pi / 2]), None],
    'aff': [np.array([-2, -1, -0.5, 0.5, 1, 2]), np.array([-2, -1, -0.5, 0.5, 2])],
}
BOUNDS = {
    'mono': [(-4, 4)], 'lin': [(-50, 50)], 'exp': [(-50, 50)],
    'aff': [(-50, 50), (-4, 4)], 'sin': [(-2, 2), (1e-3, 100)], 'log': [],
    'poly': [(-50, 50)] * 8,
}
EXPONENT = {'pow': 0, 'mono': 0, 'aff': 1}      # which parameter of a node is an exponent
SCALED = {'lin': 0, 'exp': 0, 'aff': 0, 'sin': 1}   # ... and which multiplies the argument, as th/scale
OUTER_TEXT = {'log': 'exp({m})', 'exp': 'log({m}) + {c}', 'expneg': '{c} - log({m})', 'inv': '1/({m})',
              'sqrt': '({m})^2', 'square': 'sqrt({m})'}
# The relative noise is at most 30%: with more, a constant explains any data as noise.
LOG_BOUNDS = {'amp': (-7, 4), 'ls': (-3, 5), 'const': (-7, 4), 'noise': (-9, 1), 'outside': (-9, 1),
              'relative': (-9, np.log(0.3))}


def nice(node, k):
    """The simple values of parameter k of a node, or None."""
    if node.kind == 'pow':
        ints = np.arange(1, CFG['max_power'] + 1)
        return np.concatenate([-ints, ints, [-1.5, -0.5, 0.5, 1.5]])
    values = NICE[node.kind][k] if node.kind in NICE else None
    return values * node.scale if values is not None and SCALED.get(node.kind) == k else values


def warp(name, y):
    """The target z of a model with the function `name` around it, log|dz/dy|
    at each point and the constant of the function. None if it does not apply."""
    with np.errstate(all='ignore'):
        c = 0.0
        if name == 'log':                   # y = exp(m)
            ok, z, jac = (y > 0).all(), np.log(y), -np.log(y)
        elif name == 'exp':                 # y = log(m) + c
            c = y.max()
            ok, z, jac = True, np.exp(y - c), y - c
        elif name == 'expneg':              # y = c - log(m)
            c = y.min()
            ok, z, jac = True, np.exp(c - y), c - y
        elif name == 'inv':                 # y = 1/m
            ok, z, jac = (y > 0).all() or (y < 0).all(), 1 / y, -2 * np.log(np.abs(y))
        elif name == 'sqrt':                # y = m^2
            ok, z, jac = (y > 0).all(), np.sqrt(y), -np.log(2) - 0.5 * np.log(y)
        elif name == 'square':              # y = sqrt(m)
            ok, z, jac = (y > 0).all(), y ** 2, np.log(2) + np.log(y)
    if not ok or not (np.isfinite(z).all() and np.isfinite(jac).all()) or z.std() == 0:
        return None
    return z, jac, c


def unwarp(name, z, c):
    """y from the value z of the expression inside the function `name`."""
    with np.errstate(all='ignore'):
        return {'log': lambda: np.exp(z), 'exp': lambda: c + np.log(z), 'expneg': lambda: c - np.log(z),
                'inv': lambda: 1 / z, 'sqrt': lambda: z ** 2, 'square': lambda: np.sqrt(z)}[name]()


def num(v):
    return f"{v:.3g}"


def wrap(s):
    """Parentheses around s if it is a sum, product or power."""
    if s.startswith('(') or not any(c in s for c in '+-*/^ '):
        return s
    if s[:4] in ('exp(', 'sin(', 'cos(', 'log(') and s.endswith(')') and s.count('(') == 1:
        return s
    return f"({s})"


class Node:
    """A symbolic function of the inputs, with parameters th. A parameter that
    multiplies the argument is kept as th*scale, with scale the size of the
    argument, so that the fitted values are of order one."""

    def __init__(self, kind, kids=(), th=(), free=None, j=None, bounds=None, scale=1.0):
        self.kind, self.kids, self.j, self.scale = kind, list(kids), j, scale
        self.th = np.array(th, float)
        self.free = np.ones(len(self.th), bool) if free is None else np.array(free, bool)
        if kind == 'pow':
            bounds = [(-CFG['max_power'] - 1, CFG['max_power'] + 1)]
        self.bounds = bounds or BOUNDS.get(kind, [])

    def __call__(self, X):
        k, th = self.kind, self.th
        if k == 'var':
            return X[:, self.j]
        u = self.kids[0](X)
        if k == 'mono':
            return u * self.kids[1](X) ** th[0]
        if k == 'lin':
            return u + th[0] / self.scale * self.kids[1](X)
        if k == 'pow':
            return u ** th[0]
        if k == 'exp':
            return np.exp(th[0] / self.scale * u)
        if k == 'aff':
            return (1 + th[0] / self.scale * u) ** th[1]
        if k == 'sin':
            return np.sin(th[0] + th[1] / self.scale * u)
        if k == 'log':
            return np.log(u)
        if k == 'logabs':
            return np.log(np.abs(u))
        if k == 'poly':                 # 1 + th[0]*v + th[1]*v^2 + ..., v = u/scale
            v = u / self.scale
            return 1 + sum(c * v ** (i + 1) for i, c in enumerate(th))

    def walk(self):
        yield self
        for kid in self.kids:
            yield from kid.walk()

    def text(self, key=False):
        """The expression as a string. With key=True free parameters are shown
        as '?', so that models with the same structure get the same string."""
        k = self.kind
        raw = [v / self.scale if SCALED.get(k) == i else v for i, v in enumerate(self.th)]
        p = ['?' if (key and f) else num(v) for v, f in zip(raw, self.free)]
        if k == 'var':
            return f"x{self.j + 1}"
        u = self.kids[0].text(key)
        if k == 'mono':
            v = self.kids[1].text(key)
            if p[0] == '1':
                return f"{wrap(u)}*{wrap(v)}"
            if p[0] == '-1':
                return f"{wrap(u)}/{wrap(v)}"
            if p[0].startswith('-'):
                return f"{wrap(u)}/{wrap(v)}^{p[0][1:]}"
            return f"{wrap(u)}*{wrap(v)}^{p[0]}"
        if k == 'lin':
            v = self.kids[1].text(key)
            if p[0] == '1':
                return f"({u} + {v})"
            if p[0] == '-1':
                return f"({u} - {wrap(v)})"
            return f"({u} + {p[0]}*{wrap(v)})"
        if k == 'pow':
            if p[0] == '1':
                return u
            if p[0] == '-1':
                return f"1/{wrap(u)}"
            if p[0].startswith('-'):
                return f"1/{wrap(u)}^{p[0][1:]}"
            return f"{wrap(u)}^{p[0]}"
        # c*u, written c/v if u is 1/v
        times = lambda c: f"{c}/{u[2:]}" if u.startswith('1/') else f"{c}*{wrap(u)}"
        if k == 'exp':
            return f"exp({times(p[0])})"
        if k == 'aff':
            return f"(1 + {times(p[0])})^{p[1]}"
        if k == 'sin':
            if p[0] == '0':
                return f"sin({times(p[1])})"
            if p[0] == num(np.pi / 2):
                return f"cos({times(p[1])})"
            return f"sin({p[0]} + {times(p[1])})"
        if k == 'log':
            return f"log({u})"
        if k == 'logabs':
            return f"log|{u}|"
        if k == 'poly':
            c = ['?' if (key and f) else num(v / self.scale ** (i + 1))
                 for i, (v, f) in enumerate(zip(self.th, self.free))]
            terms = [f"{c[0]}*{wrap(u)}"] + [f"{ci}*{wrap(u)}^{i + 2}" for i, ci in enumerate(c[1:])]
            return '(' + ' + '.join(['1'] + terms).replace('+ -', '- ') + ')'


def var(j):
    return Node('var', j=j)


def free(roots):
    """Whether a node below the roots has a free parameter."""
    return any(n.free.any() for root in roots for n in root.walk())


def kernel(r2):
    """The kernel of the holes at squared distances r2, and g = -2 dk/d(r2)."""
    if CFG['kernel'] == 'rbf':
        k = np.exp(-0.5 * r2)
        return k, k
    r = np.sqrt(5 * r2)
    e = np.exp(-r)
    return (1 + r + r * r / 3) * e, 5 / 3 * (1 + r) * e


class Term:
    """a(x) * H(u(x)) if feats is a list of nodes, w * a(x) if feats is None.
    a(x) is the product of the factors."""

    def __init__(self, factors=(), feats=None, amp=-0.5, ls=None):
        self.factors = list(factors)
        self.feats = None if feats is None else list(feats)
        self.amp = np.array([amp])                      # log amplitude
        n = 0 if feats is None else len(self.feats)
        self.ls = np.zeros(n) if ls is None else np.array(ls, float)   # log length scales

    def nodes(self):
        for root in self.factors + (self.feats or []):
            yield from root.walk()

    def factor_key(self):
        return '*'.join(sorted(f.text(True) for f in self.factors))

    def feat_keys(self):
        return None if self.feats is None else frozenset(f.text(True) for f in self.feats)

    def key(self):
        fk = self.feat_keys()
        hole = '' if fk is None else 'H(' + ', '.join(sorted(fk)) + ')'
        return '*'.join(s for s in (self.factor_key(), hole) if s) or '1'

    def scale(self, X):
        """a(x), unnormalised."""
        a = np.ones(len(X))
        for f in self.factors:
            a = a * f(X)
        return a

    def factor(self, X, ref):
        """a(x) in units of its rms over ref."""
        a = self.scale(X)
        return a / np.sqrt(np.mean((a if ref is X else self.scale(ref)) ** 2))

    def inputs(self, X, ref):
        """The hole inputs, standardised over ref. None without a hole."""
        if self.feats is None:
            return None
        U = np.column_stack([f(X) for f in self.feats])
        R = U if ref is X else np.column_stack([f(ref) for f in self.feats])
        return (U - R.mean(0)) / R.std(0)

    def K(self, Xa, Xb, ref):
        """The term's kernel matrix."""
        aa, Va = self.factor(Xa, ref), self.inputs(Xa, ref)
        ab, Vb = (aa, Va) if Xb is Xa else (self.factor(Xb, ref), self.inputs(Xb, ref))
        K = np.exp(2 * self.amp[0]) * np.outer(aa, ab)
        if Va is None:
            return K
        return K * kernel((Va[:, None] - Vb[None]) ** 2 @ np.exp(-2 * self.ls))[0]

    def train(self, X, c):
        """The term's kernel matrix K on the training inputs X and its pieces,
        into c: a(x) and its outer product P, the standardised hole inputs V
        and the squared differences E of each, and k and g of the hole (see
        kernel). What depends on no free parameter of a node is kept from the
        last call."""
        if 'P' not in c or c['free_a']:
            c['a'] = self.factor(X, X)
            c['P'] = np.outer(c['a'], c['a'])
        c['K'] = np.exp(2 * self.amp[0]) * c['P']
        if self.feats is None:
            return
        if 'E' not in c or c['free_u']:
            c['V'] = self.inputs(X, X)
            c['E'] = (c['V'][:, None] - c['V'][None]) ** 2
        c['k'], c['g'] = kernel(c['E'] @ np.exp(-2 * self.ls))
        c['K'] = c['K'] * c['k']

    def grad(self, W, c):
        """The derivatives of the negative log evidence with respect to the
        log amplitude, the log length scales, a(x) at each point and the
        standardised hole inputs at each point, from the pieces in c and
        W = K^-1 - alpha alpha^T of the whole model."""
        A2 = np.exp(2 * self.amp[0])
        d_amp = np.sum(W * c['K'])
        if self.feats is None:
            return d_amp, None, A2 * (W @ c['a']), None
        w = np.exp(-2 * self.ls)
        Q = A2 * W * c['P'] * c['g']
        d_ls = 0.5 * w * (Q.ravel() @ c['E'].reshape(-1, len(w)))
        d_a = (A2 * W * c['k']) @ c['a']
        d_V = (Q @ c['V'] - Q.sum(1)[:, None] * c['V']) * w
        return d_amp, d_ls, d_a, d_V


class Model:
    """A sum of terms and a constant, with noise of the same size at every
    point and noise in proportion to the size of the target. With warp set,
    the sum is the expression inside that function (see warp), fitted to the
    transformed target: offset is then the term that makes its score one of the
    target itself, warp_args the scale of the transformed target and the
    constant of the function. The noise of the same size is then the noise
    inside the function, and there is noise of the same size outside it as
    well. The noise outside the function is carried through the transform."""

    def __init__(self, terms, const=-1.0, noise=-2.0):
        self.terms = terms
        # log const amplitude, log noise: of the same size inside, outside, relative
        self.hyp = np.array([const, noise, -2.0, -3.0])
        self.score = -np.inf
        self.warp, self.offset, self.warp_args, self.y_scale = None, 0.0, None, 1.0

    def key(self):
        inner = ' + '.join(sorted(t.key() for t in self.terms)) or '0'
        return OUTER_TEXT[self.warp].format(m=inner, c='?') if self.warp else inner

    def n_holes(self):
        return sum(t.feats is not None for t in self.terms)

    def n_nodes(self):
        """The number of symbolic functions (the inputs themselves do not count)."""
        return sum(node.kind != 'var' for t in self.terms for node in t.nodes())

    def n_free_powers(self):
        return sum(node.free[EXPONENT[node.kind]] for t in self.terms for node in t.nodes()
                   if node.kind in EXPONENT)

    def slots(self):
        """One (array, index, bounds, tag, term index, length scale index) per
        free parameter. For a parameter of a node the last entry tells whether
        the node is in a(x) and whether it is in the hole inputs."""
        out = []
        for i, t in enumerate(self.terms):
            out.append((t.amp, 0, LOG_BOUNDS['amp'], 'amp', i, None))
            out += [(t.ls, j, LOG_BOUNDS['ls'], 'ls', i, j) for j in range(len(t.ls))]
            in_a = [n for f in t.factors for n in f.walk()]
            in_u = [n for f in t.feats or [] for n in f.walk()]
            for node in in_a + in_u:
                where = any(n is node for n in in_a), any(n is node for n in in_u)
                out += [(node.th, k, node.bounds[k], 'th', i, where)
                        for k in range(len(node.th)) if node.free[k]]
        out.append((self.hyp, 0, LOG_BOUNDS['const'], 'const', None, None))
        out.append((self.hyp, 1, LOG_BOUNDS['noise'], 'noise', None, None))
        if self.warp:
            out.append((self.hyp, 2, LOG_BOUNDS['outside'], 'outside', None, None))
        out.append((self.hyp, 3, LOG_BOUNDS['relative'], 'relative', None, None))
        return out

    def noise(self, n, w):
        """The noise variance of each of n points. w has the variance at each
        point of noise outside the function and of relative noise (see search)."""
        s2 = np.exp(2 * self.hyp[1:])
        var = s2[0] + 1e-8 + s2[2] * w[1] * np.ones(n)
        return var + s2[1] * w[0] if self.warp else var

    def nll_grad(self, p, X, y, slots, w, cache):
        """Negative log evidence and its gradient at the parameters p. cache
        has a dict for each term (see Term.train)."""
        for v, s in zip(p, slots):
            s[0][s[1]] = v
        n = len(y)
        with np.errstate(all='ignore'):
            for t, c in zip(self.terms, cache):
                t.train(X, c)
            c2 = np.exp(2 * self.hyp[0])
            K = sum(c['K'] for c in cache) + c2 + np.diag(self.noise(n, w))
            try:
                chol = cho_factor(K, lower=True, check_finite=True)
            except (np.linalg.LinAlgError, ValueError):
                return 1e10, np.zeros(len(p))
            alpha = cho_solve(chol, y)
            W = cho_solve(chol, np.eye(n)) - np.outer(alpha, alpha)
            nll = 0.5 * y @ alpha + np.log(np.diag(chol[0])).sum() + 0.5 * n * np.log(2 * np.pi)
            diag = np.diag(W)
            d = [t.grad(W, c) for t, c in zip(self.terms, cache)]
            grad = np.zeros(len(p))
            for q, (arr, idx, _, tag, i, j) in enumerate(slots):
                if tag == 'amp':
                    grad[q] = d[i][0]
                elif tag == 'ls':
                    grad[q] = d[i][1][j]
                elif tag == 'const':
                    grad[q] = c2 * W.sum()
                elif tag == 'noise':
                    grad[q] = np.exp(2 * self.hyp[1]) * diag.sum()
                elif tag == 'outside':
                    grad[q] = np.exp(2 * self.hyp[2]) * (diag @ w[0])
                elif tag == 'relative':
                    grad[q] = np.exp(2 * self.hyp[3]) * (diag @ w[1])
                else:           # a parameter of a symbolic node, from differences of a(x) and the hole inputs
                    t, v, eps = self.terms[i], arr[idx], 1e-6 * max(1.0, abs(arr[idx]))
                    arr[idx] = v + eps
                    ap, Vp = j[0] and t.factor(X, X), j[1] and t.inputs(X, X)
                    arr[idx] = v - eps
                    am, Vm = j[0] and t.factor(X, X), j[1] and t.inputs(X, X)
                    arr[idx] = v
                    if j[0]:
                        grad[q] += (ap - am) @ d[i][2] / (2 * eps)
                    if j[1]:
                        grad[q] += np.sum((Vp - Vm) * d[i][3]) / (2 * eps)
            if not (np.isfinite(nll) and np.isfinite(grad).all()):
                return 1e10, np.zeros(len(p))
        return nll, grad

    def fit(self, X, y, w=None, maxiter=200, restarts=0, rng=None):
        """Maximise the evidence over the free parameters. The score is the log
        evidence with a BIC penalty on the number of parameters and a cost for
        each symbolic function and each free exponent. score0 leaves out the
        cost of the exponents, for models that have not been through snap.
        Without w the relative noise follows y (see search)."""
        w = w or (None, y ** 2)
        slots = self.slots()
        cache = [dict(free_a=free(t.factors), free_u=free(t.feats or [])) for t in self.terms]
        bounds = [s[2] for s in slots]
        p0 = np.clip([s[0][s[1]] for s in slots], [b[0] for b in bounds], [b[1] for b in bounds])
        starts = [p0]
        for _ in range(restarts):
            p = p0.copy()
            for q, s in enumerate(slots):
                if s[3] != 'th':
                    p[q] = (rng.uniform(-3, -1) if s[3] in ('noise', 'outside', 'relative')
                            else rng.uniform(-1.5, 1.5))
            starts.append(np.clip(p, [b[0] for b in bounds], [b[1] for b in bounds]))
        best = None
        for p in starts:
            r = minimize(self.nll_grad, p, args=(X, y, slots, w, cache), jac=True, method='L-BFGS-B',
                         bounds=bounds, options={'maxiter': maxiter})
            if best is None or r.fun < best.fun:
                best = r
        for v, s in zip(best.x, slots):
            s[0][s[1]] = v
        self.logml, self.n_par = -best.fun, len(slots)
        self.score0 = (self.logml - 0.5 * self.n_par * np.log(len(y))
                       - CFG['node_cost'] * self.n_nodes() + self.offset)
        self.score = self.score0 - CFG['power_cost'] * self.n_free_powers()
        return self

    def condition(self, X, y, w=None):
        """Store what predict and text need."""
        w = w or (None, y ** 2)
        self.X = X
        K = sum((t.K(X, X, X) for t in self.terms), np.full((len(X), len(X)), np.exp(2 * self.hyp[0])))
        self.alpha = np.linalg.solve(K + np.diag(self.noise(len(y), w)), y)
        return self

    def predict(self, Xs):
        """The target at Xs, in the units the search was given it in."""
        with np.errstate(all='ignore'):
            Ks = sum((t.K(Xs, self.X, self.X) for t in self.terms),
                     np.full((len(Xs), len(self.X)), np.exp(2 * self.hyp[0])))
        z = Ks @ self.alpha
        if self.warp is None:
            return z
        return unwarp(self.warp, z * self.warp_args[0], self.warp_args[1]) / self.y_scale

    def text(self, ystd=1.0):
        """The model as a string, with the posterior mean of the weights."""
        if self.warp:
            return OUTER_TEXT[self.warp].format(m=self.inner_text(self.warp_args[0]),
                                                c=num(self.warp_args[1]))
        return self.inner_text(ystd)

    def inner_text(self, ystd):
        out = []
        for t in self.terms:
            a = '*'.join(wrap(f.text()) for f in t.factors)
            if t.feats is None:
                an = t.scale(self.X)
                rms = np.sqrt(np.mean(an ** 2))
                w = np.exp(2 * t.amp[0]) * (an / rms) @ self.alpha / rms * ystd
                out.append(f"{num(w)}*{a}")
            else:
                hole = 'H(' + ', '.join(f.text() for f in t.feats) + ')'
                out.append(f"{a}*{hole}" if a else hole)
        out.append(num(np.exp(2 * self.hyp[0]) * self.alpha.sum() * ystd))
        return ' + '.join(out).replace('+ -', '- ')


def tidy(model):
    """Remove holes that another hole with the same factors contains, and
    repeated symbolic terms."""
    keep = []
    for i, t in enumerate(model.terms):
        fk = t.feat_keys()
        absorbed = False
        for j, o in enumerate(model.terms):
            if i == j or o.factor_key() != t.factor_key():
                continue
            ok = o.feat_keys()
            if fk is None:
                absorbed |= ok is None and j < i
            elif ok is not None:
                absorbed |= fk < ok or (fk == ok and j < i)
        if not absorbed:
            keep.append(t)
    model.terms = keep
    return model


def size_of(v):
    """The typical size of the values v."""
    return np.median(np.abs(v)) or np.abs(v).max()


def library(f, X):
    """Candidate functions of the feature f: nodes with starting values."""
    u = f(X)
    inner = [f, Node('pow', [f], [2.0], [False])]
    if (u > 0).all():
        inner.append(Node('pow', [f], [-1.0], [False]))
        out = [Node('pow', [f], [p]) for p in (1.0, -1.0)] + [Node('log', [f])]
    else:               # a fitted exponent needs a positive argument
        out = [Node('pow', [f], [p], [False]) for p in (1.0, 2.0, 3.0, -1.0, -2.0)] + [Node('logabs', [f])]
    for g in inner:
        v = g(X)
        size, top = size_of(v), np.abs(v).max()
        out += [Node('exp', [g], [c * size / v.std()], scale=size) for c in (-1, 1)]
        # In units of the largest |v|, 1 + th*v/scale stays positive for |th| < 1.
        out += [Node('aff', [g], [c, -1.0], scale=top) for c in (-0.5, 0.5)]
        # Polynomials of degree 2 to 4 with constant term 1.
        out += [Node('poly', [g], [c] + [0.0] * (deg - 1), scale=top)
                for deg in (2, 3, 4) for c in (-0.5, 0.5)]
    # A sine has to show at least half a period over the data, and at most 60.
    size, span = size_of(u), np.ptp(u)
    out += [Node('sin', [f], [0.0, 2 * np.pi * k * size / span], scale=size,
                 bounds=[(-2, 2), (np.pi * size / span, 120 * np.pi * size / span)])
            for k in (0.5, 1, 2, 4)]
    return [n for n in out if n.kind in CFG['library']]


def pairings(a, b, X):
    """Candidate combinations of two features into one: a + c*b and a*b^p."""
    with np.errstate(all='ignore'):
        size = size_of(b(X)) / size_of(a(X))
    out = [Node('lin', [a, b], [c], scale=size) for c in (1.0, -1.0)]
    if not (b(X) > 0).all():
        a, b = b, a
    if (b(X) > 0).all():
        out += [Node('mono', [a, b], [p]) for p in (1.0, -1.0)]
    else:
        out += [Node('mono', [a, b], [p], [False]) for p in (1.0, -1.0)]
    return out


def moves(model, X, outer=()):
    """All one-step refinements of a model, as (rule, new model). outer has
    the functions that can go around a model without symbolic functions, as
    (name, offset, warp_args)."""
    out = []
    if model.warp is None and model.n_nodes() == 0:
        for name, offset, args in outer:
            m = copy.deepcopy(model)
            m.warp, m.offset, m.warp_args, m.fresh = name, offset, args, True
            out.append(('outer function', m))

    def emit(rule, i, new_terms):
        m = copy.deepcopy(model)
        m.terms[i:i + 1] = copy.deepcopy(new_terms)
        out.append((rule, tidy(m)))

    for i, t in enumerate(model.terms):
        emit('drop term', i, [])
        if len(t.factors) > 1 or (t.factors and t.feats):
            for j in range(len(t.factors)):
                new = copy.deepcopy(t)
                del new.factors[j]
                emit('drop factor', i, [new])
        if t.feats is None:
            continue
        q = len(t.feats)

        def without(*drop, amp=t.amp[0]):
            """The term with some of the hole's inputs removed."""
            feats = [f for k, f in enumerate(t.feats) if k not in drop]
            ls = [l for k, l in enumerate(t.ls) if k not in drop]
            return Term(t.factors, feats or None, amp, ls)

        for j in range(q):
            lower = without(j)
            # The hole does not depend on input j.
            if q > 1 or t.factors:
                emit('drop input', i, [lower])
            # The hole is simpler in the logarithm of input j.
            if t.feats[j].kind != 'log' and (t.feats[j](X) > 0).all():
                new = copy.deepcopy(t)
                new.feats[j] = Node('log', [t.feats[j]])
                emit('change input', i, [new])
            # The hole is a known function of input j times a smaller hole.
            lib = library(t.feats[j], X)
            for f in lib:
                new = copy.deepcopy(lower)
                new.factors.append(f)
                emit('factor', i, [new])
                if q == 1 and f.kind != 'pow':          # ... or that function times a power
                    for p in (n for n in lib if n.kind == 'pow'):
                        both = copy.deepcopy(new)
                        both.factors.append(p)
                        emit('factor', i, [both])
                # ... plus a smaller hole (the hole is linear in that function).
                if q > 1 or t.factors:
                    emit('linear', i, [new, lower])
                if q == 1:                              # ... plus the hole itself
                    emit('peel', i, [new, t])
        for j, k in itertools.combinations(range(q), 2):
            # Inputs j and k do not interact.
            emit('no interaction', i, [without(j, amp=t.amp[0] - 0.35),
                                       without(k, amp=t.amp[0] - 0.35)])
            # The hole depends on inputs j and k through one combination.
            for pair in pairings(t.feats[j], t.feats[k], X):
                new = without(j, k)
                new.feats = (new.feats or []) + [pair]
                new.ls = np.append(new.ls, 0.5 * (t.ls[j] + t.ls[k]))
                emit('merge', i, [new])
                # ... and, beside it, through one of the two.
                for keep in (j, k):
                    new = copy.deepcopy(t)
                    new.feats[k if keep == j else j] = pair
                    emit('change input', i, [new])
    return out


def snap(model, X, y, w=None):
    """Round parameters to simple values where that does not lower the score."""
    improved = True
    while improved:
        improved = False
        nodes = [n for t in model.terms for n in t.nodes()]
        for a, node in enumerate(nodes):
            for k in np.flatnonzero(node.free):
                values = nice(node, k)
                if values is None:
                    continue
                trial = copy.deepcopy(model)
                tn = [n for t in trial.terms for n in t.nodes()][a]
                # A phase of -pi/2 is one of pi/2 with the sign of the weight changed.
                v = abs(node.th[k]) if node.kind == 'sin' else node.th[k]
                tn.th[k] = values[np.argmin(np.abs(values - v))]
                tn.free[k] = False
                trial.fit(X, y, w, maxiter=100)
                if trial.score > model.score:
                    model, improved = trial, True
                    break
            if improved:
                break
    return model


def _init_worker(cfg):
    # The matrices are small, so BLAS threads in each worker only compete.
    global _limit
    _limit = threadpool_limits(1)
    CFG.update(cfg)


def _quick(args):
    model, X, (y, w) = args
    fresh = model.__dict__.pop('fresh', False)      # a new target: the parameters are redrawn too
    return model.fit(X, y, w, maxiter=25, restarts=3 if fresh else 0, rng=np.random.default_rng(0))


def _full(args):
    model, X, (y, w) = args
    return snap(model.fit(X, y, w, maxiter=300), X, y, w)


def transforms(y, y_scale):
    """The target of each function around the expression (None: no function),
    in units of its spread, with the variance there of noise outside the
    function (with mean 1) and of noise of 100% of the target, so that the
    relative noise is in units of the target. Also the functions that apply,
    as (name, offset, warp_args)."""
    targets, outer = {None: (y, (None, y ** 2))}, []
    for name in CFG['outer']:
        warped = warp(name, y * y_scale)
        if warped is not None:
            z, jac, c = warped
            w = np.exp(2 * (jac - jac.max()))
            with np.errstate(divide='ignore'):
                rel = np.exp(2 * (np.log(np.abs(y * y_scale)) + jac)) / z.var()
            targets[name] = (z / z.std(), (w / w.mean(), rel))
            outer.append((name, jac.sum() + len(y) * np.log(y_scale / z.std()) - CFG['node_cost'],
                          (z.std(), c)))
    return targets, outer


def search(X, y, y_scale=1.0, beam=4, top_k=12, patience=3, own_steps=6, max_steps=40, workers=4,
           seed=0, verbose=True, **cfg):
    """Beam search from a single hole. Each step refines the `beam` best
    models and, for its first `own_steps` steps, the best model of each
    function around the expression, until all of these have been refined or
    none of them has improved for `patience` steps. Returns the model with the
    best score, the trace of the best score and some counts, among them the 5
    best models. y should be scaled to unit variance, with y_scale the factor
    that gives it its units back. cfg replaces entries of CFG."""
    CFG.update(cfg)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    d = X.shape[1]
    targets, outer = transforms(y, y_scale)
    root = Model([Term(feats=[var(j) for j in range(d)])])
    root.y_scale = y_scale
    root.fit(X, *targets[None], restarts=6, rng=rng)
    seen = {root.key(): root}          # structure key -> fitted model
    tried, expanded, n_quick = {root.key()}, set(), {}
    best_of, stale = {}, 0             # the best score of each function around the expression
    own = {w: 0 for w in targets}      # ... and how often its best model was refined
    trace, n_fits = [(root.score, root.key(), 'start')], 1
    say = print if verbose else (lambda *a: None)
    say(f"   0 {root.score:9.2f}  {root.key()}")
    with Pool(workers, initializer=_init_worker, initargs=(dict(CFG),)) as pool:
        for step in range(1, max_steps + 1):
            ranked = sorted(seen.values(), key=lambda m: -m.score)
            front = ranked[:beam]
            for w in targets:
                first = next((m for m in ranked if m.warp == w), None)
                if first is not None and own[w] < own_steps and first.key() not in expanded:
                    front.append(first)
                    own[w] += 1
            todo = list({m.key(): m for m in front if m.key() not in expanded}.values())
            if not todo:
                break
            cands = []
            for m in todo:
                expanded.add(m.key())
                cands += [(rule, c) for rule, c in moves(m, X, outer) if c.key() not in tried]
            # A structure gets quick fits in at most two steps.
            for key in {c.key() for _, c in cands}:
                n_quick[key] = n_quick.get(key, 0) + 1
            cands = [(rule, c) for rule, c in cands if n_quick[c.key()] <= 2]
            t1 = time.time()
            quick = pool.map(_quick, [(c, X, targets[c.warp]) for _, c in cands], chunksize=4)
            t2 = time.time()
            n_fits += len(quick)
            best = {}                  # the best quick fit of each structure
            for (rule, _), m in zip(cands, quick):
                if m.key() not in best or m.score0 > best[m.key()][1].score0:
                    best[m.key()] = (rule, m)
            # A full fit for the best structures, and for the 3 best of each outer function.
            order = sorted(best.values(), key=lambda rm: -rm[1].score0)
            short = {id(rm[1]): rm for rm in order[:top_k]}
            for w in targets:
                short.update({id(rm[1]): rm for rm in [rm for rm in order if rm[1].warp == w][:3]})
            short = list(short.values())
            if verbose > 1:
                for rule, m in short:
                    print(f"        quick {m.score0:9.2f}  {rule:14s} {m.key()}")
            full = pool.map(_full, [(m, X, targets[m.warp]) for _, m in short])
            n_fits += len(full)
            for (rule, k), m in zip(short, full):
                tried.update((k.key(), m.key()))
                if verbose > 1:
                    print(f"        full  {m.score:9.2f}  {rule:14s} {m.key()}")
                if m.key() not in seen or m.score > seen[m.key()].score:
                    seen[m.key()] = m
                    m.rule = rule
            top = max(seen.values(), key=lambda m: m.score)
            now = {w: max((m.score for m in seen.values() if m.warp == w), default=-np.inf)
                   for w in targets}
            stale = 0 if any(now[w] > best_of.get(w, -np.inf) + 0.1 for w in now) else stale + 1
            best_of = now
            if top.key() != trace[-1][1]:
                trace.append((top.score, top.key(), getattr(top, 'rule', '')))
            say(f"{step:4d} {top.score:9.2f}  {top.key()}   [{len(cands)} candidates, "
                f"{t2 - t1:.0f} + {time.time() - t2:.0f} s]")
            if stale >= patience:
                break
    best = [m.condition(X, *targets[m.warp]) for m in sorted(seen.values(), key=lambda m: -m.score)[:5]]
    return best[0], trace, dict(n_fits=n_fits, n_models=len(tried), seconds=time.time() - t0,
                                candidates=best)
