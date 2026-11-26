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

# Values a fitted parameter is rounded to if the evidence allows.
NICE = {
    'pow': [np.array([-3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3])],
    'mono': [np.array([-3, -2, -1, -0.5, 0.5, 1, 2, 3])],
    'lin': [np.array([-2, -1, -0.5, 0.5, 1, 2])],
    'sin': [np.array([0, np.pi / 2]), None],
    'aff': [np.array([-2, -1, -0.5, 0.5, 1, 2]), np.array([-2, -1, -0.5, 0.5, 2])],
}
BOUNDS = {
    'pow': [(-4, 4)], 'mono': [(-4, 4)], 'lin': [(-50, 50)], 'exp': [(-50, 50)],
    'aff': [(-50, 50), (-4, 4)], 'sin': [(-2, 2), (1e-3, 100)], 'log': [],
}
KERNEL = 'matern52'     # kernel of the holes: 'rbf' or 'matern52'
NODE_COST = 3.0         # cost in log evidence of each symbolic function in a model
POWER_COST = 10.0       # ... and of each exponent that is not one of the NICE values
EXPONENT = {'pow': 0, 'mono': 0, 'aff': 1}      # which parameter of a node is an exponent
LOG_BOUNDS = {'amp': (-7, 4), 'ls': (-3, 5), 'const': (-7, 4), 'noise': (-9, 1)}


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
    """A symbolic function of the inputs, with parameters th."""

    def __init__(self, kind, kids=(), th=(), free=None, j=None, bounds=None):
        self.kind, self.kids, self.j = kind, list(kids), j
        self.th = np.array(th, float)
        self.free = np.ones(len(self.th), bool) if free is None else np.array(free, bool)
        self.bounds = bounds or BOUNDS.get(kind, [])

    def __call__(self, X):
        k, th = self.kind, self.th
        if k == 'var':
            return X[:, self.j]
        u = self.kids[0](X)
        if k == 'mono':
            return u * self.kids[1](X) ** th[0]
        if k == 'lin':
            return u + th[0] * self.kids[1](X)
        if k == 'pow':
            return u ** th[0]
        if k == 'exp':
            return np.exp(th[0] * u)
        if k == 'aff':
            return (1 + th[0] * u) ** th[1]
        if k == 'sin':
            return np.sin(th[0] + th[1] * u)
        if k == 'log':
            return np.log(u)

    def walk(self):
        yield self
        for kid in self.kids:
            yield from kid.walk()

    def text(self, key=False):
        """The expression as a string. With key=True free parameters are shown
        as '?', so that models with the same structure get the same string."""
        k = self.kind
        p = ['?' if (key and f) else num(v) for v, f in zip(self.th, self.free)]
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


def var(j):
    return Node('var', j=j)


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

    def parts(self, X, ref):
        """a(x) in units of its rms over ref, and the hole inputs standardised
        over ref and divided by the length scales."""
        a = self.scale(X)
        a = a / np.sqrt(np.mean((a if ref is X else self.scale(ref)) ** 2))
        if self.feats is None:
            return a, None
        U = np.column_stack([f(X) for f in self.feats])
        R = U if ref is X else np.column_stack([f(ref) for f in self.feats])
        return a, (U - R.mean(0)) / (R.std(0) * np.exp(self.ls))

    def K(self, Xa, Xb, ref, grad=False):
        """The term's kernel matrix and, with grad=True, its derivatives with
        respect to the log length scales."""
        aa, Za = self.parts(Xa, ref)
        ab, Zb = (aa, Za) if Xb is Xa else self.parts(Xb, ref)
        K = np.exp(2 * self.amp[0]) * np.outer(aa, ab)
        dK = []
        if Za is not None:
            d2 = [(Za[:, None, j] - Zb[None, :, j]) ** 2 for j in range(Za.shape[1])]
            if KERNEL == 'rbf':
                k = g = np.exp(-0.5 * sum(d2))
            else:
                r = np.sqrt(5 * sum(d2))
                k, g = (1 + r + r * r / 3) * np.exp(-r), 5 / 3 * (1 + r) * np.exp(-r)
            dK = [K * g * d for d in d2] if grad else []
            K = K * k
        return (K, dK) if grad else K


class Model:
    def __init__(self, terms, const=-1.0, noise=-2.0):
        self.terms = terms
        self.hyp = np.array([const, noise])             # log const amplitude, log noise
        self.score = -np.inf

    def key(self):
        return ' + '.join(sorted(t.key() for t in self.terms)) or '0'

    def n_holes(self):
        return sum(t.feats is not None for t in self.terms)

    def n_nodes(self):
        """The number of symbolic functions (the inputs themselves do not count)."""
        return sum(node.kind != 'var' for t in self.terms for node in t.nodes())

    def n_free_powers(self):
        return sum(node.free[EXPONENT[node.kind]] for t in self.terms for node in t.nodes()
                   if node.kind in EXPONENT)

    def slots(self):
        """One (array, index, bounds, tag, term index, length scale index) per free parameter."""
        out = []
        for i, t in enumerate(self.terms):
            out.append((t.amp, 0, LOG_BOUNDS['amp'], 'amp', i, None))
            out += [(t.ls, j, LOG_BOUNDS['ls'], 'ls', i, j) for j in range(len(t.ls))]
            for node in t.nodes():
                out += [(node.th, k, node.bounds[k], 'th', i, None)
                        for k in range(len(node.th)) if node.free[k]]
        out.append((self.hyp, 0, LOG_BOUNDS['const'], 'const', None, None))
        out.append((self.hyp, 1, LOG_BOUNDS['noise'], 'noise', None, None))
        return out

    def nll_grad(self, p, X, y, slots):
        """Negative log evidence and its gradient at the parameters p."""
        for v, s in zip(p, slots):
            s[0][s[1]] = v
        n = len(y)
        with np.errstate(all='ignore'):
            parts = [t.K(X, X, X, grad=True) for t in self.terms]
            c2, s2 = np.exp(2 * self.hyp)
            K = sum(k for k, _ in parts) + c2 + (s2 + 1e-8) * np.eye(n)
            try:
                chol = cho_factor(K, lower=True, check_finite=True)
            except (np.linalg.LinAlgError, ValueError):
                return 1e10, np.zeros(len(p))
            alpha = cho_solve(chol, y)
            W = cho_solve(chol, np.eye(n)) - np.outer(alpha, alpha)
            nll = 0.5 * y @ alpha + np.log(np.diag(chol[0])).sum() + 0.5 * n * np.log(2 * np.pi)
            grad = np.zeros(len(p))
            for q, (arr, idx, _, tag, i, j) in enumerate(slots):
                if tag == 'amp':
                    dK = 2 * parts[i][0]
                elif tag == 'ls':
                    dK = parts[i][1][j]
                elif tag == 'const':
                    dK = 2 * c2 * np.ones((n, n))
                elif tag == 'noise':
                    dK = 2 * s2 * np.eye(n)
                else:                                   # a parameter of a symbolic node
                    v, eps = arr[idx], 1e-6 * max(1.0, abs(arr[idx]))
                    arr[idx] = v + eps
                    Kp = self.terms[i].K(X, X, X)
                    arr[idx] = v - eps
                    dK = (Kp - self.terms[i].K(X, X, X)) / (2 * eps)
                    arr[idx] = v
                grad[q] = 0.5 * np.sum(W * dK)
            if not (np.isfinite(nll) and np.isfinite(grad).all()):
                return 1e10, np.zeros(len(p))
        return nll, grad

    def fit(self, X, y, maxiter=200, restarts=0, rng=None):
        """Maximise the evidence over the free parameters. The score is the log
        evidence with a BIC penalty on the number of parameters and a cost for
        each symbolic function and each free exponent. score0 leaves out the
        cost of the exponents, for models that have not been through snap."""
        slots = self.slots()
        bounds = [s[2] for s in slots]
        p0 = np.clip([s[0][s[1]] for s in slots], [b[0] for b in bounds], [b[1] for b in bounds])
        starts = [p0]
        for _ in range(restarts):
            p = p0.copy()
            for q, s in enumerate(slots):
                if s[3] != 'th':
                    p[q] = rng.uniform(-3, -1) if s[3] == 'noise' else rng.uniform(-1.5, 1.5)
            starts.append(p)
        best = None
        for p in starts:
            r = minimize(self.nll_grad, p, args=(X, y, slots), jac=True, method='L-BFGS-B',
                         bounds=bounds, options={'maxiter': maxiter})
            if best is None or r.fun < best.fun:
                best = r
        for v, s in zip(best.x, slots):
            s[0][s[1]] = v
        self.logml, self.n_par = -best.fun, len(slots)
        self.score0 = (self.logml - 0.5 * self.n_par * np.log(len(y))
                       - NODE_COST * self.n_nodes())
        self.score = self.score0 - POWER_COST * self.n_free_powers()
        return self

    def condition(self, X, y):
        """Store what predict and text need."""
        self.X = X
        K = sum(t.K(X, X, X) for t in self.terms) + np.exp(2 * self.hyp[0])
        self.alpha = np.linalg.solve(K + (np.exp(2 * self.hyp[1]) + 1e-8) * np.eye(len(y)), y)
        return self

    def predict(self, Xs):
        with np.errstate(all='ignore'):
            Ks = sum(t.K(Xs, self.X, self.X) for t in self.terms) + np.exp(2 * self.hyp[0])
        return Ks @ self.alpha

    def text(self, ystd=1.0):
        """The model as a string, with the posterior mean of the weights."""
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


def library(f, X):
    """Candidate functions of the feature f: nodes with starting values."""
    u = f(X)
    inner = [f, Node('pow', [f], [2.0], [False])]
    if (u > 0).all():
        inner.append(Node('pow', [f], [-1.0], [False]))
        out = [Node('pow', [f], [p]) for p in (1.0, -1.0)] + [Node('log', [f])]
    else:
        out = [Node('pow', [f], [p], [False]) for p in (1.0, 2.0)]
    for g in inner:
        v = g(X)
        out += [Node('exp', [g], [c / v.std()]) for c in (-1, 1)]
        out += [Node('aff', [g], [c / np.abs(v).max(), -1.0]) for c in (-0.5, 0.5)]
    # A sine has to show at least half a period over the data.
    span = np.ptp(u)
    out += [Node('sin', [f], [0.0, 2 * np.pi * k / span], bounds=[(-2, 2), (np.pi / span, 100)])
            for k in (0.5, 1, 2, 4)]
    return out


def pairings(a, b, X):
    """Candidate combinations of two features into one: a + c*b and a*b^p."""
    out = [Node('lin', [a, b], [c]) for c in (1.0, -1.0)]
    if not (b(X) > 0).all():
        a, b = b, a
    if (b(X) > 0).all():
        out += [Node('mono', [a, b], [p]) for p in (1.0, -1.0)]
    else:
        out += [Node('mono', [a, b], [1.0], [False])]
    return out


def moves(model, X):
    """All one-step refinements of a model, as (rule, new model)."""
    out = []

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


def snap(model, X, y):
    """Round parameters to simple values where that does not lower the score."""
    improved = True
    while improved:
        improved = False
        nodes = [n for t in model.terms for n in t.nodes()]
        for a, node in enumerate(nodes):
            for k in np.flatnonzero(node.free):
                nice = NICE.get(node.kind, [None] * len(node.th))[k]
                if nice is None:
                    continue
                trial = copy.deepcopy(model)
                tn = [n for t in trial.terms for n in t.nodes()][a]
                # A phase of -pi/2 is one of pi/2 with the sign of the weight changed.
                v = abs(node.th[k]) if node.kind == 'sin' else node.th[k]
                tn.th[k] = nice[np.argmin(np.abs(nice - v))]
                tn.free[k] = False
                trial.fit(X, y, maxiter=100)
                if trial.score > model.score:
                    model, improved = trial, True
                    break
            if improved:
                break
    return model


def _init_worker(kernel, node_cost, power_cost):
    # The matrices are small, so BLAS threads in each worker only compete.
    global _limit, KERNEL, NODE_COST, POWER_COST
    _limit, KERNEL, NODE_COST, POWER_COST = threadpool_limits(1), kernel, node_cost, power_cost


def _quick(args):
    model, X, y = args
    return model.fit(X, y, maxiter=25)


def _full(args):
    model, X, y = args
    return snap(model.fit(X, y, maxiter=300), X, y)


def search(X, y, beam=4, top_k=12, patience=3, max_steps=40, workers=4, seed=0,
           kernel='matern52', node_cost=3.0, power_cost=10.0, verbose=True):
    """Beam search from a single hole, until the models in the beam have all
    been refined or the best score has not risen for `patience` steps. Returns
    the model with the best score, the trace of the best score and some
    counts. y should be scaled to unit variance."""
    global KERNEL, NODE_COST, POWER_COST
    KERNEL, NODE_COST, POWER_COST = kernel, node_cost, power_cost
    rng = np.random.default_rng(seed)
    t0 = time.time()
    d = X.shape[1]
    root = Model([Term(feats=[var(j) for j in range(d)])]).fit(X, y, restarts=6, rng=rng)
    seen = {root.key(): root}          # structure key -> fitted model
    tried, expanded, n_quick = {root.key()}, set(), {}
    best_score, stale = root.score, 0
    trace, n_fits = [(root.score, root.key(), 'start')], 1
    say = print if verbose else (lambda *a: None)
    say(f"   0 {root.score:9.2f}  {root.key()}")
    with Pool(workers, initializer=_init_worker, initargs=(kernel, node_cost, power_cost)) as pool:
        for step in range(1, max_steps + 1):
            front = sorted(seen.values(), key=lambda m: -m.score)[:beam]
            todo = [m for m in front if m.key() not in expanded]
            if not todo:
                break
            cands = []
            for m in todo:
                expanded.add(m.key())
                cands += [(rule, c) for rule, c in moves(m, X) if c.key() not in tried]
            # A structure gets quick fits in at most two steps.
            for key in {c.key() for _, c in cands}:
                n_quick[key] = n_quick.get(key, 0) + 1
            cands = [(rule, c) for rule, c in cands if n_quick[c.key()] <= 2]
            t1 = time.time()
            quick = pool.map(_quick, [(c, X, y) for _, c in cands], chunksize=4)
            t2 = time.time()
            n_fits += len(quick)
            best = {}                  # the best quick fit of each structure
            for (rule, _), m in zip(cands, quick):
                if m.key() not in best or m.score0 > best[m.key()][1].score0:
                    best[m.key()] = (rule, m)
            short = sorted(best.values(), key=lambda rm: -rm[1].score0)[:top_k]
            if verbose > 1:
                for rule, m in short:
                    print(f"        quick {m.score0:9.2f}  {rule:14s} {m.key()}")
            full = pool.map(_full, [(m, X, y) for _, m in short])
            n_fits += len(full)
            for (rule, k), m in zip(short, full):
                tried.update((k.key(), m.key()))
                if verbose > 1:
                    print(f"        full  {m.score:9.2f}  {rule:14s} {m.key()}")
                if m.key() not in seen or m.score > seen[m.key()].score:
                    seen[m.key()] = m
                    m.rule = rule
            top = max(seen.values(), key=lambda m: m.score)
            stale = 0 if top.score > best_score + 0.1 else stale + 1
            best_score = max(best_score, top.score)
            if top.key() != trace[-1][1]:
                trace.append((top.score, top.key(), getattr(top, 'rule', '')))
            say(f"{step:4d} {top.score:9.2f}  {top.key()}   [{len(cands)} candidates, "
                f"{t2 - t1:.0f} + {time.time() - t2:.0f} s]")
            if stale >= patience:
                break
    top = max(seen.values(), key=lambda m: m.score).condition(X, y)
    return top, trace, dict(n_fits=n_fits, n_models=len(tried), seconds=time.time() - t0)
