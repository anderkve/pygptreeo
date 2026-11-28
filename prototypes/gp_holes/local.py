"""Holes of local GPs for the GP-hole search (prototype).

The points of a model are divided into blocks of at most `nbar` points: the
leaves of a tree of cuts through the inputs of its holes, as in pygptreeo. A
hole has no covariance between points of different blocks, so it is a GP of
its own in each block. The weights of the symbolic terms and the constant
stay those of the whole model. The covariance of the targets is then a
matrix of blocks plus one of low rank, and the evidence and its gradient cost
one Cholesky factor per block: the cost grows in proportion to the number of
points. A model without a hole needs no blocks.

    search(X, y, model=LocalModel)
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.lapack import dpotri

from gpholes import Model, hole_values, kernel, unwarp


def partition(V, nbar, theta, rng):
    """Blocks of at most nbar of the points V, as lists of their numbers. A
    set of more points is cut at its median along the input of largest
    range. A point within theta times that range of the cut goes to either
    side, more likely to the one it is on (the overlap of pygptreeo)."""
    out, todo = [], [np.arange(len(V))]
    while todo:
        idx = todo.pop()
        if len(idx) <= nbar:
            out.append(np.sort(idx))
            continue
        j = np.argmax(np.ptp(V[idx], 0))
        v = V[idx, j]
        width = theta * np.ptp(v)
        p = np.clip((v - np.median(v)) / width + 0.5, 0, 1) if width > 0 else 0.5
        right = rng.random(len(idx)) < p
        if right.all() or not right.any():          # inputs that are all the same: any halves
            right = np.arange(len(idx)) % 2 == 0
        todo += [idx[right], idx[~right]]
    return out


class LocalModel(Model):
    """A model whose holes are GPs within blocks of points (see partition).
    The blocks are those of the inputs of all its holes together, and are
    made anew for each fit."""
    nbar, theta = 100, 0.05

    def space(self, X, ref):
        """The standardised inputs of all the holes, each once. None without a hole."""
        cols = {f.text(): v for t in self.terms if t.feats is not None
                for f, v in zip(t.feats, t.inputs(X, ref).T)}
        return np.column_stack(list(cols.values())) if cols else None

    def divide(self, X):
        """Make the blocks for the points X, unless the blocks are for them already."""
        made_for = (len(X), tuple(f.text() for t in self.terms for f in t.feats or []))
        if getattr(self, 'made_for', None) != made_for:
            V = self.space(X, X)
            self.blocks = [None] if V is None else partition(V, self.nbar, self.theta, np.random.default_rng(0))
            self.made_for = made_for

    def factors(self, X, y, w):
        """The Cholesky factors of the blocks and what follows from them, or
        None if a block has none. With K = B + U U^T, where B has the holes
        and the noise and U a column for each symbolic term and the constant:
        B^-1 y, B^-1 U, the factor of M = 1 + U^T B^-1 U, alpha = K^-1 y, the
        negative log evidence, U, and for each block its points, its factor
        and the pieces of each hole (see Term.train)."""
        n = len(y)
        noise = self.noise(n, w)
        a = [t.factor(X, X) for t in self.terms]
        V = [t.inputs(X, X) for t in self.terms]
        hol = [i for i, t in enumerate(self.terms) if t.feats is not None]
        U = np.column_stack([np.exp(t.amp[0]) * a[i] for i, t in enumerate(self.terms) if t.feats is None]
                            + [np.full(n, np.exp(self.hyp[0]))])
        if not np.isfinite(U).all():
            return None
        By, BU, logdet, pieces = np.empty(n), np.empty_like(U), 0.0, []
        for idx in self.blocks:
            if idx is None:                     # no hole: B is the noise
                By, BU, logdet = y / noise, U / noise[:, None], np.log(noise).sum()
                break
            K, cs = np.diag(noise[idx]), {}
            for i in hol:
                t, c = self.terms[i], dict(a=a[i][idx], V=V[i][idx])
                c['P'], c['E'] = np.outer(c['a'], c['a']), (c['V'][:, None] - c['V'][None]) ** 2
                c['k'], c['g'] = kernel(c['E'] @ np.exp(-2 * t.ls))
                c['K'] = np.exp(2 * t.amp[0]) * c['P'] * c['k']
                K, cs[i] = K + c['K'], c
            try:
                chol = cho_factor(K, lower=True, check_finite=True)
            except (np.linalg.LinAlgError, ValueError):
                return None
            By[idx], BU[idx] = cho_solve(chol, y[idx]), cho_solve(chol, U[idx])
            logdet += 2 * np.log(np.diag(chol[0])).sum()
            pieces.append((idx, chol, cs))
        try:
            cm = cho_factor(np.eye(U.shape[1]) + U.T @ BU, lower=True, check_finite=True)
        except (np.linalg.LinAlgError, ValueError):
            return None
        alpha = By - BU @ cho_solve(cm, U.T @ By)
        nll = 0.5 * y @ alpha + 0.5 * logdet + np.log(np.diag(cm[0])).sum() + 0.5 * n * np.log(2 * np.pi)
        return By, BU, cm, alpha, nll, U, pieces

    def solve(self, X, y, w, cache=None):
        """None for the Cholesky factor, which is one per block, and alpha =
        K^-1 y; or None if there is no factor."""
        self.divide(X)
        f = self.factors(X, y, w)
        return None if f is None else (None, f[3])

    def holes(self, X, y, w):
        self.divide(X)
        f = self.factors(X, y, w)
        if f is None:
            return None, None
        alpha, pieces, n = f[3], f[6], len(y)
        sigma, out = np.sqrt(self.noise(n, w)), []
        for i, t in enumerate(self.terms):
            if t.feats is None:
                out.append(None)
                continue
            q = len(t.feats)
            h, err, D, D2 = np.empty(n), np.empty(n), np.empty((n, q)), np.empty((n, q))
            for idx, _, cs in pieces:
                h[idx], err[idx], D[idx], D2[idx] = hole_values(t, cs[i], alpha[idx], sigma[idx])
            out.append((h, err, D, D2))
        return out, np.exp(2 * self.hyp[0]) * alpha.sum()

    def nll_grad(self, p, X, y, slots, w, cache):
        for v, s in zip(p, slots):
            s[0][s[1]] = v
        n = len(y)
        with np.errstate(all='ignore'):
            f = self.factors(X, y, w)
            if f is None:
                return 1e10, np.zeros(len(p))
            By, BU, cm, alpha, nll, U, pieces = f
            Minv = cho_solve(cm, np.eye(U.shape[1]))
            KU, u = BU @ Minv, U.T @ alpha                      # K^-1 U and U^T alpha
            UWU = np.eye(len(u)) - Minv - np.outer(u, u)        # U^T (K^-1 - alpha alpha^T) U
            # For each term: the derivatives of Term.grad, summed or put together over the blocks.
            d, column = {}, 0
            for i, t in enumerate(self.terms):
                if t.feats is None:
                    d[i] = [UWU[column, column], None, np.exp(t.amp[0]) * (KU[:, column] - alpha * u[column]), None]
                    column += 1
                else:
                    d[i] = [0.0, np.zeros(len(t.ls)), np.empty(n), np.empty((n, len(t.feats)))]
            if pieces:
                diag = np.empty(n)
                for idx, chol, cs in pieces:
                    inv = dpotri(chol[0], lower=True)[0]
                    W = np.tril(inv) + np.tril(inv, -1).T - BU[idx] @ Minv @ BU[idx].T - np.outer(alpha[idx], alpha[idx])
                    diag[idx] = np.diag(W)
                    for i, c in cs.items():
                        d_amp, d_ls, d[i][2][idx], d[i][3][idx] = self.terms[i].grad(W, c)
                        d[i][0] += d_amp
                        d[i][1] += d_ls
            else:
                diag = 1 / self.noise(n, w) - np.einsum('ij,jk,ik->i', BU, Minv, BU) - alpha ** 2
            grad = np.zeros(len(p))
            for q, (arr, idx, _, tag, i, j) in enumerate(slots):
                if tag == 'amp':
                    grad[q] = d[i][0]
                elif tag == 'ls':
                    grad[q] = d[i][1][j]
                elif tag == 'const':
                    grad[q] = UWU[-1, -1]
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

    def fit(self, X, y, w=None, **kw):
        self.divide(X)
        return super().fit(X, y, w, **kw)

    def condition(self, X, y, w=None):
        w = w or (None, y ** 2)
        self.X, self.alpha = X, self.solve(X, y, w)[1]
        return self

    def predict(self, Xs):
        """The target at Xs. A hole is predicted from the block whose points
        are nearest in the mean."""
        X, alpha = self.X, self.alpha
        with np.errstate(all='ignore'):
            z = np.full(len(Xs), np.exp(2 * self.hyp[0]) * alpha.sum())
            V, Vs = self.space(X, X), self.space(Xs, X)
            if V is not None:
                centres = np.array([V[idx].mean(0) for idx in self.blocks])
                block = np.argmin(((Vs[:, None] - centres[None]) ** 2).sum(2), 1)
            for t in self.terms:
                a, a_s = t.factor(X, X), t.factor(Xs, X)
                if t.feats is None:
                    z += np.exp(2 * t.amp[0]) * a_s * (a @ alpha)
                    continue
                U, Us = t.inputs(X, X), t.inputs(Xs, X)
                for b, idx in enumerate(self.blocks):
                    at = block == b
                    k = kernel((Us[at][:, None] - U[idx][None]) ** 2 @ np.exp(-2 * t.ls))[0]
                    z[at] += np.exp(2 * t.amp[0]) * a_s[at] * (k @ (a[idx] * alpha[idx]))
        if self.warp is None:
            return z
        return unwarp(self.warp, z * self.warp_args[0], self.warp_args[1]) / self.y_scale
