"""Larger data sets for the GP-hole search (prototype).

A fit of the search costs n^3 for n points, and a search has thousands of
fits, so the search itself stays on about 100 points. The other points come in
before it and after it, through trees of local GPs (GPTree of pygptreeo):

- `distil`: a GPTree is fitted to all the points. The search then runs on its
  predictions at some of them, which have less noise than the points have.
- `validate`: a model of the search is fitted to all the points but those
  held out, where each hole is a GPTree of the inputs of the hole, and scored
  by the likelihood of the held-out points.

Needs pygptreeo and scikit-learn.
"""
import copy
import warnings

import numpy as np
from scipy.optimize import least_squares, minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from gpholes import CFG, OUTER_TEXT, num, unwarp, warp, wrap
from pygptreeo import Default_GPR, GPTree

THETA = 0.05        # the overlap of the two children of a leaf that is split


def tree(U, z, rng, nbar=100):
    """A GPTree fitted to the targets z at the inputs U, with leaves of at
    most nbar points. The noise is not known, so each leaf learns its own
    level, from a start at half of the variance of its targets."""
    kernel = ConstantKernel() * Matern(nu=2.5, length_scale=np.ones(U.shape[1])) + WhiteKernel(0.5, (1e-8, 1e2))
    gpt = GPTree(GPR=Default_GPR(kernel=kernel), Nbar=nbar, theta=THETA,
                 use_calibrated_sigma=False, split_dimension_criteria='max_spread')
    order = rng.permutation(len(z))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        gpt.fit(U[order], z[order], np.full(len(z), 1e-6 * z.std()), shuffle=False)
    return gpt


def mean(gpt, U):
    """The prediction of a GPTree at the inputs U."""
    return gpt.predict(U)[0].ravel()


def distil(X, y, m, rng, nbar=300):
    """m of the points X with the prediction of a GPTree of all the points in
    place of their targets. The leaves are larger than those of a hole: a
    leaf of more points averages more of the noise away."""
    gpt = tree(X, y, rng, nbar)
    pick = rng.choice(len(y), m, replace=False)
    return X[pick], mean(gpt, X[pick])


def noise_shape(e, yhat):
    """The sizes (a, r) of noise with variance a^2 + r^2*yhat^2 that are most
    likely for the residuals e, and their log likelihood."""
    scale = np.sqrt(np.mean(e ** 2)) + 1e-300

    def nll(p):
        v = np.exp(2 * p[0]) + np.exp(2 * p[1]) * yhat ** 2
        return 0.5 * np.sum(e ** 2 / v + np.log(2 * np.pi * v))

    starts = [(np.log(scale), np.log(scale) - 20), (np.log(scale) - 20, np.log(scale / np.sqrt(np.mean(yhat ** 2))))]
    best = min((minimize(nll, s, method='Nelder-Mead') for s in starts), key=lambda r: r.fun)
    return np.exp(best.x), -best.fun


class Fitted:
    """A model of the search fitted to more points (see validate): its
    prediction, its text and its score."""

    def __init__(self, model, predict, text, score, holes):
        self.model, self.predict, self.text, self.score, self.holes = model, predict, text, score, holes


def validate(model, X, y, Xv, yv, rng, sweeps=2):
    """Fit a model of the search to the points (X, y), with y in its own
    units, and score it on the held-out points (Xv, yv). The weights of the
    terms and the free parameters of the symbolic functions are fitted by
    least squares. A hole is a GPTree of its inputs, fitted to what the other
    terms leave of the target inside the function around the expression, in
    `sweeps` rounds over the holes; the symbolic parameters of a model with a
    hole are kept. The score is the log likelihood of the held-out points, for
    the most likely noise of the same size and in proportion to the
    prediction, minus the costs of the search for the symbolic functions, the
    holes and the parameters. Returns None for a model that cannot be
    evaluated at these points."""
    model = copy.deepcopy(model)
    terms = model.terms
    sym = [t for t in terms if t.feats is None]
    holes = [t for t in terms if t.feats is not None]
    c0 = model.warp_args[1] if model.warp else 0.0
    outer = (lambda z: unwarp(model.warp, z, c0)) if model.warp else (lambda z: z)
    slots = [(node.th, k, node.bounds[k]) for t in terms for node in t.nodes() for k in np.flatnonzero(node.free)]

    def columns(Z):
        return np.column_stack([t.scale(Z) for t in sym] + [np.ones(len(Z))])

    with np.errstate(all='ignore'):
        z = y if model.warp is None else (warp(model.warp, y) or [None])[0]
        if z is None or not np.isfinite(z).all() or not np.isfinite(columns(X)).all():
            return None
        if model.warp in ('exp', 'expneg'):     # with the constant of the function as it was in the search
            z = np.exp(y - c0) if model.warp == 'exp' else np.exp(c0 - y)
        w = np.linalg.lstsq(columns(X), z, rcond=None)[0]
        trees = []
        if holes:
            parts = np.zeros((len(holes), len(y)))
            for _ in range(sweeps):
                for i, t in enumerate(holes):
                    a, U = t.scale(X), np.column_stack([f(X) for f in t.feats])
                    if not (np.isfinite(a).all() and np.isfinite(U).all() and a.any()):
                        return None
                    floor = 1e-3 * np.sqrt(np.mean(a ** 2))         # a hole is not read off where its factor vanishes
                    a = np.where(np.abs(a) < floor, np.where(a < 0, -floor, floor), a)
                    rest = z - columns(X) @ w - parts.sum(0) + parts[i]
                    gpt = tree(U, rest / a, rng)
                    parts[i] = a * mean(gpt, U)
                    trees = trees[:i] + [gpt] + trees[i + 1:]
                    w = np.linalg.lstsq(columns(X), z - parts.sum(0), rcond=None)[0]

            def inner(Z):
                return columns(Z) @ w + sum(t.scale(Z) * mean(g, np.column_stack([f(Z) for f in t.feats]))
                                            for t, g in zip(holes, trees))
        else:
            # Weights and free parameters together, in the units of the target, in two
            # rounds: the second with the noise that the residuals of the first show.
            p = np.concatenate([w, [arr[k] for arr, k, _ in slots]])
            lo = np.concatenate([np.full(len(w), -np.inf), [b[0] for _, _, b in slots]])
            hi = np.concatenate([np.full(len(w), np.inf), [b[1] for _, _, b in slots]])

            def predict_with(p, Z):
                for v, (arr, k, _) in zip(p[len(w):], slots):
                    arr[k] = v
                return outer(columns(Z) @ p[:len(w)])

            def residual(p, s):
                r = (predict_with(p, X) - y) / s
                return np.where(np.isfinite(r), r, 1e6)

            s = np.full(len(y), y.std())
            for _ in range(2):
                p = least_squares(residual, np.clip(p, lo, hi), args=(s,), bounds=(lo, hi), x_scale='jac').x
                yhat = predict_with(p, X)
                if not np.isfinite(yhat).all():
                    return None
                (a_, r_), _ = noise_shape(yhat - y, yhat)
                s = np.sqrt(a_ ** 2 + r_ ** 2 * yhat ** 2)
            w = p[:len(w)]

            def inner(Z):
                return columns(Z) @ w

        predict = lambda Z: outer(inner(Z))
        yhat, yvhat = predict(X), predict(Xv)
        if not (np.isfinite(yhat).all() and np.isfinite(yvhat).all()):
            return None
        loglik = noise_shape(yvhat - yv, yvhat)[1]
    n_par = len(w) + len(slots) + 2
    cost = (CFG['node_cost'] * (model.n_nodes() + len(holes) + bool(model.warp)) + 0.5 * n_par * np.log(len(y))
            + CFG['power_cost'] * model.n_free_powers())
    parts = ['*'.join([num(wk)] + [wrap(f.text()) for f in t.factors]) for wk, t in zip(w, sym)]
    parts += ['*'.join([wrap(f.text()) for f in t.factors] + ['H(' + ', '.join(f.text() for f in t.feats) + ')'])
              for t in holes]
    text = ' + '.join(parts + [num(w[-1])]).replace('+ -', '- ')
    if model.warp:
        text = OUTER_TEXT[model.warp].format(m=text, c=num(c0))
    return Fitted(model, predict, text, loglik - cost, len(holes))
