"""Trainer: chooses the points at which to evaluate a target, to train a GPTree.

``Trainer.suggest`` returns points, the user evaluates the target at them, and
``Trainer.add`` passes the results to the tree:

    trainer = Trainer(gpt, bounds=(lower, upper), rtol=0.01)
    X = trainer.suggest(50)
    trainer.add(X, y, sigma)        # the target at X and its uncertainty
"""

from collections import deque

import numpy as np
from scipy.special import ndtr

CANDIDATES_PER_POINT = 20   # candidates drawn for each point asked for, at least
CHECKS_BY_RANGE = 0.5       # share of the random checks drawn by the probability of the range
N_CHECKS = 200              # the random checks kept for the report, the latest ones


class Trainer:
    """Suggests the points at which a tree needs the target most.

    The tree is accurate enough at a point when its predicted sigma is below
    the tolerance ``atol + rtol * |prediction|``. The need at a point is the
    squared sigma in units of the tolerance, times the probability that the
    target lies in ``target_range``: above 1 the tree is not yet accurate
    enough there. ``suggest`` draws points from a set of candidates with a
    probability proportional to their need.

    A share of the suggested points is drawn without regard to the need.
    ``add`` predicts these random checks before it adds them, and ``report``
    measures the tree by their errors.

    Args:
        tree (GPTree): The tree to train. It may hold points already.
        bounds (tuple): ``(lower, upper)``, each with one entry per input. The
            candidates are uniform in this box.
        candidates (callable): In place of ``bounds``: ``candidates(n)``
            returns ``n`` candidate points, one per row, e.g. draws from a
            prior.
        atol, rtol (float or array): The tolerance, for all outputs or with
            one entry per output. With ``rtol`` alone the tolerance vanishes
            where the target is zero.
        target_range (tuple): ``(lowest, highest)`` target value of interest,
            for all outputs or with one entry per output. Defaults to all
            values.
        random_share (float): Share of the suggested points that are random
            checks. Defaults to 0.1.
        n_candidates (int): Candidates drawn for a call of ``suggest``.
            Defaults to 2000.
        n_initial (int): Every suggested point is a random candidate until
            the tree has seen this many points. Defaults to ``2 * Nbar``.
        seed: Seed of the random draws.
    """

    def __init__(self, tree, bounds=None, candidates=None, atol=0.0, rtol=0.0, target_range=(-np.inf, np.inf),
                 random_share=0.1, n_candidates=2000, n_initial=None, seed=None):
        if (bounds is None) == (candidates is None):
            raise ValueError("Give either bounds or candidates.")
        if not np.any(atol) and not np.any(rtol):
            raise ValueError("Give a tolerance: atol, rtol or both.")
        self.tree = tree
        self.rng = np.random.default_rng(seed)
        if candidates is None:
            lower, upper = (np.asarray(bound, dtype=float) for bound in bounds)
            candidates = lambda n: self.rng.uniform(lower, upper, (n, lower.size))
        self.candidates = candidates
        self.atol, self.rtol = np.asarray(atol, dtype=float), np.asarray(rtol, dtype=float)
        self.target_range = tuple(np.asarray(limit, dtype=float) for limit in target_range)
        self.random_share = random_share
        self.n_candidates = n_candidates
        self.n_initial = 2 * tree.root.Nbar if n_initial is None else n_initial
        # The random checks that were suggested and not yet added, with their weights, and those added
        self._pending = {}
        self._checks = deque(maxlen=N_CHECKS)

    def _score(self, X):
        """For each row of ``X``: the probability that the target lies in the
        range, and the squared sigma in units of the tolerance, the largest
        over the outputs."""
        mu, sigma = self.tree.predict(X, mode='loop')
        sigma = np.maximum(sigma, 1e-300)
        lowest, highest = self.target_range
        in_range = (ndtr((highest - mu) / sigma) - ndtr((lowest - mu) / sigma)).prod(axis=1)
        return in_range, ((sigma / (self.atol + self.rtol * np.abs(mu))) ** 2).max(axis=1)

    def need(self, X):
        """The need of the tree for the target at each row of ``X``."""
        in_range, excess = self._score(X)
        return in_range * excess

    def suggest(self, n):
        """``n`` points at which to evaluate the target, one per row."""
        if self.tree.n_points_seen < self.n_initial:
            return self.candidates(n)
        X = np.ascontiguousarray(self.candidates(max(self.n_candidates, CANDIDATES_PER_POINT * n)), dtype=float)
        in_range, excess = self._score(X)

        # A random check is drawn uniformly or, the share CHECKS_BY_RANGE of them, by
        # the probability of the range. Its weight makes up for the latter in the report.
        by_range = in_range + 1e-300
        chance = (1.0 - CHECKS_BY_RANGE) / len(X) + CHECKS_BY_RANGE * by_range / by_range.sum()
        checks = self.rng.choice(len(X), self.rng.binomial(n, self.random_share), replace=False, p=chance)
        for i in checks:
            self._pending[X[i].tobytes()] = 1.0 / (len(X) * chance[i])

        others = np.delete(np.arange(len(X)), checks)
        weights = in_range[others] * excess[others]
        weights += 1e-12 * weights.max() + 1e-300     # so that there are enough candidates to draw from
        picks = self.rng.choice(others, n - len(checks), replace=False, p=weights / weights.sum())
        return X[np.concatenate((checks, picks))]

    def add(self, X, y, sigma):
        """Add evaluated points to the tree, as ``GPTree.batch_update_tree``
        does. The random checks among them are found by their inputs, which
        have to be as ``suggest`` returned them."""
        X = np.ascontiguousarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(len(X), -1)
        rows = [i for i, x in enumerate(X) if x.tobytes() in self._pending]
        if rows:
            mu, predicted_sigma = self.tree.predict(X[rows])
            weights = [self._pending.pop(X[i].tobytes()) for i in rows]
            self._checks.extend(zip(y[rows] - mu, predicted_sigma, y[rows], weights))
        self.tree.batch_update_tree(X, y, sigma)

    def report(self):
        """How far the training has come, as a dict.

            n_points: Points the tree has seen.
        From the predictions at a new set of candidates:
            share_in_range: Share of the candidates expected to lie in the
                target range.
            share_above_tolerance: Share of that with a sigma above the
                tolerance.
        From the latest random checks with a target value in the range, each
        predicted before it was added:
            n_checks: Their number.
            coverage: Share of their errors within one predicted sigma.
            rms_error: Their RMS error, per output.
            share_within_tolerance: Share of them with an error within the
                tolerance.
        """
        in_range, excess = self._score(self.candidates(self.n_candidates))
        report = {'n_points': self.tree.n_points_seen,
                  'share_in_range': in_range.mean(),
                  'share_above_tolerance': in_range @ (excess > 1.0) / max(in_range.sum(), 1e-300),
                  'n_checks': 0, 'coverage': np.nan, 'rms_error': np.nan, 'share_within_tolerance': np.nan}
        if self._checks:
            error, sigma, y, weight = (np.array(part) for part in zip(*self._checks))
            lowest, highest = self.target_range
            inside = ((y >= lowest) & (y <= highest)).all(axis=1)
            if inside.any():
                error, sigma, y = np.abs(error[inside]), sigma[inside], y[inside]
                weight = weight[inside] / weight[inside].sum()
                report.update(n_checks=int(inside.sum()),
                              coverage=weight @ (error <= sigma).mean(axis=1),
                              rms_error=np.sqrt(weight @ error ** 2),
                              share_within_tolerance=weight @ (error <= self.atol + self.rtol * np.abs(y)).all(axis=1))
        return report
