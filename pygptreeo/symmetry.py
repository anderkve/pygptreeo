"""Prior symmetries of the target function.

``Symmetry`` is a finite group of input transformations (reflections, permutations
of the inputs, a finite set of periodic shifts, ...). ``SymmetrizedKernel`` is a
leaf kernel that is partly invariant under it, with the invariant fraction
``alpha`` a fitted hyperparameter. ``GPTree(symmetry=...)`` partitions the input
space in the symmetry's canonical coordinates, so that symmetric points share a
leaf and the symmetrized kernel can use them.
"""

import itertools
import math
import warnings

import numpy as np
from sklearn.gaussian_process.kernels import (Kernel, Hyperparameter, Sum, Product, Exponentiation,
                                              ConstantKernel, WhiteKernel, RBF, Matern)


# Picklable group elements (GPTree.save pickles the tree, kernels and symmetry).

class _Identity:
    def __call__(self, X):
        return X


class _Permute:
    """Column permutation ``X[:, perm]``."""

    def __init__(self, perm):
        self.perm = np.asarray(perm, dtype=int)

    def __call__(self, X):
        return X[:, self.perm]


class _Shift:
    """Translation of the columns ``dims`` by ``offsets``."""

    def __init__(self, dims, offsets):
        self.dims = np.asarray(dims, dtype=int)
        self.offsets = np.asarray(offsets, dtype=float)

    def __call__(self, X):
        Y = np.array(X, dtype=float, copy=True)
        Y[:, self.dims] += self.offsets
        return Y


class _Modulo:
    """Reduce the columns ``dims`` modulo ``periods`` into [0, period)."""

    def __init__(self, dims, periods):
        self.dims = np.asarray(dims, dtype=int)
        self.periods = np.asarray(periods, dtype=float)

    def __call__(self, X):
        Y = np.array(X, dtype=float, copy=True)
        Y[:, self.dims] = np.mod(Y[:, self.dims], self.periods)
        return Y


class _Composed:
    """``outer(inner(X))``."""

    def __init__(self, outer, inner):
        self.outer = outer
        self.inner = inner

    def __call__(self, X):
        return self.outer(self.inner(X))


def _compose(outer, inner):
    if isinstance(outer, _Identity):
        return inner
    if isinstance(inner, _Identity):
        return outer
    if isinstance(outer, _Permute) and isinstance(inner, _Permute):
        return _Permute(inner.perm[outer.perm])
    return _Composed(outer, inner)


class Symmetry:
    """A finite group of transformations of the input space under which the target
    function is (approximately) invariant.

    Parameters
    ----------
    generators : list of callables, optional
        Each maps an array of inputs of shape (n, d) to their images, shape (n, d).
        The group they generate is found numerically by composing them until no
        new element appears, so they must be exact to floating-point precision and
        generate at most ``max_order`` elements. Use module-level functions (not
        lambdas) if the tree is to be saved with ``GPTree.save``.
    permutations : list of index arrays, optional
        Generators that permute the input columns: ``X[:, perm]``. A kernel with
        one length scale per input then ties the length scales within each orbit
        of the group (see ``SymmetrizedKernel``).
    canonical : callable, optional
        Picks one image per orbit, ``canonical(X) -> (n, d)``. By default the
        lexicographically smallest image, which maps reflections onto a folded
        domain and permutations onto a sorted order.
    max_order : int, default=1024
        Largest group accepted.
    n_features : int, optional
        Input dimension; otherwise taken from the permutations or the first inputs seen.

    Examples
    --------
    >>> def flip(X):            # x1 -> 1 - x1
    ...     Y = X.copy(); Y[:, 1] = 1 - X[:, 1]; return Y
    >>> sym = Symmetry(generators=[flip])
    >>> sym = Symmetry(permutations=[[1, 0, 2]])   # the target is symmetric in x0 <-> x1
    """

    def __init__(self, generators=None, permutations=None, canonical=None, max_order=1024, n_features=None):
        gens = list(generators or [])
        for p in permutations or []:
            p = np.asarray(p, dtype=int)
            if sorted(p.tolist()) != list(range(p.size)):
                raise ValueError(f"not a permutation: {p.tolist()}")
            gens.append(_Permute(p))
        if not gens:
            raise ValueError("give at least one generator or permutation")
        self._generators = gens
        self._canonical = canonical
        self.max_order = max_order
        self._elements = None
        self._orbits = None
        self._n_features = None
        if permutations:
            self._n_features = int(np.asarray(permutations[0]).size)
        elif n_features is not None:
            self._n_features = int(n_features)

    @classmethod
    def periodic(cls, dims, periods, n_images=1):
        """The target is periodic in the inputs ``dims`` with the given periods.

        Uses the ``(2 n_images + 1) ** len(dims)`` translations by up to
        ``n_images`` periods in each direction, a finite part of the (infinite)
        group of shifts. Canonical coordinates reduce each input modulo its period.
        """
        dims = [int(i) for i in np.atleast_1d(dims)]
        periods = np.broadcast_to(np.asarray(periods, dtype=float), (len(dims),)).copy()
        shifts = [_Shift(dims, np.asarray(ks, dtype=float) * periods)
                  for ks in itertools.product(range(-n_images, n_images + 1), repeat=len(dims)) if any(ks)]
        sym = cls(generators=shifts, canonical=_Modulo(dims, periods))
        sym._elements = [_Identity()] + shifts
        return sym

    @property
    def n_features(self):
        """Input dimension, once known (from permutations, or from the first inputs seen)."""
        return self._n_features

    def _closure(self, d):
        """All group elements, identity first, by composing the generators until
        nothing new appears. Elements are told apart by their action on probe points."""
        if all(isinstance(g, _Permute) for g in self._generators):
            probes = np.arange(d, dtype=float)[None, :]
        else:
            probes = np.random.default_rng(0).uniform(-0.5, 0.5, size=(8, d)) + 0.5
        for g in self._generators:
            img = np.asarray(g(probes))
            if img.shape != probes.shape:
                raise ValueError(f"a generator maps shape {probes.shape} to {img.shape}")
        elements, actions = [_Identity()], [probes.copy()]
        frontier = list(elements)
        while frontier:
            new = []
            for a in frontier:
                for g in self._generators:
                    c = _compose(g, a)
                    img = np.asarray(c(probes), dtype=float)
                    if any(np.allclose(img, seen, rtol=1e-9, atol=1e-9) for seen in actions):
                        continue
                    if len(elements) >= self.max_order:
                        raise ValueError(f"the generators give more than max_order={self.max_order} elements; "
                                         "for periodic shifts use Symmetry.periodic")
                    elements.append(c); actions.append(img); new.append(c)
            frontier = new
        return elements

    def elements(self, n_features=None):
        """The group elements as callables, identity first."""
        if self._elements is None:
            d = n_features if n_features is not None else self._n_features
            if d is None:
                raise ValueError("the input dimension is not known yet; pass n_features")
            self._elements = self._closure(int(d))
            self._n_features = int(d)
        return self._elements

    @property
    def n_elements(self):
        return len(self.elements())

    def images(self, X):
        """All images g(X) of the inputs, shape (n_elements, n, d), the identity first."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.stack([np.asarray(g(X), dtype=float) for g in self.elements(X.shape[1])])

    def canonical(self, X):
        """One representative per orbit: the lexicographically smallest image, or the
        user's ``canonical`` map."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self._canonical is not None:
            return np.atleast_2d(np.asarray(self._canonical(X), dtype=float))
        imgs = self.images(X)
        G, n, d = imgs.shape
        if G == 1:
            return X
        alive = np.ones((n, G), dtype=bool)
        for j in range(d):
            col = imgs[:, :, j].T
            m = np.where(alive, col, np.inf).min(axis=1, keepdims=True)
            alive &= col <= m + 1e-12 * (1.0 + np.abs(m))
        return imgs[alive.argmax(axis=1), np.arange(n)]

    @property
    def orbits(self):
        """Orbits of the input indices under the group when every element permutes
        the inputs (a list of index lists covering every input), else None."""
        if self._orbits is None:
            elems = self.elements()
            if not all(isinstance(g, (_Identity, _Permute)) for g in elems):
                return None
            d = self._n_features
            parent = list(range(d))

            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]; i = parent[i]
                return i
            for g in elems:
                if isinstance(g, _Permute):
                    for i in range(d):
                        parent[find(i)] = find(int(g.perm[i]))
            groups = {}
            for i in range(d):
                groups.setdefault(find(i), []).append(i)
            self._orbits = sorted(groups.values())
        return self._orbits

    def __repr__(self):
        n = len(self._elements) if self._elements is not None else "?"
        return f"Symmetry(n_generators={len(self._generators)}, n_elements={n})"


def _pair_eval(kernel, X, Y):
    """k(X, Y) and its gradient w.r.t. the kernel's log hyperparameters for Y != X,
    shape (n, m, n_dims). scikit-learn's kernels only give gradients for Y = X, so
    the common ones are evaluated here; others are evaluated on the stacked set."""
    n, m = X.shape[0], Y.shape[0]
    if isinstance(kernel, Sum):
        K1, G1 = _pair_eval(kernel.k1, X, Y); K2, G2 = _pair_eval(kernel.k2, X, Y)
        return K1 + K2, np.concatenate([G1, G2], axis=2)
    if isinstance(kernel, Product):
        K1, G1 = _pair_eval(kernel.k1, X, Y); K2, G2 = _pair_eval(kernel.k2, X, Y)
        return K1 * K2, np.concatenate([G1 * K2[..., None], K1[..., None] * G2], axis=2)
    if isinstance(kernel, Exponentiation):
        K, G = _pair_eval(kernel.kernel, X, Y)
        e = kernel.exponent
        return K ** e, e * (K ** (e - 1))[..., None] * G
    if isinstance(kernel, ConstantKernel):
        K = np.full((n, m), float(kernel.constant_value))
        return K, (np.empty((n, m, 0)) if kernel.hyperparameter_constant_value.fixed else K[..., None].copy())
    if isinstance(kernel, WhiteKernel):
        return np.zeros((n, m)), np.zeros((n, m, 0 if kernel.hyperparameter_noise_level.fixed else 1))
    nu = kernel.nu if isinstance(kernel, Matern) else (np.inf if isinstance(kernel, RBF) else None)
    if nu in (0.5, 1.5, 2.5, np.inf):
        ls = np.atleast_1d(np.asarray(kernel.length_scale, dtype=float))
        D = ((X[:, None, :] - Y[None, :, :]) / ls) ** 2       # per-dimension scaled squared differences
        d2 = D.sum(axis=2)
        if ls.size == 1:
            D = d2[..., None]
        if nu == np.inf:
            K = np.exp(-0.5 * d2); G = D * K[..., None]
        elif nu == 0.5:
            r = np.sqrt(d2); K = np.exp(-r)
            G = np.divide(D, r[..., None], out=np.zeros_like(D), where=r[..., None] != 0) * K[..., None]
        elif nu == 1.5:
            r = np.sqrt(3.0 * d2); K = (1.0 + r) * np.exp(-r); G = 3.0 * D * np.exp(-r)[..., None]
        else:
            r = np.sqrt(5.0 * d2); K = (1.0 + r + r ** 2 / 3.0) * np.exp(-r)
            G = 5.0 / 3.0 * D * ((r + 1.0) * np.exp(-r))[..., None]
        if kernel.hyperparameter_length_scale.fixed:
            G = np.empty((n, m, 0))
        return K, G
    try:
        return kernel(X, Y, eval_gradient=True)
    except ValueError:
        Kz, Gz = kernel(np.vstack([X, Y]), eval_gradient=True)
        return Kz[:n, n:], Gz[:n, n:, :]


class SymmetrizedKernel(Kernel):
    """Kernel ``k(x, y) = alpha * sum_g k0(x, g y) + (1 - alpha) * k0(x, y)`` for a
    base kernel ``k0`` and the elements ``g`` of a ``Symmetry``.

    ``alpha = 1`` gives a kernel invariant under the symmetry (every sample
    function is symmetric), ``alpha = 0`` the base kernel; in between the prior is
    the sum of a symmetric part and an unconstrained part with variance fractions
    ``alpha`` and ``1 - alpha`` (away from the symmetry's fixed points), so an
    approximate symmetry is expressed by fitting ``alpha`` with the other
    hyperparameters.

    Parameters
    ----------
    kernel : Kernel
        The base kernel.
    symmetry : Symmetry
    alpha : float, default=0.9
        Initial invariant fraction.
    alpha_bounds : pair of floats or "fixed", default=(0.0, 1.0)
        Bounds on ``alpha``. scikit-learn fits log-hyperparameters, so a lower
        bound of 0 is treated as 1e-6.
    tie_length_scales : bool, default=True
        With a permutation symmetry, share the base kernel's per-input length
        scales within each orbit of the inputs (the length scales of exchangeable
        inputs are then one hyperparameter).
    input_transform : (shift, scale) or None
        Standardization the inputs carry when the kernel sees them, as arrays of
        length d: the symmetry acts on ``shift + scale * x``. Set by ``GPNode``
        for its leaf when it standardizes the inputs.
    """

    _ALPHA_FLOOR = 1e-6

    def __init__(self, kernel, symmetry, alpha=0.9, alpha_bounds=(0.0, 1.0), tie_length_scales=True,
                 input_transform=None):
        self.kernel = kernel
        self.symmetry = symmetry
        self.alpha = alpha
        self.alpha_bounds = alpha_bounds
        self.tie_length_scales = tie_length_scales
        self.input_transform = input_transform

    # -- parameters and hyperparameters (sklearn's Kernel API) --------------------

    def get_params(self, deep=True):
        params = dict(kernel=self.kernel, symmetry=self.symmetry, alpha=self.alpha, alpha_bounds=self.alpha_bounds,
                      tie_length_scales=self.tie_length_scales, input_transform=self.input_transform)
        if deep:
            params.update(("kernel__" + k, v) for k, v in self.kernel.get_params().items())
        return params

    @property
    def hyperparameter_alpha(self):
        b = self.alpha_bounds
        if not (isinstance(b, str) and b == "fixed"):
            b = (max(float(b[0]), self._ALPHA_FLOOR), float(b[1]))
        return Hyperparameter("alpha", "numeric", b)

    def _tied_orbits(self):
        if not self.tie_length_scales:
            return None
        orbits = self.symmetry.orbits if self.symmetry.n_features is not None else None
        if orbits is None or len(orbits) == self.symmetry.n_features:
            return None
        return orbits

    def _blocks(self):
        """(hyperparameter, offset in the base theta, orbits or None) of each free base hyperparameter."""
        orbits = self._tied_orbits()
        blocks, off = [], 0
        for hp in self.kernel.hyperparameters:
            if hp.fixed:
                continue
            tied = orbits if (orbits is not None and hp.name.endswith("length_scale")
                              and hp.n_elements == self.symmetry.n_features) else None
            blocks.append((hp, off, tied)); off += hp.n_elements
        return blocks, off

    def _index_maps(self):
        """``rep``: base-theta index of each free parameter; ``expand``: free index of
        each base-theta entry (tied length scales share one free parameter)."""
        blocks, n_full = self._blocks()
        rep, expand = [], np.empty(n_full, dtype=int)
        for hp, off, tied in blocks:
            if tied is None:
                for i in range(hp.n_elements):
                    expand[off + i] = len(rep); rep.append(off + i)
            else:
                for orbit in tied:
                    for i in orbit:
                        expand[off + i] = len(rep)
                    rep.append(off + orbit[0])
        return np.asarray(rep, dtype=int), expand

    @property
    def hyperparameters(self):
        out = []
        for hp in self.kernel.hyperparameters:
            tied = None
            if not hp.fixed:
                tied = next((t for h, _, t in self._blocks()[0] if h.name == hp.name), None)
            if tied is None:
                out.append(Hyperparameter("kernel__" + hp.name, hp.value_type, hp.bounds, hp.n_elements))
            else:
                bounds = np.atleast_2d(hp.bounds)
                bounds = bounds[[o[0] for o in tied]] if bounds.shape[0] == hp.n_elements else bounds
                out.append(Hyperparameter("kernel__" + hp.name, hp.value_type, bounds, len(tied)))
        out.append(self.hyperparameter_alpha)
        return out

    def _ensure_tied(self):
        """Set tied length scales of the base kernel to their orbit representative."""
        rep, expand = self._index_maps()
        if expand.size > rep.size:
            t = np.asarray(self.kernel.theta)
            tied = t[rep][expand]
            if not np.array_equal(t, tied):
                self.kernel.theta = tied

    @property
    def theta(self):
        self._ensure_tied()
        rep, _ = self._index_maps()
        t = np.asarray(self.kernel.theta)[rep]
        if not self.hyperparameter_alpha.fixed:
            t = np.append(t, np.log(self.alpha))
        return t

    @theta.setter
    def theta(self, theta):
        rep, expand = self._index_maps()
        theta = np.asarray(theta, dtype=float)
        if rep.size:
            self.kernel.theta = theta[:rep.size][expand]
        if not self.hyperparameter_alpha.fixed:
            if theta.size != rep.size + 1:
                raise ValueError(f"theta has {theta.size} entries, expected {rep.size + 1}")
            self.alpha = float(np.exp(theta[rep.size]))

    @property
    def bounds(self):
        rep, _ = self._index_maps()
        b = np.asarray(self.kernel.bounds).reshape(-1, 2)[rep]
        if not self.hyperparameter_alpha.fixed:
            b = np.vstack([b, np.log(np.asarray(self.hyperparameter_alpha.bounds, dtype=float).reshape(1, 2))])
        return b

    def is_stationary(self):
        return False

    def __repr__(self):
        return f"SymmetrizedKernel({self.kernel!r}, alpha={self.alpha:.3g})"

    # -- evaluation -----------------------------------------------------------------

    def _images(self, Z):
        """Images of kernel-space inputs Z under the symmetry, shape (n_elements, n, d)."""
        if self.input_transform is None:
            return self.symmetry.images(Z)
        shift, scale = (np.asarray(a, dtype=float) for a in self.input_transform)
        return (self.symmetry.images(Z * scale + shift) - shift) / scale

    def __call__(self, X, Y=None, eval_gradient=False):
        self._ensure_tied()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        alpha = float(self.alpha)
        if Y is None:
            if eval_gradient:
                K0, G0 = self.kernel(X, eval_gradient=True)
            else:
                K0 = self.kernel(X)
            imgs = self._images(X)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            Y = np.atleast_2d(np.asarray(Y, dtype=float))
            K0 = self.kernel(X, Y)
            imgs = self._images(Y)
        S = np.zeros_like(K0)
        GS = np.zeros_like(G0) if eval_gradient else None
        for g in range(1, imgs.shape[0]):
            if eval_gradient:
                Kg, Gg = _pair_eval(self.kernel, X, imgs[g]); S += Kg; GS += Gg
            else:
                S += self.kernel(X, imgs[g])
        K = K0 + alpha * S
        if not eval_gradient:
            return K
        rep, expand = self._index_maps()
        M = np.zeros((expand.size, rep.size))
        M[np.arange(expand.size), expand] = 1.0
        parts = [(G0 + alpha * GS) @ M]
        if not self.hyperparameter_alpha.fixed:
            parts.append((alpha * S)[..., None])
        return K, np.concatenate(parts, axis=2)

    def diag(self, X):
        self._ensure_tied()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.array(self.kernel.diag(X), dtype=float)
        imgs = self._images(X)
        alpha = float(self.alpha)
        for g in range(1, imgs.shape[0]):
            for s in range(0, X.shape[0], 256):
                out[s:s + 256] += alpha * np.diag(self.kernel(X[s:s + 256], imgs[g][s:s + 256]))
        return out


def contains_symmetrized_kernel(kernel):
    """True if ``kernel`` is, or contains, a SymmetrizedKernel."""
    if isinstance(kernel, SymmetrizedKernel):
        return True
    try:
        return any(isinstance(v, SymmetrizedKernel) for v in kernel.get_params(deep=True).values())
    except Exception:
        return False
