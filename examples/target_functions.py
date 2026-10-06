"""Provides a collection of common benchmark target functions.

These functions are typically used for testing optimization and regression
algorithms. The input `x` for each function is generally expected to be
a 1-D NumPy array representing a point in a d-dimensional space, with
values in each dimension initially scaled to the `[0,1]` range. This
input is then mapped to the function's standard domain before computation.
"""
import numpy as np


def Eggholder_2d(x):
    """Computes the 2-dimensional Eggholder function.

    This is a helper function used by the N-dimensional `Eggholder` function.
    It is known for its complex landscape with many local minima.

    Args:
        x (np.ndarray): A 1-D NumPy array of length 2, representing a point
            in 2D space (e.g., `[x0, x1]`). Values are assumed to be
            already scaled to the function's standard domain (typically [-512, 512]).

    Returns:
        float: The value of the 2D Eggholder function at point `x`.
    """
    term1 = -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[1] + x[0]/2. + 47.)))
    term2 = -x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47.))))
    func = term1 + term2
    func += 959.6407
    return func

def Eggholder(x):
    """Computes the N-dimensional Eggholder function.

    The Eggholder function is a common benchmark for optimization algorithms,
    characterized by a large number of local minima, making it challenging
    to optimize. This implementation expects input `x` to be a 1-D NumPy array
    with values in the range `[0,1]` for each dimension. These values are then
    scaled to the standard Eggholder domain of `[-512, 512]` for each dimension.
    The N-dimensional function is computed by summing 2D Eggholder results
    for adjacent pairs of dimensions.

    Args:
        x (np.ndarray): A 1-D NumPy array where each element `x[i]` represents
            the value for the i-th dimension. Values are expected to be in `[0,1]`.

    Returns:
        float: The value of the N-dimensional Eggholder function at point `x`.
    """
    xmin, xmax = -512, 512
    x = xmin + x*(xmax - xmin)
    dim = len(x)
    func = 0

    for i in range(dim-1):
        func += Eggholder_2d(x[i:i+2])

    return func


def Himmelblau(x):
    """Computes the N-dimensional Himmelblau function.

    The Himmelblau function is often used as a benchmark for optimization.
    It has a relatively small number of local minima (typically 4 in its 2D form).
    This implementation expects input `x` to be a 1-D NumPy array with values
    in the range `[0,1]` for each dimension. These values are then scaled to
    the standard Himmelblau domain of `[-5, 5]`. The N-dimensional version
    is constructed by summing 2D Himmelblau-like terms for adjacent pairs of
    dimensions. The result is log-transformed and shifted to have a minimum of zero
    for known dimensionalities (2 to 6).

    Args:
        x (np.ndarray): A 1-D NumPy array where each element `x[i]` represents
            the value for the i-th dimension. Values are expected to be in `[0,1]`.

    Returns:
        float: The value of the N-dimensional Himmelblau function at point `x`.

    Raises:
        Exception: If the dimensionality is greater than 6 and the minimum
            value for that dimensionality is not predefined.
    """
    xmin, xmax = -5, 5
    x = xmin + x*(xmax - xmin)
    dim = len(x)
    func = 0

    for i in range(dim-1):
        func += (x[i]**2 + x[i+1] - 11)**2 + (x[i] + x[i+1]**2 -7)**2
    func += 1
    func = np.log(func)

    if dim == 2:
        pass
    elif dim == 3:
        func -= 0.265331837897597
    elif dim == 4:
        func -= 1.7010318616354436
    elif dim == 5:
        func -= 2.3001107745553155
    elif dim == 6:
        func -= 2.8576426513378994
    else:
        raise Exception("We don't know the minimum value for Himmelblau in this number of dimensions.")

    return func


def Rosenbrock(x):
    """Computes the N-dimensional Rosenbrock function.

    The Rosenbrock function, also known as Rosenbrock's valley or Rosenbrock's
    banana function, is a non-convex function used as a performance test problem
    for optimization algorithms. It has a global minimum inside a long, narrow,
    parabolic-shaped flat valley.
    This implementation expects input `x` to be a 1-D NumPy array with values
    in the range `[0,1]` for each dimension. These values are then scaled to
    the domain `[-5, 10]`.

    Args:
        x (np.ndarray): A 1-D NumPy array where each element `x[i]` represents
            the value for the i-th dimension. Values are expected to be in `[0,1]`.

    Returns:
        float: The value of the N-dimensional Rosenbrock function at point `x`.
    """
    xmin, xmax = -5, 10
    x = xmin + x*(xmax - xmin)
    dim = len(x)
    func = 0

    for i in range(dim-1):
        func += 100*(x[i+1]-x[i]**2)**2 + (1 - x[i])**2
    
    return func


def Rastrigin(x):
    """Computes the N-dimensional Rastrigin function.

    The Rastrigin function is a non-convex function used as a performance test
    problem for optimization algorithms. It is known for having many local
    minima, arranged in a regular grid. It has a global minimum at x_i = 0.
    This implementation expects input `x` to be a 1-D NumPy array with values
    in the range `[0,1]` for each dimension. These values are then scaled to
    the standard Rastrigin domain of `[-5.12, 5.12]`.

    Args:
        x (np.ndarray): A 1-D NumPy array where each element `x[i]` represents
            the value for the i-th dimension. Values are expected to be in `[0,1]`.

    Returns:
        float: The value of the N-dimensional Rastrigin function at point `x`.
    """
    xmin, xmax = -5.12, 5.12
    x = xmin + x*(xmax - xmin)
    dim = len(x)
    func = 0

    func = 10 * dim
    for i in range(dim):
        func += x[i]**2 - 10 * np.cos(2 * np.pi * x[i])
    return func


def Levy(x):
    """Computes the N-dimensional Levy function.

    The Levy function is a benchmark problem for global optimization. It is
    characterized by many local minima.
    This implementation expects input `x` to be a 1-D NumPy array with values
    in the range `[0,1]` for each dimension. These values are then scaled to
    the standard Levy domain of `[-10, 10]`.

    Args:
        x (np.ndarray): A 1-D NumPy array where each element `x[i]` represents
            the value for the i-th dimension. Values are expected to be in `[0,1]`.

    Returns:
        float: The value of the N-dimensional Levy function at point `x`.
    """
    xmin, xmax = -10, 10
    x = xmin + x*(xmax - xmin)
    dim = len(x)
    func = 0

    w = []
    for i in range(dim):
        w.append(1 + (x[i]-1)/4)
    term1 = (np.sin(np.pi * w[0]))**2
    term_sum = 0
    for i in range(dim-1):
        term_sum += ((w[i] - 1)**2) * (1 + 10 * (np.sin(np.pi * w[i] + 1))**2)
    
    term_end = ((w[dim-1] - 1)**2) * (1 + (np.sin(2 * np.pi * w[dim-1])**2))
    
    func = term1 + term_sum + term_end

    return func



def Custom(x):
    """Computes a custom composite function.

    This function is a weighted sum of other benchmark functions defined in
    this module: Rastrigin, Eggholder, and Levy. It serves as an example of
    how more complex target functions can be constructed. The input `x` is
    passed to each component function, which assumes `x` contains values in
    `[0,1]` for each dimension before their respective domain scaling.

    Args:
        x (np.ndarray): A 1-D NumPy array where each element `x[i]` represents
            the value for the i-th dimension. Values are expected to be in `[0,1]`.

    Returns:
        float: The value of the custom composite function at point `x`.
    """
    func = 0
    func += Rastrigin(x)
    func += Eggholder(x) / 6.
    func += Levy(x)
    return func

# --------------------------------------------------------------------------- #
# Non-additive targets
#
# Every N-dimensional function above is a sum of terms that each involve only
# one or two adjacent coordinates, so all of them have an exact low-order
# additive decomposition in the given coordinates. The two targets below do
# not: a fixed random rotation couples every input dimension to every other,
# so no low-order additive model (additive kernels, additive global trends)
# can represent them exactly. Both are deterministic for a given (d, seed).
# --------------------------------------------------------------------------- #
_ROTATION_CACHE = {}


def _random_rotation(dim, seed):
    """A fixed random orthogonal matrix for (dim, seed), cached."""
    key = (int(dim), int(seed))
    if key not in _ROTATION_CACHE:
        rng = np.random.RandomState(seed)
        Q, _ = np.linalg.qr(rng.randn(dim, dim))
        _ROTATION_CACHE[key] = Q
    return _ROTATION_CACHE[key]


def RotatedRosenbrock(x, seed=13):
    """Computes the Rosenbrock function in randomly rotated coordinates.

    The unit-cube coordinates are rotated about the cube centre by a fixed
    random orthogonal matrix before the standard :func:`Rosenbrock` mapping
    and evaluation. In the given coordinates the quartic valley terms then
    involve products of up to four different inputs, so the function has no
    exact order-1 or order-2 additive decomposition (unlike the unrotated
    Rosenbrock, which is a sum of terms in adjacent coordinate pairs). It
    stays smooth and polynomial, so it is learnable, and it keeps a large-scale
    quadratic trend that a low-order model can partly capture.

    Args:
        x (np.ndarray): Array of shape (d,) or (d, N) with values in `[0,1]`.
        seed (int): Seed of the fixed rotation (one rotation per (d, seed)).

    Returns:
        float or np.ndarray: Function value(s).
    """
    x = np.asarray(x, dtype=float)
    R = _random_rotation(x.shape[0], seed)
    z = 0.5 + R @ (x - 0.5)
    return Rosenbrock(z)


def GaussianPeaks(x, n_peaks=3, seed=7):
    """Negative log of a mixture of anisotropic, mutually rotated Gaussian peaks.

    A smooth, multi-modal, likelihood-like landscape in `[0,1]^d`::

        f(x) = -log( sum_k w_k exp(-0.5 (x - mu_k)^T Sigma_k^-1 (x - mu_k)) )

    with peak centres `mu_k` in `[0.2, 0.8]^d`, full covariance matrices
    `Sigma_k = Q_k diag(s_k^2) Q_k^T` built from random rotations `Q_k` and
    principal widths `s_k` in `[0.05, 0.25]`, and weights `w_k` in `[0.3, 1]`.
    The rotated covariances couple all input dimensions, and the log-sum-exp
    between peaks adds higher-order structure in the transition regions; near
    a single peak the function is an (order-2) quadratic form. The minimum is
    close to the centre of the heaviest, narrowest peak.

    Args:
        x (np.ndarray): Array of shape (d,) or (d, N) with values in `[0,1]`.
        n_peaks (int): Number of peaks.
        seed (int): Seed of the fixed peak parameters (per (d, n_peaks, seed)).

    Returns:
        float or np.ndarray: Function value(s), non-negative up to a constant.
    """
    x = np.asarray(x, dtype=float)
    dim = x.shape[0]
    key = ('peaks', int(dim), int(n_peaks), int(seed))
    if key not in _ROTATION_CACHE:
        rng = np.random.RandomState(seed)
        peaks = []
        for _ in range(n_peaks):
            mu = rng.uniform(0.2, 0.8, dim)
            Q, _ = np.linalg.qr(rng.randn(dim, dim))
            s = rng.uniform(0.05, 0.25, dim)
            prec = Q @ np.diag(1.0 / s ** 2) @ Q.T           # Sigma^-1
            w = rng.uniform(0.3, 1.0)
            peaks.append((mu, prec, w))
        _ROTATION_CACHE[key] = peaks
    peaks = _ROTATION_CACHE[key]

    X = x.reshape(dim, -1)                                   # (d, N)
    log_terms = []
    for mu, prec, w in peaks:
        dxv = X - mu[:, None]
        m = np.einsum('in,ij,jn->n', dxv, prec, dxv)          # Mahalanobis^2 per point
        log_terms.append(np.log(w) - 0.5 * m)
    log_terms = np.array(log_terms)                           # (K, N)
    mx = log_terms.max(axis=0)
    f = -(mx + np.log(np.exp(log_terms - mx).sum(axis=0)))    # -log-sum-exp, stable
    return f[0] if x.ndim == 1 else f


# --------------------------------------------------------------------------- #
# Harder targets for the hybrid-vs-tree comparison: a low effective dimension
# hidden in many inputs, steep narrow valleys, ripples on a bowl, an all-order
# interaction, a discontinuity, and a length scale that varies across the box.
# --------------------------------------------------------------------------- #
def ActiveSubspacePeaks(x, seed=21):
    """The 2D :func:`GaussianPeaks` landscape of a fixed random 2-plane of the inputs.

    ``z = 0.5 + 0.55 * P (x - 0.5)`` with ``P`` two orthonormal rows, clipped to
    the unit square, then ``GaussianPeaks(z)``. Every input matters but only
    two directions do, so a model that can learn the projection (a feature
    network) has a 2D problem and one that cannot has a ``d``-dimensional one.
    """
    x = np.asarray(x, dtype=float)
    dim = x.shape[0]
    key = ('plane', int(dim), int(seed))
    if key not in _ROTATION_CACHE:
        rng = np.random.RandomState(seed)
        Q, _ = np.linalg.qr(rng.randn(dim, 2))
        _ROTATION_CACHE[key] = Q.T                            # (2, d), orthonormal rows
    P = _ROTATION_CACHE[key]
    z = np.clip(0.5 + 0.55 * (P @ (x.reshape(dim, -1) - 0.5)), 0, 1)
    out = GaussianPeaks(z)
    return out if x.ndim > 1 else float(np.squeeze(out))


def Michalewicz(x, m=10):
    """The Michalewicz function on ``[0, pi]^d``: steep, narrow valleys in a flat plain.

    ``f = 4 - sum_i sin(x_i) sin(i x_i^2 / pi)^(2m)``; the exponent ``2m = 20``
    makes each valley a few percent of the box wide, with the valleys of the
    higher coordinates narrower and more numerous. The offset of 4 keeps the
    function positive, so relative errors stay meaningful on the plain.
    """
    x = np.asarray(x, dtype=float) * np.pi
    i = np.arange(1, x.shape[0] + 1).reshape(-1, *([1] * (x.ndim - 1)))
    return 4.0 - np.sum(np.sin(x) * np.sin(i * x ** 2 / np.pi) ** (2 * m), axis=0)


def Ackley(x):
    """The Ackley function on ``[-2, 2]^d``: a cosine ripple of unit period on an exponential bowl."""
    x = np.asarray(x, dtype=float) * 4.0 - 2.0
    d = x.shape[0]
    return (-20.0 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2, axis=0) / d))
            - np.exp(np.sum(np.cos(2 * np.pi * x), axis=0) / d) + 20.0 + np.e)


def Griewank(x):
    """The Griewank function on ``[-10, 10]^d``: a product of cosines (an interaction of every order) on a shallow bowl."""
    x = np.asarray(x, dtype=float) * 20.0 - 10.0
    i = np.arange(1, x.shape[0] + 1).reshape(-1, *([1] * (x.ndim - 1)))
    return np.sum(x ** 2, axis=0) / 4000.0 - np.prod(np.cos(x / np.sqrt(i)), axis=0) + 1.0


def StepRidge(x, seed=5):
    """A rotated quadratic bowl with a jump across a rotated hyperplane.

    ``f = 4 |R (x - 0.5)|^2 + 1.5 * [w . (x - 0.5) > 0]`` with ``R`` a fixed
    random rotation and ``w`` a fixed random unit vector; the jump is about
    the size of the bowl's spread, and no smooth model represents it.
    """
    x = np.asarray(x, dtype=float)
    dim = x.shape[0]
    key = ('step', int(dim), int(seed))
    if key not in _ROTATION_CACHE:
        rng = np.random.RandomState(seed)
        Q, _ = np.linalg.qr(rng.randn(dim, dim))
        w = rng.randn(dim); w /= np.linalg.norm(w)
        _ROTATION_CACHE[key] = (Q, w)
    Q, w = _ROTATION_CACHE[key]
    X = x.reshape(dim, -1) - 0.5
    z = Q @ X
    out = 4.0 * np.sum(z ** 2, axis=0) + 1.5 * (w @ X > 0)
    return out if x.ndim > 1 else float(np.squeeze(out))


def Chirp(x):
    """A sum of sines whose frequency grows with the first coordinate: a length scale that varies across the box.

    ``f = 5 + sum_i sin(2 pi (1 + 5 x_1) x_i)``; near ``x_1 = 0`` one period
    per unit in every direction, near ``x_1 = 1`` six. The offset keeps the
    function positive.
    """
    x = np.asarray(x, dtype=float)
    freq = 1.0 + 5.0 * x[0]
    return 5.0 + np.sum(np.sin(2 * np.pi * freq * x), axis=0)
