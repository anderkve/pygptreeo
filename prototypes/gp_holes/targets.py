"""Target functions for the GP-hole prototype, mostly from the Feynman
symbolic regression database. Each entry: name, formula as text, function of
the input array, and the range of each input."""
import numpy as np

R15 = (1.0, 5.0)

TARGETS = [
    # Within the reach of the rewrite rules.
    ('toy', 'x2*sin(2*x1) + 0.5*x3^2',
     lambda X: X[:, 1] * np.sin(2 * X[:, 0]) + 0.5 * X[:, 2] ** 2, [(0.5, 3.0)] * 3),
    ('I.14.3', 'x1*x2*x3',
     lambda X: X[:, 0] * X[:, 1] * X[:, 2], [R15] * 3),
    ('I.12.2', 'x1*x2/(4*pi*x3^2)',
     lambda X: X[:, 0] * X[:, 1] / (4 * np.pi * X[:, 2] ** 2), [R15] * 3),
    ('I.34.8', 'x1*x2*x3/x4',
     lambda X: X[:, 0] * X[:, 1] * X[:, 2] / X[:, 3], [R15] * 4),
    ('I.13.4', '0.5*x1*(x2^2 + x3^2 + x4^2)',
     lambda X: 0.5 * X[:, 0] * (X[:, 1] ** 2 + X[:, 2] ** 2 + X[:, 3] ** 2), [R15] * 4),
    ('I.24.6', '0.25*x1*(x2^2 + x3^2)*x4^2',
     lambda X: 0.25 * X[:, 0] * (X[:, 1] ** 2 + X[:, 2] ** 2) * X[:, 3] ** 2, [R15] * 4),
    ('I.6.2', 'exp(-(x1/x2)^2/2)/(sqrt(2*pi)*x2)',
     lambda X: np.exp(-(X[:, 0] / X[:, 1]) ** 2 / 2) / (np.sqrt(2 * np.pi) * X[:, 1]), [(1.0, 3.0)] * 2),
    ('I.12.11', 'x1*(x2 + x3*x4*sin(x5))',
     lambda X: X[:, 0] * (X[:, 1] + X[:, 2] * X[:, 3] * np.sin(X[:, 4])), [R15] * 5),
    ('I.37.4', 'x1 + x2 + 2*sqrt(x1*x2)*cos(x3)',
     lambda X: X[:, 0] + X[:, 1] + 2 * np.sqrt(X[:, 0] * X[:, 1]) * np.cos(X[:, 2]), [R15] * 3),
    ('I.40.1', 'x1*exp(-x2*x3/x4)',
     lambda X: X[:, 0] * np.exp(-X[:, 1] * X[:, 2] / X[:, 3]), [R15] * 4),
    ('isothermal', 'x1*x2*log(x3/x4)',
     lambda X: X[:, 0] * X[:, 1] * np.log(X[:, 2] / X[:, 3]), [R15] * 4),
    ('I.50.26', 'x1*(cos(x2*x3) + x4*cos(x2*x3)^2)',
     lambda X: X[:, 0] * (np.cos(X[:, 1] * X[:, 2]) + X[:, 3] * np.cos(X[:, 1] * X[:, 2]) ** 2),
     [(1.0, 3.0)] * 4),
    # A part is outside the library, so a hole should remain.
    ('I.10.7', 'x1/sqrt(1 - x2^2/x3^2)',
     lambda X: X[:, 0] / np.sqrt(1 - X[:, 1] ** 2 / X[:, 2] ** 2), [R15, (1.0, 2.0), (3.0, 10.0)]),
    ('I.30.5', 'arcsin(x1/(x2*x3))',
     lambda X: np.arcsin(X[:, 0] / (X[:, 1] * X[:, 2])), [(1.0, 2.0), (2.0, 5.0), (2.0, 5.0)]),
    ('I.8.14', 'sqrt((x2 - x1)^2 + (x4 - x3)^2)',
     lambda X: np.sqrt((X[:, 1] - X[:, 0]) ** 2 + (X[:, 3] - X[:, 2]) ** 2), [R15] * 4),
    ('I.18.4', '(x1*x3 + x2*x4)/(x1 + x2)',
     lambda X: (X[:, 0] * X[:, 2] + X[:, 1] * X[:, 3]) / (X[:, 0] + X[:, 1]), [R15] * 4),
    ('II.11.28', '1 + x1*x2/(1 - x1*x2/3)',
     lambda X: 1 + X[:, 0] * X[:, 1] / (1 - X[:, 0] * X[:, 1] / 3), [(0.2, 1.0)] * 2),
]


def sample(target, n, rng, widen=0.0):
    """n inputs, uniform in the target's box. With widen > 0 the box is that
    much wider on each side (as a fraction of its width, staying positive),
    and the points lie outside the original box."""
    lo, hi = np.array(target[3]).T
    if not widen:
        return rng.uniform(lo, hi, (n, len(lo)))
    wlo = np.maximum(lo - widen * (hi - lo), 0.5 * lo)
    X = rng.uniform(wlo, hi + widen * (hi - lo), (20 * n, len(lo)))
    return X[((X < lo) | (X > hi)).any(1)][:n]
