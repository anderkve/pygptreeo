"""Target functions for the GP-hole prototype. Each entry: name, formula as
text, function of the input array, and the range of each input."""
import numpy as np
from scipy.special import ellipk, erf, j0

R15 = (1.0, 5.0)

# Mostly from the Feynman symbolic regression database.
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

# Targets without a closed form. The further entries give the series: the
# input it is in, the powers, their coefficients and the factor in front.
SQPI = np.sqrt(np.pi)
SERIES = [
    ('erf', 'erf(x1) = 1.128*x1 - 0.376*x1^3 + 0.113*x1^5 - 0.0269*x1^7 + ...',
     lambda X: erf(X[:, 0]), [(0.1, 1.5)],
     0, [1, 3, 5, 7, 9], [2 / SQPI, -2 / (3 * SQPI), 2 / (10 * SQPI), -2 / (42 * SQPI), 2 / (216 * SQPI)],
     lambda X: 1.0),
    ('bessel', 'J0(x1) = 1 - x1^2/4 + x1^4/64 - x1^6/2304 + ...',
     lambda X: j0(X[:, 0]), [(0.1, 2.5)],
     0, [0, 2, 4, 6, 8], [1, -1 / 4, 1 / 64, -1 / 2304, 1 / 147456],
     lambda X: 1.0),
    ('pendulum', '4*sqrt(x1/x2)*K(sin(x3/2)) = 2*pi*sqrt(x1/x2)*(1 + x3^2/16 + 11*x3^4/3072 + ...)',
     lambda X: 4 * np.sqrt(X[:, 0] / X[:, 1]) * ellipk(np.sin(X[:, 2] / 2) ** 2),
     [R15, R15, (0.1, 2.0)],
     2, [0, 2, 4, 6, 8], [2 * np.pi * c for c in (1, 1 / 16, 11 / 3072, 173 / 737280, 22931 / 1321205760)],
     lambda X: np.sqrt(X[:, 0] / X[:, 1])),
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
