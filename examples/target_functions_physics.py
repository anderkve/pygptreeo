"""Test targets with the structure of quantities emulated in particle physics.

Each target is cheap (closed form, a small quadrature or a 4 x 4 eigenproblem) but built
to have the features that make the real computation hard to emulate: values over many
orders of magnitude, kinematic thresholds, resonances on oblique lines of the parameter
space, interference zeros, oscillations whose frequency depends on a parameter, ratios
that are invariant along a direction, Monte Carlo noise, and a likelihood with nuisance
parameters. The formulas are schematic: right in structure, not in normalisation.

Every target takes inputs in the unit cube, one row per point, and maps them to physical
ranges (masses and energies logarithmically). ``TARGETS[name]`` is a :class:`Target`;
``Target.sample(X, rng)`` returns ``(y_observed, sigma_declared, y_true)``. For a target
without Monte Carlo noise the observation is exact and the declared sigma is the
numerical precision a user would quote: 0.1% of the quantity itself for the targets
given as a logarithm, a fixed small number in the target's unit otherwise.
"""

import numpy as np
from scipy.special import roots_genlaguerre

SQRT_S = 13000.0                         # collider energy, GeV
M_W, M_Z, M_TOP, M_B, M_TAU = 80.4, 91.19, 173.0, 4.2, 1.78
SIN2_W = 0.231


def _lin(u, lo, hi):
    return lo + u * (hi - lo)


def _log(u, lo, hi):
    return lo * (hi / lo) ** u


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _parton_luminosity(m):
    """Steeply falling partonic flux at invariant mass ``m`` (arbitrary units)."""
    x = np.clip(m / SQRT_S, 1e-6, 0.999)
    return x ** -5.5 * (1.0 - x) ** 12


# --------------------------------------------------------------------------- #
# 1. Resonant production cross-section
# --------------------------------------------------------------------------- #
def resonance_xsec(X):
    """log10 of sigma(pp -> mediator -> invisible) in a simplified dark-matter model (4D).

    Inputs: mediator mass (0.2 to 5 TeV), dark-matter mass (10 GeV to 2.5 TeV), quark
    coupling, dark-matter coupling. ``sigma = g_q^2 L(M) BR(M, m, g_q, g_chi)`` plus a small
    off-shell piece. Sixteen orders of magnitude between the 1st and the 99th percentile,
    from the parton flux and from a square-root threshold on the oblique line ``M = 2 m``,
    and a branching ratio that is a ratio of couplings.
    """
    M = _log(X[:, 0], 200.0, 5000.0); m = _log(X[:, 1], 10.0, 2500.0)
    gq = _lin(X[:, 2], 0.05, 1.0); gx = _lin(X[:, 3], 0.1, 1.5)
    beta2 = np.clip(1.0 - 4 * m ** 2 / M ** 2, 0.0, None)
    width_q = 18 * gq ** 2 * M / (12 * np.pi)
    width_x = gx ** 2 * M / (12 * np.pi) * (1 + 2 * m ** 2 / M ** 2) * np.sqrt(beta2)
    on_shell = gq ** 2 * _parton_luminosity(M) * width_x / (width_q + width_x)
    off_shell = 1e-3 * gq ** 2 * gx ** 2 * _parton_luminosity(np.maximum(M, 2 * m)) * (M ** 2 / (M ** 2 + 4 * m ** 2)) ** 2
    return np.log10(on_shell + off_shell)


# --------------------------------------------------------------------------- #
# 2. Differential cross-section with a resonance and interference
# --------------------------------------------------------------------------- #
def dilepton_spectrum(X):
    """log10 of d sigma / d m_ll with a Z' interfering with the continuum (4D).

    Inputs: dilepton mass (0.5 to 5 TeV), Z' mass (1 to 4.5 TeV), Z' coupling, extra
    (invisible) width. A peak of 0.5 to 12% relative width on the diagonal ``m_ll = M``,
    a destructive dip below it, on a continuum that falls over nine orders of magnitude.
    """
    mll = _log(X[:, 0], 500.0, 5000.0); M = _lin(X[:, 1], 1000.0, 4500.0)
    g = _lin(X[:, 2], 0.1, 1.0); extra = _lin(X[:, 3], 0.0, 3.0)
    width = M * g ** 2 * (1.0 + extra) * 0.03 + 0.005 * M
    s = mll ** 2
    amplitude = 1.0 + (g ** 2 / 0.1) * s / (s - M ** 2 + 1j * M * width)
    return np.log10(_parton_luminosity(mll) / s * np.abs(amplitude) ** 2)


# --------------------------------------------------------------------------- #
# 3. Inclusive jet spectrum
# --------------------------------------------------------------------------- #
def jet_spectrum(X):
    """log10 of d sigma / d p_T d y for inclusive jets (4D).

    Inputs: p_T (50 GeV to 2 TeV), rapidity as a fraction of its kinematic limit,
    alpha_s(M_Z), renormalisation scale over p_T. A power law times a kinematic
    suppression that couples p_T and rapidity, a running coupling and a scale-dependent
    K-factor. Close to a power law over most of the cube, with a steep fall towards the
    kinematic edge at large rapidity and a kink where that edge starts to cut the
    rapidity range (p_T near 550 GeV); fourteen orders of magnitude between the 1st and
    the 99th percentile.
    """
    pt = _log(X[:, 0], 50.0, 2000.0); xt = 2 * pt / SQRT_S
    y = X[:, 1] * np.minimum(3.0, 0.95 * np.arccosh(1.0 / xt))
    alpha_mz = _lin(X[:, 2], 0.110, 0.125); xi = _log(X[:, 3], 0.5, 2.0)
    b0 = 23.0 / (12 * np.pi)
    alpha = alpha_mz / (1.0 + b0 * alpha_mz * np.log((xi * pt) ** 2 / M_Z ** 2))
    k_factor = 1.0 + alpha * (3.0 - 4.0 * b0 * np.pi * np.log(xi)) + 8.0 * alpha ** 2 * np.log(xi) ** 2
    return np.log10(alpha ** 2 * pt ** -4.5 * (1.0 - xt * np.cosh(y)) ** 9 / (1.0 + xt) ** 2 * k_factor) + 12.0


# --------------------------------------------------------------------------- #
# 4. Neutrino oscillation probability
# --------------------------------------------------------------------------- #
def oscillation_probability(X):
    """P(nu_mu -> nu_e) at 1300 km in constant-density matter (5D).

    Inputs: energy (0.3 to 6 GeV), theta_23, theta_13, delta_CP (0 to 2 pi),
    Delta m^2_31. The expansion to second order in ``alpha = dm21 / dm31``. Oscillates in
    ``1 / E`` with a phase proportional to ``dm31 / E``, is periodic in delta_CP, and has
    amplitudes that are products of functions of single angles.
    """
    E = _log(X[:, 0], 0.3, 6.0); th23 = _lin(X[:, 1], 0.67, 0.90); th13 = _lin(X[:, 2], 0.13, 0.17)
    delta = _lin(X[:, 3], 0.0, 2 * np.pi); dm31 = _lin(X[:, 4], 2.3e-3, 2.7e-3)
    th12, dm21, L, rho = 0.584, 7.4e-5, 1300.0, 2.85
    D = 1.267 * dm31 * L / E; alpha = dm21 / dm31; A = 7.6e-5 * rho * E / dm31
    f1 = np.sin((1 - A) * D) / (1 - A); f2 = np.sin(A * D) / A
    return (np.sin(th23) ** 2 * np.sin(2 * th13) ** 2 * f1 ** 2
            + alpha * np.sin(2 * th13) * np.sin(2 * th12) * np.sin(2 * th23) * np.cos(th13) * f1 * f2 * np.cos(D + delta)
            + alpha ** 2 * np.cos(th23) ** 2 * np.sin(2 * th12) ** 2 * f2 ** 2)


# --------------------------------------------------------------------------- #
# 5. Branching ratio with thresholds
# --------------------------------------------------------------------------- #
def _partial_widths(mS, kf, kV, gx, mx):
    def fermion(mf, colour, yukawa):
        beta2 = np.clip(1.0 - 4 * mf ** 2 / mS ** 2, 0.0, None)
        return colour * (kf * yukawa) ** 2 * mS / (8 * np.pi) * beta2 ** 1.5
    x = 4 * M_W ** 2 / mS ** 2
    on = np.sqrt(np.clip(1.0 - x, 0.0, None)) * (1.0 - x + 0.75 * x ** 2)
    below = 3e-3 * np.exp(-np.clip(2 * M_W - mS, 0.0, None) / 12.0)          # off-shell tail below the WW threshold
    vector = kV ** 2 * mS ** 3 / (64 * np.pi * 246.0 ** 2) * 3 * np.where(x < 1.0, on + below, below)
    gluon = (kf * 0.12) ** 2 * mS ** 3 / (72 * np.pi ** 3 * 246.0 ** 2)
    invisible = gx ** 2 * mS / (8 * np.pi) * np.clip(1.0 - 4 * mx ** 2 / mS ** 2, 0.0, None) ** 1.5
    return dict(bb=fermion(M_B, 3, M_B / 246.0), tautau=fermion(M_TAU, 1, M_TAU / 246.0), tt=fermion(M_TOP, 3, M_TOP / 246.0),
                VV=vector, gg=gluon, invisible=invisible)


def branching_ratio(X):
    """Invisible branching ratio of a Higgs-like scalar (5D).

    Inputs: scalar mass (20 GeV to 1 TeV), fermion coupling scale, vector coupling
    scale, coupling to a dark fermion, dark fermion mass (5 to 400 GeV). A ratio of
    partial widths: between 0 and 1, unchanged when all couplings are scaled together,
    with thresholds at ``m_S = 2 m`` for the W, the top and (on a diagonal) the dark fermion.
    """
    mS = _log(X[:, 0], 20.0, 1000.0); kf = _lin(X[:, 1], 0.1, 2.0); kV = _lin(X[:, 2], 0.1, 2.0)
    gx = _lin(X[:, 3], 0.01, 0.3); mx = _log(X[:, 4], 5.0, 400.0)
    w = _partial_widths(mS, kf, kV, gx, mx)
    return w["invisible"] / sum(w.values())


# --------------------------------------------------------------------------- #
# 6. Loop-induced decay width
# --------------------------------------------------------------------------- #
def _f_loop(tau):
    tau = np.asarray(tau, dtype=complex)
    below = np.arcsin(np.sqrt(np.clip(tau.real, 0.0, 1.0))) ** 2
    root = np.sqrt(np.clip(1.0 - 1.0 / np.maximum(tau.real, 1.0 + 1e-12), 0.0, None))
    above = -0.25 * (np.log((1 + root) / np.maximum(1 - root, 1e-300)) - 1j * np.pi) ** 2
    return np.where(tau.real <= 1.0, below, above)


def loop_decay_width(X):
    """log10 of a diphoton width through fermion, vector and new-particle loops (4D).

    Inputs: scalar mass (50 GeV to 1 TeV), top coupling (-2 to 2), vector coupling
    (0 to 2), mass of a new charged fermion in the loop (100 to 600 GeV).
    ``|c_V A_1 + c_t N_c Q^2 A_1/2 + A_1/2(new)|^2 m^3``: cusps where the scalar crosses a
    pair threshold (one of them on a diagonal), and valleys where the loops cancel.
    """
    mS = _log(X[:, 0], 50.0, 1000.0); ct = _lin(X[:, 1], -2.0, 2.0); cV = _lin(X[:, 2], 0.0, 2.0)
    mF = _log(X[:, 3], 100.0, 600.0)

    def a_half(tau):
        return 2 * (tau + (tau - 1) * _f_loop(tau)) / tau ** 2

    def a_one(tau):
        return -(2 * tau ** 2 + 3 * tau + 3 * (2 * tau - 1) * _f_loop(tau)) / tau ** 2

    amplitude = (cV * a_one(mS ** 2 / (4 * M_W ** 2)) + ct * 3 * (2 / 3) ** 2 * a_half(mS ** 2 / (4 * M_TOP ** 2))
                 + a_half(mS ** 2 / (4 * mF ** 2)))
    return np.log10(1e-3 + np.abs(amplitude) ** 2 * (mS / 125.0) ** 3 / 42.0)


# --------------------------------------------------------------------------- #
# 7. Relic density with a resonance funnel
# --------------------------------------------------------------------------- #
_EPS, _EPS_W = roots_genlaguerre(24, 0.5)        # thermal average over the kinetic energy


def relic_density(X):
    """log10 of the dark-matter relic density (4D).

    Inputs: dark-matter mass (10 GeV to 1 TeV), mediator mass (20 GeV to 2 TeV),
    coupling (0.05 to 2), mediator width over mass (0.005 to 0.1). The thermally
    averaged annihilation through an s-channel mediator plus annihilation into mediator
    pairs: a funnel several orders of magnitude deep along ``m_A = 2 m``, asymmetric from
    the thermal average, a threshold at ``m = m_A``, and a coupling that enters as a power.
    """
    m = _log(X[:, 0], 10.0, 1000.0); mA = _log(X[:, 1], 20.0, 2000.0)
    g = _log(X[:, 2], 0.05, 2.0); width = _log(X[:, 3], 0.005, 0.1) * mA
    x_f = 22.0
    s = 4 * m[:, None] ** 2 * (1.0 + _EPS[None, :] / x_f)
    resonant = g[:, None] ** 4 / (2 * np.pi) * s / ((s - mA[:, None] ** 2) ** 2 + (mA * width)[:, None] ** 2)
    sigma_v = (resonant * _EPS_W[None, :]).sum(axis=1) / _EPS_W.sum()
    sigma_v = sigma_v + g ** 4 / (64 * np.pi * m ** 2) * np.clip(1.0 - mA ** 2 / m ** 2, 0.0, None) ** 1.5
    return np.log10(0.12 * 2.2e-26 / (sigma_v * 1.17e-17))


# --------------------------------------------------------------------------- #
# 8. Mass spectrum from a mixing matrix
# --------------------------------------------------------------------------- #
def neutralino_mass(X):
    """Mass of the lightest neutralino in GeV (4D).

    Inputs: M_1 (50 GeV to 1 TeV), M_2 (100 GeV to 1 TeV), mu (100 GeV to 1 TeV),
    tan beta (2 to 50). The smallest absolute eigenvalue of the 4 x 4 mass matrix:
    close to the smallest of the three mass parameters, with kinks where two of them
    cross (rounded by the mixing) and no dependence on the others away from the crossings.
    """
    M1 = _log(X[:, 0], 50.0, 1000.0); M2 = _log(X[:, 1], 100.0, 1000.0); mu = _log(X[:, 2], 100.0, 1000.0)
    beta = np.arctan(_lin(X[:, 3], 2.0, 50.0)); sw, cw = np.sqrt(SIN2_W), np.sqrt(1 - SIN2_W)
    cb, sb = np.cos(beta), np.sin(beta); zero = np.zeros_like(M1)
    a, b, c, d = -M_Z * sw * cb, M_Z * sw * sb, M_Z * cw * cb, -M_Z * cw * sb
    matrix = np.array([[M1, zero, a, b], [zero, M2, c, d], [a, c, zero, -mu], [b, d, -mu, zero]]).transpose(2, 0, 1)
    return np.abs(np.linalg.eigvalsh(matrix)).min(axis=1)


# --------------------------------------------------------------------------- #
# 9. EFT signal strength
# --------------------------------------------------------------------------- #
def _eft_coefficients():
    rng = np.random.RandomState(42)
    out = []
    for _ in range(4):                               # production, and three partial widths
        lin = rng.uniform(-0.5, 0.5, 6); M = rng.uniform(-0.25, 0.25, (6, 3))
        out.append((lin, 0.25 * np.outer(lin, lin) + M @ M.T))
    return out


_EFT = _eft_coefficients()


def eft_signal_strength(X):
    """Signal strength ``sigma BR / (sigma BR)_SM`` in an effective field theory (6D).

    Inputs: six Wilson coefficients in [-1, 1]. Production and three partial widths are
    quadratic polynomials in the coefficients (a squared amplitude each); the branching
    ratio divides one width by the sum. Smooth, with every pair of inputs coupled.
    """
    c = 2 * X - 1
    quad = [1.0 + c @ lin + np.einsum("ni,ij,nj->n", c, Q, c) for lin, Q in _EFT]
    widths = np.array([0.58, 0.22, 0.20])[:, None] * np.array(quad[1:])
    return quad[0] * (widths[1] / widths.sum(axis=0)) / 0.22


# --------------------------------------------------------------------------- #
# 10. Signal yield from Monte Carlo
# --------------------------------------------------------------------------- #
N_GENERATED = 10000


def _yield_and_efficiency(X):
    m1 = _lin(X[:, 0], 300.0, 1500.0); m2 = m1 * _lin(X[:, 1], 0.0, 0.97)
    b = _lin(X[:, 2], 0.05, 1.0); theta = _lin(X[:, 3], -2.0, 2.0)
    dm = m1 - m2
    turn_on = _sigmoid((dm - 150.0 * (1 + 0.1 * theta)) / 45.0) * _sigmoid((m1 - 350.0) / 100.0)
    eff_a = 0.35 * turn_on * (0.3 + 0.7 * np.exp(-(dm / m1 - 0.65) ** 2 / 0.1))
    eff_b = 0.12 * _sigmoid((dm - 90.0 * (1 + 0.1 * theta)) / 30.0) * np.exp(-m1 / 900.0)
    efficiency = np.clip(b ** 2 * eff_a + 2 * b * (1 - b) * eff_b, 1e-7, 1.0)
    cross_section = 0.5 * (m1 / 500.0) ** -7.6 * 1e3                 # fb
    return 140.0 * cross_section, efficiency


def signal_yield(X):
    """log10 of the expected signal events in a search region, without noise (4D).

    Inputs: parent mass (300 GeV to 1.5 TeV), daughter over parent mass (0 to 0.97),
    branching ratio into the visible channel, a jet-energy-scale nuisance (-2 to 2 sigma).
    A cross-section that falls as a power, times an efficiency that turns on in the mass
    splitting (a diagonal of the mass plane) and is a polynomial in the branching ratio.
    """
    scale, efficiency = _yield_and_efficiency(X)
    return np.log10(scale * efficiency)


def signal_yield_mc(X, rng):
    """The yield of :func:`signal_yield` estimated from ``N_GENERATED`` simulated events
    per point: ``(log10 estimate, declared sigma, true value)``. The relative noise is
    below a percent where the efficiency is high and of order one where a few events pass."""
    scale, efficiency = _yield_and_efficiency(X)
    passed = rng.binomial(N_GENERATED, efficiency)
    estimate = (passed + 0.5) / (N_GENERATED + 1.0)
    sigma = np.sqrt((1.0 - estimate) / (passed + 0.5)) / np.log(10.0)
    return np.log10(scale * estimate), sigma, np.log10(scale * efficiency)


# --------------------------------------------------------------------------- #
# 11. Global-fit likelihood
# --------------------------------------------------------------------------- #
def global_fit_loglike(X):
    """Log-likelihood of a toy global fit, saturating at -40 (10D).

    Inputs: dark-matter mass (100 GeV to 2 TeV), two couplings, and seven nuisance
    parameters (top mass, alpha_s, luminosity, two backgrounds, signal efficiency, local
    dark-matter density). Poisson terms for three search regions, a relic-density
    upper limit (a curved boundary in mass and coupling), a one-sided direct-detection
    limit, a W-mass-like precision observable, and Gaussian constraints on the nuisances,
    which also enter the predictions.
    """
    m = _log(X[:, 0], 100.0, 2000.0); g = _lin(X[:, 1], 0.1, 2.0); lam = _lin(X[:, 2], 0.0, 1.0)
    mt = _lin(X[:, 3], 170.6, 174.8); alpha_s = _lin(X[:, 4], 0.115, 0.121)
    lumi, bkg1, bkg2, eff, = (_lin(X[:, k], -3.0, 3.0) for k in (5, 6, 7, 8))
    rho = _lin(X[:, 9], 0.2, 0.6)
    logl = np.zeros(len(X))
    # three search regions: observed counts equal to the nominal background
    sigma = 4.0 * g ** 4 / (g ** 2 + lam ** 2 + 0.1) * (m / 500.0) ** -6 * (alpha_s / 0.118) ** 2
    for b0, db, theta, m0, w in ((120.0, 0.08, bkg1, 250.0, 80.0), (35.0, 0.12, bkg2, 600.0, 150.0), (6.0, 0.20, bkg2, 1100.0, 200.0)):
        s = 140.0 * sigma * 0.3 * _sigmoid((m - m0) / w) * (1 + 0.1 * eff) * (1 + 0.02 * lumi)
        b = b0 * (1 + db * theta) * (1 + 0.02 * lumi)
        logl += b0 * np.log((b + s) / b0) - (b + s - b0)
    # relic density as an upper limit, 0.3 dex theory error
    omega = 0.12 * (m / 700.0) ** 2 / (6 * g ** 4 + 4 * lam ** 4 + 1e-3)
    logl += -0.5 * (np.clip(np.log10(omega / 0.12), 0.0, None) / 0.3) ** 2
    # direct detection: rate over the limit
    rate = rho / 0.4 * (lam / 0.7) ** 2 * (500.0 / m)
    logl += -0.5 * (np.clip(rate - 1.0, 0.0, None) / 0.5) ** 2
    # precision observable
    mw = 80.360 + 0.006 * (mt - 173.0) - 0.7 * (alpha_s - 0.118) + 0.02 * lam ** 2 * (500.0 / m) ** 2
    logl += -0.5 * ((mw - 80.377) / 0.012) ** 2
    # nuisance constraints
    logl += -0.5 * (((mt - 172.7) / 0.7) ** 2 + ((alpha_s - 0.118) / 0.001) ** 2 + lumi ** 2 + bkg1 ** 2 + bkg2 ** 2 + eff ** 2
                    + (np.log(rho / 0.4) / 0.3) ** 2)
    return 40.0 * np.tanh(logl / 40.0)


# --------------------------------------------------------------------------- #
PRECISION = {"dex": 0.001 / np.log(10.0), "probability": 1e-4, "branching ratio": 1e-4, "GeV": 0.01, "signal strength": 1e-3,
             "log-likelihood": 1e-3}


class Target:
    def __init__(self, func, d, unit, noisy=None):
        self.func, self.d, self.unit, self.noisy = func, d, unit, noisy

    def __call__(self, X):
        return self.func(np.atleast_2d(X))

    def sample(self, X, rng):
        """``(y_observed, sigma_declared, y_true)``."""
        X = np.atleast_2d(X)
        if self.noisy is not None:
            return self.noisy(X, rng)
        y = self.func(X)
        return y, np.full(len(y), PRECISION[self.unit]), y


TARGETS = {
    "resonance_xsec": Target(resonance_xsec, 4, "dex"),
    "dilepton_spectrum": Target(dilepton_spectrum, 4, "dex"),
    "jet_spectrum": Target(jet_spectrum, 4, "dex"),
    "oscillation_probability": Target(oscillation_probability, 5, "probability"),
    "branching_ratio": Target(branching_ratio, 5, "branching ratio"),
    "loop_decay_width": Target(loop_decay_width, 4, "dex"),
    "relic_density": Target(relic_density, 4, "dex"),
    "neutralino_mass": Target(neutralino_mass, 4, "GeV"),
    "eft_signal_strength": Target(eft_signal_strength, 6, "signal strength"),
    "signal_yield_mc": Target(signal_yield, 4, "dex", noisy=signal_yield_mc),
    "global_fit_loglike": Target(global_fit_loglike, 10, "log-likelihood"),
}


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    for name, target in TARGETS.items():
        X = rng.rand(20000, target.d)
        y, sigma, truth = target.sample(X, rng)
        q = np.percentile(truth, [0, 1, 50, 99, 100])
        print(f"{name:24s} d = {target.d:2d}  [{target.unit}]  min {q[0]:9.3g}  1% {q[1]:9.3g}  median {q[2]:9.3g}  99% {q[3]:9.3g}  max {q[4]:9.3g}"
              f"  sd {truth.std():8.3g}" + (f"  noise sd {np.std(y - truth):.3g} (declared {np.sqrt(np.mean(sigma ** 2)):.3g})" if target.noisy else ""))
