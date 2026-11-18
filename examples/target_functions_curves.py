"""Families of curves f(x; theta) with the structure of functions emulated in physics.

Each family is a function of a physics quantity x (an energy, a mass, a wavenumber, a
time) that depends on a few parameters theta. A family takes the parameters in the unit
cube, one row per point, maps them to physical ranges and returns the curve on a fixed
grid in x, one row per parameter point: the multi-output target of a ``GPTree`` with
``n_outputs = len(family.x)``. The formulas are schematic: right in structure, not in
normalisation.

The families differ in what makes a curve hard to learn as a vector of outputs:

    smooth        the curve changes shape slowly and by a few modes
    range         values over many orders of magnitude (given as log10)
    transport     a peak, an edge or a cutoff whose position in x depends on theta
    oscillation   a phase or a frequency in x that depends on theta
    noise         Monte Carlo noise that differs from bin to bin

``FAMILIES[name]`` is a :class:`CurveFamily`; ``family.sample(U, rng)`` returns
``(Y_observed, sigma_declared, Y_true)``, each of shape ``(len(U), len(family.x))``.
"""

import numpy as np
from scipy.special import erf

from target_functions_physics import _lin, _log, _parton_luminosity, M_Z, SQRT_S


def _col(u, lo, hi, log=False):
    """A parameter column mapped to its range, as a column that broadcasts against x."""
    return (_log(u, lo, hi) if log else _lin(u, lo, hi))[:, None]


# --------------------------------------------------------------------------- #
# smooth
# --------------------------------------------------------------------------- #
def matter_power(U, k):
    """log10 of a matter power spectrum P(k) (4D).

    Parameters: shape parameter Gamma (0.12 to 0.30), spectral index, amplitude
    (0.6 dex), baryon fraction (the amplitude of wiggles at fixed positions). A BBKS
    transfer function: the turnover moves with Gamma, everything else is a tilt or a
    scaling.
    """
    gamma = _col(U[:, 0], 0.12, 0.30); ns = _col(U[:, 1], 0.90, 1.05)
    amplitude = _col(U[:, 2], -0.3, 0.3); fb = _col(U[:, 3], 0.0, 0.2)
    q = k / gamma
    transfer = np.log(1 + 2.34 * q) / (2.34 * q) * (1 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4) ** -0.25
    wiggles = 1.0 + fb * np.sin(105.0 * k) * np.exp(-(k / 0.2) ** 1.4)
    return np.log10(k ** ns * transfer ** 2 * wiggles) + amplitude + 4.0


def eft_distribution(U, x):
    """A differential cross-section in an effective field theory (3D).

    Parameters: three Wilson coefficients in [-1, 1]. A squared amplitude, so each bin
    is a quadratic polynomial in the coefficients, with energy-growing coefficient
    functions: the curves span a space of ten dimensions exactly.
    """
    c = 2 * U[:, :3] - 1
    c1, c2, c3 = c[:, 0:1], c[:, 1:2], c[:, 2:3]
    sm = 100.0 * np.exp(-4.0 * x)
    real = 1.0 + 1.5 * c1 * x ** 2 + 0.6 * c2 * x + 0.8 * c3 * (x ** 2 - 0.3 * x)
    imag = 0.7 * c2 * x ** 2 - 0.9 * c3 * x
    return sm * (real ** 2 + imag ** 2)


def jet_spectrum(U, pt):
    """log10 of an inclusive jet spectrum d sigma / d p_T d y (3D).

    Parameters: rapidity as a fraction of its kinematic limit, alpha_s(M_Z),
    renormalisation scale over p_T. Close to a power law over thirteen orders of
    magnitude, with a kinematic suppression that sets in at a p_T depending on the
    rapidity.
    """
    xt = 2 * pt / SQRT_S
    y = U[:, 0:1] * np.minimum(3.0, 0.95 * np.arccosh(1.0 / xt))
    alpha_mz = _col(U[:, 1], 0.110, 0.125); xi = _col(U[:, 2], 0.5, 2.0, log=True)
    b0 = 23.0 / (12 * np.pi)
    alpha = alpha_mz / (1.0 + b0 * alpha_mz * np.log((xi * pt) ** 2 / M_Z ** 2))
    k_factor = 1.0 + alpha * (3.0 - 4.0 * b0 * np.pi * np.log(xi)) + 8.0 * alpha ** 2 * np.log(xi) ** 2
    return np.log10(alpha ** 2 * pt ** -4.5 * (1.0 - xt * np.cosh(y)) ** 9 / (1.0 + xt) ** 2 * k_factor) + 12.0


# --------------------------------------------------------------------------- #
# range and transport
# --------------------------------------------------------------------------- #
def dust_sed(U, nu):
    """log10 of a modified black-body spectrum nu^beta B_nu(T) (3D).

    Parameters: temperature (10 to 40 K), emissivity index (1 to 2.5), optical depth
    (two decades). A power law with an exponential cutoff whose position scales with
    the temperature: a shift of the curve along log(nu).
    """
    T = _col(U[:, 0], 10.0, 40.0, log=True); beta = _col(U[:, 1], 1.0, 2.5); tau = _col(U[:, 2], 1e-5, 1e-3, log=True)
    return np.log10(tau * (nu / 353.0) ** beta * nu ** 3 / np.expm1(0.04799 * nu / T))


def z_lineshape(U, e):
    """Cross-section in nb around the Z pole (3D).

    Parameters: mass (90.2 to 92.2 GeV), width (2 to 3 GeV), peak cross-section. A
    Breit-Wigner with an energy-dependent width, whose peak moves by less than its
    width.
    """
    M = _col(U[:, 0], 90.2, 92.2); width = _col(U[:, 1], 2.0, 3.0); peak = _col(U[:, 2], 30.0, 45.0)
    s = e ** 2
    return peak * s * width ** 2 / ((s - M ** 2) ** 2 + s ** 2 * width ** 2 / M ** 2)


def zprime_spectrum(U, mll):
    """log10 of d sigma / d m_ll with a Z' interfering with the continuum (3D).

    Parameters: Z' mass (1 to 4.5 TeV), coupling, extra (invisible) width. A peak of
    2 to 14% relative width that moves across the grid, with a destructive dip below
    it, on a continuum that falls over nine orders of magnitude.
    """
    M = _col(U[:, 0], 1000.0, 4500.0); g = _col(U[:, 1], 0.1, 1.0); extra = _col(U[:, 2], 0.0, 3.0)
    width = M * (0.03 * g ** 2 * (1.0 + extra) + 0.02)
    s = mll ** 2
    amplitude = 1.0 + (g ** 2 / 0.1) * s / (s - M ** 2 + 1j * M * width)
    return np.log10(_parton_luminosity(mll) / s * np.abs(amplitude) ** 2)


def kinematic_edge(U, m):
    """An invariant-mass distribution with a kinematic endpoint (3D).

    Parameters: endpoint (60 to 110 GeV), mass resolution (1 to 4 GeV), background
    fraction. A triangle that ends at the endpoint, smeared by the resolution, on an
    exponential background: an edge a few bins wide that moves across the grid.
    """
    end = _col(U[:, 0], 60.0, 110.0); res = _col(U[:, 1], 1.0, 4.0); fb = _col(U[:, 2], 0.1, 0.5)
    a, b = -m / res, (end - m) / res
    cdf = lambda z: 0.5 * (1 + erf(z / np.sqrt(2)))
    pdf = lambda z: np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    triangle = 2.0 / end ** 2 * (m * (cdf(b) - cdf(a)) + res * (pdf(a) - pdf(b)))
    return 100.0 * ((1 - fb) * triangle + fb * np.exp(-m / 40.0) / 40.0)


# --------------------------------------------------------------------------- #
# oscillation
# --------------------------------------------------------------------------- #
def oscillation_probability(U, E):
    """P(nu_mu -> nu_e) at 1300 km as a function of the neutrino energy (4D).

    Parameters: theta_23, theta_13, delta_CP (0 to 2 pi), Delta m^2_31. Oscillates in
    1 / E with a phase proportional to Delta m^2_31 / E, so the peaks move with
    Delta m^2_31, most at low energy.
    """
    th23 = _col(U[:, 0], 0.67, 0.90); th13 = _col(U[:, 1], 0.13, 0.17)
    delta = _col(U[:, 2], 0.0, 2 * np.pi); dm31 = _col(U[:, 3], 2.3e-3, 2.7e-3)
    th12, dm21, L, rho = 0.584, 7.4e-5, 1300.0, 2.85
    D = 1.267 * dm31 * L / E; alpha = dm21 / dm31; A = 7.6e-5 * rho * E / dm31
    f1 = np.sin((1 - A) * D) / (1 - A); f2 = np.sin(A * D) / A
    return (np.sin(th23) ** 2 * np.sin(2 * th13) ** 2 * f1 ** 2
            + alpha * np.sin(2 * th13) * np.sin(2 * th12) * np.sin(2 * th23) * np.cos(th13) * f1 * f2 * np.cos(D + delta)
            + alpha ** 2 * np.cos(th23) ** 2 * np.sin(2 * th12) ** 2 * f2 ** 2)


def cmb_peaks(U, ell):
    """An acoustic-peak power spectrum D_ell (4D).

    Parameters: acoustic scale (285 to 320), baryon loading, damping scale, spectral
    index. Eight peaks whose spacing follows the acoustic scale, with alternating
    heights set by the baryon loading, under a damping envelope.
    """
    l_a = _col(U[:, 0], 285.0, 320.0); R = _col(U[:, 1], 0.3, 0.9)
    l_d = _col(U[:, 2], 1100.0, 1700.0); ns = _col(U[:, 3], 0.92, 1.0)
    phase = np.pi * (ell / l_a - 0.27)
    monopole = (1 + R) * np.cos(phase) - R
    doppler = np.sin(phase) / np.sqrt(1 + R)
    return 3.0 * (monopole ** 2 + doppler ** 2) * np.exp(-(ell / l_d) ** 1.4) * (ell / 1000.0) ** (ns - 1) + 0.3 * (ell / 30.0) ** -0.5


def form_factor(U, q):
    """log10 of a squared nuclear form factor (3D).

    Parameters: nuclear radius (3.5 to 6.5 fm), surface thickness (0.7 to 1.1 fm), a
    floor that fills the minima (1e-7 to 1e-5). The Helm form factor: diffraction
    minima at positions proportional to 1 / R, under a Gaussian envelope, over seven
    orders of magnitude.
    """
    R = _col(U[:, 0], 3.5, 6.5); s = _col(U[:, 1], 0.7, 1.1); floor = _col(U[:, 2], 1e-7, 1e-5, log=True)
    z = q * R
    form = 3.0 * (np.sin(z) - z * np.cos(z)) / z ** 3 * np.exp(-0.5 * (q * s) ** 2)
    return np.log10(form ** 2 + floor)


def ringdown(U, t):
    """A damped oscillation exp(-t / tau) cos(2 pi f t + phi) (3D).

    Parameters: frequency (0.8 to 1.25), damping time (2 to 6), initial phase (0 to
    pi / 2). Ten cycles on the grid: the frequency range moves the last ones by more
    than a full cycle.
    """
    f = _col(U[:, 0], 0.8, 1.25); tau = _col(U[:, 1], 2.0, 6.0); phi = _col(U[:, 2], 0.0, np.pi / 2)
    return np.exp(-t / tau) * np.cos(2 * np.pi * f * t + phi)


# --------------------------------------------------------------------------- #
# six parameters
# --------------------------------------------------------------------------- #
def zprime_spectrum_6d(U, mll):
    """:func:`zprime_spectrum` with three more parameters (6D): a tilt of the
    spectrum, its normalisation, and the phase of the Z' amplitude relative to the
    continuum, which moves the dip from below the peak to above it."""
    M = _col(U[:, 0], 1000.0, 4500.0); g = _col(U[:, 1], 0.1, 1.0); extra = _col(U[:, 2], 0.0, 3.0)
    tilt = _col(U[:, 3], -0.5, 0.5); normalisation = _col(U[:, 4], -0.2, 0.2); phase = _col(U[:, 5], 0.0, np.pi / 2)
    width = M * (0.03 * g ** 2 * (1.0 + extra) + 0.02)
    s = mll ** 2
    amplitude = 1.0 + np.exp(1j * phase) * (g ** 2 / 0.1) * s / (s - M ** 2 + 1j * M * width)
    return np.log10(_parton_luminosity(mll) / s * np.abs(amplitude) ** 2) + tilt * np.log10(mll / 1000.0) + normalisation


def _eft_shapes():
    rng = np.random.RandomState(7)
    return rng.uniform(-1.0, 1.0, (2, 6, 3))         # real and imaginary part: coefficients of 1, x, x^2 per operator


def eft_distribution_6d(U, x):
    """:func:`eft_distribution` with six Wilson coefficients (6D): the curves span
    a space of 28 dimensions."""
    c = 2 * U[:, :6] - 1
    powers = np.stack([np.ones_like(x), x, x ** 2])                  # (3, p)
    real, imag = (c @ (shape @ powers) for shape in _eft_shapes())   # (n, p) each
    return 100.0 * np.exp(-4.0 * x) * ((1.0 + 0.6 * real) ** 2 + (0.6 * imag) ** 2)


# --------------------------------------------------------------------------- #
# noise
# --------------------------------------------------------------------------- #
MC_OVER_DATA = 5.0      # simulated events per expected event


def peak_counts(U, m):
    """Expected events per bin for a narrow peak on a falling background (3D).

    Parameters: peak mass (115 to 135 GeV), signal strength (0.5 to 2), background
    slope. 20 000 background and 600 signal events at unit strength.
    """
    mass = _col(U[:, 0], 115.0, 135.0); mu = _col(U[:, 1], 0.5, 2.0); slope = _col(U[:, 2], 0.02, 0.04)
    width = m[1] - m[0]
    background = np.exp(-slope * (m - 100.0))
    background = 20000.0 * background / background.sum(axis=1, keepdims=True)
    signal = 600.0 * mu * width * np.exp(-0.5 * ((m - mass) / 1.7) ** 2) / (1.7 * np.sqrt(2 * np.pi))
    return background + signal


def peak_counts_mc(U, m, rng):
    """The counts of :func:`peak_counts` estimated from ``MC_OVER_DATA`` simulated events
    per expected event: ``(estimate, declared sigma, true value)``."""
    truth = peak_counts(U, m)
    generated = rng.poisson(MC_OVER_DATA * truth)
    return generated / MC_OVER_DATA, np.sqrt(np.maximum(generated, 1.0)) / MC_OVER_DATA, truth


# --------------------------------------------------------------------------- #
class CurveFamily:
    """A family of curves on the grid ``x``. ``precision`` is the sigma declared for an
    exact observation, in the unit of the curve."""

    def __init__(self, func, d, x, xlabel, unit, precision, kind, noisy=None):
        self.func, self.d, self.x, self.xlabel, self.unit = func, d, np.asarray(x, dtype=float), xlabel, unit
        self.precision, self.kind, self.noisy = precision, kind, noisy

    @property
    def n_outputs(self):
        return len(self.x)

    def __call__(self, U):
        return self.func(np.atleast_2d(U), self.x)

    def sample(self, U, rng):
        """``(Y_observed, sigma_declared, Y_true)``."""
        U = np.atleast_2d(U)
        if self.noisy is not None:
            return self.noisy(U, self.x, rng)
        Y = self.func(U, self.x)
        return Y, np.full(Y.shape, self.precision), Y


DEX = 0.001 / np.log(10.0)      # 0.1% of the quantity itself

FAMILIES = {
    "matter_power": CurveFamily(matter_power, 4, np.geomspace(1e-3, 1.0, 64), "k [h/Mpc]", "dex", DEX, "smooth"),
    "eft_distribution": CurveFamily(eft_distribution, 3, np.linspace(0.0, 1.0, 48), "scaled energy", "events", 1e-3, "smooth"),
    "jet_spectrum": CurveFamily(jet_spectrum, 3, np.geomspace(50.0, 2000.0, 64), "p_T [GeV]", "dex", DEX, "smooth, range"),
    "dust_sed": CurveFamily(dust_sed, 3, np.geomspace(30.0, 3000.0, 64), "frequency [GHz]", "dex", DEX, "range, transport"),
    "z_lineshape": CurveFamily(z_lineshape, 3, np.linspace(88.0, 94.0, 64), "energy [GeV]", "nb", 1e-3, "transport (mild)"),
    "zprime_spectrum": CurveFamily(zprime_spectrum, 3, np.geomspace(500.0, 5000.0, 128), "m_ll [GeV]", "dex", DEX, "range, transport"),
    "kinematic_edge": CurveFamily(kinematic_edge, 3, np.linspace(20.0, 140.0, 64), "mass [GeV]", "events", 1e-4, "transport"),
    "oscillation_probability": CurveFamily(oscillation_probability, 4, np.geomspace(0.3, 6.0, 96), "energy [GeV]", "probability", 1e-4, "oscillation"),
    "cmb_peaks": CurveFamily(cmb_peaks, 4, np.linspace(30.0, 2500.0, 128), "multipole", "power", 1e-4, "oscillation"),
    "form_factor": CurveFamily(form_factor, 3, np.linspace(0.05, 3.0, 96), "q [1/fm]", "dex", DEX, "range, oscillation"),
    "ringdown": CurveFamily(ringdown, 3, np.linspace(0.0, 10.0, 128), "time", "amplitude", 1e-4, "oscillation"),
    "peak_counts_mc": CurveFamily(peak_counts, 3, np.linspace(100.5, 159.5, 60), "mass [GeV]", "events", 1e-3, "noise, transport",
                                  noisy=peak_counts_mc),
    # two of the families with six parameters
    "zprime_spectrum_6d": CurveFamily(zprime_spectrum_6d, 6, np.geomspace(500.0, 5000.0, 128), "m_ll [GeV]", "dex", DEX, "range, transport"),
    "eft_distribution_6d": CurveFamily(eft_distribution_6d, 6, np.linspace(0.0, 1.0, 48), "scaled energy", "events", 1e-3, "smooth"),
}


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    for name, family in FAMILIES.items():
        U = rng.rand(2000, family.d)
        Y, sigma, truth = family.sample(U, rng)
        # how many principal components carry all but 1e-4 and 1e-6 of the variance
        s = np.linalg.svd(truth - truth.mean(axis=0), compute_uv=False) ** 2
        tail = 1.0 - np.cumsum(s) / s.sum()
        rank = [int(np.searchsorted(-tail, -level)) + 1 for level in (1e-4, 1e-6)]
        print(f"{name:24s} d = {family.d}  {family.n_outputs:3d} points in x  [{family.unit:11s}]  min {truth.min():9.3g}  max {truth.max():9.3g}"
              f"  components for 1e-4, 1e-6 of the variance: {rank[0]:3d} {rank[1]:3d}   ({family.kind})"
              + (f"  noise sd {np.std(Y - truth):.3g}" if family.noisy else ""))
