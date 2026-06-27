"""Physics-style target functions defined by *numerical* computation.

Unlike the closed-form analytic benchmarks (target_functions.py), each target
here is the result of a small numerical computation — a quadrature or an ODE
solve — with no elementary closed form, mimicking the kind of "expensive
evaluation" encountered in a physics study (an emulator surrogate for a code
that integrates something per parameter point).

Every target takes inputs in the unit cube ``[0,1]^d`` (one row per point),
maps them to physical parameter ranges, and returns a 1-D array of values.
Values are arranged to be positive and O(1)-ish so the relative-error metrics
stay well-defined.

Datasets (Gaussian-sampled inputs + computed targets) are cached to ``data/``
so the numerical cost is paid once and reused across all kernels in a group.
"""

import os
import numpy as np
from math import comb
from scipy import integrate
from scipy.linalg import eigh_tridiagonal
from scipy.special import gamma as _gamma_fn

# Gaussian input sampling used by the optional make_dataset() cache helper.
GAUSS_MEAN = 0.5
GAUSS_STD = 0.15

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# --------------------------------------------------------------------------- #
# 1. damped_response (2D) — a parametric Laplace-type integral to +inf.
#    f(a,b) = ∫_0^inf e^{-a t} sin(t) ln(1 + b t) / (1 + t^2) dt + 1
#    (adapted from code_experiments/.../run_numerical_integral_demo.py)
# --------------------------------------------------------------------------- #
def _damped_response_scalar(a, b):
    val, _ = integrate.quad(
        lambda t: np.exp(-a * t) * np.sin(t) * np.log1p(b * t) / (1.0 + t * t),
        0.0, np.inf, limit=120, epsabs=1e-10, epsrel=1e-10)
    return val + 1.0


def damped_response(X):
    """2D: (a in [0.3,5], b in [0.1,5]); damped oscillatory response integral."""
    a = 0.3 + X[:, 0] * (5.0 - 0.3)
    b = 0.1 + X[:, 1] * (5.0 - 0.1)
    return np.array([_damped_response_scalar(ai, bi) for ai, bi in zip(a, b)])


# --------------------------------------------------------------------------- #
# 2. planck_band (3D) — blackbody power in a finite spectral band.
#    f(T,x0,w) = ∫_{x0}^{x0+w} x^3 / (e^{x/T} - 1) dx
# --------------------------------------------------------------------------- #
def _planck_band_scalar(T, x0, w):
    val, _ = integrate.quad(
        lambda x: x ** 3 / np.expm1(x / T),
        x0, x0 + w, limit=120, epsabs=1e-10, epsrel=1e-10)
    return val


def planck_band(X):
    """3D: (T in [0.4,4], x0 in [0.1,6], width in [0.5,6]); log Planck band power.

    The band power spans many orders of magnitude (exponential Wien suppression
    at low T / high frequency), so the target is the *log* of the integral —
    the natural, well-conditioned quantity a physics emulator would learn.
    """
    T = 0.4 + X[:, 0] * (4.0 - 0.4)
    x0 = 0.1 + X[:, 1] * (6.0 - 0.1)
    w = 0.5 + X[:, 2] * (6.0 - 0.5)
    vals = np.array([_planck_band_scalar(Ti, x0i, wi)
                     for Ti, x0i, wi in zip(T, x0, w)])
    return np.log(vals)


# --------------------------------------------------------------------------- #
# 3. projectile_drag (3D) — range of a projectile with quadratic air drag,
#    obtained by integrating the equations of motion (vectorised RK4). No
#    closed form.  f(theta, v0, k) = horizontal distance at ground impact.
# --------------------------------------------------------------------------- #
def _projectile_range(theta, v0, k, g=9.81, dt=0.01, max_steps=4000):
    n = theta.shape[0]
    x = np.zeros(n); y = np.zeros(n)
    vx = v0 * np.cos(theta); vy = v0 * np.sin(theta)
    rng = np.zeros(n); done = np.zeros(n, dtype=bool)

    def deriv(s):
        sx, sy, svx, svy = s
        sp = np.sqrt(svx * svx + svy * svy)
        return np.stack([svx, svy, -k * sp * svx, -g - k * sp * svy])

    s = np.stack([x, y, vx, vy])
    for step in range(max_steps):
        y_prev = s[1].copy(); x_prev = s[0].copy()
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        crossed = (~done) & (y_prev >= 0.0) & (s[1] < 0.0)
        if np.any(crossed):
            frac = y_prev / (y_prev - s[1])           # linear interp to y=0
            rng = np.where(crossed, x_prev + frac * (s[0] - x_prev), rng)
            done |= crossed
        if done.all():
            break
    # Any that never crossed (shouldn't happen): use current x.
    rng = np.where(done, rng, s[0])
    return rng


def projectile_drag(X):
    """3D: (theta in [20,70] deg, v0 in [8,22], k in [0,0.1]); range with drag."""
    theta = np.deg2rad(20.0 + X[:, 0] * (70.0 - 20.0))
    v0 = 8.0 + X[:, 1] * (22.0 - 8.0)
    k = 0.0 + X[:, 2] * (0.1 - 0.0)
    return _projectile_range(theta, v0, k)


# --------------------------------------------------------------------------- #
# 4. spectral_density (4D) — a 4-parameter damped, modulated spectral integral.
#    f(a,b,c,d) = ∫_0^inf e^{-a t} (1 + c sin(b t)) / (1 + d t^2) dt
# --------------------------------------------------------------------------- #
def _spectral_density_scalar(a, b, c, d):
    val, _ = integrate.quad(
        lambda t: np.exp(-a * t) * (1.0 + c * np.sin(b * t)) / (1.0 + d * t * t),
        0.0, np.inf, limit=160, epsabs=1e-10, epsrel=1e-10)
    return val


def spectral_density(X):
    """4D: (a in [0.3,4], b in [0.5,6], c in [0,0.8], d in [0.2,4]) damped spectral integral."""
    a = 0.3 + X[:, 0] * (4.0 - 0.3)
    b = 0.5 + X[:, 1] * (6.0 - 0.5)
    c = 0.0 + X[:, 2] * (0.8 - 0.0)
    d = 0.2 + X[:, 3] * (4.0 - 0.2)
    return np.array([_spectral_density_scalar(ai, bi, ci, di)
                     for ai, bi, ci, di in zip(a, b, c, d)])


# =========================================================================== #
# Harder targets (round 3): designed to actually discriminate kernels via
# high-frequency oscillation, a sharp near-singular ridge, high-dimensional
# non-separable coupling, and an eigenvalue problem with avoided crossings.
# =========================================================================== #

# 5. oscillatory_chirp (3D) — a damped *chirped* oscillatory integral. The
#    result is a Fresnel-type coefficient that oscillates many times across the
#    (chirp-rate, frequency) plane: high-frequency structure is hard to emulate.
def _oscillatory_chirp_scalar(a, w, phi):
    val, _ = integrate.quad(
        lambda t: np.exp(-a * t) * np.cos(2 * np.pi * (w * t * t + phi * t)),
        0.0, 1.0, limit=200, epsabs=1e-11, epsrel=1e-11)
    return val + 1.0


def oscillatory_chirp(X):
    """3D: (a in [0.1,1.2], chirp w in [2,16], freq phi in [0,8]); chirped oscillatory integral.

    Light damping keeps the oscillatory (Fresnel-type) structure prominent; the
    result wiggles many times across the (w, phi) plane.
    """
    a = 0.1 + X[:, 0] * (1.2 - 0.1)
    w = 2.0 + X[:, 1] * (16.0 - 2.0)
    phi = 0.0 + X[:, 2] * (8.0 - 0.0)
    return np.array([_oscillatory_chirp_scalar(ai, wi, pi)
                     for ai, wi, pi in zip(a, w, phi)])


# 6. ring_potential (3D) — electrostatic potential of a unit charged ring at a
#    field point (x, y, z=h). The integral is a (complete-elliptic-type) function
#    with no elementary closed form; as the point approaches the ring (radius 1)
#    and the height h shrinks, it develops a sharp ridge — sharp moving features.
def _ring_potential_scalar(x, y, h):
    val, _ = integrate.quad(
        lambda p: 1.0 / np.sqrt((x - np.cos(p)) ** 2 + (y - np.sin(p)) ** 2 + h * h),
        0.0, 2 * np.pi, limit=200, epsabs=1e-10, epsrel=1e-10)
    return val


def ring_potential(X):
    """3D: (x in [-1.5,1.5], y in [-1.5,1.5], height h in [0.05,1.2]); near-singular ring potential."""
    x = -1.5 + X[:, 0] * 3.0
    y = -1.5 + X[:, 1] * 3.0
    h = 0.05 + X[:, 2] * (1.2 - 0.05)
    return np.array([_ring_potential_scalar(xi, yi, hi)
                     for xi, yi, hi in zip(x, y, h)])


# 7. coupled_anisotropic (6D) — a high-dimensional, *non-separable* integral.
#    g(t) = Σ_i p_i cos(iπt); f = ∫_0^1 exp(-½ g(t)²) dt. The square couples all
#    parameters pairwise (Σ_ij p_i p_j …), so there is no additive shortcut, and
#    6 input dimensions make local data sparse — the curse of dimensionality.
_COUPLED_T = np.linspace(0.0, 1.0, 257)
_COUPLED_B = np.stack([np.cos((i + 1) * np.pi * _COUPLED_T) for i in range(6)])  # (6, 257)


def coupled_anisotropic(X):
    """6D: p_i in [-1,1]; non-separable integral exp(-0.5 (Σ p_i cos(iπt))²) over t."""
    P = -1.0 + 2.0 * X[:, :6]                  # (n, 6)
    g = P @ _COUPLED_B                          # (n, 257) — Σ_i p_i cos(iπt)
    integrand = np.exp(-0.5 * g * g)            # (n, 257)
    return np.trapezoid(integrand, _COUPLED_T, axis=1)


def coupled_nd(X):
    """Dimension-general version of coupled_anisotropic (matches it at d=6).

    p_i in [-1,1]; non-separable integral exp(-0.5 (Σ_{i=1}^d p_i cos(iπt))²) dt.
    The square couples all d parameters pairwise, so there is no additive shortcut;
    used to probe the curse of dimensionality at d = 6, 8, 10.
    """
    d = X.shape[1]
    B = np.stack([np.cos((i + 1) * np.pi * _COUPLED_T) for i in range(d)])  # (d, 257)
    g = (-1.0 + 2.0 * X) @ B                    # (n, 257)
    return np.trapezoid(np.exp(-0.5 * g * g), _COUPLED_T, axis=1)


# --- Hard *additive* numerical targets (for the Matern-vs-RBF additive-base test) #
# The per-dimension / per-pair terms are Bessel J0 functions, evaluated by their
# integral representation J0(z) = (1/pi) integral_0^pi cos(z sin theta) dtheta — a
# genuine numerical integral (and a real wave/diffraction quantity) whose *value*
# oscillates many times across the parameter domain. That makes the low-order
# structure rough, the regime where the additive base (RBF vs Matern) matters.
_BESS_TH = np.linspace(0.0, np.pi, 300)
_BESS_SIN = np.sin(_BESS_TH)


def _j0(z):
    """Bessel J0 by its integral representation; z of shape (n,)."""
    return np.trapezoid(np.cos(z[:, None] * _BESS_SIN[None, :]), _BESS_TH, axis=1) / np.pi


def additive_bessel(X):
    """Order-1 additive, rough: f = Σ_i J0(16 p_i) + 2 (oscillatory main effects)."""
    out = np.zeros(X.shape[0])
    for i in range(X.shape[1]):
        out += _j0(16.0 * X[:, i])
    return out + 2.0                            # offset -> positive (for within-4%)


def pairwise_bessel(X):
    """Order-2 additive, rough: f = Σ_{i<j} J0(14·||(p_i,p_j)−(½,½)||) (circular waves)."""
    d = X.shape[1]
    out = np.zeros(X.shape[0])
    for i in range(d):
        for j in range(i + 1, d):
            r = np.sqrt((X[:, i] - 0.5) ** 2 + (X[:, j] - 0.5) ** 2)
            out += _j0(14.0 * r)
    return out + 0.5 * comb(d, 2)               # offset -> positive


# 8. quantum_well (4D) — ground-state energy of a 1-D Schrödinger operator with a
#    tunable double-well + tilt + ripple potential, by numerical diagonalisation
#    of the finite-difference Hamiltonian. Avoided crossings between the two wells
#    create sharp curvature in E0(p): an expensive, structured physics emulation.
_QW_X = np.linspace(-7.0, 7.0, 320)            # wide enough to contain the wells
_QW_DX = _QW_X[1] - _QW_X[0]
_QW_KIN_DIAG = 1.0 / _QW_DX ** 2               # -ψ'' FD: 2/(2dx²) per side -> 1/dx² diag
_QW_OFF = np.full(_QW_X.size - 1, -0.5 / _QW_DX ** 2)


def _quantum_well_scalar(p1, p2, p3, p4):
    V = p1 * _QW_X ** 4 - p2 * _QW_X ** 2 + p3 * _QW_X + p4 * np.sin(2.0 * _QW_X)
    diag = _QW_KIN_DIAG + V
    w = eigh_tridiagonal(diag, _QW_OFF, select="i", select_range=(0, 0),
                         eigvals_only=True)
    return float(w[0])


def quantum_well(X):
    """4D: (p1 in [0.1,0.5], p2 in [0.5,3], p3 in [-1.5,1.5], p4 in [0,2]); ground-state energy E0.

    Ranges keep the double-well minima (x² = p2/2p1) inside the grid so E0 is a
    smooth (no boundary-artifact) function with avoided-crossing structure.
    """
    p1 = 0.1 + X[:, 0] * (0.5 - 0.1)
    p2 = 0.5 + X[:, 1] * (3.0 - 0.5)
    p3 = -1.5 + X[:, 2] * 3.0
    p4 = 0.0 + X[:, 3] * 2.0
    return np.array([_quantum_well_scalar(a, b, c, dd)
                     for a, b, c, dd in zip(p1, p2, p3, p4)])


# =========================================================================== #
# Round-7 additions: a much broader, smoothness-diverse physics zoo. The key new
# additions are *rough* (low-differentiability) additive targets — |amplitude|
# with kinks, and sharp resonance peaks — the regime the round-6 Bessel test did
# not probe. Plus several more non-additive physics computations for breadth.
# =========================================================================== #
_ABS_T = np.linspace(0.0, 1.0, 400)
_ABS_T2 = _ABS_T ** 2


def additive_absint(X):
    """Order-1 additive, ROUGH (kinked): f = Σ_i |∫₀¹ cos(2π(1+6 p_i) t) dt|.

    The Fourier cosine coefficient crosses zero, so |·| introduces V-shaped kinks
    (C0 but not C1) — a structure-factor / form-factor magnitude with sharp zeros.
    """
    out = np.zeros(X.shape[0])
    for i in range(X.shape[1]):
        phi = 1.0 + 6.0 * X[:, i]
        coeff = np.trapezoid(np.cos(2 * np.pi * phi[:, None] * _ABS_T[None, :]),
                             _ABS_T, axis=1)
        out += np.abs(coeff)
    return out + 0.2


def pairwise_absint(X):
    """Order-2 additive, ROUGH: f = Σ_{i<j} |∫₀¹ cos(2π((1+5 p_i)t+(1+5 p_j)t²)) dt|."""
    d = X.shape[1]
    out = np.zeros(X.shape[0])
    for i in range(d):
        for j in range(i + 1, d):
            a = 1.0 + 5.0 * X[:, i]
            b = 1.0 + 5.0 * X[:, j]
            O = np.trapezoid(np.cos(2 * np.pi * (a[:, None] * _ABS_T[None, :]
                                                 + b[:, None] * _ABS_T2[None, :])),
                             _ABS_T, axis=1)
            out += np.abs(O)
    return out + 0.2


_LOR_TAU = np.linspace(0.0, 30.0, 800)
_LOR_E = np.exp(-0.3 * _LOR_TAU)


def _lorentz_num(detune):
    """|∫₀^∞ e^{-(γ−i·detune)τ} dτ| computed numerically (≈ 1/√(γ²+detune²))."""
    c = np.trapezoid(_LOR_E[None, :] * np.cos(detune[:, None] * _LOR_TAU[None, :]),
                     _LOR_TAU, axis=1)
    s = np.trapezoid(_LOR_E[None, :] * np.sin(detune[:, None] * _LOR_TAU[None, :]),
                     _LOR_TAU, axis=1)
    return np.sqrt(c * c + s * s)


def additive_lorentzian(X):
    """Order-1 additive, ROUGH (sharp peaks): f = Σ_i Σ_k L(10 p_i − ω0_k), ω0∈{2,5,8}.

    A spectroscopy-style sum of resonance line shapes (each a numerically-computed
    Lorentzian) — sharp peaks where the swept frequency hits a resonance.
    """
    out = np.zeros(X.shape[0])
    for i in range(X.shape[1]):
        w = 10.0 * X[:, i]
        for c0 in (2.0, 5.0, 8.0):
            out += _lorentz_num(w - c0)
    return out


def kepler_anomaly(X):
    """2D non-additive: solve Kepler M = E − e sinE (Newton), return r = 1 − e cosE."""
    M = 2 * np.pi * X[:, 0]
    e = 0.8 * X[:, 1]
    E = M.copy()
    for _ in range(60):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return 1.0 - e * np.cos(E)


def fermi_dirac(X):
    """2D non-additive: Fermi-Dirac integral F_j(η)=∫₀^∞ t^j/(1+e^{t−η}) dt / Γ(j+1)."""
    eta = -3.0 + 8.0 * X[:, 0]
    j = 0.5 + 2.5 * X[:, 1]
    out = np.empty(X.shape[0])
    for k in range(X.shape[0]):
        val, _ = integrate.quad(lambda t: t ** j[k] / (1.0 + np.exp(np.clip(t - eta[k], -700, 700))),
                                0.0, np.inf, limit=120)
        out[k] = val / _gamma_fn(j[k] + 1.0)
    return np.log(out)                            # spans orders of magnitude -> log


def saha_ionization(X):
    """2D non-additive, sharp transition: ionization fraction x solving x²/(1−x)=K(T,ρ)."""
    T = 0.3 + 2.7 * X[:, 0]
    logrho = -3.0 + 6.0 * X[:, 1]
    K = (T ** 1.5) * np.exp(-1.0 / T) / np.exp(logrho)
    x = (-K + np.sqrt(K * K + 4.0 * K)) / 2.0
    return x + 0.1


def diffraction_grating(X):
    """3D non-additive, ROUGH (sharp peaks): log N-slit grating intensity (softened)."""
    theta = -1.0 + 2.0 * X[:, 0]
    N = 3.0 + 7.0 * X[:, 1]
    w = 0.2 + 1.3 * X[:, 2]
    delta = np.pi * 6.0 * theta
    grating = np.sin(N * delta / 2.0) ** 2 / (np.sin(delta / 2.0) ** 2 + 0.03)
    beta = np.pi * w * theta
    single = np.sinc(beta / np.pi) ** 2          # np.sinc(x)=sin(πx)/(πx)
    return np.log1p(grating * single) + 0.1      # offset -> away from 0 (within-4%)


# Registry: name -> (callable taking X of shape (n, d), dimensionality, description)
PHYSICS_TARGETS = {
    "damped_response": (damped_response, 2, "parametric Laplace-type integral to +inf"),
    "planck_band":     (planck_band,     3, "blackbody power in a finite spectral band"),
    "projectile_drag": (projectile_drag, 3, "projectile range with quadratic drag (ODE)"),
    "spectral_density": (spectral_density, 4, "4-parameter damped modulated spectral integral"),
    # --- harder (round 3) ---
    "oscillatory_chirp": (oscillatory_chirp, 3, "damped chirped oscillatory integral (high frequency)"),
    "ring_potential":    (ring_potential,    3, "near-singular charged-ring potential (sharp ridge)"),
    "coupled_anisotropic": (coupled_anisotropic, 6, "6D non-separable coupled integral (curse of dim)"),
    "coupled_nd":        (coupled_nd,        6, "dimension-general non-separable coupled integral"),
    "quantum_well":      (quantum_well,      4, "Schrodinger ground-state energy via diagonalisation"),
    # smooth additive numerical targets (oscillatory but analytic, via Bessel J0)
    "additive_bessel":   (additive_bessel,   6, "order-1 additive, SMOOTH oscillatory (Bessel J0)"),
    "pairwise_bessel":   (pairwise_bessel,   4, "order-2 additive, SMOOTH oscillatory (Bessel J0)"),
    # rough additive numerical targets (kinks / sharp peaks — low differentiability)
    "additive_absint":   (additive_absint,   5, "order-1 additive, ROUGH (|amplitude| kinks)"),
    "pairwise_absint":   (pairwise_absint,   4, "order-2 additive, ROUGH (|amplitude| kinks)"),
    "additive_lorentzian": (additive_lorentzian, 5, "order-1 additive, ROUGH (sharp resonance peaks)"),
    # more non-additive physics computations (root-find / integral / transition / optics)
    "kepler_anomaly":    (kepler_anomaly,    2, "non-additive: Kepler equation root-find"),
    "fermi_dirac":       (fermi_dirac,       2, "non-additive: Fermi-Dirac integral"),
    "saha_ionization":   (saha_ionization,   2, "non-additive, sharp transition: Saha ionization"),
    "diffraction_grating": (diffraction_grating, 3, "non-additive, ROUGH: N-slit grating intensity"),
}


def make_dataset(func_name, d, N, seed):
    """Gaussian-sampled inputs in [0,1]^d and numerically-computed targets, cached."""
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, f"{func_name}_d{d}_N{N}_s{seed}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["X"], z["y"]
    rng = np.random.RandomState(seed)
    X = rng.normal(GAUSS_MEAN, GAUSS_STD, size=(N, d))
    np.clip(X, 0.0, 1.0, out=X)
    func = PHYSICS_TARGETS[func_name][0]
    y = np.asarray(func(X), dtype=float)
    np.savez(cache, X=X, y=y)
    return X, y


if __name__ == "__main__":
    # Quick self-test: evaluate a few points of each target.
    rng = np.random.RandomState(0)
    for name, (func, d, desc) in PHYSICS_TARGETS.items():
        Xt = rng.uniform(0, 1, (5, d))
        yt = func(Xt)
        print(f"{name:18s} d={d}  {desc}")
        print(f"   sample values: {np.array2string(yt, precision=4)}")
