"""A one-zone leptonic (synchrotron self-Compton) blazar model run with AM3.

The emulated quantity is the steady-state comoving photon spectrum on AM3's
fixed log-energy grid, as a function of the six physical parameters of the
emitting region. The Doppler factor, redshift and luminosity distance only
rescale energies and fluxes, so they are applied analytically on top and
never enter the emulator.

Parameters (``PARAM_NAMES``, in this order):
    log10 B [G], log10 R [cm], log10 L_e [erg/s] (comoving electron
    injection luminosity), p (injection index), log10 gamma_min, log10 gamma_max.
"""

import contextlib
import os
import time

import numpy as np

C_CGS = 2.99792458e10            # cm/s
M_E_EV = 510998.95               # eV
EV_TO_ERG = 1.602176634e-12
H_EV_S = 4.135667696e-15         # eV s

PARAM_NAMES = ['log10_B', 'log10_R', 'log10_Le', 'p', 'log10_gmin', 'log10_gmax']
# Scan box; a classic SSC blazar range.
BOUNDS = [(-2.0, 1.0), (15.0, 17.0), (39.5, 43.5), (1.5, 3.0), (0.5, 2.5), (3.5, 6.0)]


@contextlib.contextmanager
def _silence_fd(fd=1):
    """Silence C++-level printing on a file descriptor (AM3 prints on stdout)."""
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)


class AM3SSCModel:
    """Steady-state SSC spectra from AM3 for the parameters in ``PARAM_NAMES``.

    One AM3 instance and one set of kernels are reused for all points; only the
    magnetic field, escape time, time step and injection change between points.

    Parameters
    ----------
    e_ph_min, e_nu_min, e_max : float
        AM3 energy grid: min photon energy, min neutrino energy, max energy of all
        particles [eV]. The default photon grid is 1e-7 eV to ~8e13 eV (482 bins).
    n_tesc : float
        Simulated time in units of the light-crossing time R/c.
    dt_frac : float
        Solver time step in units of R/c.
    annihilation : bool
        Include gamma-gamma pair production (internal absorption and the cascade).
    """

    def __init__(self, e_ph_min=1e-7, e_nu_min=1e9, e_max=1e14, n_tesc=10.0, dt_frac=0.05,
                 annihilation=True):
        import am3
        self.n_tesc = float(n_tesc)
        self.dt_frac = float(dt_frac)
        self.n_steps = int(round(self.n_tesc / self.dt_frac))
        sim = am3.AM3()
        with _silence_fd():
            sim.set_verbosity_level(0)
            sim.update_energy_grid(e_ph_min, e_nu_min, e_max)
            sim.set_estimate_max_energies(0)
            sim.set_process_parse_sed(0)
            sim.set_process_hadronic(0)
            sim.set_process_escape(1)
            sim.set_process_expansion(0)
            sim.set_process_adiabatic_cooling(0)
            sim.set_process_electron_syn(1)
            sim.set_process_ssa(1)
            sim.set_process_electron_compton(1)
            sim.set_process_annihilation(1 if annihilation else 0)
            sim.set_process_quantum_syn(0)
            t0 = time.perf_counter()
            sim.init_kernels()
            self.t_init = time.perf_counter() - t0
        self.sim = sim
        self.E = np.asarray(sim.get_egrid_photons(), dtype=float).copy()   # eV, comoving
        self.n_calls = 0
        self.t_total = 0.0

    # ------------------------------------------------------------------ #
    def photon_density(self, theta):
        """Comoving steady-state photon density per unit ln E, ``E n(E)`` [cm^-3], on ``self.E``
        (what AM3's ``get_photons`` returns)."""
        log_B, log_R, log_Le, p, log_gmin, log_gmax = [float(v) for v in theta]
        B, R, Le = 10.0 ** log_B, 10.0 ** log_R, 10.0 ** log_Le
        gmin, gmax = 10.0 ** log_gmin, 10.0 ** log_gmax
        sim = self.sim
        t0 = time.perf_counter()
        with _silence_fd():
            sim.set_mag_field(B)
            tesc = R / C_CGS
            sim.set_escape_timescale(tesc)
            sim.set_solver_time_step(self.dt_frac * tesc)
            sim.clear_particle_densities()
            sim.set_powerlaw_injection_parameters_electrons(
                4.0 / 3.0 * np.pi * R ** 3, Le, gmin * M_E_EV, gmin * M_E_EV, gmax * M_E_EV, p, p, 1.0)
            for _ in range(self.n_steps):
                sim.evolve_step()
        n_ph = np.asarray(sim.get_photons(), dtype=float).copy()
        self.n_calls += 1
        self.t_total += time.perf_counter() - t0
        return n_ph

    def comoving_sed(self, theta):
        """E^2 n(E) [eV cm^-3] on ``self.E``: the quantity the emulator learns (in log10)."""
        return self.E * self.photon_density(theta)


# ---------------------------------------------------------------------- #
# Observer-frame conversion (applied on top of the comoving SED)
# ---------------------------------------------------------------------- #
def luminosity_distance_cm(z, H0=70.0, Om=0.3):
    """Flat LambdaCDM luminosity distance [cm] by numerical integration."""
    from scipy.integrate import quad
    c_km_s = 299792.458
    integrand = lambda zz: 1.0 / np.sqrt(Om * (1 + zz) ** 3 + (1 - Om))
    dc = c_km_s / H0 * quad(integrand, 0, z)[0]       # Mpc
    return (1 + z) * dc * 3.085677581e24


def observed_sed(E_comoving, e2n_comoving, log_R, doppler, z, d_L):
    """Map a comoving E^2 n(E) [eV cm^-3] to an observed nu F_nu [erg cm^-2 s^-1].

    Returns the observed photon energies [eV] and nu F_nu on them. The comoving
    luminosity per ln E is E^2 n(E) V / t_esc with V = 4/3 pi R^3 and t_esc = R/c;
    it is boosted by delta^4 and energies by delta / (1 + z).
    """
    R = 10.0 ** log_R
    L = e2n_comoving * EV_TO_ERG * 4.0 * np.pi * R ** 2 * C_CGS / 3.0        # erg/s per ln E
    E_obs = E_comoving * doppler / (1.0 + z)
    nuFnu = L * doppler ** 4 / (4.0 * np.pi * d_L ** 2)
    return E_obs, nuFnu


def log_interp(x_new, x, y):
    """Interpolate y(x) in log10(x), linearly in y, returning -inf outside the range."""
    lx, lxn = np.log10(x), np.log10(x_new)
    out = np.interp(lxn, lx, y, left=np.nan, right=np.nan)
    return out


if __name__ == '__main__':
    model = AM3SSCModel()
    print(f"init {model.t_init:.2f}s; photon grid {model.E.size} bins {model.E[0]:.2e}..{model.E[-1]:.2e} eV")
    rng = np.random.RandomState(0)
    for i in range(5):
        theta = [lo + (hi - lo) * rng.rand() for lo, hi in BOUNDS]
        t0 = time.perf_counter()
        sed = model.comoving_sed(theta)
        dt = time.perf_counter() - t0
        pos = sed > 0
        print(f"theta={np.round(theta, 2)} {dt:.3f}s  peak {sed.max():.2e} eV/cm3 at {model.E[sed.argmax()]:.2e} eV; "
              f"positive bins {pos.sum()}; min positive {sed[pos].min():.1e}")
    print(f"mean AM3 time per call: {model.t_total / model.n_calls:.3f}s")
