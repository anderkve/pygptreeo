"""Differential-evolution SED fit with AM3, with a GPTree learning the spectra on the fly.

The scan fits a synthetic observed SED with a one-zone SSC model (``am3_model``):
seven parameters, the six comoving parameters AM3 sees plus the Doppler factor.
A GPTree (``output_model='pca'``) learns the comoving log-spectrum from every
AM3 call made by the scan. Two modes:

* ``shadow``: AM3 is called at every point; the emulator only predicts first and
  its prediction is scored against the AM3 result. This is the pure-AM3 reference
  run plus the emulator's learning curve along the scan's own trajectory.
* ``active``: the emulator is consulted first and replaces AM3 whenever its
  propagated uncertainty on chi^2 passes the gate; otherwise AM3 is run and the
  tree updated. A random fraction of the accepted points is also verified with AM3.

Usage: python scan_emulator.py --mode shadow|active [--out results.npz] [options]
"""

import argparse
import contextlib
import io
import os
import sys
import time
import warnings

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from am3_model import AM3SSCModel, BOUNDS, PARAM_NAMES, observed_sed, luminosity_distance_cm  # noqa: E402

warnings.filterwarnings('ignore')

SCAN_NAMES = PARAM_NAMES + ['doppler']
SCAN_BOUNDS = BOUNDS + [(5.0, 50.0)]
THETA_TRUE = np.array([-0.7, 16.2, 41.8, 2.4, 1.3, 4.9, 20.0])
Z_SRC = 0.1
LOG_FLUX_CAP = -30.0      # observed log10 nuFnu is clipped from below here (floor bins)


# ---------------------------------------------------------------------- #
# Synthetic observation and likelihood
# ---------------------------------------------------------------------- #
def make_observation(model, d_L, sigma_dex=0.05, seed=123, f_min=1e-14):
    """Synthetic SED: log-spaced observed energies with the far-IR and MeV gaps removed."""
    E_obs = np.logspace(-5, 12, 35)
    keep = ~(((E_obs > 1e-3) & (E_obs < 1e-1)) | ((E_obs > 1e5) & (E_obs < 1e8)))
    E_obs = E_obs[keep]
    log_f = model_log_flux(model.comoving_sed(THETA_TRUE[:6]), model.E, THETA_TRUE, d_L, E_obs)
    keep = log_f > np.log10(f_min)
    E_obs, log_f = E_obs[keep], log_f[keep]
    rng = np.random.RandomState(seed)
    log_f_obs = log_f + sigma_dex * rng.randn(E_obs.size)
    return E_obs, log_f_obs, np.full(E_obs.size, sigma_dex)


def model_log_flux(e2n, E, theta, d_L, E_obs, floor=None):
    """log10 nuFnu at the observed energies from a comoving E^2 n(E) (or its log10 with a floor)."""
    doppler = theta[6]
    if floor is not None:                      # e2n is log10(E^2 n + floor)
        e2n = np.maximum(10.0 ** e2n - floor, 0.0)
    E_o, nuFnu = observed_sed(E, e2n, theta[1], doppler, Z_SRC, d_L)
    with np.errstate(divide='ignore'):
        log_nuFnu = np.log10(nuFnu)
    log_nuFnu = np.maximum(log_nuFnu, LOG_FLUX_CAP)
    return np.interp(np.log10(E_obs), np.log10(E_o), log_nuFnu, left=LOG_FLUX_CAP, right=LOG_FLUX_CAP)


def chi2(log_f_model, log_f_obs, sig_obs):
    return float(np.sum(((log_f_model - log_f_obs) / sig_obs) ** 2))


# ---------------------------------------------------------------------- #
# The objective with the on-the-fly emulator
# ---------------------------------------------------------------------- #
class EmulatedObjective:
    def __init__(self, model, E_obs, log_f_obs, sig_obs, d_L, args):
        from pygptreeo import GPTree
        self.model, self.E_obs, self.log_f_obs, self.sig_obs, self.d_L = model, E_obs, log_f_obs, sig_obs, d_L
        self.args = args
        self.floor = args.floor
        self.sigma_train = args.sigma_train
        self.rng = np.random.RandomState(args.seed + 1)
        tree_kwargs = dict(Nbar=args.nbar, n_outputs=model.E.size, output_model=args.output_model,
                           output_basis_components=args.components, retrain_every_n_points=args.retrain_every)
        if args.min_points is not None:
            tree_kwargs['output_basis_min_points'] = args.min_points
        if args.global_mean != 'none':
            tree_kwargs['global_mean'] = args.global_mean
            if args.global_mean == 'net':
                tree_kwargs['global_mean_kwargs'] = dict(steps_per_update=args.gm_steps_per_update, random_state=args.seed)
        if args.load_tree:
            import joblib
            self.tree = joblib.load(args.load_tree)
            print(f"loaded tree from {args.load_tree}: {self._n_points()} points, {len(self.tree.root.leaves)} leaves")
        else:
            self.tree = GPTree(**tree_kwargs)
        self.records = []
        self.n_am3 = 0
        self.n_emulated = 0
        self.t_am3 = self.t_pred = self.t_update = 0.0
        self.best = np.inf
        self.best_true = np.inf      # best chi2 among AM3-evaluated points

    # -- helpers ---------------------------------------------------------
    def _n_points(self):
        return int(sum(leaf.n_points for leaf in self.tree.root.leaves))

    def _emulator_ready(self):
        return self.tree.output_basis.current is not None and self._n_points() >= self.args.min_train

    def _predict(self, theta):
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            mu, sd = self.tree.predict(np.asarray(theta[:6])[None, :])
        self.t_pred += time.perf_counter() - t0
        return mu[0], sd[0]

    def _update(self, theta, y):
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            self.tree.update_tree(np.asarray(theta[:6])[None, :], y[None, :], self.sigma_train)
        self.t_update += time.perf_counter() - t0

    def _am3(self, theta):
        t0 = time.perf_counter()
        e2n = self.model.comoving_sed(theta[:6])
        self.t_am3 += time.perf_counter() - t0
        self.n_am3 += 1
        return e2n

    def _flux_and_sigma_from_emulator(self, theta, mu, sd):
        """Observed log-flux at the data energies from the emulated log-spectrum, and its
        1-sigma band propagated bin-wise (interpolated like the mean)."""
        log_f = model_log_flux(mu, self.model.E, theta, self.d_L, self.E_obs, floor=self.floor)
        # sigma of log10 nuFnu: d log10(10^y - floor)/dy = 10^y / (10^y - floor); use it where defined
        with np.errstate(divide='ignore', invalid='ignore'):
            amp = 10.0 ** mu / np.maximum(10.0 ** mu - self.floor, 1e-300)
        sd_flux = np.minimum(sd * amp, 10.0)
        E_o = self.model.E * theta[6] / (1.0 + Z_SRC)
        sd_at = np.interp(np.log10(self.E_obs), np.log10(E_o), sd_flux, left=10.0, right=10.0)
        return log_f, sd_at

    # -- objective -------------------------------------------------------
    def __call__(self, theta):
        theta = np.asarray(theta, dtype=float)
        rec = dict(theta=theta.copy(), chi2_true=np.nan, chi2_emu=np.nan, sig_chi2=np.nan,
                   rms_vis=np.nan, max_vis=np.nan, rms_data=np.nan, max_data=np.nan,
                   z_med=np.nan, z_max=np.nan, used_emulator=False, verified=False,
                   n_points=self._n_points(), n_leaves=len(self.tree.root.leaves),
                   k=(self.tree.output_basis.current.n_components if self.tree.output_basis.current else 0))
        ready = self._emulator_ready()
        if ready:
            mu, sd = self._predict(theta)
            log_f_emu, sd_at = self._flux_and_sigma_from_emulator(theta, mu, sd)
            c2_emu = chi2(log_f_emu, self.log_f_obs, self.sig_obs)
            # first-order propagated uncertainty of chi^2 (conservative: absolute terms summed)
            r = (log_f_emu - self.log_f_obs) / self.sig_obs
            sig_chi2 = float(np.sum(2.0 * np.abs(r) * sd_at / self.sig_obs))
            rec.update(chi2_emu=c2_emu, sig_chi2=sig_chi2, sd_data_max=float(sd_at.max()))
        run_am3 = True
        if self.args.mode == 'active' and ready:
            # Emulate only when the propagated chi2 bound is small compared with chi2 itself and the
            # point is clearly worse than the best AM3-evaluated point (far_from_best = 0 disables that).
            gate = (sig_chi2 < self.args.chi2_tol_abs + self.args.chi2_tol_rel * c2_emu
                    and c2_emu > self.args.far_from_best * self.best_true)
            if gate:
                rec['used_emulator'] = True
                run_am3 = self.rng.rand() < self.args.verify_frac
                rec['verified'] = run_am3
        if run_am3:
            e2n = self._am3(theta)
            y = np.log10(e2n + self.floor)
            log_f_true = model_log_flux(e2n, self.model.E, theta, self.d_L, self.E_obs)
            c2_true = chi2(log_f_true, self.log_f_obs, self.sig_obs)
            rec['chi2_true'] = c2_true
            self.best_true = min(self.best_true, c2_true)
            if ready:
                vis = e2n > 100 * self.floor
                err = mu - y
                rec.update(rms_vis=float(np.sqrt(np.mean(err[vis] ** 2))), max_vis=float(np.abs(err[vis]).max()),
                           rms_data=float(np.sqrt(np.mean((log_f_emu - log_f_true) ** 2))),
                           max_data=float(np.abs(log_f_emu - log_f_true).max()),
                           z_med=float(np.median(np.abs(err[vis]) / sd[vis])),
                           z_max=float(np.max(np.abs(err[vis]) / sd[vis])))
            self._update(theta, y)
            value = c2_true
        else:
            self.n_emulated += 1
            value = c2_emu
        if rec['used_emulator'] and not run_am3:
            self.n_emulated_only = getattr(self, 'n_emulated_only', 0) + 1
        self.best = min(self.best, value)
        rec['value'] = value
        rec['t_am3'] = self.t_am3; rec['t_pred'] = self.t_pred; rec['t_update'] = self.t_update
        self.records.append(rec)
        i = len(self.records)
        if i % self.args.print_every == 0:
            n_em = sum(r['used_emulator'] for r in self.records)
            recent = [r for r in self.records[-self.args.print_every:] if np.isfinite(r['rms_data'])]
            rd = np.median([r['rms_data'] for r in recent]) if recent else np.nan
            md = np.max([r['max_data'] for r in recent]) if recent else np.nan
            print(f"[{i:5d}] best chi2 {self.best:10.2f} | AM3 calls {self.n_am3:5d}, emulated {n_em:5d} | "
                  f"tree: {rec['n_points']} pts, {rec['n_leaves']} leaves, k={rec['k']} | recent emu err at data: "
                  f"median rms {rd:.3f}, worst max {md:.3f} dex | t: am3 {self.t_am3:.0f}s pred {self.t_pred:.0f}s upd {self.t_update:.0f}s",
                  flush=True)
        return value


# ---------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['shadow', 'active'], default='shadow')
    ap.add_argument('--out', default='scan_results.npz')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--popsize', type=int, default=12)
    ap.add_argument('--maxiter', type=int, default=40)
    ap.add_argument('--nbar', type=int, default=100)
    ap.add_argument('--retrain-every', type=int, default=10)
    ap.add_argument('--output-model', default='pca')
    ap.add_argument('--components', default='noise')
    ap.add_argument('--min-points', type=int, default=None, help='output_basis_min_points')
    ap.add_argument('--min-train', type=int, default=60, help='tree points before the emulator is consulted')
    ap.add_argument('--floor', type=float, default=1e-2, help='soft floor in E^2 n [eV/cm3]')
    ap.add_argument('--sigma-train', type=float, default=1e-2, help='noise std [dex] given to the tree')
    ap.add_argument('--chi2-tol-abs', type=float, default=1.0)
    ap.add_argument('--chi2-tol-rel', type=float, default=0.02)
    ap.add_argument('--verify-frac', type=float, default=0.1)
    ap.add_argument('--far-from-best', type=float, default=0.0,
                    help='emulate only points whose emulated chi2 exceeds this factor times the best AM3 chi2')
    ap.add_argument('--global-mean', default='none', choices=['none', 'net', 'additive_gp'],
                    help="tree-wide global model whose residual the leaves learn ('net' needs PyTorch)")
    ap.add_argument('--gm-steps-per-update', type=int, default=1, help="amortised refit steps per update for 'net'")
    ap.add_argument('--print-every', type=int, default=100)
    ap.add_argument('--save-tree', default=None, help='joblib file to save the trained GPTree to')
    ap.add_argument('--load-tree', default=None, help='start from a GPTree saved by an earlier scan')
    ap.add_argument('--theta-true', default=None, help='comma-separated true scan parameters (7 values)')
    ap.add_argument('--obs-seed', type=int, default=123, help='seed of the synthetic observation noise')
    args = ap.parse_args()
    global THETA_TRUE
    if args.theta_true is not None:
        THETA_TRUE = np.array([float(v) for v in args.theta_true.split(',')])
        assert THETA_TRUE.size == len(SCAN_BOUNDS)
    try:
        args.components = int(args.components)
    except ValueError:
        try:
            args.components = float(args.components)
        except ValueError:
            pass

    model = AM3SSCModel()
    d_L = luminosity_distance_cm(Z_SRC)
    E_obs, log_f_obs, sig_obs = make_observation(model, d_L, seed=args.obs_seed)
    print(f"AM3 ready ({model.t_init:.1f}s init). Synthetic SED: {E_obs.size} points, "
          f"log10 E_obs {np.log10(E_obs).min():.1f}..{np.log10(E_obs).max():.1f} eV, "
          f"log10 nuFnu {log_f_obs.min():.2f}..{log_f_obs.max():.2f}")
    obj = EmulatedObjective(model, E_obs, log_f_obs, sig_obs, d_L, args)
    t0 = time.perf_counter()
    res = differential_evolution(obj, SCAN_BOUNDS, popsize=args.popsize, maxiter=args.maxiter, tol=0.0,
                                 mutation=(0.5, 1.0), recombination=0.7, seed=args.seed, polish=False,
                                 init='latinhypercube', updating='immediate')
    t_wall = time.perf_counter() - t0
    # Final verification of the reported optimum with AM3
    e2n = model.comoving_sed(res.x[:6])
    c2_best_true = chi2(model_log_flux(e2n, model.E, res.x, d_L, E_obs), log_f_obs, sig_obs)
    print(f"\nDE finished: {len(obj.records)} evaluations in {t_wall:.0f}s wall; AM3 calls {obj.n_am3}, "
          f"emulator-only evaluations {obj.n_emulated}; time in AM3 {obj.t_am3:.0f}s, "
          f"tree predict {obj.t_pred:.0f}s, tree update {obj.t_update:.0f}s")
    print(f"best objective {res.fun:.3f} (true chi2 of that point {c2_best_true:.3f}); "
          f"chi2 at theta_true {chi2(model_log_flux(model.comoving_sed(THETA_TRUE[:6]), model.E, THETA_TRUE, d_L, E_obs), log_f_obs, sig_obs):.3f}")
    for n, v, t in zip(SCAN_NAMES, res.x, THETA_TRUE):
        print(f"   {n:12s} {v:9.3f}   (true {t:9.3f})")
    keys = [k for k in obj.records[0] if k != 'theta']
    out = {k: np.array([r.get(k, np.nan) for r in obj.records], dtype=float) for k in keys}
    out['theta'] = np.array([r['theta'] for r in obj.records])
    out.update(best_x=res.x, best_fun=res.fun, best_true=c2_best_true, theta_true=THETA_TRUE,
               E_obs=E_obs, log_f_obs=log_f_obs, sig_obs=sig_obs, t_wall=t_wall, E=model.E,
               mode=args.mode, components=str(args.components), nbar=args.nbar)
    np.savez(args.out, **out)
    print(f"saved {args.out}")
    if args.save_tree:
        obj.tree.save(args.save_tree)
        print(f"saved tree to {args.save_tree}")


if __name__ == '__main__':
    main()
