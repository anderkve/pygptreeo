"""Nested sampling (dynesty) of the SSC parameters with AM3, with a GPTree
learning the spectra on the fly and optionally answering likelihood calls.

Same synthetic SED, model and tree as ``scan_emulator.py``; log L = -chi^2 / 2
with uniform priors over the scan box. Gates (``--gate``):

* ``none``: every call goes to AM3 (reference run; the tree only learns).
* ``sdex:<tol>``: emulate when the predicted 1-sigma uncertainty of the
  observed log-flux, at every data energy, is below ``tol`` dex. The simple
  gate: prediction uncertainty only, no reference to the likelihood value.
* ``slogl:<tol>``: emulate when the propagated uncertainty of log L
  (first order, absolute terms summed) is below ``tol``.
* ``tailored:<abs>,<rel>,<far>``: the DE-style gate of scan_emulator.py.
* ``thr:<k>,<tol>``: nested-sampling aware, still a pure uncertainty gate:
  emulate when the propagated log L uncertainty is below ``tol``, or when the
  emulated log L is more than ``k`` such uncertainties *below* the sampler's
  current threshold (the lowest live point), i.e. the proposal is rejected
  whatever the exact value. Points that may be accepted always go to AM3, so
  every live and dead point carries an AM3 likelihood unless its sigma < tol.

A fraction ``--verify-frac`` of the gated calls is still sent to AM3 (and the
true value returned). After the run, ``--validate N`` draws N equal-weight
posterior samples, evaluates AM3 there and reports the log L error of the
posterior and the importance-reweighted posterior moments.

Usage: python ns_emulator.py --gate sdex:0.01 --nlive 1000 --out ns_sdex.npz [options]
"""

import argparse
import contextlib
import io
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import scan_emulator as se  # noqa: E402
from am3_model import AM3SSCModel, luminosity_distance_cm  # noqa: E402

warnings.filterwarnings('ignore')


class EmulatedLogLike:
    def __init__(self, model, E_obs, log_f_obs, sig_obs, d_L, args):
        from pygptreeo import GPTree
        self.model, self.E_obs, self.log_f_obs, self.sig_obs, self.d_L = model, E_obs, log_f_obs, sig_obs, d_L
        self.args = args
        self.floor = args.floor
        self.rng = np.random.RandomState(args.seed + 1)
        self.gate_kind, self.gate_par = self._parse_gate(args.gate)
        if args.load_tree:
            import joblib
            self.tree = joblib.load(args.load_tree)
        else:
            tree_kwargs = dict(Nbar=args.nbar, n_outputs=model.E.size, output_model='pca',
                               output_basis_components=args.components, retrain_every_n_points=args.retrain_every)
            if args.global_mean != 'none':
                tree_kwargs['global_mean'] = args.global_mean
                if args.global_mean == 'net':
                    tree_kwargs['global_mean_kwargs'] = dict(steps_per_update=args.gm_steps_per_update, random_state=args.seed)
            self.tree = GPTree(**tree_kwargs)
        self.rec = {k: [] for k in ('theta', 'logl_true', 'logl_emu', 'slogl', 'sdex_max', 'used', 'verified', 'n_points')}
        self.n_calls = self.n_am3 = 0
        self.t_am3 = self.t_pred = self.t_update = 0.0
        self.best_chi2_true = np.inf
        self.sampler = None            # set by main() for the 'thr' gate
        self.t_start = time.perf_counter()

    @staticmethod
    def _parse_gate(spec):
        if spec == 'none':
            return 'none', None
        kind, _, par = spec.partition(':')
        vals = [float(v) for v in par.split(',')] if par else []
        if kind == 'sdex' or kind == 'slogl':
            return kind, vals[0]
        if kind == 'tailored':
            return kind, tuple(vals)   # abs, rel, far
        if kind == 'thr':
            return kind, tuple(vals)   # k, tol
        raise ValueError(f"unknown gate {spec!r}")

    def _n_points(self):
        return int(sum(leaf.n_points for leaf in self.tree.root.leaves))

    def _ready(self):
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
            self.tree.update_tree(np.asarray(theta[:6])[None, :], y[None, :], self.args.sigma_train)
        self.t_update += time.perf_counter() - t0

    def _am3(self, theta):
        t0 = time.perf_counter()
        e2n = self.model.comoving_sed(theta[:6])
        self.t_am3 += time.perf_counter() - t0
        self.n_am3 += 1
        return e2n

    def _emulated(self, theta, mu, sd):
        """Observed log-flux at the data energies from the emulated spectrum, its 1-sigma band,
        chi^2 and the propagated (conservative) chi^2 uncertainty."""
        log_f = se.model_log_flux(mu, self.model.E, theta, self.d_L, self.E_obs, floor=self.floor)
        with np.errstate(divide='ignore', invalid='ignore'):
            amp = 10.0 ** mu / np.maximum(10.0 ** mu - self.floor, 1e-300)
        sd_flux = np.minimum(sd * amp, 10.0)
        E_o = self.model.E * theta[6] / (1.0 + se.Z_SRC)
        sd_at = np.interp(np.log10(self.E_obs), np.log10(E_o), sd_flux, left=10.0, right=10.0)
        c2 = se.chi2(log_f, self.log_f_obs, self.sig_obs)
        r = (log_f - self.log_f_obs) / self.sig_obs
        sig_chi2 = float(np.sum(2.0 * np.abs(r) * sd_at / self.sig_obs))
        return log_f, sd_at, c2, sig_chi2

    def _gate_open(self, c2_emu, sig_chi2, sdex_max):
        k, p = self.gate_kind, self.gate_par
        if k == 'none':
            return False
        if k == 'sdex':
            return sdex_max < p
        if k == 'slogl':
            return 0.5 * sig_chi2 < p
        if k == 'tailored':
            a, rel, far = p
            return sig_chi2 < a + rel * c2_emu and c2_emu > far * self.best_chi2_true
        if k == 'thr':
            kk, tol = p
            slogl = 0.5 * sig_chi2
            if slogl < tol:
                return True
            live = getattr(self.sampler, 'live_logl', None)
            if live is None or not np.isfinite(live).all():
                return False
            return (-0.5 * c2_emu + kk * slogl) < float(np.min(live))
        return False

    def __call__(self, theta):
        theta = np.asarray(theta, dtype=float)
        self.n_calls += 1
        logl_emu = slogl = sdex_max = np.nan
        used = verified = False
        ready = self._ready()
        if ready:
            mu, sd = self._predict(theta)
            log_f_emu, sd_at, c2_emu, sig_chi2 = self._emulated(theta, mu, sd)
            logl_emu, slogl, sdex_max = -0.5 * c2_emu, 0.5 * sig_chi2, float(sd_at.max())
            used = self._gate_open(c2_emu, sig_chi2, sdex_max)
        run_am3 = not used or (self.rng.rand() < self.args.verify_frac)
        verified = used and run_am3
        logl_true = np.nan
        if run_am3:
            e2n = self._am3(theta)
            log_f_true = se.model_log_flux(e2n, self.model.E, theta, self.d_L, self.E_obs)
            c2_true = se.chi2(log_f_true, self.log_f_obs, self.sig_obs)
            logl_true = -0.5 * c2_true
            self.best_chi2_true = min(self.best_chi2_true, c2_true)
            self._update(theta, np.log10(e2n + self.floor))
            value = logl_true
        else:
            value = logl_emu
        for k, v in (('theta', theta.copy()), ('logl_true', logl_true), ('logl_emu', logl_emu), ('slogl', slogl),
                     ('sdex_max', sdex_max), ('used', used), ('verified', verified), ('n_points', self._n_points())):
            self.rec[k].append(v)
        if self.n_calls % self.args.print_every == 0:
            n_used = int(np.sum(self.rec['used']))
            lt, le, v = np.array(self.rec['logl_true'][-self.args.print_every:]), np.array(self.rec['logl_emu'][-self.args.print_every:]), np.array(self.rec['verified'][-self.args.print_every:])
            both = np.isfinite(lt) & np.isfinite(le)
            d = np.abs(lt[both] - le[both])
            dv = np.abs(lt[both & v] - le[both & v])
            live = getattr(self.sampler, 'live_logl', None)
            it = getattr(self.sampler, 'it', 0)
            lmin = float(np.min(live)) if live is not None and np.isfinite(live).all() else np.nan
            print(f"[{self.n_calls:7d}] iter {it:6d} (eff {100.0 * it / max(self.n_calls, 1):.1f}%), logL_min {lmin:9.2f} | "
                  f"AM3 {self.n_am3:6d}, emulated {n_used:6d} ({n_used / self.n_calls:.0%}) | tree {self._n_points()} pts, "
                  f"{len(self.tree.root.leaves)} leaves | recent |dlogL|: all-scored median {np.median(d) if d.size else np.nan:.2f}, "
                  f"verified median {np.median(dv) if dv.size else np.nan:.2f} max {dv.max() if dv.size else np.nan:.2f} | "
                  f"best chi2 {self.best_chi2_true:.1f} | t: am3 {self.t_am3:.0f}s pred {self.t_pred:.0f}s upd {self.t_update:.0f}s "
                  f"wall {time.perf_counter() - self.t_start:.0f}s", flush=True)
        return float(value)


def posterior_moments(samples, logwt):
    w = np.exp(logwt - logwt.max()); w /= w.sum()
    mean = (samples * w[:, None]).sum(0)
    std = np.sqrt(((samples - mean) ** 2 * w[:, None]).sum(0))
    return mean, std, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', default='none')
    ap.add_argument('--out', default='ns_results.npz')
    ap.add_argument('--nlive', type=int, default=1000)
    ap.add_argument('--dlogz', type=float, default=0.5)
    ap.add_argument('--sample', default='rwalk')
    ap.add_argument('--bound', default='multi')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--nbar', type=int, default=100)
    ap.add_argument('--retrain-every', type=int, default=10)
    ap.add_argument('--components', default='16')
    ap.add_argument('--min-train', type=int, default=60)
    ap.add_argument('--floor', type=float, default=1e-2)
    ap.add_argument('--sigma-train', type=float, default=1e-2)
    ap.add_argument('--verify-frac', type=float, default=0.1)
    ap.add_argument('--global-mean', default='none', choices=['none', 'net', 'additive_gp'],
                    help="tree-wide global model whose residual the leaves learn ('net' needs PyTorch)")
    ap.add_argument('--gm-steps-per-update', type=int, default=1, help="amortised refit steps per update for 'net'")
    ap.add_argument('--print-every', type=int, default=2000)
    ap.add_argument('--validate', type=int, default=400, help='posterior samples to re-evaluate with AM3 (0: skip)')
    ap.add_argument('--save-tree', default=None)
    ap.add_argument('--load-tree', default=None)
    ap.add_argument('--theta-true', default=None)
    ap.add_argument('--obs-seed', type=int, default=123)
    ap.add_argument('--max-calls', type=int, default=None, help='stop the run after this many likelihood calls')
    args = ap.parse_args()
    try:
        args.components = int(args.components)
    except ValueError:
        pass
    if args.theta_true is not None:
        se.THETA_TRUE = np.array([float(v) for v in args.theta_true.split(',')])

    import dynesty
    model = AM3SSCModel()
    d_L = luminosity_distance_cm(se.Z_SRC)
    E_obs, log_f_obs, sig_obs = se.make_observation(model, d_L, seed=args.obs_seed)
    lo, hi = np.array(se.SCAN_BOUNDS).T
    prior_transform = lambda u: lo + u * (hi - lo)   # noqa: E731
    ll = EmulatedLogLike(model, E_obs, log_f_obs, sig_obs, d_L, args)
    print(f"AM3 ready; SED {E_obs.size} points; gate {args.gate}; nlive {args.nlive}, sample {args.sample}, dlogz {args.dlogz}", flush=True)
    sampler = dynesty.NestedSampler(ll, prior_transform, len(lo), nlive=args.nlive, bound=args.bound, sample=args.sample,
                                    rstate=np.random.default_rng(args.seed))
    ll.sampler = sampler
    t0 = time.perf_counter()
    if args.max_calls:
        for _ in sampler.sample(dlogz=args.dlogz):
            if ll.n_calls >= args.max_calls:
                print(f"stopping at {ll.n_calls} calls (--max-calls)", flush=True)
                break
        sampler.add_live_points()
    else:
        sampler.run_nested(dlogz=args.dlogz, print_progress=False)
    t_wall = time.perf_counter() - t0
    res = sampler.results
    mean, std, w = posterior_moments(res.samples, res.logwt)
    print(f"\nNS finished: {res.niter} iterations, {ll.n_calls} likelihood calls, eff {res.eff:.1f}%; AM3 calls {ll.n_am3} "
          f"({ll.n_am3 / ll.n_calls:.1%}); logZ {res.logz[-1]:.2f} +- {res.logzerr[-1]:.2f}; H {res.information[-1]:.1f} nats; "
          f"wall {t_wall:.0f}s (AM3 {ll.t_am3:.0f}s, tree predict {ll.t_pred:.0f}s, update {ll.t_update:.0f}s); max logL {res.logl.max():.2f}")
    print("parameter     post. mean   post. std   true")
    for n, m, s_, t in zip(se.SCAN_NAMES, mean, std, se.THETA_TRUE):
        print(f"  {n:12s} {m:10.3f} {s_:10.3f} {t:8.3f}")
    out = dict(samples=res.samples, logwt=res.logwt, logl=res.logl, logz=res.logz, logzerr=res.logzerr,
               niter=res.niter, ncall=ll.n_calls, n_am3=ll.n_am3, t_wall=t_wall, t_am3=ll.t_am3, t_pred=ll.t_pred,
               t_update=ll.t_update, gate=args.gate, nlive=args.nlive, theta_true=se.THETA_TRUE, post_mean=mean, post_std=std,
               E_obs=E_obs, log_f_obs=log_f_obs, sig_obs=sig_obs)
    for k, v in ll.rec.items():
        out['rec_' + k] = np.array(v, dtype=float)
    # Validation: AM3 at equal-weight posterior samples, importance reweighting
    if args.validate > 0:
        rng = np.random.default_rng(args.seed + 2)
        idx = rng.choice(res.samples.shape[0], size=args.validate, replace=True, p=w)
        idx = np.unique(idx)
        logl_used = res.logl[idx]
        logl_true = np.empty(idx.size)
        t0 = time.perf_counter()
        for j, i in enumerate(idx):
            e2n = model.comoving_sed(res.samples[i, :6])
            logl_true[j] = -0.5 * se.chi2(se.model_log_flux(e2n, model.E, res.samples[i], d_L, E_obs), log_f_obs, sig_obs)
        dl = logl_true - logl_used
        iw = np.exp(dl - dl.max()); iw /= iw.sum()
        ess = 1.0 / np.sum(iw ** 2)
        m2 = (res.samples[idx] * iw[:, None]).sum(0)
        s2 = np.sqrt(((res.samples[idx] - m2) ** 2 * iw[:, None]).sum(0))
        m1 = res.samples[idx].mean(0); s1 = res.samples[idx].std(0)
        print(f"\nvalidation: AM3 at {idx.size} distinct posterior samples ({time.perf_counter() - t0:.0f}s): "
              f"logL_true - logL_used: median {np.median(dl):+.3f}, rms {np.sqrt(np.mean(dl ** 2)):.3f}, "
              f"5-95% [{np.percentile(dl, 5):+.2f}, {np.percentile(dl, 95):+.2f}], max |.| {np.abs(dl).max():.2f}; "
              f"importance-reweighting ESS {ess:.0f} / {idx.size} ({ess / idx.size:.0%})")
        print("parameter     sample mean -> reweighted   sample std -> reweighted   (shift in units of std)")
        for n, a, b, c, d in zip(se.SCAN_NAMES, m1, m2, s1, s2):
            print(f"  {n:12s} {a:10.3f} -> {b:10.3f}   {c:10.3f} -> {d:10.3f}   ({(b - a) / c:+.2f})")
        out.update(val_idx=idx, val_logl_true=logl_true, val_logl_used=logl_used, val_ess=ess)
    np.savez(args.out, **out)
    print(f"saved {args.out}")
    if args.save_tree:
        ll.tree.save(args.save_tree)
        print(f"saved tree to {args.save_tree}")


if __name__ == '__main__':
    main()
