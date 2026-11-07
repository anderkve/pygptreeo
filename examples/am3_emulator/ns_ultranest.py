"""Nested sampling with UltraNest (MLFriends region, uniform draws) and the
threshold-gated GPTree emulator; the vectorised counterpart of ns_emulator.py.

UltraNest draws proposals in batches above the current threshold ``Lmin``; a
thin subclass records that threshold before each batch, and the likelihood
predicts the whole batch with the tree at once. A proposal is answered by the
emulator when its emulated log L plus ``k`` propagated sigmas is below the
threshold (a safe rejection) or its sigma is below ``tol``; the rest go to AM3
and the tree learns from every AM3 call. UltraNest's own ``log_dir`` resume is
used for the sampler; the tree and the per-call records are saved alongside.

Usage: python ns_ultranest.py --gate thr:1,0.3 --nlive 250 --log-dir un_run [options]
"""

import argparse
import contextlib
import io
import os
import sys
import time
import warnings

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import scan_emulator as se  # noqa: E402
from am3_model import AM3SSCModel, luminosity_distance_cm  # noqa: E402

warnings.filterwarnings('ignore')
REC_KEYS = ('logl_true', 'logl_emu', 'slogl', 'sdex_max', 'used', 'verified', 'lmin')


class GatedLogLike:
    """Vectorised likelihood: emulate safe rejections, run AM3 for the rest."""

    def __init__(self, model, E_obs, log_f_obs, sig_obs, d_L, args):
        from pygptreeo import GPTree
        self.model, self.E_obs, self.log_f_obs, self.sig_obs, self.d_L, self.args = model, E_obs, log_f_obs, sig_obs, d_L, args
        self.floor = args.floor
        self.k, self.tol = [float(v) for v in args.gate.partition(':')[2].split(',')]
        self.rng = np.random.RandomState(args.seed + 1)
        self.tree = GPTree(Nbar=args.nbar, n_outputs=model.E.size, output_model='pca',
                           output_basis_components=args.components, retrain_every_n_points=args.retrain_every,
                           GPR=se.make_gpr(args.kernel, 6))
        self.rec = {k: [] for k in REC_KEYS}
        self.theta_rec = []
        self.n_calls = self.n_am3 = self.n_batches = 0
        self.t_am3 = self.t_pred = self.t_update = 0.0
        self.lmin = -np.inf
        self.t_start = time.perf_counter()
        self.last_state_save = time.perf_counter()
        self.last_print = 0

    # persistence of the tree and records (UltraNest's resume covers the sampler only)
    def save_state(self, path):
        joblib.dump(dict(tree=self.tree, rec=self.rec, theta_rec=self.theta_rec, n_calls=self.n_calls, n_am3=self.n_am3,
                         t_am3=self.t_am3, t_pred=self.t_pred, t_update=self.t_update, rng=self.rng), path)

    def load_state(self, path):
        d = joblib.load(path)
        self.tree, self.rec, self.theta_rec = d['tree'], d['rec'], d['theta_rec']
        self.n_calls, self.n_am3 = d['n_calls'], d['n_am3']
        self.t_am3, self.t_pred, self.t_update, self.rng = d['t_am3'], d['t_pred'], d['t_update'], d['rng']

    def _n_points(self):
        return int(sum(leaf.n_points for leaf in self.tree.root.leaves))

    def _ready(self):
        return self.tree.output_basis.current is not None and self._n_points() >= self.args.min_train

    def _emulated_row(self, theta, mu, sd):
        log_f = se.model_log_flux(mu, self.model.E, theta, self.d_L, self.E_obs, floor=self.floor)
        with np.errstate(divide='ignore', invalid='ignore'):
            amp = 10.0 ** mu / np.maximum(10.0 ** mu - self.floor, 1e-300)
        sd_flux = np.minimum(sd * amp, 10.0)
        E_o = self.model.E * theta[6] / (1.0 + se.Z_SRC)
        sd_at = np.interp(np.log10(self.E_obs), np.log10(E_o), sd_flux, left=10.0, right=10.0)
        c2 = se.chi2(log_f, self.log_f_obs, self.sig_obs)
        r = (log_f - self.log_f_obs) / self.sig_obs
        return -0.5 * c2, 0.5 * float(np.sum(2.0 * np.abs(r) * sd_at / self.sig_obs)), float(sd_at.max())

    def _am3(self, theta):
        t0 = time.perf_counter()
        e2n = self.model.comoving_sed(theta[:6])
        self.t_am3 += time.perf_counter() - t0
        self.n_am3 += 1
        logl = -0.5 * se.chi2(se.model_log_flux(e2n, self.model.E, theta, self.d_L, self.E_obs), self.log_f_obs, self.sig_obs)
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            self.tree.update_tree(np.asarray(theta[:6])[None, :], np.log10(e2n + self.floor)[None, :], self.args.sigma_train)
        self.t_update += time.perf_counter() - t0
        return logl

    def __call__(self, Theta):
        Theta = np.atleast_2d(np.asarray(Theta, dtype=float))
        n = Theta.shape[0]
        out = np.empty(n)
        lmin = self.lmin
        ready = self._ready()
        logl_emu = np.full(n, np.nan); slogl = np.full(n, np.nan); sdex = np.full(n, np.nan)
        used = np.zeros(n, bool); verified = np.zeros(n, bool)
        if ready:
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                MU, SD = self.tree.predict(Theta[:, :6])
            self.t_pred += time.perf_counter() - t0
            for i in range(n):
                logl_emu[i], slogl[i], sdex[i] = self._emulated_row(Theta[i], MU[i], SD[i])
            used = (slogl < self.tol) | ((logl_emu + self.k * slogl) < lmin)
            verified = used & (self.rng.rand(n) < self.args.verify_frac)
        logl_true = np.full(n, np.nan)
        for i in range(n):
            if not used[i] or verified[i]:
                logl_true[i] = self._am3(Theta[i])
                out[i] = logl_true[i]
            else:
                out[i] = logl_emu[i]
        self.n_calls += n; self.n_batches += 1
        for k, v in (('logl_true', logl_true), ('logl_emu', logl_emu), ('slogl', slogl), ('sdex_max', sdex),
                     ('used', used), ('verified', verified), ('lmin', np.full(n, lmin))):
            self.rec[k].extend(v.tolist())
        self.theta_rec.extend(Theta.tolist())
        if self.n_calls - self.last_print >= self.args.print_every:
            self.last_print = self.n_calls
            u = np.array(self.rec['used'][-self.args.print_every:], bool)
            lt = np.array(self.rec['logl_true'][-self.args.print_every:]); le = np.array(self.rec['logl_emu'][-self.args.print_every:])
            both = np.isfinite(lt) & np.isfinite(le); d = np.abs(lt[both] - le[both])
            print(f"[{self.n_calls:8d}] batches {self.n_batches:6d} (last {n:5d}), Lmin {lmin:9.2f} | AM3 {self.n_am3:6d} ({self.n_am3 / self.n_calls:.1%}), "
                  f"recent emulated {u.mean():.0%} | tree {self._n_points()} pts, {len(self.tree.root.leaves)} leaves | recent |dlogL| scored median "
                  f"{np.median(d) if d.size else np.nan:.2f} | t: am3 {self.t_am3:.0f}s pred {self.t_pred:.0f}s upd {self.t_update:.0f}s wall {time.perf_counter() - self.t_start:.0f}s", flush=True)
        if self.args.state and time.perf_counter() - self.last_state_save > self.args.state_every:
            self.save_state(self.args.state); self.last_state_save = time.perf_counter()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', default='thr:1,0.3', help='thr:<k>,<tol>')
    ap.add_argument('--nlive', type=int, default=250)
    ap.add_argument('--dlogz', type=float, default=0.5)
    ap.add_argument('--log-dir', default='un_run')
    ap.add_argument('--resume', default='resume', choices=['resume', 'overwrite', 'subfolder'])
    ap.add_argument('--state', default=None, help='joblib file for the tree and records (default: <log-dir>/emulator_state.joblib)')
    ap.add_argument('--state-every', type=float, default=600.0)
    ap.add_argument('--out', default='un_results.npz')
    ap.add_argument('--seed', type=int, default=13)
    ap.add_argument('--nbar', type=int, default=500)
    ap.add_argument('--retrain-every', type=int, default=500)
    ap.add_argument('--components', default='24')
    ap.add_argument('--kernel', default='matern15')
    ap.add_argument('--min-train', type=int, default=60)
    ap.add_argument('--floor', type=float, default=1e-2)
    ap.add_argument('--sigma-train', type=float, default=1e-2)
    ap.add_argument('--verify-frac', type=float, default=0.005)
    ap.add_argument('--print-every', type=int, default=10000)
    ap.add_argument('--validate', type=int, default=400)
    ap.add_argument('--max-ncalls', type=int, default=None)
    ap.add_argument('--improvement-loops', type=int, default=0, help="UltraNest's max_num_improvement_loops")
    ap.add_argument('--frac-remain', type=float, default=0.01, help="UltraNest's frac_remain stopping criterion")
    ap.add_argument('--ndraw-min', type=int, default=128)
    ap.add_argument('--theta-true', default=None)
    ap.add_argument('--obs-seed', type=int, default=123)
    args = ap.parse_args()
    try:
        args.components = int(args.components)
    except ValueError:
        pass
    if args.theta_true is not None:
        se.THETA_TRUE = np.array([float(v) for v in args.theta_true.split(',')])
    if args.state is None:
        args.state = os.path.join(args.log_dir, 'emulator_state.joblib')

    import ultranest
    from ultranest import ReactiveNestedSampler

    class GatedSampler(ReactiveNestedSampler):
        """Hands the current threshold to the gated likelihood before each batch of draws."""

        def _create_point(self, Lmin, ndraw, active_u, active_values):
            ll.lmin = float(Lmin)
            return super()._create_point(Lmin, ndraw, active_u, active_values)

    model = AM3SSCModel()
    d_L = luminosity_distance_cm(se.Z_SRC)
    E_obs, log_f_obs, sig_obs = se.make_observation(model, d_L, seed=args.obs_seed)
    lo, hi = np.array(se.SCAN_BOUNDS).T
    transform = lambda u: lo + u * (hi - lo)   # noqa: E731
    ll = GatedLogLike(model, E_obs, log_f_obs, sig_obs, d_L, args)
    os.makedirs(args.log_dir, exist_ok=True)
    if args.resume == 'resume' and os.path.exists(args.state):
        ll.load_state(args.state)
        print(f"loaded emulator state: {ll.n_calls} calls, {ll.n_am3} AM3 calls, {ll._n_points()} tree points", flush=True)
    print(f"UltraNest {ultranest.__version__}; gate {args.gate}; nlive {args.nlive}; verify {args.verify_frac}; log_dir {args.log_dir}", flush=True)
    np.random.seed(args.seed)
    sampler = GatedSampler(se.SCAN_NAMES, ll, transform=transform, log_dir=args.log_dir, resume=args.resume,
                           vectorized=True, ndraw_min=args.ndraw_min)
    t0 = time.perf_counter()
    res = sampler.run(min_num_live_points=args.nlive, dlogz=args.dlogz, frac_remain=args.frac_remain,
                      max_num_improvement_loops=args.improvement_loops, max_ncalls=args.max_ncalls,
                      show_status=False, viz_callback=None)
    t_wall = time.perf_counter() - t0
    ws = res['weighted_samples']
    pts, w, logl = ws['points'], ws['weights'], ws['logl']
    mean = (pts * w[:, None]).sum(0); std = np.sqrt(((pts - mean) ** 2 * w[:, None]).sum(0))
    print(f"\nUltraNest finished: {res['niter']} iterations, {res['ncall']} sampler calls ({ll.n_calls} likelihood evaluations seen), "
          f"AM3 calls {ll.n_am3} ({ll.n_am3 / max(ll.n_calls, 1):.1%}); logZ {res['logz']:.2f} +- {res['logzerr']:.2f}; "
          f"wall {t_wall:.0f}s (AM3 {ll.t_am3:.0f}s, tree predict {ll.t_pred:.0f}s, update {ll.t_update:.0f}s); max logL {logl.max():.2f}")
    print("parameter     post. mean   post. std   true")
    for nme, m, s_, t in zip(se.SCAN_NAMES, mean, std, se.THETA_TRUE):
        print(f"  {nme:12s} {m:10.3f} {s_:10.3f} {t:8.3f}")
    out = dict(samples=pts, weights=w, logl=logl, logz=res['logz'], logzerr=res['logzerr'], niter=res['niter'], ncall=ll.n_calls,
               n_am3=ll.n_am3, t_wall=t_wall, t_am3=ll.t_am3, t_pred=ll.t_pred, t_update=ll.t_update, gate=args.gate,
               nlive=args.nlive, theta_true=se.THETA_TRUE, post_mean=mean, post_std=std, E_obs=E_obs, log_f_obs=log_f_obs, sig_obs=sig_obs)
    for k, v in ll.rec.items():
        out['rec_' + k] = np.array(v, dtype=float)
    out['rec_theta'] = np.array(ll.theta_rec)
    if args.validate > 0:
        rng = np.random.default_rng(args.seed + 2)
        idx = np.unique(rng.choice(pts.shape[0], size=args.validate, replace=True, p=w / w.sum()))
        lt = np.array([-0.5 * se.chi2(se.model_log_flux(model.comoving_sed(pts[i, :6]), model.E, pts[i], d_L, E_obs), log_f_obs, sig_obs) for i in idx])
        dl = lt - logl[idx]
        iw = np.exp(dl - dl.max()); iw /= iw.sum(); ess = 1.0 / np.sum(iw ** 2)
        print(f"\nvalidation: AM3 at {idx.size} distinct posterior samples: logL_true - logL_used median {np.median(dl):+.3f}, rms {np.sqrt(np.mean(dl ** 2)):.3f}, "
              f"max |.| {np.abs(dl).max():.2f}; reweighting ESS {ess:.0f} / {idx.size} ({ess / idx.size:.0%})")
        out.update(val_idx=idx, val_logl_true=lt, val_logl_used=logl[idx], val_ess=ess)
    np.savez(args.out, **out)
    ll.save_state(args.state)
    print(f"saved {args.out} and {args.state}")


if __name__ == '__main__':
    main()
