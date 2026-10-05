"""Probe: a GPTree whose leaves are Bayesian linear regressions on the features
of one shared network (a "neural-linear tree"), against the plain GP tree, on
10D targets under uniform and focusing streams.

The tree code is unchanged: NeuralLinearGPR implements GPRegressorInterface and
FeatureNet plays the part of a tree-wide learner that sees every point (as
GlobalMeanLearner does).  Metrics mirror examples/benchmark_global_mean_streams.py.
Configurations: tree (the plain GP tree), nltree<Nbar> (leaves regress the
target on the features), nlres<Nbar> (leaves regress the residual of the
network's head).  Results and the reading: docs/neural_gptree_ideas.md,
raw lines in examples/results/neural_linear_probe/.

    OMP_NUM_THREADS=1 python examples/probe_neural_linear_tree.py --target rotated_rosenbrock --stream uniform \
        --configs tree,nltree400,nlres400 --d 10 --N 8000 --seed 1 [--noise 0.05]
"""
import argparse, contextlib, io, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn
torch.set_num_threads(1)
from scipy.linalg import cho_solve

from pygptreeo import GPTree, Default_GPR
from pygptreeo.gp_interface import GPRegressorInterface
from benchmark_global_mean_streams import make_stream, TARGETS


class FeatureNet:
    """Shared feature learner: an MLP trained on everything seen so far, refit when
    the count has doubled since the last fit (warm start).  The last hidden layer
    is the feature map the leaves regress on; the head is kept only as a baseline."""

    def __init__(self, d, hidden=128, depth=3, min_points=200, steps=4000, seed=0):
        self.d, self.hidden, self.depth, self.min_points, self.steps = d, hidden, depth, min_points, steps
        self.X, self.y = [], []
        self.n_seen = 0; self.n_at_fit = 0; self.version = 0; self.net = None; self.body = None
        self.y_mu, self.y_sd = 0.0, 1.0
        self.t_fit = 0.0; self.n_refits = 0
        torch.manual_seed(seed); self.gen = torch.Generator().manual_seed(seed)
        self.m = hidden

    def observe(self, x, y):
        self.X.append(np.asarray(x, float).ravel()); self.y.append(float(y)); self.n_seen += 1
        if self.n_seen >= self.min_points and (self.net is None or self.n_seen >= 2 * self.n_at_fit):
            self.fit()

    def fit(self):
        t0 = time.time()
        X = torch.tensor(np.array(self.X), dtype=torch.float32); y = np.array(self.y)
        self.y_mu, self.y_sd = float(y.mean()), float(y.std() + 1e-12)
        yt = torch.tensor((y - self.y_mu) / self.y_sd, dtype=torch.float32)[:, None]
        if self.net is None:
            layers, w = [], self.d
            for _ in range(self.depth):
                layers += [nn.Linear(w, self.hidden), nn.SiLU()]; w = self.hidden
            self.body = nn.Sequential(*layers); self.head = nn.Linear(w, 1)
            self.net = nn.Sequential(self.body, self.head)
        n = len(X); batch = int(min(128, max(16, n // 2)))
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.steps, eta_min=0.0)
        self.net.train(); step = 0
        while step < self.steps:
            perm = torch.randperm(n, generator=self.gen)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                loss = ((self.net(2 * (X[idx] - 0.5)) - yt[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
                if step >= self.steps:
                    break
        self.net.eval(); self.version += 1; self.n_at_fit = self.n_seen; self.n_refits += 1
        self.t_fit += time.time() - t0

    def features(self, X):
        with torch.no_grad():
            return self.body(2 * (torch.tensor(np.atleast_2d(X), dtype=torch.float32) - 0.5)).double().numpy()

    def head_predict(self, X):
        with torch.no_grad():
            out = self.net(2 * (torch.tensor(np.atleast_2d(X), dtype=torch.float32) - 0.5))
        return out.double().numpy().ravel() * self.y_sd + self.y_mu

    def feature_gradients(self, X, w):
        """d/dx of w . phi(x) at each row of X: (n, d)."""
        Xt = torch.tensor(np.atleast_2d(X), dtype=torch.float32, requires_grad=True)
        f = self.body(2 * (Xt - 0.5)) @ torch.tensor(w, dtype=torch.float32)
        g, = torch.autograd.grad(f.sum(), Xt)
        return g.double().numpy()


class NeuralLinearGPR(GPRegressorInterface):
    """Bayesian linear regression on the shared network's features plus an intercept.
    The prior variance and an extra (model-misfit) noise variance are chosen by the
    evidence on a small grid.  The predicted sigma is the posterior sigma of the
    latent function (no noise term), as for the GP leaves.  A fit remembers the
    feature version it used and re-solves on the current features when asked to
    predict after the network has been refit (the tree's refresh rule)."""

    LOG_S2 = np.linspace(-8, 0, 9)    # extra noise variance, relative to the leaf's y variance
    LOG_T2 = np.linspace(-3, 3, 13)   # prior variance of the (standardised) feature weights

    def __init__(self, learner, residual=False):
        self.learner = learner; self.residual = residual; self.alpha = 1e-10
        self.mu = None; self.Sigma = None; self.version = -1
        self.X_fit = None; self.y_fit = None; self.alpha_fit = None
        self.tau2 = None; self.s2 = None; self.y_shift = 0.0; self.y_scale = 1.0
        self.f_mu = None; self.f_sd = None; self.n_solves = 0

    def _phi(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        if self.learner.net is None:
            F = np.hstack([X - 0.5, (X - 0.5) ** 2])          # before the first feature fit
        else:
            F = self.learner.features(X)
        return np.hstack([F, np.ones((F.shape[0], 1))])

    def _std(self, Phi):
        out = Phi.copy(); out[:, :-1] = (Phi[:, :-1] - self.f_mu) / self.f_sd
        return out

    def set_observation_noise(self, alpha):
        self.alpha = np.asarray(alpha, float).ravel() if isinstance(alpha, np.ndarray) else float(alpha)

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, float)); y = np.asarray(y, float).ravel()
        alpha = np.broadcast_to(np.asarray(self.alpha, float).ravel(), (len(y),)).astype(float)
        self.X_fit, self.y_fit, self.alpha_fit = X, y, alpha
        self._solve(); return self

    def _head(self, X):
        if self.residual and self.learner.net is not None:
            return self.learner.head_predict(X)
        return np.zeros(np.atleast_2d(X).shape[0])

    def _solve(self):
        X, alpha = self.X_fit, self.alpha_fit
        y = self.y_fit - self._head(X)
        self.y_shift = float(y.mean()); sd = float(y.std()); self.y_scale = sd if sd > 0 else 1.0
        yc = (y - self.y_shift) / self.y_scale; a = alpha / self.y_scale ** 2
        Phi = self._phi(X)
        self.f_mu = Phi[:, :-1].mean(0); self.f_sd = Phi[:, :-1].std(0) + 1e-8
        Phi = self._std(Phi); n, m = Phi.shape
        best = (-np.inf, None)
        for log_s2 in self.LOG_S2:
            lam = a + 10.0 ** log_s2; Li = 1.0 / lam
            PtL = Phi.T * Li; G = PtL @ Phi; b = PtL @ yc; yy = float((yc ** 2 * Li).sum()); ld = float(np.log(lam).sum())
            for log_t2 in self.LOG_T2:
                t2 = 10.0 ** log_t2
                A = G + np.eye(m) / t2
                try:
                    L = np.linalg.cholesky(A)
                except np.linalg.LinAlgError:
                    continue
                mu = cho_solve((L, True), b)
                ev = -0.5 * (yy - mu @ A @ mu + 2 * np.log(np.diag(L)).sum() + ld + m * np.log(t2) + n * np.log(2 * np.pi))
                if ev > best[0]:
                    best = (ev, (t2, 10.0 ** log_s2, L, mu))
        _, (t2, s2, L, mu) = best
        self.tau2, self.s2, self.mu = t2, s2, mu
        self.Sigma = cho_solve((L, True), np.eye(m))
        self.version = self.learner.version; self.n_solves += 1

    def predict(self, X, return_std=False):
        X = np.atleast_2d(np.asarray(X, float))
        if self.mu is None:
            mean = np.full(X.shape[0], self.learner.y_mu if self.learner.n_seen else 0.0)
            std = np.full(X.shape[0], self.learner.y_sd if self.learner.n_seen else 1.0)
            return (mean, std) if return_std else mean
        if self.version != self.learner.version and self.X_fit is not None:
            self._solve()                                      # the features changed since this fit
        Phi = self._std(self._phi(X))
        mean = Phi @ self.mu * self.y_scale + self.y_shift + self._head(X)
        if not return_std:
            return mean
        var = np.einsum('ij,jk,ik->i', Phi, self.Sigma, Phi)
        return mean, np.sqrt(np.maximum(var, 0.0)) * self.y_scale

    def is_trained(self):
        return self.mu is not None

    def get_kernel_covariance(self, X):
        Phi = self._std(self._phi(X)) if self.f_mu is not None else self._phi(X)
        return (self.tau2 or 1.0) * Phi @ Phi.T

    def clone(self):
        c = NeuralLinearGPR(self.learner, self.residual)
        for k in ('alpha', 'mu', 'Sigma', 'version', 'X_fit', 'y_fit', 'alpha_fit', 'tau2', 's2',
                  'y_shift', 'y_scale', 'f_mu', 'f_sd'):
            v = getattr(self, k); setattr(c, k, v.copy() if isinstance(v, np.ndarray) else v)
        return c

    def get_kernel(self):
        return (self.tau2, self.s2)

    def set_kernel(self, kernel):
        self.tau2, self.s2 = kernel

    def get_length_scales(self, n_features):
        """Inverse rms gradient of the fitted leaf mean per input dimension: the
        analogue of an ARD length scale, from the network's Jacobian."""
        if self.mu is None or self.learner.net is None or self.X_fit is None:
            return None
        w = self.mu[:-1] / self.f_sd
        g = self.learner.feature_gradients(self.X_fit, w)
        return 1.0 / (np.sqrt((g ** 2).mean(0)) * self.y_scale + 1e-12)


def run_one(target_name, stream, config, seed, d, N, noise_rel=1e-3):
    target = TARGETS[target_name]
    rng = np.random.RandomState(seed)
    X, X_focus = make_stream(stream, target, d, N, rng)
    y_true = target(X.T); sig = np.maximum(noise_rel * np.abs(y_true), 1e-6)
    y = y_true + (sig * rng.randn(N) if noise_rel > 0 else 0.0)
    X_uni = rng.rand(3000, d); y_uni = target(X_uni.T); y_focus = target(X_focus.T)
    yrange = y_uni.max() - y_uni.min()
    warmup = max(500, N // 6)
    np.random.seed(seed)
    learner = None
    if config == 'tree':
        gpt = GPTree(GPR=Default_GPR(n_restarts_optimizer=1), Nbar=100, theta=1e-4,
                     retrain_every_n_points=25, splitting_strategy='gradual', use_calibrated_sigma=True)
    elif config.startswith('nltree') or config.startswith('nlres'):
        residual = config.startswith('nlres')
        nbar = int(config[len('nlres' if residual else 'nltree'):])
        learner = FeatureNet(d, seed=seed)
        gpt = GPTree(GPR=NeuralLinearGPR(learner, residual=residual), Nbar=nbar, theta=1e-4,
                     retrain_every_n_points=25, splitting_strategy='gradual', use_calibrated_sigma=True,
                     use_standard_scaling=False)
    else:
        raise ValueError(config)
    t0 = time.time(); errs = np.empty(N); sds = np.empty(N); errs_net = np.full(N, np.nan)
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(N):
            xi = X[i:i + 1]
            mu, sd = gpt.predict(xi)
            errs[i] = mu[0, 0] - y_true[i]; sds[i] = sd[0, 0]
            if learner is not None:
                if learner.net is not None:
                    errs_net[i] = learner.head_predict(xi)[0] - y_true[i]
                learner.observe(xi, y[i])
            gpt.update_tree(xi, np.array([[y[i]]]), np.array([[sig[i]]]))
        t_stream = time.time() - t0
        P_uni, S_uni = gpt.predict(X_uni, mode='loop')
        P_focus, S_focus = gpt.predict(X_focus, mode='loop')
    elapsed = time.time() - t0
    e_uni = P_uni[:, 0] - y_uni; e_focus = P_focus[:, 0] - y_focus
    out = {
        'target': target_name, 'stream': stream, 'config': config, 'seed': seed, 'd': d, 'N': N,
        'noise_rel': noise_rel,
        'prequential_nrmse': float(np.sqrt(np.mean(errs[warmup:] ** 2)) / yrange),
        'uniform_nrmse': float(np.sqrt(np.mean(e_uni ** 2)) / yrange),
        'focus_nrmse': float(np.sqrt(np.mean(e_focus ** 2)) / yrange),
        'coverage_prequential': float(np.mean(np.abs(errs[warmup:]) <= sds[warmup:])),
        'coverage_uniform': float(np.mean(np.abs(e_uni) <= S_uni[:, 0])),
        'coverage_focus': float(np.mean(np.abs(e_focus) <= S_focus[:, 0])),
        'sigma_over_rmse_prequential': float(np.sqrt(np.mean(sds[warmup:] ** 2)) / np.sqrt(np.mean(errs[warmup:] ** 2))),
        'sigma_over_rmse_uniform': float(np.sqrt(np.mean(S_uni[:, 0] ** 2)) / np.sqrt(np.mean(e_uni ** 2))),
        'sigma_over_rmse_focus': float(np.sqrt(np.mean(S_focus[:, 0] ** 2)) / np.sqrt(np.mean(e_focus ** 2))),
        'leaves': len(gpt.root.leaves), 'seconds': round(elapsed, 1), 'stream_seconds': round(t_stream, 1),
    }
    if learner is not None:
        ok = ~np.isnan(errs_net[warmup:])
        out.update(net_refits=learner.n_refits, net_fit_seconds=round(learner.t_fit, 1),
                   net_prequential_nrmse=float(np.sqrt(np.mean(errs_net[warmup:][ok] ** 2)) / yrange),
                   net_uniform_nrmse=float(np.sqrt(np.mean((learner.head_predict(X_uni) - y_uni) ** 2)) / yrange),
                   net_focus_nrmse=float(np.sqrt(np.mean((learner.head_predict(X_focus) - y_focus) ** 2)) / yrange))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='rotated_rosenbrock'); ap.add_argument('--stream', default='uniform')
    ap.add_argument('--configs', default='tree,nltree100,nltree400'); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--d', type=int, default=10); ap.add_argument('--N', type=int, default=8000)
    ap.add_argument('--noise', type=float, default=1e-3)
    a = ap.parse_args()
    for cfg in a.configs.split(','):
        r = run_one(a.target, a.stream, cfg, a.seed, a.d, a.N, a.noise)
        print('RESULT ' + json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
