"""Streaming sparse variational GP adapter built on GPyTorch.

Sparse variational GPs (SVGPs) with inducing points are the de-facto scalable
GP technique and can be trained with mini-batches, which makes them a natural
candidate for continual-emulation benchmarks. Each ``update`` appends the new
point to an internal replay buffer. Every ``retrain_every`` points the adapter
runs ``n_epochs`` passes of stochastic variational inference over a bounded
window of recent data. Inducing points are re-initialised by a k-means-like
subsample from the current buffer whenever the buffer grows meaningfully.
"""

from __future__ import annotations

import numpy as np
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)

from .base import OnlineRegressor


class _SVGPModel(ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=1.5, ard_num_dims=inducing_points.size(1)
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPyTorchSVGPAdapter(OnlineRegressor):
    name = "gpytorch_svgp"
    supports_uncertainty = True

    def __init__(self, n_dims: int, n_inducing: int = 256,
                 retrain_every: int = 200, n_epochs: int = 60,
                 batch_size: int = 128, max_buffer: int = 5000,
                 lr: float = 5e-3, device: str = "cpu",
                 max_steps_per_refit: int = 500):
        self.n_dims = n_dims
        self.n_inducing = n_inducing
        self.retrain_every = retrain_every
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_buffer = max_buffer
        self.lr = lr
        self.max_steps_per_refit = max_steps_per_refit
        self.device = torch.device(device)
        self._X_buf: list[np.ndarray] = []
        self._y_buf: list[float] = []
        self._steps_since_refit = 0
        self._model: _SVGPModel | None = None
        self._likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        self._trained = False
        self._rng = np.random.default_rng(0)

    def _init_model(self, X):
        # Initialise inducing points as random subsample of X.
        n = X.shape[0]
        m = min(self.n_inducing, n)
        idx = self._rng.choice(n, size=m, replace=False)
        Z = torch.tensor(X[idx], dtype=torch.float32, device=self.device)
        self._model = _SVGPModel(Z).to(self.device)
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

    def _refit(self):
        n = len(self._X_buf)
        if n < 10:
            return
        X = np.vstack(self._X_buf)
        y = np.asarray(self._y_buf, dtype=np.float32)

        if n > self.max_buffer:
            # Keep recent half + random older points for stability.
            n_recent = self.max_buffer // 2
            n_old = self.max_buffer - n_recent
            idx_old = self._rng.choice(n - n_recent, size=n_old, replace=False)
            idx = np.concatenate([idx_old, np.arange(n - n_recent, n)])
            X = X[idx]
            y = y[idx]

        y_mean = float(y.mean())
        y_std = float(y.std() + 1e-8)
        y_norm = (y - y_mean) / y_std
        self._y_mean = y_mean
        self._y_std = y_std

        if self._model is None:
            self._init_model(X)

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_norm, dtype=torch.float32, device=self.device)

        self._model.train()
        self._likelihood.train()
        optimizer = torch.optim.Adam(
            list(self._model.parameters()) + list(self._likelihood.parameters()),
            lr=self.lr,
        )
        mll = gpytorch.mlls.VariationalELBO(
            self._likelihood, self._model, num_data=X_t.size(0)
        )

        n_data = X_t.size(0)
        bs = min(self.batch_size, n_data)
        total_steps = 0
        budget = self.max_steps_per_refit
        for _ in range(self.n_epochs):
            if total_steps >= budget:
                break
            perm = torch.randperm(n_data)
            for i in range(0, n_data, bs):
                if total_steps >= budget:
                    break
                j = perm[i:i + bs]
                optimizer.zero_grad()
                out = self._model(X_t[j])
                loss = -mll(out, y_t[j])
                loss.backward()
                optimizer.step()
                total_steps += 1
        self._trained = True

    def predict(self, X):
        if not self._trained:
            return (np.zeros((X.shape[0], 1)),
                    np.full((X.shape[0], 1), np.nan))
        self._model.eval()
        self._likelihood.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self._likelihood(self._model(X_t))
            mean = pred.mean.cpu().numpy().reshape(-1, 1)
            std = pred.stddev.cpu().numpy().reshape(-1, 1)
        mean = mean * self._y_std + self._y_mean
        std = std * self._y_std
        return mean, std

    def update(self, x, y):
        self._X_buf.append(x.reshape(1, -1).astype(np.float32).copy())
        self._y_buf.append(float(y.ravel()[0]))
        self._steps_since_refit += 1
        if self._steps_since_refit >= self.retrain_every:
            self._refit()
            self._steps_since_refit = 0
