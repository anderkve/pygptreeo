"""
Adapter for GPyTorch Gaussian Process models.

This module provides an adapter that wraps GPyTorch models to conform
to pygptreeo's GPRegressorInterface. GPyTorch uses PyTorch tensors and
has a different API structure, so this adapter handles the conversions.
"""

from copy import deepcopy
from typing import Union, Tuple, Optional

import numpy as np
import torch
import gpytorch
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import Mean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP

from ..gp_interface import GPRegressorInterface

MIN_NOISE = 1e-4  # Lowest noise variance of the likelihood, for numerical stability


class SimpleGPModel(ExactGP):
    """An exact GP with the given mean and kernel: by default a constant mean and
    a scaled RBF kernel."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: GaussianLikelihood,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module if mean_module is not None else gpytorch.means.ConstantMean()
        self.covar_module = (covar_module if covar_module is not None
                             else gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class GPyTorchAdapter(GPRegressorInterface):
    """Adapter for GPyTorch. Each fit builds a ``SimpleGPModel`` on the training
    data and optimises its hyperparameters.

    Parameters
    ----------
    mean_module : gpytorch.means.Mean, optional
        Mean of the model. A constant mean by default.
    covar_module : gpytorch.kernels.Kernel, optional
        Kernel of the model. A scaled RBF kernel by default. A mean or kernel
        given is used by every fit, which so starts from the hyperparameters of
        the fit before it.
    optimizer : {'adam', 'lbfgs'}, default='adam'
        Optimiser of the hyperparameters.
    learning_rate : float, default=0.1
        Learning rate of the optimiser.
    training_iterations : int, default=50
        Steps of the optimiser in a fit.
    n_restarts_optimizer : int, default=0
        Further optimisations in a fit, from random hyperparameters. The result
        with the lowest loss is kept.
    device : str, default='cpu'
        'cpu' or 'cuda'.
    normalize_y : bool, default=False
        Standardise the targets within the adapter.

    Examples
    --------
    >>> import gpytorch
    >>> from pygptreeo import GPTree
    >>> from pygptreeo.adapters import GPyTorchAdapter
    >>> kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
    >>> gpt = GPTree(GPR=GPyTorchAdapter(covar_module=kernel, training_iterations=100))
    """

    def __init__(
        self,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        optimizer: str = 'adam',
        learning_rate: float = 0.1,
        training_iterations: int = 50,
        n_restarts_optimizer: int = 0,
        device: str = 'cpu',
        normalize_y: bool = False
    ):
        self._mean_module = mean_module
        self._covar_module = covar_module
        self._optimizer_type = optimizer.lower()
        self._learning_rate = learning_rate
        self._training_iterations = training_iterations
        self._n_restarts_optimizer = n_restarts_optimizer
        self._device = device
        self._normalize_y = normalize_y

        # The model and the likelihood of the last fit
        self._model = None
        self._likelihood = None
        self._observation_noise = None
        # Mean and standard deviation of the targets, with normalize_y
        self._y_mean = None
        self._y_std = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GPyTorchAdapter':
        """Build a new model on the data and optimise its hyperparameters. The
        likelihood has one noise variance for all points. It starts from the
        mean of those set with ``set_observation_noise``, not below 1e-4, and
        is optimised with the other hyperparameters."""
        train_x = torch.from_numpy(X).to(self._device)
        y = y.flatten()
        if self._normalize_y:
            self._y_mean = y.mean()
            self._y_std = y.std()
            if self._y_std < 1e-10:  # Avoid division by zero
                self._y_std = 1.0
            y = (y - self._y_mean) / self._y_std
        train_y = torch.from_numpy(y).to(self._device)

        noise = MIN_NOISE
        if self._observation_noise is not None:
            noise = float(np.mean(self._observation_noise))
            if self._normalize_y:
                noise = noise / self._y_std ** 2
            noise = max(noise, MIN_NOISE)
        # A new likelihood for each fit: one that L-BFGS has changed the parameters
        # of gives numerical problems when it is used again
        self._likelihood = GaussianLikelihood(noise_constraint=GreaterThan(MIN_NOISE)).to(self._device)
        self._likelihood.noise = noise
        self._model = SimpleGPModel(train_x, train_y, self._likelihood,
                                    self._mean_module, self._covar_module).to(self._device)
        self._model.train()
        self._likelihood.train()
        mll = ExactMarginalLogLikelihood(self._likelihood, self._model)

        best_loss, best_state = float('inf'), None
        for attempt in range(self._n_restarts_optimizer + 1):
            if attempt > 0:
                # A restart: from random hyperparameters
                with torch.no_grad():
                    for name, param in self._model.named_parameters():
                        if 'raw_' in name or 'noise' in name:
                            param.data.uniform_(-2.0, 2.0)

            if self._optimizer_type == 'adam':
                optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
            elif self._optimizer_type == 'lbfgs':
                optimizer = torch.optim.LBFGS(self._model.parameters(), lr=self._learning_rate)
            else:
                raise ValueError(f"Unknown optimizer: {self._optimizer_type}")

            def closure():
                optimizer.zero_grad()
                loss = -mll(self._model(train_x), train_y)
                loss.backward()
                return loss

            # Jitter for the numerical stability of the Cholesky decomposition
            with gpytorch.settings.cholesky_jitter(1e-3):
                for _ in range(self._training_iterations):
                    optimizer.step(closure)

            with torch.no_grad():
                loss = -mll(self._model(train_x), train_y).item()
            if loss < best_loss:
                best_loss, best_state = loss, deepcopy(self._model.state_dict())

        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """The mean at ``X`` and, with ``return_std``, the standard deviation of an
        observation (the noise included), each of shape (n_samples, 1). Before the
        first fit: the prior."""
        if self._model is None:
            mean, std = np.zeros((X.shape[0], 1)), np.ones((X.shape[0], 1))
        else:
            self._model.eval()
            self._likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred = self._likelihood(self._model(torch.from_numpy(X).to(self._device)))
                mean = pred.mean.cpu().numpy().reshape(-1, 1)
                std = pred.stddev.cpu().numpy().reshape(-1, 1)
            if self._normalize_y:
                mean = mean * self._y_std + self._y_mean
                std = std * self._y_std
        return (mean, std) if return_std else mean

    def is_trained(self) -> bool:
        """Whether ``fit`` has been called."""
        return self._model is not None

    def nbytes(self) -> int:
        """Size of the training data and of the n x n factor that GPyTorch caches
        for predictions."""
        if self._model is None:
            return 0
        train_x = self._model.train_inputs[0]
        n = train_x.shape[0]
        return (train_x.nelement() + n + n * n) * train_x.element_size()

    def set_observation_noise(self, alpha: Union[float, np.ndarray]) -> None:
        """Keep the noise variances of the training points for the next ``fit``."""
        self._observation_noise = alpha

    def clone(self) -> 'GPyTorchAdapter':
        """A copy with the same settings, and with a copy of the fitted model if
        there is one."""
        new_adapter = GPyTorchAdapter(
            mean_module=deepcopy(self._mean_module),
            covar_module=deepcopy(self._covar_module),
            optimizer=self._optimizer_type,
            learning_rate=self._learning_rate,
            training_iterations=self._training_iterations,
            n_restarts_optimizer=self._n_restarts_optimizer,
            device=self._device,
            normalize_y=self._normalize_y
        )
        new_adapter._model = deepcopy(self._model)
        new_adapter._likelihood = deepcopy(self._likelihood)
        new_adapter._y_mean = self._y_mean
        new_adapter._y_std = self._y_std
        return new_adapter

    def get_kernel(self):
        """The kernel of the model, or the one given if there is no model yet."""
        if self._model is not None:
            return self._model.covar_module
        return self._covar_module

    def set_kernel(self, kernel) -> None:
        """Set the kernel of the model and of later fits."""
        if self._model is not None:
            self._model.covar_module = kernel
        self._covar_module = kernel

    @property
    def model(self) -> Optional[ExactGP]:
        """The GPyTorch model of the last fit."""
        return self._model

    @property
    def likelihood(self) -> Optional[GaussianLikelihood]:
        """The likelihood of the last fit."""
        return self._likelihood

    def __repr__(self) -> str:
        return f"GPyTorchAdapter(trained={self.is_trained()}, device={self._device})"
