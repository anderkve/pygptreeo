"""
Adapter for GPyTorch Gaussian Process models.

This module provides an adapter that wraps GPyTorch models to conform
to pygptreeo's GPRegressorInterface. GPyTorch uses PyTorch tensors and
has a different API structure, so this adapter handles the conversions.
"""

from typing import Union, Tuple, Optional
import numpy as np
from copy import deepcopy

try:
    import torch
    import gpytorch
    from gpytorch.models import ExactGP
    from gpytorch.means import Mean
    from gpytorch.kernels import Kernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.mlls import ExactMarginalLogLikelihood
    GPYTORCH_AVAILABLE = True
except ImportError:
    GPYTORCH_AVAILABLE = False
    # Define dummy classes for type hints when GPyTorch is not available
    ExactGP = object
    Mean = object
    Kernel = object
    GaussianLikelihood = object

from ..gp_interface import GPRegressorInterface


if not GPYTORCH_AVAILABLE:
    raise ImportError(
        "GPyTorch is not installed. Please install it with: pip install gpytorch"
    )


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
        super(SimpleGPModel, self).__init__(train_x, train_y, likelihood)

        if mean_module is None:
            self.mean_module = gpytorch.means.ConstantMean()
        else:
            self.mean_module = mean_module

        if covar_module is None:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )
        else:
            self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPyTorchAdapter(GPRegressorInterface):
    """Adapter for GPyTorch. Each fit builds a ``SimpleGPModel`` on the training
    data and optimises its hyperparameters.

    Parameters
    ----------
    model, likelihood : optional
        Kept until the first fit, which replaces them: a fit builds a new model
        and a new ``GaussianLikelihood``.
    mean_module : gpytorch.means.Mean, optional
        Mean of the model. A constant mean by default.
    covar_module : gpytorch.kernels.Kernel, optional
        Kernel of the model. A scaled RBF kernel by default.
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
        model: Optional[ExactGP] = None,
        likelihood: Optional[GaussianLikelihood] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        optimizer: str = 'adam',
        learning_rate: float = 0.1,
        training_iterations: int = 50,
        n_restarts_optimizer: int = 0,
        device: str = 'cpu',
        normalize_y: bool = False
    ):
        self._model = model

        # By default the noise cannot fall below 1e-4, for numerical stability
        if likelihood is not None:
            self._likelihood = likelihood
        else:
            from gpytorch.constraints import GreaterThan
            self._likelihood = GaussianLikelihood(
                noise_constraint=GreaterThan(1e-4)
            )
            self._likelihood.noise = 1e-4

        self._mean_module = mean_module
        self._covar_module = covar_module
        self._optimizer_type = optimizer.lower()
        self._learning_rate = learning_rate
        self._training_iterations = training_iterations
        self._n_restarts_optimizer = n_restarts_optimizer
        self._device = device
        self._trained = False
        self._observation_noise = None

        # Mean and standard deviation of the targets, with normalize_y
        self._normalize_y = normalize_y
        self._y_mean = None
        self._y_std = None

        self._likelihood = self._likelihood.to(device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GPyTorchAdapter':
        """Build a new model on the data and optimise its hyperparameters. The
        likelihood has one noise variance for all points. It starts from the
        mean of those set with ``set_observation_noise``, not below 1e-4, and
        is optimised with the other hyperparameters."""
        train_x = torch.from_numpy(X).to(self._device)

        y_array = y.flatten()
        if self._normalize_y:
            self._y_mean = y_array.mean()
            self._y_std = y_array.std()
            if self._y_std < 1e-10:  # Avoid division by zero
                self._y_std = 1.0
            y_normalized = (y_array - self._y_mean) / self._y_std
            train_y = torch.from_numpy(y_normalized).to(self._device)
        else:
            train_y = torch.from_numpy(y_array).to(self._device)

        # A new likelihood for each fit: one that L-BFGS has changed the parameters
        # of gives numerical problems when it is used again
        from gpytorch.constraints import GreaterThan
        fresh_likelihood = GaussianLikelihood(
            noise_constraint=GreaterThan(1e-4)
        ).to(self._device)

        if self._observation_noise is not None:
            # The default GaussianLikelihood is homoskedastic (one noise variance for
            # all points), so per-point noise variances (as handed over by GPTree)
            # are reduced to their mean.
            if isinstance(self._observation_noise, np.ndarray):
                obs_noise = float(np.mean(self._observation_noise))
            else:
                obs_noise = float(self._observation_noise)
            # Scale the noise *variance* if y is normalized
            if self._normalize_y and self._y_std is not None:
                obs_noise = obs_noise / (self._y_std ** 2)
            # Ensure noise doesn't get too small (match likelihood constraint)
            noise = max(obs_noise, 1e-4)
            with torch.no_grad():
                fresh_likelihood.noise = noise
        else:
            fresh_likelihood.noise = 1e-4

        # TODO: the kernel and mean modules given are used again by every fit;
        # each fit should have new ones
        if self._covar_module is None:
            fresh_covar = None
        else:
            fresh_covar = self._covar_module

        if self._mean_module is None:
            fresh_mean = None
        else:
            fresh_mean = self._mean_module

        self._model = SimpleGPModel(
            train_x, train_y, fresh_likelihood,
            mean_module=fresh_mean,
            covar_module=fresh_covar
        ).to(self._device)

        self._likelihood = fresh_likelihood

        self._model.train()
        self._likelihood.train()

        mll = ExactMarginalLogLikelihood(self._likelihood, self._model)

        best_loss = float('inf')
        best_state_dict = None

        n_attempts = self._n_restarts_optimizer + 1  # 1 initial + n restarts

        for attempt in range(n_attempts):
            # Randomize hyperparameters for restarts (skip first attempt)
            if attempt > 0:
                with torch.no_grad():
                    for param_name, param in self._model.named_parameters():
                        if 'raw_' in param_name or 'noise' in param_name:
                            param.data.uniform_(-2.0, 2.0)

            if self._optimizer_type == 'adam':
                optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
            elif self._optimizer_type == 'lbfgs':
                optimizer = torch.optim.LBFGS(self._model.parameters(), lr=self._learning_rate)
            else:
                raise ValueError(f"Unknown optimizer: {self._optimizer_type}")

            # Add jitter for numerical stability in Cholesky decomposition
            with gpytorch.settings.cholesky_jitter(1e-3):
                if self._optimizer_type == 'lbfgs':
                    # LBFGS requires a closure function
                    def closure():
                        optimizer.zero_grad()
                        output = self._model(train_x)
                        loss = -mll(output, train_y)
                        loss.backward()
                        return loss

                    for i in range(self._training_iterations):
                        loss = optimizer.step(closure)
                else:
                    for i in range(self._training_iterations):
                        optimizer.zero_grad()
                        output = self._model(train_x)
                        loss = -mll(output, train_y)
                        loss.backward()
                        optimizer.step()

            with torch.no_grad():
                output = self._model(train_x)
                final_loss = -mll(output, train_y).item()

            if final_loss < best_loss:
                best_loss = final_loss
                best_state_dict = deepcopy(self._model.state_dict())

        if best_state_dict is not None:
            self._model.load_state_dict(best_state_dict)

        self._trained = True
        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """The mean at ``X`` and, with ``return_std``, the standard deviation of an
        observation (the noise included), each of shape (n_samples, 1). Before the
        first fit: the prior."""
        if not self._trained:
            n_samples = X.shape[0]
            if self._normalize_y and self._y_mean is not None and self._y_std is not None:
                mean = np.full((n_samples, 1), self._y_mean)
                std = np.full((n_samples, 1), self._y_std)
            else:
                mean = np.zeros((n_samples, 1))
                std = np.ones((n_samples, 1))

            if return_std:
                return mean, std
            else:
                return mean

        test_x = torch.from_numpy(X).to(self._device)

        self._model.eval()
        self._likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self._likelihood(self._model(test_x))
            mean = observed_pred.mean.cpu().numpy().reshape(-1, 1)
            std = observed_pred.stddev.cpu().numpy().reshape(-1, 1)

            if self._normalize_y and self._y_mean is not None and self._y_std is not None:
                mean = mean * self._y_std + self._y_mean
                std = std * self._y_std  # std scales but doesn't shift

            if return_std:
                return mean, std
            else:
                return mean

    def is_trained(self) -> bool:
        """Whether ``fit`` has been called."""
        return self._trained

    def nbytes(self) -> int:
        """Size of the training data and of the n x n factor that GPyTorch caches
        for predictions."""
        if not self._trained:
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
            model=None,  # Will be created on first fit
            likelihood=deepcopy(self._likelihood),
            mean_module=deepcopy(self._mean_module) if self._mean_module is not None else None,
            covar_module=deepcopy(self._covar_module) if self._covar_module is not None else None,
            optimizer=self._optimizer_type,
            learning_rate=self._learning_rate,
            training_iterations=self._training_iterations,
            n_restarts_optimizer=self._n_restarts_optimizer,
            device=self._device,
            normalize_y=self._normalize_y
        )

        if self._trained and self._model is not None:
            new_adapter._model = deepcopy(self._model)
            new_adapter._trained = True

        new_adapter._y_mean = self._y_mean
        new_adapter._y_std = self._y_std

        return new_adapter

    def get_kernel(self):
        """The kernel of the model, or the one given if there is no model yet."""
        if self._model is not None:
            return self._model.covar_module
        else:
            return self._covar_module

    def set_kernel(self, kernel) -> None:
        """Set the kernel of the model and of later fits."""
        if self._model is not None:
            self._model.covar_module = kernel
        self._covar_module = kernel

    @property
    def model(self) -> Optional[ExactGP]:
        """The GPyTorch model: that of the last fit."""
        return self._model

    @property
    def likelihood(self) -> GaussianLikelihood:
        """The likelihood: that of the last fit."""
        return self._likelihood

    def __repr__(self) -> str:
        return f"GPyTorchAdapter(trained={self._trained}, device={self._device})"
