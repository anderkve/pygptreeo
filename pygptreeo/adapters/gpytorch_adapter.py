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
    """
    A simple GPyTorch exact GP model.

    This is a basic implementation that can be used when the user doesn't
    provide their own custom GPyTorch model.

    Parameters
    ----------
    train_x : torch.Tensor
        Training inputs.
    train_y : torch.Tensor
        Training targets.
    likelihood : GaussianLikelihood
        The likelihood function.
    mean_module : Mean, optional
        The mean function. If None, uses ConstantMean.
    covar_module : Kernel, optional
        The kernel/covariance function. If None, uses ScaleKernel(RBFKernel).
    """

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
    """
    Adapter for GPyTorch GP models.

    This class wraps a GPyTorch model to provide the interface required
    by pygptreeo. It handles conversion between numpy arrays and PyTorch
    tensors, and manages training with optimization.

    Parameters
    ----------
    model : ExactGP or None
        A GPyTorch ExactGP model instance. If None, a SimpleGPModel will
        be created on first fit.
    likelihood : GaussianLikelihood or None
        The likelihood function. If None, creates a default GaussianLikelihood.
    mean_module : Mean, optional
        Mean function for SimpleGPModel (only used if model is None).
    covar_module : Kernel, optional
        Kernel for SimpleGPModel (only used if model is None).
    optimizer : str, default='adam'
        Optimizer to use for training ('adam' or 'lbfgs').
    learning_rate : float, default=0.1
        Learning rate for the optimizer.
    training_iterations : int, default=50
        Number of optimization iterations during fit().
    device : str, default='cpu'
        Device to use for computation ('cpu' or 'cuda').

    Attributes
    ----------
    _model : ExactGP or None
        The underlying GPyTorch model.
    _likelihood : GaussianLikelihood
        The likelihood function.
    _trained : bool
        Whether the model has been trained.

    Examples
    --------
    >>> import gpytorch
    >>> from pygptreeo.adapters import GPyTorchAdapter
    >>>
    >>> # Create adapter with default model (will be initialized on first fit)
    >>> adapter = GPyTorchAdapter()
    >>>
    >>> # Or create with custom kernel
    >>> kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
    >>> adapter = GPyTorchAdapter(covar_module=kernel, training_iterations=100)
    >>>
    >>> # Use with pygptreeo
    >>> from pygptreeo import GPTree
    >>> gpt = GPTree(GPR=adapter)
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
        device: str = 'cpu'
    ):
        """Initialize the GPyTorch adapter."""
        self._model = model
        self._likelihood = likelihood if likelihood is not None else GaussianLikelihood()
        self._mean_module = mean_module
        self._covar_module = covar_module
        self._optimizer_type = optimizer.lower()
        self._learning_rate = learning_rate
        self._training_iterations = training_iterations
        self._device = device
        self._trained = False
        self._observation_noise = None

        # Move likelihood to device
        self._likelihood = self._likelihood.to(device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GPyTorchAdapter':
        """
        Train the GP on the provided data.

        Parameters
        ----------
        X : np.ndarray
            Training input data of shape (n_samples, n_features).
        y : np.ndarray
            Training target values of shape (n_samples,) or (n_samples, 1).

        Returns
        -------
        self : GPyTorchAdapter
            The fitted adapter instance.
        """
        # Convert numpy arrays to PyTorch tensors
        train_x = torch.from_numpy(X).float().to(self._device)
        train_y = torch.from_numpy(y.flatten()).float().to(self._device)

        # Set observation noise if specified
        if self._observation_noise is not None:
            if isinstance(self._observation_noise, np.ndarray):
                noise = torch.from_numpy(self._observation_noise.flatten()).float().to(self._device)
            else:
                noise = torch.tensor(self._observation_noise).float().to(self._device)
            self._likelihood.noise = noise

        # Create model if it doesn't exist
        if self._model is None:
            self._model = SimpleGPModel(
                train_x, train_y, self._likelihood,
                mean_module=self._mean_module,
                covar_module=self._covar_module
            ).to(self._device)
        else:
            # Update the model's training data
            self._model.set_train_data(train_x, train_y, strict=False)

        # Set to training mode
        self._model.train()
        self._likelihood.train()

        # Create marginal log likelihood
        mll = ExactMarginalLogLikelihood(self._likelihood, self._model)

        # Create optimizer
        if self._optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
        elif self._optimizer_type == 'lbfgs':
            optimizer = torch.optim.LBFGS(self._model.parameters(), lr=self._learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self._optimizer_type}")

        # Training loop
        for i in range(self._training_iterations):
            optimizer.zero_grad()
            output = self._model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        self._trained = True
        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions at test points.

        Parameters
        ----------
        X : np.ndarray
            Test input data of shape (n_samples, n_features).
        return_std : bool, default=False
            If True, return both mean predictions and standard deviations.

        Returns
        -------
        y_mean : np.ndarray
            Mean predictions of shape (n_samples, 1).
        y_std : np.ndarray, optional
            Standard deviations of shape (n_samples, 1).
            Only returned if return_std=True.
        """
        if not self._trained:
            raise RuntimeError("Model must be trained before making predictions")

        # Convert to tensor
        test_x = torch.from_numpy(X).float().to(self._device)

        # Set to evaluation mode
        self._model.eval()
        self._likelihood.eval()

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self._likelihood(self._model(test_x))
            mean = observed_pred.mean.cpu().numpy().reshape(-1, 1)

            if return_std:
                std = observed_pred.stddev.cpu().numpy().reshape(-1, 1)
                return mean, std
            else:
                return mean

    def is_trained(self) -> bool:
        """
        Check whether the GP has been trained on data.

        Returns
        -------
        bool
            True if the GP has been fitted, False otherwise.
        """
        return self._trained

    def set_observation_noise(self, alpha: Union[float, np.ndarray]) -> None:
        """
        Set the observation noise levels for training data.

        Parameters
        ----------
        alpha : float or np.ndarray
            If float: a single noise level applied to all observations.
            If array: per-observation noise levels of shape (n_samples,).
        """
        self._observation_noise = alpha

    def get_kernel_covariance(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the kernel covariance matrix K(X, X).

        Parameters
        ----------
        X : np.ndarray
            Input points of shape (n_samples, n_features).

        Returns
        -------
        K : np.ndarray
            Covariance matrix of shape (n_samples, n_samples).
        """
        if not self._trained:
            raise RuntimeError("Model must be trained before computing kernel covariance")

        test_x = torch.from_numpy(X).float().to(self._device)

        self._model.eval()
        with torch.no_grad():
            # Get the covariance matrix from the model's kernel
            covar = self._model.covar_module(test_x).evaluate()
            return covar.cpu().numpy()

    def clone(self) -> 'GPyTorchAdapter':
        """
        Create a deep copy of this GP regressor.

        Returns
        -------
        GPyTorchAdapter
            A deep copy of this adapter instance.
        """
        # Create a new adapter with the same configuration
        new_adapter = GPyTorchAdapter(
            model=None,  # Will be created on first fit
            likelihood=deepcopy(self._likelihood),
            mean_module=deepcopy(self._mean_module) if self._mean_module is not None else None,
            covar_module=deepcopy(self._covar_module) if self._covar_module is not None else None,
            optimizer=self._optimizer_type,
            learning_rate=self._learning_rate,
            training_iterations=self._training_iterations,
            device=self._device
        )

        # If current model is trained, copy its state
        if self._trained and self._model is not None:
            new_adapter._model = deepcopy(self._model)
            new_adapter._trained = True

        return new_adapter

    def get_kernel(self):
        """
        Get the kernel/covariance module.

        Returns
        -------
        kernel
            The kernel object (gpytorch.kernels.Kernel).
        """
        if self._model is not None:
            return self._model.covar_module
        else:
            return self._covar_module

    def set_kernel(self, kernel) -> None:
        """
        Set the kernel/covariance module.

        This is used for hyperparameter inheritance when creating child nodes.

        Parameters
        ----------
        kernel
            The kernel object to set (gpytorch.kernels.Kernel).
        """
        if self._model is not None:
            self._model.covar_module = kernel
        self._covar_module = kernel

    @property
    def model(self) -> Optional[ExactGP]:
        """
        Access the underlying GPyTorch model.

        Returns
        -------
        ExactGP or None
            The underlying GPyTorch model.
        """
        return self._model

    @property
    def likelihood(self) -> GaussianLikelihood:
        """
        Access the likelihood function.

        Returns
        -------
        GaussianLikelihood
            The likelihood function.
        """
        return self._likelihood

    def __repr__(self) -> str:
        """String representation of the adapter."""
        return f"GPyTorchAdapter(trained={self._trained}, device={self._device})"
