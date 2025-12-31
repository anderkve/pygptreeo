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
    n_restarts_optimizer : int, default=0
        Number of times to restart hyperparameter optimization from random
        initializations. The best result (lowest loss) is kept. Similar to
        sklearn's n_restarts_optimizer.
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
        n_restarts_optimizer: int = 0,
        device: str = 'cpu',
        normalize_y: bool = False
    ):
        """Initialize the GPyTorch adapter."""
        self._model = model

        # Initialize likelihood with better numerical stability
        # Default noise constraint prevents noise from getting too small (1e-6 minimum)
        if likelihood is not None:
            self._likelihood = likelihood
        else:
            from gpytorch.constraints import GreaterThan
            # Use more conservative noise floor for better numerical stability
            self._likelihood = GaussianLikelihood(
                noise_constraint=GreaterThan(1e-4)
            )
            # Initialize noise to a reasonable value
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

        # Y normalization parameters (for numerical stability)
        # Only used if normalize_y=True
        self._normalize_y = normalize_y
        self._y_mean = None
        self._y_std = None

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
        # Use the default dtype (respects torch.set_default_dtype)
        train_x = torch.from_numpy(X).to(self._device)

        # Normalize y values for numerical stability if requested
        y_array = y.flatten()
        if self._normalize_y:
            self._y_mean = y_array.mean()
            self._y_std = y_array.std()
            if self._y_std < 1e-10:  # Avoid division by zero
                self._y_std = 1.0
            y_normalized = (y_array - self._y_mean) / self._y_std
            train_y = torch.from_numpy(y_normalized).to(self._device)
        else:
            # No normalization - use y values as-is
            train_y = torch.from_numpy(y_array).to(self._device)

        # Always create a fresh model AND fresh likelihood for each fit
        # Reusing the likelihood causes numerical issues because LBFGS modifies its parameters
        # Create a completely new likelihood from scratch instead of deepcopy to avoid state issues
        from gpytorch.constraints import GreaterThan
        fresh_likelihood = GaussianLikelihood(
            noise_constraint=GreaterThan(1e-4)
        ).to(self._device)

        # Set observation noise on the fresh likelihood if specified
        # Note: Noise must be scaled if y is normalized
        if self._observation_noise is not None:
            if isinstance(self._observation_noise, np.ndarray):
                noise_array = self._observation_noise.flatten()
                # Scale the noise if y is normalized
                if self._normalize_y and self._y_std is not None:
                    noise_array = noise_array / self._y_std
                noise = torch.from_numpy(noise_array).to(self._device)
                # Ensure noise doesn't get too small (match likelihood constraint)
                noise = torch.clamp(noise, min=1e-4)
            else:
                # Scale scalar noise if y is normalized
                if self._normalize_y and self._y_std is not None:
                    noise = max(float(self._observation_noise) / self._y_std, 1e-4)
                else:
                    noise = max(float(self._observation_noise), 1e-4)
            # Set the noise value
            with torch.no_grad():
                fresh_likelihood.noise = noise
        else:
            # Default noise
            fresh_likelihood.noise = 1e-4

        # Create fresh kernel instances for each fit
        # Reusing the same kernel object across models causes numerical issues
        # Need to recreate the kernel structure from scratch
        if self._covar_module is None:
            fresh_covar = None
        else:
            # TODO: This is a hacky way to create a fresh kernel - we should have a better method
            # For now, just use the original (this works if it's not reused across fits)
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

        # Update our reference to use the fresh likelihood
        self._likelihood = fresh_likelihood

        # Set to training mode
        self._model.train()
        self._likelihood.train()

        # Create marginal log likelihood
        mll = ExactMarginalLogLikelihood(self._likelihood, self._model)

        # Hyperparameter optimization with restarts
        # Similar to sklearn's n_restarts_optimizer
        best_loss = float('inf')
        best_state_dict = None

        n_attempts = self._n_restarts_optimizer + 1  # 1 initial + n restarts

        for attempt in range(n_attempts):
            # Randomize hyperparameters for restarts (skip first attempt)
            if attempt > 0:
                # Re-initialize hyperparameters to random values within bounds
                with torch.no_grad():
                    for param_name, param in self._model.named_parameters():
                        if 'raw_' in param_name or 'noise' in param_name:
                            # Randomize within a reasonable range
                            param.data.uniform_(-2.0, 2.0)

            # Create optimizer for this attempt
            if self._optimizer_type == 'adam':
                optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
            elif self._optimizer_type == 'lbfgs':
                optimizer = torch.optim.LBFGS(self._model.parameters(), lr=self._learning_rate)
            else:
                raise ValueError(f"Unknown optimizer: {self._optimizer_type}")

            # Training loop with numerical stability settings
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
                    # Standard training loop for other optimizers (e.g., Adam)
                    for i in range(self._training_iterations):
                        optimizer.zero_grad()
                        output = self._model(train_x)
                        loss = -mll(output, train_y)
                        loss.backward()
                        optimizer.step()

            # Evaluate final loss for this attempt
            with torch.no_grad():
                output = self._model(train_x)
                final_loss = -mll(output, train_y).item()

            # Keep the best result
            if final_loss < best_loss:
                best_loss = final_loss
                best_state_dict = deepcopy(self._model.state_dict())

        # Restore the best model state
        if best_state_dict is not None:
            self._model.load_state_dict(best_state_dict)

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
        # If not trained, return prior predictions
        if not self._trained:
            n_samples = X.shape[0]
            # If we normalized y, return priors in original scale
            if self._normalize_y and self._y_mean is not None and self._y_std is not None:
                mean = np.full((n_samples, 1), self._y_mean)
                std = np.full((n_samples, 1), self._y_std)
            else:
                # No normalization or not yet fitted - return standard priors
                mean = np.zeros((n_samples, 1))
                std = np.ones((n_samples, 1))

            if return_std:
                return mean, std
            else:
                return mean

        # Convert to tensor (use default dtype)
        test_x = torch.from_numpy(X).to(self._device)

        # Set to evaluation mode
        self._model.eval()
        self._likelihood.eval()

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self._likelihood(self._model(test_x))
            mean = observed_pred.mean.cpu().numpy().reshape(-1, 1)
            std = observed_pred.stddev.cpu().numpy().reshape(-1, 1)

            # Denormalize predictions if we normalized during training
            if self._normalize_y and self._y_mean is not None and self._y_std is not None:
                mean = mean * self._y_std + self._y_mean
                std = std * self._y_std  # std scales but doesn't shift

            if return_std:
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

        test_x = torch.from_numpy(X).to(self._device)

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
            n_restarts_optimizer=self._n_restarts_optimizer,
            device=self._device,
            normalize_y=self._normalize_y
        )

        # If current model is trained, copy its state
        if self._trained and self._model is not None:
            new_adapter._model = deepcopy(self._model)
            new_adapter._trained = True

        # Copy normalization parameters
        new_adapter._y_mean = self._y_mean
        new_adapter._y_std = self._y_std

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
