from copy import deepcopy
from typing import Optional, Tuple, Union
import jax
import jax.numpy as jnp
import numpy as np
import tinygp as tgp

from ..gp_interface import GPRegressorInterface


class TinyGPAdapter(GPRegressorInterface):
    """Minimal TinyGP adapter implementing GPRegressorInterface."""

    def __init__(
        self,
        kernel: tgp.kernels.Kernel,
        alpha: Union[float, np.ndarray] = 1e-6,
    ):
        self._kernel = kernel
        self.set_observation_noise(alpha)
        self._gp: Optional[tgp.GaussianProcess] = None
        self._y_train: Optional[jnp.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TinyGPAdapter":
        X_arr = np.atleast_2d(X)
        self._y_train = jnp.asarray(y).ravel()
        self._gp = tgp.GaussianProcess(
            self._kernel,
            jnp.asarray(X_arr),
            diag=self._alpha,
        )
        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        X_arr = np.atleast_2d(X)
        n_samples = X_arr.shape[0]

        # Prior evaluation when not fitted yet
        if not self.is_trained():
            y_mean = np.zeros(n_samples)
            if not return_std:
                return y_mean

            X_jax = jnp.asarray(X_arr)
            var = jax.vmap(lambda x: self._kernel.evaluate(x, x))(X_jax)
            y_std = np.sqrt(np.maximum(np.asarray(var).ravel(), 0.0))
            return y_mean, y_std

        # Posterior evaluation
        _, cond_gp = self._gp.condition(self._y_train, jnp.asarray(X_arr))
        y_mean = np.asarray(cond_gp.loc).ravel()

        if not return_std:
            return y_mean

        y_std = np.sqrt(np.maximum(np.asarray(cond_gp.variance).ravel(), 0.0))
        return y_mean, y_std

    def is_trained(self) -> bool:
        return self._gp is not None

    def set_observation_noise(self, alpha: Union[float, np.ndarray]) -> None:
        arr = np.asarray(alpha, dtype=float)
        if arr.ndim == 0:
            self._alpha = float(arr)
        else:
            self._alpha = arr.ravel()

    def get_kernel_covariance(self, X: np.ndarray) -> np.ndarray:
        X_jax = jnp.atleast_2d(jnp.asarray(X))
        cov = jax.vmap(lambda x1: jax.vmap(lambda x2: self._kernel.evaluate(x1, x2))(X_jax))(X_jax)
        return np.asarray(cov)

    def clone(self) -> "TinyGPAdapter":
        return TinyGPAdapter(
            kernel=deepcopy(self._kernel),
            alpha=deepcopy(self._alpha),
        )

    def get_kernel(self) -> tgp.kernels.Kernel:
        return self._kernel

    def set_kernel(self, kernel: tgp.kernels.Kernel) -> None:
        self._kernel = kernel