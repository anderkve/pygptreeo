"""Custom kernels for GP regression using explicit feature maps."""

import numpy as np
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter
from sklearn.base import clone


class DotProductFeatureKernel(Kernel):
    """Kernel defined by an explicit feature map: k(x, x') = φ(x)ᵀ φ(x').

    Given a list of basis functions [φ₁, φ₂, ..., φ_d], this kernel computes
    the Gram matrix as the dot product in the feature space spanned by those
    basis functions.

    Parameters
    ----------
    basis_functions : list of callable
        Each callable takes an array of shape (n_samples, n_features) and
        returns an array of shape (n_samples,).
    """

    def __init__(self, basis_functions):
        self.basis_functions = basis_functions

    @property
    def hyperparameter_constant(self):
        # No free hyperparameters in this kernel itself; amplitude is handled
        # externally via ConstantKernel.
        return Hyperparameter("constant", "fixed", 0.0)

    @property
    def n_dims(self):
        return 0

    @property
    def hyperparameters(self):
        return []

    @property
    def theta(self):
        return np.array([])

    @theta.setter
    def theta(self, value):
        pass

    @property
    def bounds(self):
        return np.empty((0, 2))

    def _compute_features(self, X):
        """Compute the feature matrix Φ of shape (n_samples, n_basis)."""
        cols = [bf(X) for bf in self.basis_functions]
        return np.column_stack(cols)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Compute the kernel matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features)
        Y : array-like of shape (n_samples_Y, n_features), optional
        eval_gradient : bool
            Not supported (no free hyperparameters).

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
        """
        phi_X = self._compute_features(X)
        if Y is None:
            K = phi_X @ phi_X.T
        else:
            phi_Y = self._compute_features(Y)
            K = phi_X @ phi_Y.T

        if eval_gradient:
            # No hyperparameters, so gradient is empty.
            return K, np.empty((X.shape[0], X.shape[0], 0))
        return K

    def diag(self, X):
        """Return the diagonal of the kernel matrix."""
        phi_X = self._compute_features(X)
        return np.sum(phi_X ** 2, axis=1)

    def is_stationary(self):
        return False

    def __repr__(self):
        return f"DotProductFeatureKernel(n_basis={len(self.basis_functions)})"

    def clone_with_theta(self, theta):
        """Return a clone of this kernel (no hyperparameters to update)."""
        cloned = DotProductFeatureKernel(
            basis_functions=self.basis_functions,
        )
        return cloned

    def get_params(self, deep=True):
        return {"basis_functions": self.basis_functions}


class ARDDotProductFeatureKernel(Kernel):
    """Dot-product feature kernel with per-feature amplitudes (ARD).

    k(x, x') = Σᵢ αᵢ² φᵢ(x) φᵢ(x')

    Each basis function gets its own amplitude parameter αᵢ, optimised via
    marginal-likelihood maximisation.  This is equivalent to Bayesian linear
    regression with independent prior variances on the weights.

    Parameters
    ----------
    basis_functions : list of callable
        Each callable takes (n_samples, n_features) and returns (n_samples,).
    log_amplitudes : array-like of shape (n_basis,) or None
        Initial log-amplitudes (ln αᵢ).  Defaults to 0 (αᵢ = 1) for each.
    amplitude_bounds : pair of floats
        Bounds on each αᵢ (not log), applied identically to every component.
    """

    def __init__(self, basis_functions, log_amplitudes=None,
                 amplitude_bounds=(1e-5, 1e5)):
        self.basis_functions = basis_functions
        n = len(basis_functions)
        if log_amplitudes is None:
            self.log_amplitudes = np.zeros(n)
        else:
            self.log_amplitudes = np.asarray(log_amplitudes, dtype=float)
        self.amplitude_bounds = amplitude_bounds

    # -- sklearn hyperparameter plumbing ----------------------------------

    @property
    def n_dims(self):
        return len(self.basis_functions)

    @property
    def hyperparameters(self):
        return [Hyperparameter(f"log_amplitudes_{i}", "numeric",
                               (np.log(self.amplitude_bounds[0]),
                                np.log(self.amplitude_bounds[1])))
                for i in range(len(self.basis_functions))]

    @property
    def theta(self):
        return self.log_amplitudes.copy()

    @theta.setter
    def theta(self, value):
        self.log_amplitudes = np.asarray(value, dtype=float)

    @property
    def bounds(self):
        lb = np.log(self.amplitude_bounds[0])
        ub = np.log(self.amplitude_bounds[1])
        return np.array([[lb, ub]] * len(self.basis_functions))

    # -- kernel computation -----------------------------------------------

    def _compute_features(self, X):
        return np.column_stack([bf(X) for bf in self.basis_functions])

    def __call__(self, X, Y=None, eval_gradient=False):
        phi_X = self._compute_features(X)
        alphas = np.exp(self.log_amplitudes)            # (d,)
        phi_X_scaled = phi_X * alphas[np.newaxis, :]    # scale columns

        if Y is None:
            phi_Y_scaled = phi_X_scaled
        else:
            phi_Y = self._compute_features(Y)
            phi_Y_scaled = phi_Y * alphas[np.newaxis, :]

        K = phi_X_scaled @ phi_Y_scaled.T

        if eval_gradient:
            # ∂K/∂(ln αᵢ) = 2 αᵢ² φᵢ(x) φᵢ(x')   (chain rule on exp)
            n_X = phi_X.shape[0]
            n_Y = phi_Y_scaled.shape[0]
            d = len(self.basis_functions)
            K_grad = np.empty((n_X, n_Y, d))
            for i in range(d):
                # outer product of the i-th feature, times 2 αᵢ²
                K_grad[:, :, i] = 2.0 * alphas[i] ** 2 * np.outer(
                    phi_X[:, i], phi_Y_scaled[:, i] / alphas[i]
                )
                # Simplifies to: 2 αᵢ² φᵢ(x) φᵢ(x')
            return K, K_grad
        return K

    def diag(self, X):
        phi_X = self._compute_features(X)
        alphas = np.exp(self.log_amplitudes)
        return np.sum((phi_X * alphas[np.newaxis, :]) ** 2, axis=1)

    def is_stationary(self):
        return False

    def __repr__(self):
        amps = np.exp(self.log_amplitudes)
        amp_str = ", ".join(f"{a:.3g}" for a in amps)
        return f"ARDDotProductFeatureKernel(amplitudes=[{amp_str}])"

    def clone_with_theta(self, theta):
        cloned = ARDDotProductFeatureKernel(
            basis_functions=self.basis_functions,
            log_amplitudes=theta,
            amplitude_bounds=self.amplitude_bounds,
        )
        return cloned

    def get_params(self, deep=True):
        return {
            "basis_functions": self.basis_functions,
            "log_amplitudes": self.log_amplitudes,
            "amplitude_bounds": self.amplitude_bounds,
        }
