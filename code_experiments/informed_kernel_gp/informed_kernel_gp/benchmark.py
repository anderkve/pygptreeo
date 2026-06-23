"""Benchmark utilities for comparing GP kernel performance."""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from .kernels import ARDDotProductFeatureKernel


def compute_rmse(y_true, y_pred):
    """Root mean squared error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_mlpd(y_true, y_pred_mean, y_pred_std):
    """Mean log predictive density under a Gaussian predictive distribution.

    MLPD = (1/N) Σ log N(y_i | μ_i, σ_i²)
    """
    variance = y_pred_std ** 2
    # Clamp variance to avoid log(0)
    variance = np.maximum(variance, 1e-10)
    log_densities = (
        -0.5 * np.log(2 * np.pi * variance)
        - 0.5 * (y_true - y_pred_mean) ** 2 / variance
    )
    return np.mean(log_densities)


def _phi_constant(X):
    """Constant basis function (intercept)."""
    return np.ones(X.shape[0])


def build_kernels(basis_functions, rbf_length_scale=1.0, rbf_length_scale_bounds=(1e-2, 1e2)):
    """Build the kernel variants for benchmarking.

    A constant basis function is automatically prepended to the basis set
    so that all specific kernels can capture a non-zero mean.

    Parameters
    ----------
    basis_functions : list of callable
        Basis functions for the specific kernels (shared-amplitude and ARD).
    rbf_length_scale : float
    rbf_length_scale_bounds : tuple

    Returns
    -------
    kernels : dict
    """
    basis_functions = [_phi_constant] + list(basis_functions)

    def _rbf():
        return RBF(length_scale=rbf_length_scale, length_scale_bounds=rbf_length_scale_bounds)

    def _const():
        return ConstantKernel(1.0, (1e-3, 1e3))

    def _white():
        return WhiteKernel(1e-5, (1e-10, 1e-1))

    kernels = {
        # k = k_specific_ard + noise  (ARD has per-feature amplitudes)
        "k_specific_ard": ARDDotProductFeatureKernel(basis_functions) + _white(),
        # k = k_specific_ard + α² · k_RBF + noise
        "k_total_ard": (
            ARDDotProductFeatureKernel(basis_functions)
            + _const() * _rbf()
            + _white()
        ),
        # k = k_specific_ard · k_RBF + α² · k_RBF + noise
        "k_product_ard": (
            ARDDotProductFeatureKernel(basis_functions) * _rbf()
            + _const() * _rbf()
            + _white()
        ),
        # k = α² · k_RBF + noise
        "k_rbf": _const() * _rbf() + _white(),
    }

    return kernels


def run_single_trial(X_train, y_train, X_test, y_test, kernels, random_state=0):
    """Fit all kernel variants on one training set and evaluate on the test set.

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test : test data
    kernels : dict of kernel objects (as returned by build_kernels)
    random_state : int

    Returns
    -------
    results : dict
        Keys are kernel names; values are dicts with 'rmse', 'mlpd', 'gp'.
    """
    # Pure specific kernel (dot-product only) has zero prior mean, so
    # normalize_y distorts the fit when the true function has non-zero mean.
    # Use normalize_y only for kernels that include an RBF component.
    _pure_specific = {"k_specific_ard"}

    results = {}
    for name, kernel in kernels.items():
        use_normalize = name not in _pure_specific
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=random_state,
            normalize_y=use_normalize,
        )
        gp.fit(X_train, y_train)
        y_pred, y_std = gp.predict(X_test, return_std=True)

        rmse = compute_rmse(y_test, y_pred)
        mlpd = compute_mlpd(y_test, y_pred, y_std)

        results[name] = {"rmse": rmse, "mlpd": mlpd, "gp": gp}

    return results


def run_benchmark(
    true_function,
    basis_functions,
    domain_bounds,
    n_train_values,
    n_trials=20,
    n_test_per_dim=50,
    noise_std=0.0,
    random_seed=42,
):
    """Run the full benchmark across training set sizes and trials.

    Parameters
    ----------
    true_function : callable
        Takes (n_samples, n_features) array, returns (n_samples,) array.
    basis_functions : list of callable
        Basis functions for the specific kernel.
    domain_bounds : list of (low, high) tuples
        Bounds for each input dimension.
    n_train_values : list of int
        Training set sizes to evaluate.
    n_trials : int
        Number of random trials per training set size.
    n_test_per_dim : int
        Number of test points per dimension (total test = n_test_per_dim^n_dim).
    noise_std : float
        Standard deviation of Gaussian noise added to training targets.
    random_seed : int

    Returns
    -------
    results : dict
        Nested dict: results[n_train][kernel_name] = dict with arrays of
        'rmse', 'mlpd' (length n_trials).
    X_test, y_test : test grid arrays
    """
    rng = np.random.RandomState(random_seed)
    n_dim = len(domain_bounds)

    # Build test grid
    grids = [
        np.linspace(lo, hi, n_test_per_dim) for lo, hi in domain_bounds
    ]
    mesh = np.meshgrid(*grids, indexing="ij")
    X_test = np.column_stack([m.ravel() for m in mesh])
    y_test = true_function(X_test)

    all_results = {}

    for n_train in n_train_values:
        kernel_names = list(build_kernels(basis_functions).keys())
        trial_results = {name: [] for name in kernel_names}

        for trial in range(n_trials):
            # Sample training points uniformly
            X_train = np.column_stack([
                rng.uniform(lo, hi, size=n_train)
                for lo, hi in domain_bounds
            ])
            y_train = true_function(X_train)
            if noise_std > 0:
                y_train = y_train + rng.normal(0, noise_std, size=n_train)

            kernels = build_kernels(basis_functions)
            results = run_single_trial(
                X_train, y_train, X_test, y_test, kernels,
                random_state=trial,
            )

            for name in trial_results:
                trial_results[name].append(results[name])

        # Aggregate
        aggregated = {}
        for name in trial_results:
            aggregated[name] = {
                "rmse": np.array([r["rmse"] for r in trial_results[name]]),
                "mlpd": np.array([r["mlpd"] for r in trial_results[name]]),
            }

        all_results[n_train] = aggregated

    return all_results, X_test, y_test
