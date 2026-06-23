"""Tests for the DotProductFeatureKernel and benchmark utilities."""

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from informed_kernel_gp.kernels import DotProductFeatureKernel, ARDDotProductFeatureKernel
from informed_kernel_gp.benchmark import (
    compute_rmse,
    compute_mlpd,
    build_kernels,
    run_single_trial,
)


# --- Fixtures ---

def _simple_basis():
    """Two simple basis functions for testing: x₁² and sin(x₂)."""
    return [
        lambda X: X[:, 0] ** 2,
        lambda X: np.sin(X[:, 1]),
    ]


def _make_data(n=20, rng_seed=0):
    rng = np.random.RandomState(rng_seed)
    X = rng.uniform(-2, 2, size=(n, 2))
    y = X[:, 0] ** 2 + np.sin(X[:, 1]) + rng.normal(0, 0.01, size=n)
    return X, y


# --- Kernel Tests ---

class TestDotProductFeatureKernel:

    def test_kernel_shape(self):
        """Kernel matrix has correct shape."""
        basis = _simple_basis()
        kernel = DotProductFeatureKernel(basis)
        X = np.random.RandomState(0).randn(10, 2)
        K = kernel(X)
        assert K.shape == (10, 10)

    def test_kernel_shape_two_inputs(self):
        """Kernel matrix K(X, Y) has correct shape."""
        basis = _simple_basis()
        kernel = DotProductFeatureKernel(basis)
        rng = np.random.RandomState(0)
        X = rng.randn(10, 2)
        Y = rng.randn(7, 2)
        K = kernel(X, Y)
        assert K.shape == (10, 7)

    def test_kernel_symmetry(self):
        """Kernel matrix is symmetric."""
        basis = _simple_basis()
        kernel = DotProductFeatureKernel(basis)
        X = np.random.RandomState(0).randn(15, 2)
        K = kernel(X)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_kernel_positive_semidefinite(self):
        """Kernel matrix is positive semi-definite."""
        basis = _simple_basis()
        kernel = DotProductFeatureKernel(basis)
        X = np.random.RandomState(0).randn(15, 2)
        K = kernel(X)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-10)

    def test_kernel_equals_manual_dot_product(self):
        """Kernel matrix equals Φ Φᵀ computed manually."""
        basis = _simple_basis()
        kernel = DotProductFeatureKernel(basis)
        X = np.random.RandomState(0).randn(8, 2)
        K = kernel(X)
        phi = np.column_stack([bf(X) for bf in basis])
        K_manual = phi @ phi.T
        np.testing.assert_allclose(K, K_manual, atol=1e-12)

    def test_diag(self):
        """diag() matches the diagonal of the full kernel matrix."""
        basis = _simple_basis()
        kernel = DotProductFeatureKernel(basis)
        X = np.random.RandomState(0).randn(12, 2)
        K = kernel(X)
        diag = kernel.diag(X)
        np.testing.assert_allclose(diag, np.diag(K), atol=1e-12)

    def test_eval_gradient_shape(self):
        """eval_gradient returns empty gradient (no hyperparameters)."""
        basis = _simple_basis()
        kernel = DotProductFeatureKernel(basis)
        X = np.random.RandomState(0).randn(5, 2)
        K, K_grad = kernel(X, eval_gradient=True)
        assert K.shape == (5, 5)
        assert K_grad.shape == (5, 5, 0)

    def test_no_hyperparameters(self):
        """Kernel has no free hyperparameters."""
        basis = _simple_basis()
        kernel = DotProductFeatureKernel(basis)
        assert len(kernel.theta) == 0
        assert kernel.bounds.shape == (0, 2)

    def test_clone_with_theta(self):
        """clone_with_theta returns a functional clone."""
        basis = _simple_basis()
        kernel = DotProductFeatureKernel(basis)
        clone = kernel.clone_with_theta(np.array([]))
        X = np.random.RandomState(0).randn(5, 2)
        np.testing.assert_allclose(kernel(X), clone(X))

    def test_repr(self):
        basis = _simple_basis()
        kernel = DotProductFeatureKernel(basis)
        assert "n_basis=2" in repr(kernel)


# --- GP Integration Tests ---

class TestGPIntegration:

    def test_gp_fit_with_specific_kernel(self):
        """GP can fit and predict with DotProductFeatureKernel."""
        X, y = _make_data()
        basis = _simple_basis()
        kernel = ConstantKernel(1.0) * DotProductFeatureKernel(basis) + WhiteKernel(1e-5)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                       random_state=0, normalize_y=True)
        gp.fit(X, y)
        y_pred, y_std = gp.predict(X, return_std=True)
        # Sanity check: predictions should be finite and not wildly off
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        assert rmse < 2.0, f"RMSE too large: {rmse}"
        assert np.all(np.isfinite(y_pred))

    def test_gp_fit_with_combined_kernel(self):
        """GP can fit and predict with k_total = k_specific + k_RBF."""
        X, y = _make_data()
        basis = _simple_basis()
        kernel = (
            ConstantKernel(1.0) * DotProductFeatureKernel(basis)
            + ConstantKernel(1.0) * RBF(1.0)
            + WhiteKernel(1e-5)
        )
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                       random_state=0, normalize_y=True)
        gp.fit(X, y)
        y_pred = gp.predict(X)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        assert rmse < 0.5, f"RMSE too large: {rmse}"


# --- Metric Tests ---

class TestMetrics:

    def test_rmse_zero_for_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(y, y) == pytest.approx(0.0)

    def test_rmse_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        expected = np.sqrt(1.0 / 3)
        assert compute_rmse(y_true, y_pred) == pytest.approx(expected, rel=1e-6)

    def test_mlpd_higher_for_better_prediction(self):
        """MLPD is higher when predictions are closer to truth."""
        y_true = np.array([1.0, 2.0, 3.0])
        good_mean = np.array([1.01, 2.01, 3.01])
        bad_mean = np.array([2.0, 3.0, 4.0])
        std = np.array([0.5, 0.5, 0.5])
        mlpd_good = compute_mlpd(y_true, good_mean, std)
        mlpd_bad = compute_mlpd(y_true, bad_mean, std)
        assert mlpd_good > mlpd_bad


# --- Build Kernels Test ---

class TestBuildKernels:

    def test_returns_four_kernels(self):
        basis = _simple_basis()
        kernels = build_kernels(basis)
        assert set(kernels.keys()) == {
            "k_specific_ard", "k_total_ard", "k_product_ard", "k_rbf",
        }

    def test_all_kernels_callable(self):
        basis = _simple_basis()
        kernels = build_kernels(basis)
        X = np.random.RandomState(0).randn(5, 2)
        for name, kernel in kernels.items():
            K = kernel(X)
            assert K.shape == (5, 5), f"Kernel {name} produced wrong shape"


# --- Synthetic Function Test ---

class TestSyntheticFunction:
    """Test that the full pipeline runs without error on the new test function."""

    def test_run_single_trial(self):
        def true_fn(X):
            x, y = X[:, 0], X[:, 1]
            return (
                np.cos(3 * x) * np.exp(-y ** 2 / 2)
                + np.tanh(x * y) / (1 + x ** 2)
                + 0.5 * np.sin(np.pi * y) * np.log1p(x ** 2)
            )

        basis = [
            lambda X: np.cos(3 * X[:, 0]) * np.exp(-X[:, 1] ** 2 / 2),
            lambda X: np.tanh(X[:, 0] * X[:, 1]) / (1 + X[:, 0] ** 2),
            lambda X: np.sin(np.pi * X[:, 1]) * np.log1p(X[:, 0] ** 2),
            lambda X: X[:, 0] ** 2 + X[:, 1] ** 2,
        ]

        rng = np.random.RandomState(42)
        X_train = np.column_stack([rng.uniform(-2, 3, 15), rng.uniform(-2, 2, 15)])
        y_train = true_fn(X_train)

        x_test = np.linspace(-2, 3, 10)
        y_test_grid = np.linspace(-2, 2, 10)
        Xg, Yg = np.meshgrid(x_test, y_test_grid)
        X_test = np.column_stack([Xg.ravel(), Yg.ravel()])
        y_test = true_fn(X_test)

        kernels = build_kernels(basis)
        results = run_single_trial(X_train, y_train, X_test, y_test, kernels)

        for name in ["k_specific_ard", "k_total_ard", "k_product_ard", "k_rbf"]:
            assert "rmse" in results[name]
            assert "mlpd" in results[name]
            assert results[name]["rmse"] >= 0
            assert np.isfinite(results[name]["rmse"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
