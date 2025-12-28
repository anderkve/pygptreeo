"""
Multi-Output GP Example: Learning Functions f(t; x) using B-splines

This example demonstrates how to use pygptreeo's multi-output functionality
to learn a function f(t; x) where:
- x = (x1, x2) is a 2D parameter vector
- t is a 1D variable (e.g., time)
- f(t; x) is the target function that depends on both t and x

Approach:
1. For each training point x, we observe f(t; x) at many t values
2. We represent f(t; x) using cubic B-spline basis functions
3. We fit B-splines to get coefficients c(x) for each x
4. We train a multi-output GP to learn the mapping: x → c(x)
5. At test points x_test, we predict c(x_test) and reconstruct f(t; x_test)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, splrep
from pygptreeo import GPTree, Default_GPR

# Set random seed for reproducibility
np.random.seed(42)


def target_function(t, x1, x2):
    """
    Target function f(t; x1, x2) to learn.

    A function that varies with both the parameter (x1, x2) and the variable t.
    This represents a realistic scenario where we want to predict an entire
    function f(t) for any given parameter values (x1, x2).

    Parameters:
    -----------
    t : np.ndarray
        The independent variable (e.g., time). Shape: (n_t,)
    x1 : float
        First parameter
    x2 : float
        Second parameter

    Returns:
    --------
    np.ndarray
        Function values f(t; x1, x2). Shape: (n_t,)
    """
    # A combination of oscillations modulated by parameters
    return (
        np.sin(2 * np.pi * t) * x1
        + np.cos(4 * np.pi * t) * x2
        + 0.5 * np.sin(6 * np.pi * t) * x1 * x2
        + 0.3 * np.exp(-((t - 0.5)**2) / (0.1 + 0.1 * x1**2))
    )


def fit_bspline_coefficients(t_values, f_values, n_knots=10):
    """
    Fit cubic B-splines to function values and return coefficients.

    Parameters:
    -----------
    t_values : np.ndarray
        The t values where function is observed. Shape: (n_t,)
    f_values : np.ndarray
        The function values f(t). Shape: (n_t,)
    n_knots : int
        Number of interior knots for the B-spline

    Returns:
    --------
    coefficients : np.ndarray
        B-spline coefficients. Shape: (n_coefficients,)
    tck : tuple
        The B-spline representation (t, c, k) for reconstruction
    """
    # Fit B-spline using scipy
    # splrep returns (knots, coefficients, degree)
    tck = splrep(t_values, f_values, s=0, k=3)  # k=3 for cubic splines
    knots, coefficients, degree = tck

    return coefficients, tck


def reconstruct_from_coefficients(t_values, coefficients, knots, degree=3):
    """
    Reconstruct function f(t) from B-spline coefficients.

    Parameters:
    -----------
    t_values : np.ndarray
        The t values where to evaluate. Shape: (n_t,)
    coefficients : np.ndarray
        B-spline coefficients. Shape: (n_coefficients,)
    knots : np.ndarray
        B-spline knot vector
    degree : int
        B-spline degree (default: 3 for cubic)

    Returns:
    --------
    f_reconstructed : np.ndarray
        Reconstructed function values. Shape: (n_t,)
    """
    # Create BSpline object and evaluate
    bspline = BSpline(knots, coefficients, degree, extrapolate=False)
    f_reconstructed = bspline(t_values)

    # Handle NaN values (outside knot range)
    f_reconstructed = np.nan_to_num(f_reconstructed, nan=0.0)

    return f_reconstructed


def main():
    print("=" * 70)
    print("Multi-Output GP: Learning Functions f(t; x) with B-splines")
    print("=" * 70)

    # ======================
    # 1. Setup
    # ======================
    print("\n1. Setting up problem...")

    # Define t grid (dense for evaluation)
    t_grid = np.linspace(0, 1, 100)
    n_t = len(t_grid)

    # Parameter space: x = (x1, x2) in [0, 1]²
    n_train = 50  # Number of training points in parameter space
    n_test = 20   # Number of test points

    # Generate training points in parameter space
    np.random.seed(42)
    X_train_params = np.random.rand(n_train, 2)  # Shape: (n_train, 2)

    # Generate test points in parameter space (grid for visualization)
    x1_test = np.linspace(0, 1, int(np.sqrt(n_test)))
    x2_test = np.linspace(0, 1, int(np.sqrt(n_test)))
    X1_test, X2_test = np.meshgrid(x1_test, x2_test)
    X_test_params = np.column_stack([X1_test.ravel(), X2_test.ravel()])

    print(f"   - Training points in parameter space: {n_train}")
    print(f"   - Test points in parameter space: {n_test}")
    print(f"   - t grid points: {n_t}")

    # ======================
    # 2. Generate Training Data
    # ======================
    print("\n2. Generating training data...")
    print("   - Evaluating f(t; x) for each training x...")
    print("   - Fitting B-splines to get coefficients...")

    # For each training point, evaluate f(t; x) and fit B-splines
    coefficients_list = []
    n_coeffs = None
    reference_knots = None

    for i, (x1, x2) in enumerate(X_train_params):
        # Evaluate target function
        f_values = target_function(t_grid, x1, x2)

        # Fit B-splines
        coeffs, tck = fit_bspline_coefficients(t_grid, f_values, n_knots=8)
        coefficients_list.append(coeffs)

        # Store reference knots from first fit
        if i == 0:
            n_coeffs = len(coeffs)
            reference_knots = tck[0]
            print(f"   - Number of B-spline coefficients: {n_coeffs}")

    # Convert to numpy array
    y_train = np.array(coefficients_list)  # Shape: (n_train, n_coeffs)

    # Sigma (uncertainty) - use small values (we have "perfect" observations)
    sigma_train = np.ones((n_train, n_coeffs)) * 0.01

    print(f"   - Training data shape: X={X_train_params.shape}, y={y_train.shape}")

    # ======================
    # 3. Train Multi-Output GPTree
    # ======================
    print("\n3. Training multi-output GPTree...")
    print(f"   - Number of outputs: {n_coeffs}")

    # Create GPTree with multi-output support
    gpt = GPTree(
        GPR=Default_GPR(),
        Nbar=25,  # Max points per leaf
        theta=0.001,  # Overlap parameter
        n_outputs=n_coeffs,  # KEY: Multi-output!
        use_calibrated_sigma=False,  # Disable for simplicity
        splitting_strategy='standard',
        use_standard_scaling=True,  # Enable scaling for better GP performance
    )

    # Train
    gpt.fit(X_train_params, y_train, sigma_train, show_progress=True, shuffle=True)

    print(f"   - Tree has {len(gpt.root.leaves)} leaves")

    # ======================
    # 4. Make Predictions
    # ======================
    print("\n4. Making predictions at test points...")

    # Predict coefficients at test points
    y_pred, y_std = gpt.predict(X_test_params, mode='recursive', show_progress=True)

    print(f"   - Predicted coefficients shape: {y_pred.shape}")
    print(f"   - Predicted std shape: {y_std.shape}")

    # ======================
    # 5. Reconstruct Functions
    # ======================
    print("\n5. Reconstructing functions f(t; x_test) from predicted coefficients...")

    # Evaluate true and reconstructed functions at test points
    f_true_list = []
    f_reconstructed_list = []

    for i, (x1, x2) in enumerate(X_test_params):
        # True function
        f_true = target_function(t_grid, x1, x2)
        f_true_list.append(f_true)

        # Reconstructed function from predicted coefficients
        coeffs_pred = y_pred[i, :]
        f_recon = reconstruct_from_coefficients(t_grid, coeffs_pred, reference_knots, degree=3)
        f_reconstructed_list.append(f_recon)

    # ======================
    # 6. Compute Errors
    # ======================
    print("\n6. Computing reconstruction errors...")

    errors = []
    for f_true, f_recon in zip(f_true_list, f_reconstructed_list):
        mse = np.mean((f_true - f_recon)**2)
        errors.append(mse)

    errors = np.array(errors)
    print(f"   - Mean MSE: {np.mean(errors):.6f}")
    print(f"   - Std MSE: {np.std(errors):.6f}")
    print(f"   - Max MSE: {np.max(errors):.6f}")

    # ======================
    # 7. Visualize Results
    # ======================
    print("\n7. Creating visualizations...")

    # Select a few test points for detailed visualization
    test_indices = [0, n_test//4, n_test//2, 3*n_test//4]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for plot_idx, test_idx in enumerate(test_indices):
        ax = axes[plot_idx]

        x1, x2 = X_test_params[test_idx]
        f_true = f_true_list[test_idx]
        f_recon = f_reconstructed_list[test_idx]

        # Plot true and reconstructed functions
        ax.plot(t_grid, f_true, 'b-', linewidth=2, label='True f(t)', alpha=0.7)
        ax.plot(t_grid, f_recon, 'r--', linewidth=2, label='Reconstructed f(t)', alpha=0.7)

        ax.set_xlabel('t', fontsize=11)
        ax.set_ylabel('f(t; x₁, x₂)', fontsize=11)
        ax.set_title(f'x₁={x1:.2f}, x₂={x2:.2f} | MSE={errors[test_idx]:.2e}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/pygptreeo/examples/multioutput_bsplines_functions.png', dpi=150, bbox_inches='tight')
    print("   - Saved: multioutput_bsplines_functions.png")

    # Plot coefficient predictions
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    # Show first 6 coefficients
    for coeff_idx in range(min(6, n_coeffs)):
        ax = axes[coeff_idx]

        # True coefficients (from training data)
        # For visualization, compute true coefficients at test points
        y_test_true = []
        for x1, x2 in X_test_params:
            f_vals = target_function(t_grid, x1, x2)
            coeffs, _ = fit_bspline_coefficients(t_grid, f_vals, n_knots=8)
            y_test_true.append(coeffs)
        y_test_true = np.array(y_test_true)

        # Scatter plot: true vs predicted
        ax.scatter(y_test_true[:, coeff_idx], y_pred[:, coeff_idx], alpha=0.6, s=50)
        ax.plot([y_test_true[:, coeff_idx].min(), y_test_true[:, coeff_idx].max()],
                [y_test_true[:, coeff_idx].min(), y_test_true[:, coeff_idx].max()],
                'r--', linewidth=2, alpha=0.5, label='Perfect prediction')

        ax.set_xlabel(f'True Coefficient #{coeff_idx}', fontsize=10)
        ax.set_ylabel(f'Predicted Coefficient #{coeff_idx}', fontsize=10)
        ax.set_title(f'Coefficient #{coeff_idx}', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/pygptreeo/examples/multioutput_bsplines_coefficients.png', dpi=150, bbox_inches='tight')
    print("   - Saved: multioutput_bsplines_coefficients.png")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
