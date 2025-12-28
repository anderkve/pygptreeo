"""
Performance Analysis: Effect of Number of B-spline Coefficients

This script tests how the number of B-spline coefficients affects the quality
of function reconstruction when using multi-output GPs for learning f(t; x).

The analysis includes:
1. Training multi-output GPs with varying numbers of coefficients
2. Computing reconstruction errors (MSE, max error)
3. Visualizing true vs reconstructed functions
4. Analyzing the trade-off between accuracy and model complexity
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, splrep
from pygptreeo import GPTree, Default_GPR
import time
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
output_dir = Path('/home/user/pygptreeo/examples/coefficient_analysis')
output_dir.mkdir(exist_ok=True)


def target_function(t, x1, x2):
    """
    Complex target function f(t; x1, x2) to learn.

    This function has rich structure in t that requires adequate
    basis representation.
    """
    return (
        np.sin(2 * np.pi * t) * x1
        + np.cos(4 * np.pi * t) * x2
        + 0.5 * np.sin(6 * np.pi * t) * x1 * x2
        + 0.3 * np.exp(-((t - 0.5)**2) / (0.1 + 0.1 * x1**2))
        + 0.2 * np.sin(10 * np.pi * t) * (x1 + x2)  # High-frequency component
    )


def fit_bspline_coefficients(t_values, f_values, n_interior_knots):
    """
    Fit B-splines with specified number of interior knots.

    Parameters:
    -----------
    t_values : np.ndarray
        The t values where function is observed
    f_values : np.ndarray
        The function values f(t)
    n_interior_knots : int
        Number of interior knots (controls number of coefficients)

    Returns:
    --------
    coefficients : np.ndarray
        B-spline coefficients
    knots : np.ndarray
        Knot vector
    """
    # Place interior knots uniformly
    if n_interior_knots > 0:
        interior_knots = np.linspace(t_values[0], t_values[-1], n_interior_knots + 2)[1:-1]
        # Add knots to scipy's splrep via task parameter
        tck = splrep(t_values, f_values, s=0, k=3, t=interior_knots)
    else:
        tck = splrep(t_values, f_values, s=0, k=3)

    knots, coefficients, degree = tck
    return coefficients, knots


def reconstruct_from_coefficients(t_values, coefficients, knots, degree=3):
    """Reconstruct function from B-spline coefficients."""
    bspline = BSpline(knots, coefficients, degree, extrapolate=False)
    f_reconstructed = bspline(t_values)
    f_reconstructed = np.nan_to_num(f_reconstructed, nan=0.0)
    return f_reconstructed


def train_and_evaluate(n_interior_knots, n_train=50, n_test=25, t_grid_size=100):
    """
    Train multi-output GP with given number of coefficients and evaluate.

    Returns:
    --------
    results : dict
        Dictionary containing errors, predictions, and timing info
    """
    print(f"\n{'='*70}")
    print(f"Testing with {n_interior_knots} interior knots")
    print(f"{'='*70}")

    # Setup
    t_grid = np.linspace(0, 1, t_grid_size)

    # Generate training data
    X_train_params = np.random.rand(n_train, 2)

    coefficients_list = []
    n_coeffs = None
    reference_knots = None

    print(f"Fitting B-splines to training data...")
    for i, (x1, x2) in enumerate(X_train_params):
        f_values = target_function(t_grid, x1, x2)
        coeffs, knots = fit_bspline_coefficients(t_grid, f_values, n_interior_knots)
        coefficients_list.append(coeffs)

        if i == 0:
            n_coeffs = len(coeffs)
            reference_knots = knots
            print(f"   Number of B-spline coefficients: {n_coeffs}")

    y_train = np.array(coefficients_list)
    sigma_train = np.ones((n_train, n_coeffs)) * 0.01

    # Train GPTree
    print(f"Training multi-output GPTree...")
    start_time = time.time()

    gpt = GPTree(
        GPR=Default_GPR(),
        Nbar=25,
        theta=0.001,
        n_outputs=n_coeffs,
        use_calibrated_sigma=False,
        splitting_strategy='standard',
        use_standard_scaling=True,
    )

    gpt.fit(X_train_params, y_train, sigma_train, show_progress=False, shuffle=True)

    train_time = time.time() - start_time
    n_leaves = len(gpt.root.leaves)
    print(f"   Training time: {train_time:.2f}s")
    print(f"   Number of leaves: {n_leaves}")

    # Generate test data
    X_test_params = np.random.rand(n_test, 2)

    # Predict
    print(f"Making predictions...")
    start_time = time.time()
    y_pred, y_std = gpt.predict(X_test_params, mode='recursive', show_progress=False)
    pred_time = time.time() - start_time
    print(f"   Prediction time: {pred_time:.2f}s")

    # Evaluate reconstruction errors
    print(f"Computing reconstruction errors...")
    mse_errors = []
    max_errors = []
    mean_abs_errors = []

    f_true_list = []
    f_reconstructed_list = []

    for i, (x1, x2) in enumerate(X_test_params):
        # True function
        f_true = target_function(t_grid, x1, x2)
        f_true_list.append(f_true)

        # Reconstructed function
        coeffs_pred = y_pred[i, :]
        f_recon = reconstruct_from_coefficients(t_grid, coeffs_pred, reference_knots)
        f_reconstructed_list.append(f_recon)

        # Compute errors
        mse = np.mean((f_true - f_recon)**2)
        max_err = np.max(np.abs(f_true - f_recon))
        mae = np.mean(np.abs(f_true - f_recon))

        mse_errors.append(mse)
        max_errors.append(max_err)
        mean_abs_errors.append(mae)

    mse_errors = np.array(mse_errors)
    max_errors = np.array(max_errors)
    mean_abs_errors = np.array(mean_abs_errors)

    print(f"   Mean MSE: {np.mean(mse_errors):.6f} ± {np.std(mse_errors):.6f}")
    print(f"   Mean MAE: {np.mean(mean_abs_errors):.6f} ± {np.std(mean_abs_errors):.6f}")
    print(f"   Mean Max Error: {np.mean(max_errors):.6f} ± {np.std(max_errors):.6f}")

    return {
        'n_interior_knots': n_interior_knots,
        'n_coeffs': n_coeffs,
        'mse_errors': mse_errors,
        'max_errors': max_errors,
        'mae_errors': mean_abs_errors,
        'train_time': train_time,
        'pred_time': pred_time,
        'n_leaves': n_leaves,
        'X_test': X_test_params,
        'f_true_list': f_true_list,
        'f_recon_list': f_reconstructed_list,
        't_grid': t_grid,
    }


def plot_coefficient_comparison(all_results):
    """Plot how errors vary with number of coefficients."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    n_coeffs_list = [r['n_coeffs'] for r in all_results]

    # Plot 1: MSE vs n_coefficients
    ax = axes[0, 0]
    mean_mse = [np.mean(r['mse_errors']) for r in all_results]
    std_mse = [np.std(r['mse_errors']) for r in all_results]
    ax.errorbar(n_coeffs_list, mean_mse, yerr=std_mse, marker='o', capsize=5, linewidth=2)
    ax.set_xlabel('Number of B-spline Coefficients', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('Reconstruction Error vs Number of Coefficients', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: MAE vs n_coefficients
    ax = axes[0, 1]
    mean_mae = [np.mean(r['mae_errors']) for r in all_results]
    std_mae = [np.std(r['mae_errors']) for r in all_results]
    ax.errorbar(n_coeffs_list, mean_mae, yerr=std_mae, marker='s', capsize=5, linewidth=2, color='orange')
    ax.set_xlabel('Number of B-spline Coefficients', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('MAE vs Number of Coefficients', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 3: Max error vs n_coefficients
    ax = axes[1, 0]
    mean_max_err = [np.mean(r['max_errors']) for r in all_results]
    std_max_err = [np.std(r['max_errors']) for r in all_results]
    ax.errorbar(n_coeffs_list, mean_max_err, yerr=std_max_err, marker='^', capsize=5, linewidth=2, color='green')
    ax.set_xlabel('Number of B-spline Coefficients', fontsize=12)
    ax.set_ylabel('Maximum Absolute Error', fontsize=12)
    ax.set_title('Max Error vs Number of Coefficients', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 4: Training time vs n_coefficients
    ax = axes[1, 1]
    train_times = [r['train_time'] for r in all_results]
    ax.plot(n_coeffs_list, train_times, marker='D', linewidth=2, color='red', markersize=8)
    ax.set_xlabel('Number of B-spline Coefficients', fontsize=12)
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Computational Cost vs Number of Coefficients', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'coefficient_performance_summary.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'coefficient_performance_summary.png'}")
    plt.close()


def plot_example_reconstructions(all_results, n_examples=4):
    """Plot example function reconstructions for different coefficient counts."""
    n_configs = len(all_results)

    # Select test points to show (same across all configs)
    test_idx = [0, 5, 10, 15][:n_examples]

    fig, axes = plt.subplots(n_examples, n_configs, figsize=(4*n_configs, 3*n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    if n_configs == 1:
        axes = axes.reshape(-1, 1)

    for config_idx, result in enumerate(all_results):
        n_coeffs = result['n_coeffs']
        t_grid = result['t_grid']

        for row_idx, test_i in enumerate(test_idx):
            ax = axes[row_idx, config_idx]

            f_true = result['f_true_list'][test_i]
            f_recon = result['f_recon_list'][test_i]
            x1, x2 = result['X_test'][test_i]

            mse = np.mean((f_true - f_recon)**2)

            # Plot
            ax.plot(t_grid, f_true, 'b-', linewidth=2.5, label='True f(t)', alpha=0.7)
            ax.plot(t_grid, f_recon, 'r--', linewidth=2, label='Reconstructed', alpha=0.8)

            ax.set_xlabel('t', fontsize=10)
            ax.set_ylabel('f(t; x₁, x₂)', fontsize=10)

            # Title
            if row_idx == 0:
                ax.set_title(f'{n_coeffs} coefficients', fontsize=12, fontweight='bold')

            # Add parameter values and error on the left
            if config_idx == 0:
                ax.text(0.02, 0.98, f'x₁={x1:.2f}, x₂={x2:.2f}',
                       transform=ax.transAxes, verticalalignment='top',
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Add error on the right
            ax.text(0.98, 0.02, f'MSE={mse:.2e}',
                   transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'example_reconstructions.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'example_reconstructions.png'}")
    plt.close()


def plot_error_distributions(all_results):
    """Plot error distributions for different coefficient counts."""
    n_configs = len(all_results)

    fig, axes = plt.subplots(1, n_configs, figsize=(5*n_configs, 4))
    if n_configs == 1:
        axes = [axes]

    for idx, result in enumerate(all_results):
        ax = axes[idx]
        n_coeffs = result['n_coeffs']
        mse_errors = result['mse_errors']

        # Histogram of MSE errors
        ax.hist(mse_errors, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.mean(mse_errors), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(mse_errors):.2e}')
        ax.axvline(np.median(mse_errors), color='orange', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(mse_errors):.2e}')

        ax.set_xlabel('MSE', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{n_coeffs} Coefficients\nError Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'error_distributions.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'error_distributions.png'}")
    plt.close()


def plot_worst_best_cases(all_results):
    """Plot the best and worst reconstructions for each coefficient count."""
    n_configs = len(all_results)

    fig, axes = plt.subplots(2, n_configs, figsize=(5*n_configs, 8))
    if n_configs == 1:
        axes = axes.reshape(-1, 1)

    for config_idx, result in enumerate(all_results):
        n_coeffs = result['n_coeffs']
        t_grid = result['t_grid']
        mse_errors = result['mse_errors']

        # Find best and worst
        best_idx = np.argmin(mse_errors)
        worst_idx = np.argmax(mse_errors)

        for row_idx, (label, test_idx) in enumerate([('Best', best_idx), ('Worst', worst_idx)]):
            ax = axes[row_idx, config_idx]

            f_true = result['f_true_list'][test_idx]
            f_recon = result['f_recon_list'][test_idx]
            x1, x2 = result['X_test'][test_idx]
            mse = mse_errors[test_idx]

            # Plot
            ax.plot(t_grid, f_true, 'b-', linewidth=2.5, label='True f(t)', alpha=0.7)
            ax.plot(t_grid, f_recon, 'r--', linewidth=2.5, label='Reconstructed', alpha=0.8)
            ax.fill_between(t_grid, f_true, f_recon, alpha=0.2, color='gray', label='Error')

            ax.set_xlabel('t', fontsize=11)
            ax.set_ylabel('f(t; x₁, x₂)', fontsize=11)

            # Title
            if row_idx == 0:
                ax.set_title(f'{n_coeffs} Coefficients - Best Case', fontsize=12, fontweight='bold')
            else:
                ax.set_title(f'{n_coeffs} Coefficients - Worst Case', fontsize=12, fontweight='bold')

            # Add info
            info_text = f'x₁={x1:.2f}, x₂={x2:.2f}\nMSE={mse:.2e}'
            ax.text(0.02, 0.98, info_text,
                   transform=ax.transAxes, verticalalignment='top',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.legend(fontsize=9, loc='lower right')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'best_worst_cases.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'best_worst_cases.png'}")
    plt.close()


def main():
    print("="*70)
    print("B-spline Coefficient Count Performance Analysis")
    print("="*70)

    # Test different numbers of interior knots
    # This will result in different numbers of coefficients
    n_interior_knots_list = [2, 5, 10, 20, 40]  # Will give roughly 6, 9, 14, 24, 44 coeffs

    all_results = []

    for n_knots in n_interior_knots_list:
        result = train_and_evaluate(
            n_interior_knots=n_knots,
            n_train=50,
            n_test=25,
            t_grid_size=100
        )
        all_results.append(result)

    # Generate plots
    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)

    plot_coefficient_comparison(all_results)
    plot_example_reconstructions(all_results, n_examples=4)
    plot_error_distributions(all_results)
    plot_worst_best_cases(all_results)

    # Print summary table
    print("\n" + "="*70)
    print("Summary Table")
    print("="*70)
    print(f"{'Knots':<8} {'Coeffs':<10} {'Mean MSE':<15} {'Mean MAE':<15} {'Mean Max Err':<15} {'Train Time':<12}")
    print("-"*85)
    for r in all_results:
        print(f"{r['n_interior_knots']:<8} {r['n_coeffs']:<10} "
              f"{np.mean(r['mse_errors']):<15.6f} {np.mean(r['mae_errors']):<15.6f} "
              f"{np.mean(r['max_errors']):<15.6f} {r['train_time']:<12.2f}s")

    print("\n" + "="*70)
    print("Analysis Complete!")
    print(f"All plots saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
