"""
Coefficient Function Learning Test

This test separates two challenges:
1. Learning coefficient functions c_i(x1, x2) - how well can the GP learn smooth
   functions that map input parameters to B-spline coefficients?
2. Function reconstruction f(t) - how do errors in coefficient prediction
   propagate to errors in the reconstructed functions?

Unlike previous tests where we fit B-splines to target functions, here we:
- Explicitly define coefficient functions c_i(x1, x2)
- Use these to generate f(t; x) via B-spline reconstruction
- Train GP to learn the coefficient functions
- Analyze both coefficient prediction error AND reconstruction error

This reveals the relationship between coefficient learning accuracy and
final function reconstruction quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from pygptreeo import GPTree, Default_GPR
import time
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)


# ============================================================================
# COEFFICIENT FUNCTION DEFINITIONS
# ============================================================================

def smooth_polynomial_coefficients(x1, x2, n_coeffs=18):
    """
    Smooth polynomial functions for coefficients.
    Each coefficient is a smooth function of (x1, x2).
    """
    coeffs = np.zeros(n_coeffs)

    # Define each coefficient as a smooth function
    for i in range(n_coeffs):
        # Create diverse patterns for different coefficients
        phase = 2 * np.pi * i / n_coeffs

        coeffs[i] = (
            0.5 * np.sin(phase) * (x1 - 0.5)**2
            + 0.5 * np.cos(phase) * (x2 - 0.5)**2
            + 0.3 * np.sin(2 * phase) * x1 * x2
            + 0.2 * (x1 + x2 - 1)
            + 0.1 * np.sin(3 * phase) * (x1 - x2)
        )

    return coeffs


def oscillatory_coefficients(x1, x2, n_coeffs=18):
    """
    Oscillatory functions for coefficients.
    Creates more complex, oscillating patterns in coefficient space.
    """
    coeffs = np.zeros(n_coeffs)

    for i in range(n_coeffs):
        freq = 1 + i * 0.3

        coeffs[i] = (
            np.sin(freq * np.pi * x1) * x2
            + np.cos(freq * np.pi * x2) * x1
            + 0.5 * np.sin(freq * 1.5 * np.pi * (x1 + x2))
            + 0.3 * np.cos(freq * 0.7 * np.pi * (x1 - x2))
        )

    return coeffs


def radial_coefficients(x1, x2, n_coeffs=18):
    """
    Radial basis-like functions for coefficients.
    Coefficients vary based on distance from center and angle.
    """
    coeffs = np.zeros(n_coeffs)

    # Center point
    cx, cy = 0.5, 0.5

    # Radial distance and angle
    r = np.sqrt((x1 - cx)**2 + (x2 - cy)**2)
    theta = np.arctan2(x2 - cy, x1 - cx)

    for i in range(n_coeffs):
        mode = i % 4

        if mode == 0:
            # Radial pattern
            coeffs[i] = np.exp(-5 * r**2) * np.cos(i * theta)
        elif mode == 1:
            # Angular pattern
            coeffs[i] = r * np.sin(i * theta)
        elif mode == 2:
            # Mixed pattern
            coeffs[i] = (1 - r) * np.cos(i * theta) + r * np.sin(i * theta / 2)
        else:
            # Polynomial in r
            coeffs[i] = (r**2 - 0.5) * np.cos((i + 1) * theta)

    return coeffs


def localized_bumps_coefficients(x1, x2, n_coeffs=18):
    """
    Localized bump functions for coefficients.
    Each coefficient has localized features in parameter space.
    """
    coeffs = np.zeros(n_coeffs)

    # Define centers for bumps (spread across parameter space)
    n_centers = 4
    centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]

    for i in range(n_coeffs):
        # Each coefficient gets contributions from different bumps
        center_idx = i % n_centers
        cx, cy = centers[center_idx]

        # Gaussian bump
        r_sq = (x1 - cx)**2 + (x2 - cy)**2
        width = 0.1 + 0.1 * (i / n_coeffs)
        amplitude = 2.0 * np.sin(2 * np.pi * i / n_coeffs)

        coeffs[i] = amplitude * np.exp(-r_sq / (2 * width**2))

        # Add smooth background
        coeffs[i] += 0.2 * (x1 - 0.5) * (x2 - 0.5)

    return coeffs


def sharp_transition_coefficients(x1, x2, n_coeffs=18):
    """
    Coefficients with sharp but continuous transitions.
    Tests GP's ability to learn sharp features in coefficient space.
    """
    coeffs = np.zeros(n_coeffs)

    for i in range(n_coeffs):
        sharpness = 20
        offset = i / n_coeffs

        coeffs[i] = (
            np.tanh(sharpness * (x1 - 0.3 - offset * 0.2)) * x2
            + np.tanh(sharpness * (x2 - 0.7 + offset * 0.2)) * x1
            + 0.5 * np.tanh(sharpness * (x1 + x2 - 1 - offset * 0.1))
        )

    return coeffs


COEFFICIENT_FUNCTIONS = {
    'smooth_polynomial': {
        'func': smooth_polynomial_coefficients,
        'name': 'Smooth Polynomial',
        'description': 'Polynomial functions for each coefficient'
    },
    'oscillatory': {
        'func': oscillatory_coefficients,
        'name': 'Oscillatory',
        'description': 'Oscillating patterns in coefficient space'
    },
    'radial': {
        'func': radial_coefficients,
        'name': 'Radial Basis',
        'description': 'Radial and angular patterns'
    },
    'localized_bumps': {
        'func': localized_bumps_coefficients,
        'name': 'Localized Bumps',
        'description': 'Localized Gaussian bumps in parameter space'
    },
    'sharp_transitions': {
        'func': sharp_transition_coefficients,
        'name': 'Sharp Transitions',
        'description': 'Sharp but continuous transitions in coefficient space'
    }
}


# ============================================================================
# B-SPLINE UTILITIES
# ============================================================================

def create_bspline_basis(t_min=0, t_max=1, n_interior_knots=10, degree=3):
    """
    Create a B-spline basis with uniform knots.

    Returns:
    --------
    knots : np.ndarray
        Knot vector for the B-spline basis
    n_coeffs : int
        Number of coefficients (basis functions)
    """
    # Create uniform interior knots
    if n_interior_knots > 0:
        interior_knots = np.linspace(t_min, t_max, n_interior_knots + 2)[1:-1]
    else:
        interior_knots = np.array([])

    # Create full knot vector with multiplicity at boundaries
    knots = np.concatenate([
        [t_min] * (degree + 1),
        interior_knots,
        [t_max] * (degree + 1)
    ])

    # Number of basis functions = number of knots - degree - 1
    n_coeffs = len(knots) - degree - 1

    return knots, n_coeffs


def reconstruct_from_coefficients(t_values, coefficients, knots, degree=3):
    """Reconstruct function from B-spline coefficients."""
    bspline = BSpline(knots, coefficients, degree, extrapolate=False)
    f_reconstructed = bspline(t_values)
    f_reconstructed = np.nan_to_num(f_reconstructed, nan=0.0)
    return f_reconstructed


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def generate_training_data(coeff_func, knots, n_train=100):
    """
    Generate training data from coefficient functions.

    Returns:
    --------
    X_train : np.ndarray, shape (n_train, 2)
        Training parameter points
    y_train : np.ndarray, shape (n_train, n_coeffs)
        True coefficient values at training points
    """
    # Sample parameter space
    X_train = np.random.rand(n_train, 2)

    # Compute true coefficients for each training point
    n_coeffs = len(knots) - 4  # For cubic B-splines
    y_train = np.zeros((n_train, n_coeffs))

    for i, (x1, x2) in enumerate(X_train):
        y_train[i, :] = coeff_func(x1, x2, n_coeffs=n_coeffs)

    return X_train, y_train


def train_and_evaluate(coeff_func, n_interior_knots=10, n_train=100, n_test=50,
                      t_grid_size=100):
    """
    Train GP on coefficient functions and evaluate both coefficient prediction
    and function reconstruction.

    Returns:
    --------
    results : dict
        Comprehensive results including coefficient errors and reconstruction errors
    """
    print(f"  Creating B-spline basis with {n_interior_knots} interior knots...")

    # Create B-spline basis
    t_grid = np.linspace(0, 1, t_grid_size)
    knots, n_coeffs = create_bspline_basis(n_interior_knots=n_interior_knots)

    print(f"  Number of coefficients: {n_coeffs}")
    print(f"  Generating training data with {n_train} samples...")

    # Generate training data
    X_train, y_train = generate_training_data(coeff_func, knots, n_train=n_train)
    sigma_train = np.ones((n_train, n_coeffs)) * 0.01

    # Train GPTree
    print(f"  Training GPTree...")
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

    gpt.fit(X_train, y_train, sigma_train, show_progress=False, shuffle=True)

    train_time = time.time() - start_time
    n_leaves = len(gpt.root.leaves)

    print(f"  Training completed in {train_time:.2f}s with {n_leaves} leaves")
    print(f"  Generating test data with {n_test} samples...")

    # Generate test data
    X_test = np.random.rand(n_test, 2)
    y_test_true = np.zeros((n_test, n_coeffs))

    for i, (x1, x2) in enumerate(X_test):
        y_test_true[i, :] = coeff_func(x1, x2, n_coeffs=n_coeffs)

    # Predict coefficients
    print(f"  Predicting coefficients...")
    start_time = time.time()
    y_test_pred, y_test_std = gpt.predict(X_test, mode='recursive', show_progress=False)
    pred_time = time.time() - start_time

    print(f"  Predictions completed in {pred_time:.2f}s")
    print(f"  Evaluating errors...")

    # Compute coefficient prediction errors
    coeff_mse = np.mean((y_test_true - y_test_pred)**2, axis=1)
    coeff_mae = np.mean(np.abs(y_test_true - y_test_pred), axis=1)
    coeff_max_err = np.max(np.abs(y_test_true - y_test_pred), axis=1)

    # Per-coefficient errors (averaged over test points)
    per_coeff_mse = np.mean((y_test_true - y_test_pred)**2, axis=0)

    # Reconstruct functions and compute reconstruction errors
    recon_mse = []
    recon_mae = []
    recon_max_err = []

    f_true_list = []
    f_pred_list = []

    for i in range(n_test):
        # Reconstruct from true coefficients
        f_true = reconstruct_from_coefficients(t_grid, y_test_true[i, :], knots)
        f_true_list.append(f_true)

        # Reconstruct from predicted coefficients
        f_pred = reconstruct_from_coefficients(t_grid, y_test_pred[i, :], knots)
        f_pred_list.append(f_pred)

        # Compute reconstruction errors
        recon_mse.append(np.mean((f_true - f_pred)**2))
        recon_mae.append(np.mean(np.abs(f_true - f_pred)))
        recon_max_err.append(np.max(np.abs(f_true - f_pred)))

    recon_mse = np.array(recon_mse)
    recon_mae = np.array(recon_mae)
    recon_max_err = np.array(recon_max_err)

    print(f"  Coefficient MSE: {np.mean(coeff_mse):.6f} ± {np.std(coeff_mse):.6f}")
    print(f"  Reconstruction MSE: {np.mean(recon_mse):.6f} ± {np.std(recon_mse):.6f}")

    return {
        'n_interior_knots': n_interior_knots,
        'n_coeffs': n_coeffs,
        'n_train': n_train,
        'n_test': n_test,
        'train_time': train_time,
        'pred_time': pred_time,
        'n_leaves': n_leaves,

        # Coefficient prediction errors
        'coeff_mse': coeff_mse,
        'coeff_mae': coeff_mae,
        'coeff_max_err': coeff_max_err,
        'per_coeff_mse': per_coeff_mse,

        # Reconstruction errors
        'recon_mse': recon_mse,
        'recon_mae': recon_mae,
        'recon_max_err': recon_max_err,

        # Data for visualization
        'X_test': X_test,
        'y_test_true': y_test_true,
        'y_test_pred': y_test_pred,
        'y_test_std': y_test_std,
        'f_true_list': f_true_list,
        'f_pred_list': f_pred_list,
        't_grid': t_grid,
        'knots': knots,
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_coefficient_vs_reconstruction_error(all_results, output_dir, func_name):
    """Plot relationship between coefficient error and reconstruction error."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    n_train_list = [r['n_train'] for r in all_results]

    # Plot 1: Coefficient MSE vs training size
    ax = axes[0, 0]
    coeff_mse_mean = [np.mean(r['coeff_mse']) for r in all_results]
    coeff_mse_std = [np.std(r['coeff_mse']) for r in all_results]
    ax.errorbar(n_train_list, coeff_mse_mean, yerr=coeff_mse_std,
                marker='o', capsize=5, linewidth=2, label='Coefficient MSE')
    ax.set_xlabel('Number of Training Samples', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title(f'{func_name}: Coefficient Prediction Error', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.legend()

    # Plot 2: Reconstruction MSE vs training size
    ax = axes[0, 1]
    recon_mse_mean = [np.mean(r['recon_mse']) for r in all_results]
    recon_mse_std = [np.std(r['recon_mse']) for r in all_results]
    ax.errorbar(n_train_list, recon_mse_mean, yerr=recon_mse_std,
                marker='s', capsize=5, linewidth=2, color='orange', label='Reconstruction MSE')
    ax.set_xlabel('Number of Training Samples', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title(f'{func_name}: Function Reconstruction Error', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.legend()

    # Plot 3: Scatter plot - Coefficient MSE vs Reconstruction MSE
    ax = axes[1, 0]
    for r in all_results:
        ax.scatter(r['coeff_mse'], r['recon_mse'], alpha=0.5, s=30,
                  label=f'{r["n_train"]} samples')
    ax.set_xlabel('Coefficient MSE', fontsize=12)
    ax.set_ylabel('Reconstruction MSE', fontsize=12)
    ax.set_title('Correlation: Coefficient vs Reconstruction Error', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Add correlation line
    all_coeff_mse = np.concatenate([r['coeff_mse'] for r in all_results])
    all_recon_mse = np.concatenate([r['recon_mse'] for r in all_results])
    correlation = np.corrcoef(np.log(all_coeff_mse), np.log(all_recon_mse))[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Per-coefficient error pattern
    ax = axes[1, 1]
    for r in all_results:
        ax.plot(r['per_coeff_mse'], marker='o', linewidth=1.5,
               label=f'{r["n_train"]} samples', alpha=0.7)
    ax.set_xlabel('Coefficient Index', fontsize=12)
    ax.set_ylabel('MSE (averaged over test set)', fontsize=12)
    ax.set_title('Per-Coefficient Prediction Error', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'coefficient_vs_reconstruction_error.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_example_predictions(result, output_dir, func_name, n_examples=6):
    """Plot examples of coefficient predictions and function reconstructions."""
    n_test = result['n_test']
    test_idx = np.linspace(0, n_test - 1, n_examples, dtype=int)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(n_examples, 3, hspace=0.3, wspace=0.3)

    for row_idx, test_i in enumerate(test_idx):
        x1, x2 = result['X_test'][test_i]
        y_true = result['y_test_true'][test_i, :]
        y_pred = result['y_test_pred'][test_i, :]
        y_std = result['y_test_std'][test_i, :]

        f_true = result['f_true_list'][test_i]
        f_pred = result['f_pred_list'][test_i]
        t_grid = result['t_grid']

        coeff_mse = result['coeff_mse'][test_i]
        recon_mse = result['recon_mse'][test_i]

        # Plot 1: Coefficient comparison
        ax = fig.add_subplot(gs[row_idx, 0])
        indices = np.arange(len(y_true))
        ax.plot(indices, y_true, 'b-o', markersize=4, linewidth=1.5, label='True', alpha=0.7)
        ax.plot(indices, y_pred, 'r--s', markersize=3, linewidth=1.5, label='Predicted', alpha=0.7)
        ax.fill_between(indices, y_pred - 2*y_std, y_pred + 2*y_std,
                        alpha=0.2, color='red', label='±2σ')
        ax.set_ylabel('Coefficient Value', fontsize=9)
        if row_idx == 0:
            ax.set_title('Coefficient Predictions', fontsize=11, fontweight='bold')
        if row_idx == n_examples - 1:
            ax.set_xlabel('Coefficient Index', fontsize=9)
        ax.text(0.02, 0.98, f'x₁={x1:.2f}, x₂={x2:.2f}\nMSE={coeff_mse:.2e}',
               transform=ax.transAxes, verticalalignment='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Plot 2: Coefficient errors
        ax = fig.add_subplot(gs[row_idx, 1])
        errors = y_true - y_pred
        ax.bar(indices, errors, color='purple', alpha=0.6)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Error', fontsize=9)
        if row_idx == 0:
            ax.set_title('Coefficient Errors', fontsize=11, fontweight='bold')
        if row_idx == n_examples - 1:
            ax.set_xlabel('Coefficient Index', fontsize=9)
        max_err = np.max(np.abs(errors))
        ax.text(0.98, 0.98, f'Max: {max_err:.2e}',
               transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
               fontsize=8, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Function reconstruction
        ax = fig.add_subplot(gs[row_idx, 2])
        ax.plot(t_grid, f_true, 'b-', linewidth=2, label='True f(t)', alpha=0.7)
        ax.plot(t_grid, f_pred, 'r--', linewidth=2, label='Predicted f(t)', alpha=0.7)
        ax.fill_between(t_grid, f_true, f_pred, alpha=0.2, color='gray')
        ax.set_ylabel('f(t)', fontsize=9)
        if row_idx == 0:
            ax.set_title('Function Reconstruction', fontsize=11, fontweight='bold')
        if row_idx == n_examples - 1:
            ax.set_xlabel('t', fontsize=9)
        ax.text(0.98, 0.02, f'MSE={recon_mse:.2e}',
               transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
               fontsize=8, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{func_name}: Coefficient Predictions and Function Reconstructions\n'
                 f'(n_train={result["n_train"]}, n_coeffs={result["n_coeffs"]})',
                 fontsize=14, fontweight='bold')

    plt.savefig(output_dir / f'example_predictions_ntrain{result["n_train"]}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_distributions(all_results, output_dir, func_name):
    """Plot distributions of coefficient and reconstruction errors."""
    n_configs = len(all_results)

    fig, axes = plt.subplots(2, n_configs, figsize=(5*n_configs, 8))
    if n_configs == 1:
        axes = axes.reshape(-1, 1)

    for idx, result in enumerate(all_results):
        n_train = result['n_train']

        # Top row: Coefficient error distribution
        ax = axes[0, idx]
        ax.hist(result['coeff_mse'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.mean(result['coeff_mse']), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(result["coeff_mse"]):.2e}')
        ax.axvline(np.median(result['coeff_mse']), color='orange', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(result["coeff_mse"]):.2e}')
        ax.set_xlabel('Coefficient MSE', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{n_train} Training Samples\nCoefficient Error', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Bottom row: Reconstruction error distribution
        ax = axes[1, idx]
        ax.hist(result['recon_mse'], bins=20, alpha=0.7, color='coral', edgecolor='black')
        ax.axvline(np.mean(result['recon_mse']), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(result["recon_mse"]):.2e}')
        ax.axvline(np.median(result['recon_mse']), color='orange', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(result["recon_mse"]):.2e}')
        ax.set_xlabel('Reconstruction MSE', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Reconstruction Error', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'{func_name}: Error Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_coefficient_heatmaps(result, output_dir, func_name):
    """Plot heatmaps of true vs predicted coefficients."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: True coefficients
    ax = axes[0]
    im = ax.imshow(result['y_test_true'].T, aspect='auto', cmap='viridis',
                   interpolation='nearest')
    ax.set_xlabel('Test Sample Index', fontsize=11)
    ax.set_ylabel('Coefficient Index', fontsize=11)
    ax.set_title('True Coefficients', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax)

    # Plot 2: Predicted coefficients
    ax = axes[1]
    im = ax.imshow(result['y_test_pred'].T, aspect='auto', cmap='viridis',
                   interpolation='nearest')
    ax.set_xlabel('Test Sample Index', fontsize=11)
    ax.set_ylabel('Coefficient Index', fontsize=11)
    ax.set_title('Predicted Coefficients', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax)

    # Plot 3: Prediction errors
    ax = axes[2]
    errors = result['y_test_true'] - result['y_test_pred']
    vmax = np.max(np.abs(errors))
    im = ax.imshow(errors.T, aspect='auto', cmap='RdBu',
                   interpolation='nearest', vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Test Sample Index', fontsize=11)
    ax.set_ylabel('Coefficient Index', fontsize=11)
    ax.set_title('Prediction Errors', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax)

    fig.suptitle(f'{func_name}: Coefficient Heatmaps (n_train={result["n_train"]})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'coefficient_heatmaps_ntrain{result["n_train"]}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("Coefficient Function Learning Test")
    print("="*80)

    # Test with different training set sizes
    n_train_list = [25, 50, 100, 200]
    n_interior_knots = 10  # Fixed for this test
    n_test = 50

    # Process each coefficient function type
    for func_key, func_info in COEFFICIENT_FUNCTIONS.items():
        print(f"\n{'='*80}")
        print(f"Coefficient Function: {func_info['name']}")
        print(f"Description: {func_info['description']}")
        print(f"{'='*80}")

        # Create output directory
        output_dir = Path(f'/home/user/pygptreeo/examples/coeff_learning_{func_key}')
        output_dir.mkdir(exist_ok=True)

        all_results = []

        for n_train in n_train_list:
            print(f"\n{'─'*80}")
            print(f"Testing with {n_train} training samples")
            print(f"{'─'*80}")

            result = train_and_evaluate(
                coeff_func=func_info['func'],
                n_interior_knots=n_interior_knots,
                n_train=n_train,
                n_test=n_test
            )
            all_results.append(result)

            # Generate per-configuration plots
            plot_example_predictions(result, output_dir, func_info['name'], n_examples=6)
            plot_coefficient_heatmaps(result, output_dir, func_info['name'])

        # Generate comparison plots
        print(f"\nGenerating comparison visualizations for {func_info['name']}...")
        plot_coefficient_vs_reconstruction_error(all_results, output_dir, func_info['name'])
        plot_error_distributions(all_results, output_dir, func_info['name'])

        print(f"\nSaved plots to: {output_dir}")

        # Print summary table
        print(f"\nSummary for {func_info['name']}:")
        print(f"{'N_train':<10} {'Coeff MSE':<15} {'Recon MSE':<15} {'Train Time':<12} {'Leaves':<8}")
        print("-"*60)
        for r in all_results:
            print(f"{r['n_train']:<10} {np.mean(r['coeff_mse']):<15.6f} "
                  f"{np.mean(r['recon_mse']):<15.6f} {r['train_time']:<12.2f}s "
                  f"{r['n_leaves']:<8}")

    print("\n" + "="*80)
    print("Analysis Complete for All Coefficient Functions!")
    print("="*80)


if __name__ == "__main__":
    main()
