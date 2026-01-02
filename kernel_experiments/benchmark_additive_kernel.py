"""Benchmark script comparing standard kernels vs additive kernels.

This script systematically tests:
- Different kernel types (standard RBF/Matern, AdditiveKernel depth=1, depth=2)
- Different dimensionalities (3, 5, 8, 10, 15)
- Different target functions (Eggholder, Rastrigin, Levy, Rosenbrock)

Metrics collected:
- RMSE (Root Mean Squared Error)
- Empirical coverage (fraction within 1-sigma uncertainty)
- Average prediction time
- Average update time
- Number of leaves created
"""

import numpy as np
import time
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF

from pygptreeo import GPTree
from pygptreeo.kernels import AdditiveKernel, AnisotropicRationalQuadratic

import sys
sys.path.append('examples')
from target_functions import Eggholder, Rastrigin, Levy, Rosenbrock

# Suppress convergence warnings for cleaner output
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


class BenchmarkConfig:
    """Configuration for benchmark tests."""

    # Test parameters
    DIMENSIONS = [3, 5, 8, 10, 15]
    TARGET_FUNCTIONS = {
        'eggholder': Eggholder,
        'rastrigin': Rastrigin,
        'levy': Levy,
        'rosenbrock': Rosenbrock,
    }

    # GPTree parameters (consistent across all tests)
    N_POINTS = 10000
    NBAR = 200
    THETA = 1e-4
    RETRAIN_STEP = 200
    X_MIN = 0.0
    X_MAX = 1.0

    # Evaluation parameters
    EVAL_BATCH_SIZE = 1000  # Evaluate metrics every N points

    # Random seed for reproducibility
    RANDOM_SEED = 42


def create_gpr_class(kernel_type, n_dims):
    """Create a GPR class with specified kernel type.

    Args:
        kernel_type: One of 'matern', 'rbf', 'additive_d1', 'additive_d2', 'combo'
        n_dims: Number of input dimensions

    Returns:
        A GaussianProcessRegressor class configured with the specified kernel
    """

    class CustomGPR(GaussianProcessRegressor):
        def __init__(self, kernel=None, *, alpha=1e-6, optimizer='fmin_l_bfgs_b',
                     n_restarts_optimizer=1, normalize_y=False, copy_X_train=True,
                     n_targets=None, random_state=None):
            super().__init__()

            if kernel_type == 'matern':
                # Standard Matern kernel with ARD
                self.kernel = ConstantKernel(
                    constant_value=1.0,
                    constant_value_bounds=(1e-3, 1e8)
                ) * Matern(
                    nu=1.5,
                    length_scale=[1.0]*n_dims,
                    length_scale_bounds=[(1e-5, 1e5)]*n_dims
                )

            elif kernel_type == 'rbf':
                # Standard RBF kernel with ARD
                self.kernel = ConstantKernel(
                    constant_value=1.0,
                    constant_value_bounds=(1e-3, 1e8)
                ) * RBF(
                    length_scale=[1.0]*n_dims,
                    length_scale_bounds=[(1e-5, 1e5)]*n_dims
                )

            elif kernel_type == 'additive_d1':
                # Fully additive (GAM) - only main effects
                self.kernel = ConstantKernel(
                    constant_value=1.0,
                    constant_value_bounds=(1e-3, 1e8)
                ) * AdditiveKernel(
                    input_dim=n_dims,
                    interaction_depth=1,  # Only main effects
                    base_kernel='rbf',
                    length_scale=1.0,
                    length_scale_bounds=(1e-5, 1e5),
                )

            elif kernel_type == 'additive_d2':
                # Additive with pairwise interactions
                self.kernel = ConstantKernel(
                    constant_value=1.0,
                    constant_value_bounds=(1e-3, 1e8)
                ) * AdditiveKernel(
                    input_dim=n_dims,
                    interaction_depth=2,  # Main effects + pairwise
                    base_kernel='rbf',
                    length_scale=1.0,
                    length_scale_bounds=(1e-5, 1e5),
                )

            elif kernel_type == 'combo':
                # Combination kernel (current default in performance_test.py)
                self.kernel = ConstantKernel(
                    constant_value=1.0,
                    constant_value_bounds=(1e-3, 1e8)
                ) * (AnisotropicRationalQuadratic(
                    length_scale=[1.0]*n_dims,
                    length_scale_bounds=(1e-5, 1e5),
                    alpha=1.0,
                    alpha_bounds=(1e-4, 1e4)
                ) + Matern(
                    nu=1.5,
                    length_scale=[1.0]*n_dims,
                    length_scale_bounds=[(1e-5, 1e5)]*n_dims
                ))
            else:
                raise ValueError(f"Unknown kernel type: {kernel_type}")

            self.min_length_scale = 0.001
            self.alpha = alpha
            self.optimizer = optimizer
            self.n_restarts_optimizer = 3
            self.normalize_y = normalize_y
            self.copy_X_train = copy_X_train
            self.n_targets = n_targets
            self.random_state = random_state

    return CustomGPR


def run_single_benchmark(target_func, n_dims, kernel_type, config):
    """Run a single benchmark test.

    Args:
        target_func: Target function to approximate
        n_dims: Number of input dimensions
        kernel_type: Type of kernel to use
        config: BenchmarkConfig instance

    Returns:
        Dictionary with benchmark results
    """

    print(f"  Testing {kernel_type} kernel...", end=' ', flush=True)

    # Set random seed
    np.random.seed(config.RANDOM_SEED)

    # Generate data
    X_input = np.random.uniform(
        config.X_MIN, config.X_MAX,
        n_dims * config.N_POINTS
    ).reshape(config.N_POINTS, n_dims)
    y_input = target_func(X_input.T)

    # Create GPR class
    gpr_class = create_gpr_class(kernel_type, n_dims)

    # Create GPTree
    gpt = GPTree(
        GPR=gpr_class(),
        Nbar=config.NBAR,
        theta=config.THETA,
        split_position_method='median',
        split_dimension_criteria='max_uncertainty',
        retrain_every_n_points=config.RETRAIN_STEP,
        use_calibrated_sigma=True,
        splitting_strategy='gradual',
        max_n_pred_leaves=3,
        aggregation='moe',
        use_hyperparameter_inheritance=False,
        use_standard_scaling=True,
        enable_point_rejection=False,
        enable_point_merging=False,
        enable_split_evaluation=True,
        n_split_candidates=4,
        split_eval_train_fraction=0.4,
        split_eval_min_points=20,
    )

    # Storage for metrics
    predict_times = []
    update_times = []
    actual_values = []
    predicted_values = []
    predicted_stds = []

    # Run through all points
    start_time = time.time()
    for i, (x, y) in enumerate(zip(X_input, y_input)):
        x = x.reshape((1, x.shape[0]))
        y_val = np.array([[y]])

        # Predict
        t0 = time.time()
        y_pred, y_pred_std = gpt.predict(x, show_progress=False)
        predict_times.append(time.time() - t0)

        # Update (sigma is the uncertainty in the observation, set to small value)
        t0 = time.time()
        sigma = 0.001 * np.abs(y)  # Small relative uncertainty
        gpt.update_tree(x, y_val, sigma)
        update_times.append(time.time() - t0)

        # Store for metrics
        actual_values.append(y)
        predicted_values.append(y_pred[0][0])
        predicted_stds.append(y_pred_std[0][0])

    total_time = time.time() - start_time

    # Calculate metrics on last EVAL_BATCH_SIZE points
    eval_slice = slice(-config.EVAL_BATCH_SIZE, None)
    actual = np.array(actual_values[eval_slice])
    predicted = np.array(predicted_values[eval_slice])
    stds = np.array(predicted_stds[eval_slice])

    # RMSE
    rmse = np.sqrt(np.mean((actual - predicted)**2))

    # Normalized RMSE
    nrmse = rmse / (np.max(actual) - np.min(actual)) if np.max(actual) != np.min(actual) else 0

    # Coverage (fraction within 1-sigma)
    coverage = np.mean(np.abs(actual - predicted) <= stds)

    # Average times
    avg_predict_time = np.mean(predict_times[eval_slice])
    avg_update_time = np.mean(update_times[eval_slice])

    # Count leaves
    n_leaves = len(gpt.root.leaves)

    print(f"Done. NRMSE={nrmse:.4f}, Coverage={coverage:.3f}, Leaves={n_leaves}")

    return {
        'rmse': rmse,
        'nrmse': nrmse,
        'coverage': coverage,
        'avg_predict_time': avg_predict_time,
        'avg_update_time': avg_update_time,
        'total_time': total_time,
        'n_leaves': n_leaves,
    }


def run_all_benchmarks():
    """Run all benchmark combinations and save results."""

    config = BenchmarkConfig()

    # Define kernel types to test
    kernel_types = ['matern', 'rbf', 'additive_d1', 'additive_d2', 'combo']

    results = []

    print("=" * 80)
    print("BENCHMARK: Additive Kernel Performance")
    print("=" * 80)
    print(f"Total points per test: {config.N_POINTS}")
    print(f"Nbar: {config.NBAR}, Retrain step: {config.RETRAIN_STEP}")
    print(f"Metrics computed on last {config.EVAL_BATCH_SIZE} points")
    print("=" * 80)
    print()

    total_tests = len(config.DIMENSIONS) * len(config.TARGET_FUNCTIONS) * len(kernel_types)
    test_num = 0

    for target_name, target_func in config.TARGET_FUNCTIONS.items():
        for n_dims in config.DIMENSIONS:
            print(f"\n{target_name.upper()} - {n_dims}D:")
            print("-" * 60)

            for kernel_type in kernel_types:
                test_num += 1
                print(f"[{test_num}/{total_tests}] ", end='')

                try:
                    result = run_single_benchmark(target_func, n_dims, kernel_type, config)

                    # Add metadata
                    result['target_function'] = target_name
                    result['n_dims'] = n_dims
                    result['kernel_type'] = kernel_type

                    results.append(result)

                except Exception as e:
                    print(f"FAILED: {e}")
                    # Still add a failed result
                    results.append({
                        'target_function': target_name,
                        'n_dims': n_dims,
                        'kernel_type': kernel_type,
                        'rmse': np.nan,
                        'nrmse': np.nan,
                        'coverage': np.nan,
                        'avg_predict_time': np.nan,
                        'avg_update_time': np.nan,
                        'total_time': np.nan,
                        'n_leaves': np.nan,
                        'error': str(e)
                    })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    output_dir = Path('benchmark_results')
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / 'additive_kernel_benchmark.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n\nResults saved to: {csv_path}")

    return df


def create_visualizations(df):
    """Create visualization plots from benchmark results.

    Args:
        df: DataFrame with benchmark results
    """

    output_dir = Path('benchmark_results')
    output_dir.mkdir(exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)

    # 1. NRMSE by dimension and kernel type
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Additive Kernel Benchmark Results: NRMSE by Dimension', fontsize=16)

    for idx, (target_name, ax) in enumerate(zip(df['target_function'].unique(), axes.flat)):
        df_target = df[df['target_function'] == target_name]

        # Pivot for heatmap
        pivot = df_target.pivot(index='kernel_type', columns='n_dims', values='nrmse')

        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=ax,
                    vmin=0, vmax=pivot.max().max(), cbar_kws={'label': 'NRMSE'})
        ax.set_title(f'{target_name.upper()}')
        ax.set_xlabel('Dimensions')
        ax.set_ylabel('Kernel Type')

    plt.tight_layout()
    plt.savefig(output_dir / 'nrmse_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'nrmse_heatmap.png'}")
    plt.close()

    # 2. Coverage by dimension and kernel type
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Additive Kernel Benchmark Results: Coverage by Dimension', fontsize=16)

    for idx, (target_name, ax) in enumerate(zip(df['target_function'].unique(), axes.flat)):
        df_target = df[df['target_function'] == target_name]

        pivot = df_target.pivot(index='kernel_type', columns='n_dims', values='coverage')

        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                    vmin=0.5, vmax=0.8, cbar_kws={'label': 'Coverage'})
        ax.set_title(f'{target_name.upper()}')
        ax.set_xlabel('Dimensions')
        ax.set_ylabel('Kernel Type')

    plt.tight_layout()
    plt.savefig(output_dir / 'coverage_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'coverage_heatmap.png'}")
    plt.close()

    # 3. Prediction time comparison
    fig, ax = plt.subplots(figsize=(14, 8))

    df_grouped = df.groupby(['kernel_type', 'n_dims'])['avg_predict_time'].mean().reset_index()

    for kernel_type in df['kernel_type'].unique():
        df_kernel = df_grouped[df_grouped['kernel_type'] == kernel_type]
        ax.plot(df_kernel['n_dims'], df_kernel['avg_predict_time'] * 1000,
                marker='o', linewidth=2, markersize=8, label=kernel_type)

    ax.set_xlabel('Number of Dimensions', fontsize=12)
    ax.set_ylabel('Average Prediction Time (ms)', fontsize=12)
    ax.set_title('Prediction Time vs Dimensionality', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_time.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'prediction_time.png'}")
    plt.close()

    # 4. NRMSE comparison across dimensions (line plot)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('NRMSE vs Dimensionality by Target Function', fontsize=16)

    for idx, (target_name, ax) in enumerate(zip(df['target_function'].unique(), axes.flat)):
        df_target = df[df['target_function'] == target_name]

        for kernel_type in df['kernel_type'].unique():
            df_kernel = df_target[df_target['kernel_type'] == kernel_type]
            ax.plot(df_kernel['n_dims'], df_kernel['nrmse'],
                    marker='o', linewidth=2, markersize=8, label=kernel_type)

        ax.set_xlabel('Number of Dimensions', fontsize=11)
        ax.set_ylabel('NRMSE', fontsize=11)
        ax.set_title(f'{target_name.upper()}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'nrmse_vs_dimension.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'nrmse_vs_dimension.png'}")
    plt.close()

    print("\nAll visualizations created successfully!")


def print_summary(df):
    """Print summary statistics from benchmark results.

    Args:
        df: DataFrame with benchmark results
    """

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Average performance by kernel type
    print("\n1. Average NRMSE by Kernel Type (across all tests):")
    print("-" * 60)
    summary = df.groupby('kernel_type')['nrmse'].agg(['mean', 'std', 'min', 'max'])
    print(summary.to_string())

    # Best kernel per dimension
    print("\n2. Best Kernel by Dimensionality (lowest average NRMSE):")
    print("-" * 60)
    best_by_dim = df.groupby('n_dims').apply(
        lambda x: x.groupby('kernel_type')['nrmse'].mean().idxmin()
    )
    for dim, best_kernel in best_by_dim.items():
        avg_nrmse = df[(df['n_dims'] == dim) & (df['kernel_type'] == best_kernel)]['nrmse'].mean()
        print(f"  {dim}D: {best_kernel:15s} (NRMSE: {avg_nrmse:.4f})")

    # Coverage statistics
    print("\n3. Average Coverage by Kernel Type:")
    print("-" * 60)
    coverage_summary = df.groupby('kernel_type')['coverage'].agg(['mean', 'std'])
    print(coverage_summary.to_string())

    # Speed comparison
    print("\n4. Average Prediction Time by Kernel Type (ms):")
    print("-" * 60)
    time_summary = df.groupby('kernel_type')['avg_predict_time'].mean() * 1000
    print(time_summary.to_string())

    print("\n" + "=" * 80)


if __name__ == '__main__':
    print("\nStarting additive kernel benchmark...")
    print("This will test 5 kernel types × 5 dimensions × 4 target functions = 100 tests")
    print("Estimated time: ~30-60 minutes\n")

    # Run benchmarks
    df = run_all_benchmarks()

    # Print summary
    print_summary(df)

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df)

    print("\n✓ Benchmark complete!")
