"""Extended kernel comparison script for testing different kernels across multiple test functions and dimensionalities.

This script compares the performance of different kernel choices on various test functions
at two different dimensionalities each to determine if a complex kernel is generally better
or only suited for specific functions/dimensions.
"""
import numpy as np
import matplotlib.pyplot as plt
from pygptreeo import GPTree
from pygptreeo.adapters import SklearnGPAdapter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
import time
import sys
import os

# Add examples directory to path for target functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'examples'))
from target_functions import Eggholder, Himmelblau, Rosenbrock, Rastrigin, Levy, Custom


def Ackley(x):
    """Computes the N-dimensional Ackley function.

    The Ackley function is widely used for testing optimization algorithms.
    It is characterized by a nearly flat outer region and a large hole at the center.

    Args:
        x (np.ndarray): Input array of shape (n_dims, n_points) with values in [0,1].

    Returns:
        np.ndarray: Function values of shape (n_points,).
    """
    xmin, xmax = -5, 5
    x_scaled = xmin + x * (xmax - xmin)

    # Handle both single point and multiple points
    if x_scaled.ndim == 1:
        x_scaled = x_scaled.reshape(-1, 1)

    dim = x_scaled.shape[0]

    sum_sq = np.sum(x_scaled**2, axis=0)
    sum_cos = np.sum(np.cos(2 * np.pi * x_scaled), axis=0)

    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / dim))
    term2 = -np.exp(sum_cos / dim)

    result = term1 + term2 + 20 + np.e

    return result if result.shape[0] > 1 else result[0]


def Griewank(x):
    """Computes the N-dimensional Griewank function.

    The Griewank function has many widespread local minima, which are
    regularly distributed. It becomes more difficult in higher dimensions.

    Args:
        x (np.ndarray): Input array of shape (n_dims, n_points) with values in [0,1].

    Returns:
        np.ndarray: Function values of shape (n_points,).
    """
    xmin, xmax = -600, 600
    x_scaled = xmin + x * (xmax - xmin)

    # Handle both single point and multiple points
    if x_scaled.ndim == 1:
        x_scaled = x_scaled.reshape(-1, 1)

    dim = x_scaled.shape[0]

    sum_term = np.sum(x_scaled**2, axis=0) / 4000

    prod_term = np.ones(x_scaled.shape[1])
    for i in range(dim):
        prod_term *= np.cos(x_scaled[i] / np.sqrt(i + 1))

    result = sum_term - prod_term + 1

    return result if result.shape[0] > 1 else result[0]


def Schwefel(x):
    """Computes the N-dimensional Schwefel function.

    The Schwefel function is complex with many local minima. The global
    minimum is geometrically distant from the next best local minima.

    Args:
        x (np.ndarray): Input array of shape (n_dims, n_points) with values in [0,1].

    Returns:
        np.ndarray: Function values of shape (n_points,).
    """
    xmin, xmax = -500, 500
    x_scaled = xmin + x * (xmax - xmin)

    # Handle both single point and multiple points
    if x_scaled.ndim == 1:
        x_scaled = x_scaled.reshape(-1, 1)

    dim = x_scaled.shape[0]

    func = np.full(x_scaled.shape[1], 418.9829 * dim)
    for i in range(dim):
        func -= x_scaled[i] * np.sin(np.sqrt(np.abs(x_scaled[i])))

    return func if func.shape[0] > 1 else func[0]


# Test function dictionary
target_dict = {
    'eggholder': Eggholder,
    'himmelblau': Himmelblau,
    'rosenbrock': Rosenbrock,
    'rastrigin': Rastrigin,
    'levy': Levy,
    'ackley': Ackley,
    'griewank': Griewank,
    'schwefel': Schwefel,
    'custom': Custom,
}


def is_within_percentage(value, target, percentage):
    """Check if value is within percentage of target."""
    if target == 0:
        return value == 0
    return (np.abs(value - target) / np.abs(target)) <= (0.01 * percentage)


def create_gpr_class(kernel_type, n_dims):
    """Factory function to create GPR classes with different kernels."""

    class CustomGPR(GaussianProcessRegressor):
        def __init__(self, kernel=None, *, alpha=1e-6, optimizer='fmin_l_bfgs_b',
                     n_restarts_optimizer=1, normalize_y=False, copy_X_train=True,
                     n_targets=None, random_state=None):
            super().__init__()

            if kernel_type == 'complex':
                # Current complex kernel: ConstantKernel * (AnisotropicRationalQuadratic + Matern)
                from pygptreeo.kernels import AnisotropicRationalQuadratic
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
            elif kernel_type == 'matern':
                # Simple Matern kernel
                self.kernel = ConstantKernel(
                    constant_value=1.0,
                    constant_value_bounds=(1e-3, 1e8)
                ) * Matern(
                    nu=1.5,
                    length_scale=[1.0]*n_dims,
                    length_scale_bounds=[(1e-5, 1e5)]*n_dims
                )
            elif kernel_type == 'rbf':
                # RBF kernel
                self.kernel = ConstantKernel(
                    constant_value=1.0,
                    constant_value_bounds=(1e-3, 1e8)
                ) * RBF(
                    length_scale=[1.0]*n_dims,
                    length_scale_bounds=[(1e-5, 1e5)]*n_dims
                )
            elif kernel_type == 'anisotropic_rq':
                # AnisotropicRationalQuadratic only (no Matern sum)
                from pygptreeo.kernels import AnisotropicRationalQuadratic
                self.kernel = ConstantKernel(
                    constant_value=1.0,
                    constant_value_bounds=(1e-3, 1e8)
                ) * AnisotropicRationalQuadratic(
                    length_scale=[1.0]*n_dims,
                    length_scale_bounds=(1e-5, 1e5),
                    alpha=1.0,
                    alpha_bounds=(1e-4, 1e4)
                )
            else:
                raise ValueError(f"Unknown kernel_type: {kernel_type}")

            self.min_length_scale = 0.001
            self.alpha = alpha
            self.optimizer = optimizer
            self.n_restarts_optimizer = 3
            self.normalize_y = normalize_y
            self.copy_X_train = copy_X_train
            self.n_targets = n_targets
            self.random_state = random_state

    return CustomGPR


def run_test(target_name, kernel_type, n_dims=3, n_pts=10000, Nbar=200,
             theta=1e-4, retrain_step=200, seed=512312):
    """Run a single test with specified parameters."""

    np.random.seed(seed)

    # Get target function
    target = target_dict[target_name]

    # Generate data
    x_min, x_max = 0.0, 1.0
    X_input = np.random.uniform(x_min, x_max, n_dims * n_pts).reshape(n_pts, n_dims)
    y_input = target(X_input.T)

    # Create GPR class for this kernel
    GPR_class = create_gpr_class(kernel_type, n_dims)

    # Create GPTree
    gpt = GPTree(
        GPR=SklearnGPAdapter(GPR_class()),
        Nbar=Nbar,
        theta=theta,
        split_position_method='median',
        split_dimension_criteria='max_uncertainty',
        retrain_every_n_points=retrain_step,
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

    # Data storage
    batch_size = 2000
    all_metrics = []

    # Storage for current batch
    current_batch_actual = []
    current_batch_predicted = []
    current_batch_std = []
    current_batch_predict_times = []
    current_batch_update_times = []

    # Run through points
    point_i = 0
    for x, y in zip(X_input, y_input):
        point_i += 1

        x = x.reshape((1, x.shape[0]))
        y = y.reshape((1, 1))

        # Predict
        start_time = time.time()
        y_pred, y_pred_std, _ = gpt.predict(x, show_progress=False, return_leaf_names=True)
        predict_time = time.time() - start_time

        # Update
        start_time = time.time()
        gpt.update_tree(x, y, 0.001 * np.abs(y))
        update_time = time.time() - start_time

        # Store data
        current_batch_actual.append(y[0][0])
        current_batch_predicted.append(y_pred[0][0])
        current_batch_std.append(y_pred_std[0][0])
        current_batch_predict_times.append(predict_time)
        current_batch_update_times.append(update_time)

        # Calculate metrics every batch_size points
        if point_i % batch_size == 0 and point_i > 0:
            actual_vals = np.array(current_batch_actual)
            predicted_vals = np.array(current_batch_predicted)
            std_devs = np.array(current_batch_std)

            # Calculate metrics
            nrmse = np.sqrt(np.mean((actual_vals - predicted_vals)**2)) / (np.max(actual_vals) - np.min(actual_vals))

            within_1 = np.mean([is_within_percentage(p, a, 1) for p, a in zip(predicted_vals, actual_vals)])
            within_2 = np.mean([is_within_percentage(p, a, 2) for p, a in zip(predicted_vals, actual_vals)])
            within_4 = np.mean([is_within_percentage(p, a, 4) for p, a in zip(predicted_vals, actual_vals)])
            within_8 = np.mean([is_within_percentage(p, a, 8) for p, a in zip(predicted_vals, actual_vals)])
            within_16 = np.mean([is_within_percentage(p, a, 16) for p, a in zip(predicted_vals, actual_vals)])

            coverage = np.mean(np.abs(actual_vals - predicted_vals) <= std_devs)
            avg_predict_time = np.mean(current_batch_predict_times)
            avg_update_time = np.mean(current_batch_update_times)

            all_metrics.append({
                'points': point_i,
                'nrmse': nrmse,
                'within_1': within_1,
                'within_2': within_2,
                'within_4': within_4,
                'within_8': within_8,
                'within_16': within_16,
                'coverage': coverage,
                'avg_predict_time': avg_predict_time,
                'avg_update_time': avg_update_time,
            })

            # Clear batch
            current_batch_actual.clear()
            current_batch_predicted.clear()
            current_batch_std.clear()
            current_batch_predict_times.clear()
            current_batch_update_times.clear()

            print(f"  {target_name}/{kernel_type}/{n_dims}D: {point_i}/{n_pts} pts, "
                  f"NRMSE={nrmse:.4f}, within_4%={within_4:.3f}, within_8%={within_8:.3f}")

    return all_metrics


def main():
    """Run extended comparison tests."""

    # Test configurations
    kernel_types = ['complex', 'anisotropic_rq', 'matern', 'rbf']
    kernel_names = {
        'complex': 'Const*(AnisRQ + Matern)',
        'anisotropic_rq': 'Const*AnisRQ',
        'matern': 'Const*Matern',
        'rbf': 'Const*RBF',
    }

    # Test functions with two dimensionalities each
    # (function_name, dim_low, dim_high)
    test_configs = [
        ('eggholder', 3, 5),
        ('himmelblau', 3, 5),
        ('rosenbrock', 3, 5),
        ('rastrigin', 3, 5),
        ('levy', 3, 5),
        ('ackley', 3, 5),
        ('griewank', 3, 5),
        ('schwefel', 3, 5),
        ('custom', 3, 5),
    ]

    # Common settings
    n_pts = 10000
    Nbar = 200
    theta = 1e-4
    retrain_step = 200
    seed = 512312

    # Run tests
    all_results = {}

    for target_name, dim_low, dim_high in test_configs:
        all_results[target_name] = {}

        for n_dims in [dim_low, dim_high]:
            print(f"\nTesting {target_name} in {n_dims}D")
            all_results[target_name][n_dims] = {}

            for kernel_type in kernel_types:
                print(f"  Testing kernel: {kernel_names[kernel_type]}")

                metrics = run_test(
                    target_name=target_name,
                    kernel_type=kernel_type,
                    n_dims=n_dims,
                    n_pts=n_pts,
                    Nbar=Nbar,
                    theta=theta,
                    retrain_step=retrain_step,
                    seed=seed,
                )

                all_results[target_name][n_dims][kernel_type] = metrics

    # Save results
    import pickle
    with open('kernel_comparison_extended_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    # Print summary
    print("\n" + "="*100)
    print("SUMMARY OF FINAL BATCH RESULTS (Fraction within 4% and 8% error)")
    print("="*100)

    for target_name, dim_low, dim_high in test_configs:
        print(f"\n{target_name.upper()}:")
        print("-" * 100)

        for n_dims in [dim_low, dim_high]:
            print(f"\n  {n_dims}D:")
            print(f"  {'Kernel':<30} {'NRMSE':<10} {'<1%':<8} {'<2%':<8} {'<4%':<8} {'<8%':<8} {'<16%':<8}")
            print("  " + "-" * 94)

            for kernel_type in kernel_types:
                final_metrics = all_results[target_name][n_dims][kernel_type][-1]
                print(f"  {kernel_names[kernel_type]:<30} "
                      f"{final_metrics['nrmse']:<10.4f} "
                      f"{final_metrics['within_1']:<8.3f} "
                      f"{final_metrics['within_2']:<8.3f} "
                      f"{final_metrics['within_4']:<8.3f} "
                      f"{final_metrics['within_8']:<8.3f} "
                      f"{final_metrics['within_16']:<8.3f}")

    print("\n" + "="*100)
    print("Results saved to kernel_comparison_extended_results.pkl")
    print("="*100)

    # Create comparison plots
    create_comparison_plots(all_results, kernel_names, test_configs)
    create_summary_heatmap(all_results, kernel_names, test_configs)


def create_comparison_plots(all_results, kernel_names, test_configs):
    """Create plots comparing kernel performance across target functions and dimensionalities."""

    n_functions = len(test_configs)
    fig, axs = plt.subplots(n_functions, 2, figsize=(16, 4*n_functions))
    fig.suptitle('Extended Kernel Comparison Across Test Functions and Dimensions', fontsize=16)

    colors = {'complex': 'blue', 'anisotropic_rq': 'green', 'matern': 'orange', 'rbf': 'red'}
    linestyles_dim = {3: '-', 5: '--'}

    for i, (target_name, dim_low, dim_high) in enumerate(test_configs):
        ax_left = axs[i, 0] if n_functions > 1 else axs[0]
        ax_right = axs[i, 1] if n_functions > 1 else axs[1]

        # Left column: Fraction within 4%
        for kernel_type in ['complex', 'anisotropic_rq', 'matern', 'rbf']:
            for n_dims in [dim_low, dim_high]:
                metrics = all_results[target_name][n_dims][kernel_type]
                points = [m['points'] for m in metrics]
                within_4 = [m['within_4'] for m in metrics]

                label = f"{kernel_names[kernel_type]} ({n_dims}D)"
                ax_left.plot(points, within_4, label=label,
                           color=colors[kernel_type],
                           linestyle=linestyles_dim[n_dims],
                           linewidth=2 if n_dims == dim_low else 1.5,
                           alpha=1.0 if n_dims == dim_low else 0.7)

        ax_left.set_xlabel('Points Processed')
        ax_left.set_ylabel('Fraction Within 4%')
        ax_left.set_title(f'{target_name.title()} - Accuracy')
        ax_left.set_ylim([0, 1])
        ax_left.legend(fontsize=6, ncol=2)
        ax_left.grid(True)

        # Right column: NRMSE
        for kernel_type in ['complex', 'anisotropic_rq', 'matern', 'rbf']:
            for n_dims in [dim_low, dim_high]:
                metrics = all_results[target_name][n_dims][kernel_type]
                points = [m['points'] for m in metrics]
                nrmse = [m['nrmse'] for m in metrics]

                label = f"{kernel_names[kernel_type]} ({n_dims}D)"
                ax_right.plot(points, nrmse, label=label,
                            color=colors[kernel_type],
                            linestyle=linestyles_dim[n_dims],
                            linewidth=2 if n_dims == dim_low else 1.5,
                            alpha=1.0 if n_dims == dim_low else 0.7)

        ax_right.set_xlabel('Points Processed')
        ax_right.set_ylabel('NRMSE')
        ax_right.set_title(f'{target_name.title()} - NRMSE')
        ax_right.set_yscale('log')
        ax_right.legend(fontsize=6, ncol=2)
        ax_right.grid(True)

    plt.tight_layout()
    plt.savefig('kernel_comparison_extended_plots.png', dpi=150)
    print("Plots saved to kernel_comparison_extended_plots.png")


def create_summary_heatmap(all_results, kernel_names, test_configs):
    """Create heatmap summary showing relative performance of complex kernel vs others."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle('Complex Kernel Performance vs Alternatives (Final Batch)', fontsize=16)

    # Create matrices for 3D and 5D results
    n_funcs = len(test_configs)
    kernel_list = ['anisotropic_rq', 'matern', 'rbf']

    # Metric: within_4%
    improvement_3d = np.zeros((n_funcs, len(kernel_list)))
    improvement_5d = np.zeros((n_funcs, len(kernel_list)))

    for i, (target_name, dim_low, dim_high) in enumerate(test_configs):
        complex_perf_3d = all_results[target_name][dim_low]['complex'][-1]['within_4']
        complex_perf_5d = all_results[target_name][dim_high]['complex'][-1]['within_4']

        for j, kernel_type in enumerate(kernel_list):
            other_perf_3d = all_results[target_name][dim_low][kernel_type][-1]['within_4']
            other_perf_5d = all_results[target_name][dim_high][kernel_type][-1]['within_4']

            # Calculate relative improvement (percentage points)
            improvement_3d[i, j] = (complex_perf_3d - other_perf_3d) * 100
            improvement_5d[i, j] = (complex_perf_5d - other_perf_5d) * 100

    # Plot 3D results
    im1 = ax1.imshow(improvement_3d, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=60)
    ax1.set_xticks(range(len(kernel_list)))
    ax1.set_xticklabels([kernel_names[k] for k in kernel_list], rotation=45, ha='right')
    ax1.set_yticks(range(n_funcs))
    ax1.set_yticklabels([name.title() for name, _, _ in test_configs])
    ax1.set_title('3D: Complex Kernel Improvement Over Alternatives\n(Percentage Points in "Within 4%" Metric)')

    # Add text annotations
    for i in range(n_funcs):
        for j in range(len(kernel_list)):
            text = ax1.text(j, i, f'{improvement_3d[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im1, ax=ax1, label='Improvement (pp)')

    # Plot 5D results
    im2 = ax2.imshow(improvement_5d, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=60)
    ax2.set_xticks(range(len(kernel_list)))
    ax2.set_xticklabels([kernel_names[k] for k in kernel_list], rotation=45, ha='right')
    ax2.set_yticks(range(n_funcs))
    ax2.set_yticklabels([name.title() for name, _, _ in test_configs])
    ax2.set_title('5D: Complex Kernel Improvement Over Alternatives\n(Percentage Points in "Within 4%" Metric)')

    # Add text annotations
    for i in range(n_funcs):
        for j in range(len(kernel_list)):
            text = ax2.text(j, i, f'{improvement_5d[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im2, ax=ax2, label='Improvement (pp)')

    plt.tight_layout()
    plt.savefig('kernel_comparison_heatmap.png', dpi=150)
    print("Heatmap saved to kernel_comparison_heatmap.png")


if __name__ == '__main__':
    main()
