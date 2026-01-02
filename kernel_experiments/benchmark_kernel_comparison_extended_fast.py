"""Fast parallelized kernel comparison script for testing different kernels across multiple test functions.

This script uses multiprocessing to run tests in parallel and updates plots progressively
as tests complete.
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
import pickle
from multiprocessing import Pool, Manager
import warnings

# Add examples directory to path for target functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'examples'))
from target_functions import Eggholder, Himmelblau, Rosenbrock, Rastrigin, Levy, Custom


def Ackley(x):
    """Computes the N-dimensional Ackley function."""
    xmin, xmax = -5, 5
    x_scaled = xmin + x * (xmax - xmin)

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
    """Computes the N-dimensional Griewank function."""
    xmin, xmax = -600, 600
    x_scaled = xmin + x * (xmax - xmin)

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
    """Computes the N-dimensional Schwefel function."""
    xmin, xmax = -500, 500
    x_scaled = xmin + x * (xmax - xmin)

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
            elif kernel_type == 'rbf_matern':
                # New kernel: Const*(RBF + Matern(nu=1.5))
                self.kernel = ConstantKernel(
                    constant_value=1.0,
                    constant_value_bounds=(1e-3, 1e8)
                ) * (RBF(
                    length_scale=[1.0]*n_dims,
                    length_scale_bounds=[(1e-5, 1e5)]*n_dims
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


def run_test(args):
    """Run a single test with specified parameters. Designed for parallel execution."""

    target_name, kernel_type, n_dims, n_pts, Nbar, theta, retrain_step, seed = args

    # Suppress sklearn warnings in worker processes
    warnings.filterwarnings('ignore')

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

            print(f"  {target_name}/{kernel_type}/3D: {point_i}/{n_pts} pts, "
                  f"NRMSE={nrmse:.4f}, within_4%={within_4:.3f}, within_8%={within_8:.3f}")

    return (target_name, kernel_type, all_metrics)


def update_plot(all_results, kernel_names, test_functions):
    """Update the comparison plot with current results."""

    n_functions = len(test_functions)
    fig, axs = plt.subplots(n_functions, 2, figsize=(16, 4*n_functions))
    if n_functions == 1:
        axs = axs.reshape(1, -1)
    fig.suptitle('Extended Kernel Comparison Across Test Functions (3D Only)', fontsize=16)

    colors = {
        'complex': 'blue',
        'anisotropic_rq': 'green',
        'rbf_matern': 'purple',
        'matern': 'orange',
        'rbf': 'red'
    }

    for i, target_name in enumerate(test_functions):
        ax_left = axs[i, 0]
        ax_right = axs[i, 1]

        # Check if we have any results for this target
        has_data = False

        if target_name in all_results:
            # Left column: Fraction within 4%
            for kernel_type in ['complex', 'anisotropic_rq', 'rbf_matern', 'matern', 'rbf']:
                if kernel_type in all_results[target_name]:
                    metrics = all_results[target_name][kernel_type]
                    if len(metrics) > 0:
                        has_data = True
                        points = [m['points'] for m in metrics]
                        within_4 = [m['within_4'] for m in metrics]

                        label = kernel_names[kernel_type]
                        ax_left.plot(points, within_4, label=label,
                                   color=colors[kernel_type],
                                   linewidth=2,
                                   alpha=1.0)

            # Right column: NRMSE
            for kernel_type in ['complex', 'anisotropic_rq', 'rbf_matern', 'matern', 'rbf']:
                if kernel_type in all_results[target_name]:
                    metrics = all_results[target_name][kernel_type]
                    if len(metrics) > 0:
                        points = [m['points'] for m in metrics]
                        nrmse = [m['nrmse'] for m in metrics]

                        label = kernel_names[kernel_type]
                        ax_right.plot(points, nrmse, label=label,
                                    color=colors[kernel_type],
                                    linewidth=2,
                                    alpha=1.0)

        ax_left.set_xlabel('Points Processed')
        ax_left.set_ylabel('Fraction Within 4%')
        ax_left.set_title(f'{target_name.title()} - Accuracy')
        ax_left.set_ylim([0, 1])
        if has_data:
            ax_left.legend(fontsize=8)
        ax_left.grid(True)

        ax_right.set_xlabel('Points Processed')
        ax_right.set_ylabel('NRMSE')
        ax_right.set_title(f'{target_name.title()} - NRMSE')
        ax_right.set_yscale('log')
        if has_data:
            ax_right.legend(fontsize=8)
        ax_right.grid(True)

    plt.tight_layout()
    plt.savefig('kernel_comparison_extended_plots.png', dpi=150)
    plt.close(fig)
    print("  Plot updated: kernel_comparison_extended_plots.png")


def main():
    """Run extended comparison tests with parallelization."""

    # Test configurations
    kernel_types = ['complex', 'anisotropic_rq', 'rbf_matern', 'matern', 'rbf']
    kernel_names = {
        'complex': 'Const*(AnisRQ + Matern)',
        'anisotropic_rq': 'Const*AnisRQ',
        'rbf_matern': 'Const*(RBF + Matern)',
        'matern': 'Const*Matern',
        'rbf': 'Const*RBF',
    }

    # Test functions (3D only)
    test_functions = [
        'eggholder',
        'himmelblau',
        'rosenbrock',
        'rastrigin',
        'levy',
        'ackley',
        'griewank',
        'schwefel',
        'custom',
    ]

    # Common settings
    n_dims = 3
    n_pts = 8000  # Reduced from 10000
    Nbar = 200
    theta = 1e-4
    retrain_step = 200
    seed = 512312

    # Build all test configurations
    test_args = []
    for target_name in test_functions:
        for kernel_type in kernel_types:
            test_args.append((
                target_name, kernel_type, n_dims, n_pts,
                Nbar, theta, retrain_step, seed
            ))

    print(f"Running {len(test_args)} tests in parallel with 4 processes...")
    print(f"Test configuration: {n_pts} points, 3D, {len(test_functions)} functions, {len(kernel_types)} kernels")
    print("="*100)

    # Initialize results storage
    all_results = {target: {} for target in test_functions}

    # Run tests in parallel with 4 processes
    with Pool(processes=4) as pool:
        for target_name, kernel_type, metrics in pool.imap_unordered(run_test, test_args):
            # Store results
            all_results[target_name][kernel_type] = metrics

            print(f"\nCompleted: {target_name}/{kernel_type}")

            # Update plot after each test completes
            update_plot(all_results, kernel_names, test_functions)

    # Save final results
    with open('kernel_comparison_extended_fast_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    # Print summary
    print("\n" + "="*100)
    print("SUMMARY OF FINAL BATCH RESULTS (Fraction within 4% and 8% error)")
    print("="*100)

    for target_name in test_functions:
        print(f"\n{target_name.upper()} (3D):")
        print(f"  {'Kernel':<30} {'NRMSE':<10} {'<1%':<8} {'<2%':<8} {'<4%':<8} {'<8%':<8} {'<16%':<8}")
        print("  " + "-" * 94)

        for kernel_type in kernel_types:
            if kernel_type in all_results[target_name] and len(all_results[target_name][kernel_type]) > 0:
                final_metrics = all_results[target_name][kernel_type][-1]
                print(f"  {kernel_names[kernel_type]:<30} "
                      f"{final_metrics['nrmse']:<10.4f} "
                      f"{final_metrics['within_1']:<8.3f} "
                      f"{final_metrics['within_2']:<8.3f} "
                      f"{final_metrics['within_4']:<8.3f} "
                      f"{final_metrics['within_8']:<8.3f} "
                      f"{final_metrics['within_16']:<8.3f}")

    print("\n" + "="*100)
    print("Results saved to kernel_comparison_extended_fast_results.pkl")
    print("Final plot saved to kernel_comparison_extended_plots.png")
    print("="*100)


if __name__ == '__main__':
    main()
