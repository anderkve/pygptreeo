"""Third hyperparameter benchmark: Testing separate vs shared constants in additive kernels.

This script tests whether Const1*RBF + Const2*Matern (separate constants) performs better
than Const*(RBF + Matern) (shared constant).
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
from multiprocessing import Pool
import warnings

# Add examples directory to path for target functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'examples'))
from target_functions import Eggholder, Rastrigin, Levy


# Test function dictionary
target_dict = {
    'eggholder': Eggholder,
    'rastrigin': Rastrigin,
    'levy': Levy,
}


def is_within_percentage(value, target, percentage):
    """Check if value is within percentage of target."""
    if target == 0:
        return value == 0
    return (np.abs(value - target) / np.abs(target)) <= (0.01 * percentage)


def extract_hyperparameters(gpr, kernel_type, n_dims):
    """Extract hyperparameters from a fitted GPR model."""
    params = {}

    try:
        # Get the kernel
        kernel = gpr.kernel_

        if kernel_type == 'rbf_matern_shared':
            # Const*(RBF + Matern)
            params['constant'] = np.exp(kernel.theta[0])
            params['rbf_length_scales'] = np.exp(kernel.theta[1:1+n_dims])
            params['matern_length_scales'] = np.exp(kernel.theta[1+n_dims:1+2*n_dims])

        elif kernel_type == 'rbf_matern_separate':
            # Const1*RBF + Const2*Matern
            params['constant1'] = np.exp(kernel.theta[0])
            params['rbf_length_scales'] = np.exp(kernel.theta[1:1+n_dims])
            params['constant2'] = np.exp(kernel.theta[1+n_dims])
            params['matern_length_scales'] = np.exp(kernel.theta[2+n_dims:2+2*n_dims])

        elif kernel_type == 'anisrq_matern_shared':
            # Const*(AnisRQ + Matern)
            params['constant'] = np.exp(kernel.theta[0])
            params['anisrq_length_scales'] = np.exp(kernel.theta[1:1+n_dims])
            params['anisrq_alpha'] = np.exp(kernel.theta[1+n_dims])
            params['matern_length_scales'] = np.exp(kernel.theta[2+n_dims:2+2*n_dims])

        elif kernel_type == 'anisrq_matern_separate':
            # Const1*AnisRQ + Const2*Matern
            params['constant1'] = np.exp(kernel.theta[0])
            params['anisrq_length_scales'] = np.exp(kernel.theta[1:1+n_dims])
            params['anisrq_alpha'] = np.exp(kernel.theta[1+n_dims])
            params['constant2'] = np.exp(kernel.theta[2+n_dims])
            params['matern_length_scales'] = np.exp(kernel.theta[3+n_dims:3+2*n_dims])

    except Exception as e:
        print(f"Warning: Could not extract hyperparameters: {e}")

    return params


def create_gpr_class(kernel_type, n_dims):
    """Factory function to create GPR classes with different kernels."""

    class CustomGPR(GaussianProcessRegressor):
        def __init__(self, kernel=None, *, alpha=1e-6, optimizer='fmin_l_bfgs_b',
                     n_restarts_optimizer=1, normalize_y=False, copy_X_train=True,
                     n_targets=None, random_state=None):
            super().__init__()

            if kernel_type == 'rbf_matern_shared':
                # Const*(RBF + Matern) - shared constant
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

            elif kernel_type == 'rbf_matern_separate':
                # Const1*RBF + Const2*Matern - separate constants
                self.kernel = (
                    ConstantKernel(
                        constant_value=1.0,
                        constant_value_bounds=(1e-3, 1e8)
                    ) * RBF(
                        length_scale=[1.0]*n_dims,
                        length_scale_bounds=[(1e-5, 1e5)]*n_dims
                    )
                ) + (
                    ConstantKernel(
                        constant_value=1.0,
                        constant_value_bounds=(1e-3, 1e8)
                    ) * Matern(
                        nu=1.5,
                        length_scale=[1.0]*n_dims,
                        length_scale_bounds=[(1e-5, 1e5)]*n_dims
                    )
                )

            elif kernel_type == 'anisrq_matern_shared':
                # Const*(AnisRQ + Matern) - shared constant
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

            elif kernel_type == 'anisrq_matern_separate':
                # Const1*AnisRQ + Const2*Matern - separate constants
                from pygptreeo.kernels import AnisotropicRationalQuadratic
                self.kernel = (
                    ConstantKernel(
                        constant_value=1.0,
                        constant_value_bounds=(1e-3, 1e8)
                    ) * AnisotropicRationalQuadratic(
                        length_scale=[1.0]*n_dims,
                        length_scale_bounds=(1e-5, 1e5),
                        alpha=1.0,
                        alpha_bounds=(1e-4, 1e4)
                    )
                ) + (
                    ConstantKernel(
                        constant_value=1.0,
                        constant_value_bounds=(1e-3, 1e8)
                    ) * Matern(
                        nu=1.5,
                        length_scale=[1.0]*n_dims,
                        length_scale_bounds=[(1e-5, 1e5)]*n_dims
                    )
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
    """Run a single test with specified parameters and track hyperparameters."""

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
    all_hyperparams = []

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

            # Extract hyperparameters from all leaf nodes
            batch_hyperparams = []
            for leaf in gpt.root.leaves:
                if hasattr(leaf.my_GPR, 'sklearn_gpr') and leaf.my_GPR.is_trained():
                    params = extract_hyperparameters(leaf.my_GPR.sklearn_gpr, kernel_type, n_dims)
                    if params:
                        params['leaf_name'] = leaf.name
                        batch_hyperparams.append(params)

            all_hyperparams.append({
                'points': point_i,
                'hyperparams': batch_hyperparams
            })

            # Clear batch
            current_batch_actual.clear()
            current_batch_predicted.clear()
            current_batch_std.clear()
            current_batch_predict_times.clear()
            current_batch_update_times.clear()

            print(f"  {target_name}/{kernel_type}: {point_i}/{n_pts} pts, "
                  f"NRMSE={nrmse:.4f}, within_4%={within_4:.3f}, {len(batch_hyperparams)} leaves")

    return (target_name, kernel_type, all_metrics, all_hyperparams)


def main():
    """Run hyperparameter tracking tests with separate vs shared constants."""

    # Test configurations
    kernel_types = [
        'rbf_matern_shared',        # Const*(RBF + Matern)
        'rbf_matern_separate',      # Const1*RBF + Const2*Matern
        'anisrq_matern_shared',     # Const*(AnisRQ + Matern)
        'anisrq_matern_separate',   # Const1*AnisRQ + Const2*Matern
    ]

    kernel_names = {
        'rbf_matern_shared': 'Const*(RBF + Matern)',
        'rbf_matern_separate': 'Const1*RBF + Const2*Matern',
        'anisrq_matern_shared': 'Const*(AnisRQ + Matern)',
        'anisrq_matern_separate': 'Const1*AnisRQ + Const2*Matern',
    }

    # Test functions
    test_functions = ['eggholder', 'rastrigin', 'levy']

    # Common settings
    n_dims = 3
    n_pts = 8000
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
    print(f"Testing separate vs shared constants in additive kernels")
    print("="*100)

    # Initialize results storage
    all_results = {target: {} for target in test_functions}
    all_hyperparams = {target: {} for target in test_functions}

    # Run tests in parallel with 4 processes
    with Pool(processes=4) as pool:
        for target_name, kernel_type, metrics, hyperparams in pool.imap_unordered(run_test, test_args):
            # Store results
            all_results[target_name][kernel_type] = metrics
            all_hyperparams[target_name][kernel_type] = hyperparams

            print(f"\nCompleted: {target_name}/{kernel_type}")

    # Save results
    with open('kernel_hyperparameter_results_3.pkl', 'wb') as f:
        pickle.dump({
            'metrics': all_results,
            'hyperparams': all_hyperparams,
            'kernel_names': kernel_names,
        }, f)

    print("\n" + "="*100)
    print("Results saved to kernel_hyperparameter_results_3.pkl")
    print("="*100)

    # Create analysis plots
    analyze_hyperparameters(all_results, all_hyperparams, kernel_names, test_functions)


def analyze_hyperparameters(all_results, all_hyperparams, kernel_names, test_functions):
    """Analyze hyperparameter patterns for separate vs shared constants."""

    print("\nAnalyzing hyperparameter patterns...")

    # For each test function and kernel, compute statistics
    for target_name in test_functions:
        print(f"\n{'='*100}")
        print(f"{target_name.upper()} - Final Hyperparameter Analysis (at 8000 points)")
        print('='*100)

        for kernel_type in kernel_names.keys():
            if kernel_type not in all_hyperparams[target_name]:
                continue

            print(f"\n{kernel_names[kernel_type]}:")
            print("-" * 100)

            # Get final batch hyperparameters
            final_batch = all_hyperparams[target_name][kernel_type][-1]
            hyperparams = final_batch['hyperparams']

            if not hyperparams:
                print("  No hyperparameters available")
                continue

            # Analyze based on kernel type
            if 'shared' in kernel_type:
                # Shared constant
                constants = [h['constant'] for h in hyperparams]
                const_mean = np.mean(constants)
                print(f"  Shared constant (mean across leaves): {const_mean:.4f}")

            else:
                # Separate constants
                const1_all = [h['constant1'] for h in hyperparams]
                const2_all = [h['constant2'] for h in hyperparams]
                const1_mean = np.mean(const1_all)
                const2_mean = np.mean(const2_all)
                print(f"  Constant1 (mean across leaves): {const1_mean:.4f}")
                print(f"  Constant2 (mean across leaves): {const2_mean:.4f}")
                print(f"  Constant ratio (Const1/Const2): {const1_mean/const2_mean:.4f}")

            # Extract length scales
            if 'rbf_matern' in kernel_type:
                rbf_ls_all = [h['rbf_length_scales'] for h in hyperparams]
                matern_ls_all = [h['matern_length_scales'] for h in hyperparams]

                rbf_ls_mean = np.mean(rbf_ls_all, axis=0)
                matern_ls_mean = np.mean(matern_ls_all, axis=0)

                rbf_ls_global = np.mean([np.mean(ls) for ls in rbf_ls_all])
                matern_ls_global = np.mean([np.mean(ls) for ls in matern_ls_all])

                print(f"  RBF length scales (mean across leaves): {rbf_ls_mean}")
                print(f"  Matern length scales (mean across leaves): {matern_ls_mean}")
                print(f"  Length scale ratio (RBF/Matern): {rbf_ls_global/matern_ls_global:.4f}")

            elif 'anisrq_matern' in kernel_type:
                anisrq_ls_all = [h['anisrq_length_scales'] for h in hyperparams]
                anisrq_alpha_all = [h['anisrq_alpha'] for h in hyperparams]
                matern_ls_all = [h['matern_length_scales'] for h in hyperparams]

                anisrq_ls_mean = np.mean(anisrq_ls_all, axis=0)
                matern_ls_mean = np.mean(matern_ls_all, axis=0)
                anisrq_alpha_mean = np.mean(anisrq_alpha_all)

                anisrq_ls_global = np.mean([np.mean(ls) for ls in anisrq_ls_all])
                matern_ls_global = np.mean([np.mean(ls) for ls in matern_ls_all])

                print(f"  AnisRQ length scales (mean across leaves): {anisrq_ls_mean}")
                print(f"  AnisRQ alpha (mean across leaves): {anisrq_alpha_mean:.4f}")
                print(f"  Matern length scales (mean across leaves): {matern_ls_mean}")
                print(f"  Length scale ratio (AnisRQ/Matern): {anisrq_ls_global/matern_ls_global:.4f}")

            # Print performance for reference
            final_metrics = all_results[target_name][kernel_type][-1]
            print(f"  Performance: NRMSE={final_metrics['nrmse']:.4f}, within_4%={final_metrics['within_4']:.3f}")

    # Create visualization
    create_hyperparameter_plots(all_results, all_hyperparams, kernel_names, test_functions)


def create_hyperparameter_plots(all_results, all_hyperparams, kernel_names, test_functions):
    """Create plots showing constant and length scale evolution."""

    n_funcs = len(test_functions)
    fig, axs = plt.subplots(n_funcs, 3, figsize=(18, 5*n_funcs))
    if n_funcs == 1:
        axs = axs.reshape(1, -1)

    fig.suptitle('Separate vs Shared Constants: Does Individual Scaling Matter?', fontsize=16)

    colors = {
        'rbf_matern_shared': 'blue',
        'rbf_matern_separate': 'cyan',
        'anisrq_matern_shared': 'red',
        'anisrq_matern_separate': 'orange',
    }

    for i, target_name in enumerate(test_functions):
        # Column 1: Constant ratios (for separate) or constant values (for shared)
        ax1 = axs[i, 0]

        for kernel_type in kernel_names.keys():
            if kernel_type not in all_hyperparams[target_name]:
                continue

            batches = all_hyperparams[target_name][kernel_type]
            points = [b['points'] for b in batches]

            if 'separate' in kernel_type:
                # Plot ratio of constants
                ratios = []
                for batch in batches:
                    hyperparams = batch['hyperparams']
                    if not hyperparams:
                        ratios.append(np.nan)
                        continue
                    const1 = np.mean([h['constant1'] for h in hyperparams])
                    const2 = np.mean([h['constant2'] for h in hyperparams])
                    ratios.append(const1 / const2)

                ax1.plot(points, ratios, label=kernel_names[kernel_type],
                        color=colors[kernel_type], linewidth=2, linestyle='--')

        ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Ratio=1')
        ax1.set_xlabel('Points Processed')
        ax1.set_ylabel('Constant Ratio (Const1/Const2)')
        ax1.set_title(f'{target_name.title()}: Constant Specialization')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Column 2: Length scale ratios
        ax2 = axs[i, 1]

        for kernel_type in kernel_names.keys():
            if kernel_type not in all_hyperparams[target_name]:
                continue

            batches = all_hyperparams[target_name][kernel_type]
            points = [b['points'] for b in batches]
            ratios = []

            for batch in batches:
                hyperparams = batch['hyperparams']
                if not hyperparams:
                    ratios.append(np.nan)
                    continue

                if 'rbf_matern' in kernel_type:
                    rbf_ls = np.mean([np.mean(h['rbf_length_scales']) for h in hyperparams])
                    matern_ls = np.mean([np.mean(h['matern_length_scales']) for h in hyperparams])
                    ratios.append(rbf_ls / matern_ls)
                elif 'anisrq_matern' in kernel_type:
                    anisrq_ls = np.mean([np.mean(h['anisrq_length_scales']) for h in hyperparams])
                    matern_ls = np.mean([np.mean(h['matern_length_scales']) for h in hyperparams])
                    ratios.append(anisrq_ls / matern_ls)

            linestyle = '-' if 'shared' in kernel_type else '--'
            ax2.plot(points, ratios, label=kernel_names[kernel_type],
                    color=colors[kernel_type], linewidth=2, linestyle=linestyle)

        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Ratio=1')
        ax2.set_xlabel('Points Processed')
        ax2.set_ylabel('Length Scale Ratio (Component1/Matern)')
        ax2.set_title(f'{target_name.title()}: Length Scale Specialization')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        # Column 3: Performance comparison
        ax3 = axs[i, 2]

        for kernel_type in kernel_names.keys():
            if kernel_type not in all_results[target_name]:
                continue

            metrics = all_results[target_name][kernel_type]
            points = [m['points'] for m in metrics]
            within_4 = [m['within_4'] for m in metrics]

            linestyle = '-' if 'shared' in kernel_type else '--'
            ax3.plot(points, within_4, label=kernel_names[kernel_type],
                    color=colors[kernel_type], linewidth=2, linestyle=linestyle)

        ax3.set_xlabel('Points Processed')
        ax3.set_ylabel('Fraction Within 4%')
        ax3.set_title(f'{target_name.title()}: Performance Comparison')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('kernel_hyperparameter_analysis_3.png', dpi=150)
    print("\nHyperparameter analysis plot saved to kernel_hyperparameter_analysis_3.png")


if __name__ == '__main__':
    main()
