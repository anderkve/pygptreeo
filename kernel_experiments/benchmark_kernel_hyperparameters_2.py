"""Second hyperparameter benchmark: Testing if two instances of the same kernel can specialize.

This script tests whether additive structure alone (e.g., RBF + RBF) enables multi-scale modeling,
or if different kernel types are needed (e.g., RBF + Matern).
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

        # Extract constant
        params['constant'] = np.exp(kernel.theta[0])

        if kernel_type == 'rbf_rbf':
            # Const*(RBF + RBF): Extract both RBF length scales
            params['rbf1_length_scales'] = np.exp(kernel.theta[1:1+n_dims])
            params['rbf2_length_scales'] = np.exp(kernel.theta[1+n_dims:1+2*n_dims])

        elif kernel_type == 'matern_matern':
            # Const*(Matern + Matern): Extract both Matern length scales
            params['matern1_length_scales'] = np.exp(kernel.theta[1:1+n_dims])
            params['matern2_length_scales'] = np.exp(kernel.theta[1+n_dims:1+2*n_dims])

        elif kernel_type == 'anisrq_anisrq':
            # Const*(AnisRQ + AnisRQ): Extract both AnisRQ components
            params['anisrq1_length_scales'] = np.exp(kernel.theta[1:1+n_dims])
            params['anisrq1_alpha'] = np.exp(kernel.theta[1+n_dims])
            params['anisrq2_length_scales'] = np.exp(kernel.theta[2+n_dims:2+2*n_dims])
            params['anisrq2_alpha'] = np.exp(kernel.theta[2+2*n_dims])

        elif kernel_type == 'rbf_matern':
            # Const*(RBF + Matern)
            params['rbf_length_scales'] = np.exp(kernel.theta[1:1+n_dims])
            params['matern_length_scales'] = np.exp(kernel.theta[1+n_dims:1+2*n_dims])

        elif kernel_type == 'anisrq_matern':
            # Const*(AnisRQ + Matern)
            params['anisrq_length_scales'] = np.exp(kernel.theta[1:1+n_dims])
            params['anisrq_alpha'] = np.exp(kernel.theta[1+n_dims])
            params['matern_length_scales'] = np.exp(kernel.theta[2+n_dims:2+2*n_dims])

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

            if kernel_type == 'rbf_rbf':
                # Const*(RBF + RBF)
                self.kernel = ConstantKernel(
                    constant_value=1.0,
                    constant_value_bounds=(1e-3, 1e8)
                ) * (RBF(
                    length_scale=[1.0]*n_dims,
                    length_scale_bounds=[(1e-5, 1e5)]*n_dims
                ) + RBF(
                    length_scale=[1.0]*n_dims,
                    length_scale_bounds=[(1e-5, 1e5)]*n_dims
                ))

            elif kernel_type == 'matern_matern':
                # Const*(Matern + Matern)
                self.kernel = ConstantKernel(
                    constant_value=1.0,
                    constant_value_bounds=(1e-3, 1e8)
                ) * (Matern(
                    nu=1.5,
                    length_scale=[1.0]*n_dims,
                    length_scale_bounds=[(1e-5, 1e5)]*n_dims
                ) + Matern(
                    nu=1.5,
                    length_scale=[1.0]*n_dims,
                    length_scale_bounds=[(1e-5, 1e5)]*n_dims
                ))

            elif kernel_type == 'anisrq_anisrq':
                # Const*(AnisRQ + AnisRQ)
                from pygptreeo.kernels import AnisotropicRationalQuadratic
                self.kernel = ConstantKernel(
                    constant_value=1.0,
                    constant_value_bounds=(1e-3, 1e8)
                ) * (AnisotropicRationalQuadratic(
                    length_scale=[1.0]*n_dims,
                    length_scale_bounds=(1e-5, 1e5),
                    alpha=1.0,
                    alpha_bounds=(1e-4, 1e4)
                ) + AnisotropicRationalQuadratic(
                    length_scale=[1.0]*n_dims,
                    length_scale_bounds=(1e-5, 1e5),
                    alpha=1.0,
                    alpha_bounds=(1e-4, 1e4)
                ))

            elif kernel_type == 'rbf_matern':
                # Const*(RBF + Matern)
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

            elif kernel_type == 'anisrq_matern':
                # Const*(AnisRQ + Matern)
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
    """Run hyperparameter tracking tests with same-kernel additions."""

    # Test configurations
    kernel_types = [
        'rbf_rbf',           # Const*(RBF + RBF)
        'matern_matern',     # Const*(Matern + Matern)
        'anisrq_anisrq',     # Const*(AnisRQ + AnisRQ)
        'rbf_matern',        # Const*(RBF + Matern)
        'anisrq_matern',     # Const*(AnisRQ + Matern)
    ]

    kernel_names = {
        'rbf_rbf': 'Const*(RBF + RBF)',
        'matern_matern': 'Const*(Matern + Matern)',
        'anisrq_anisrq': 'Const*(AnisRQ + AnisRQ)',
        'rbf_matern': 'Const*(RBF + Matern)',
        'anisrq_matern': 'Const*(AnisRQ + Matern)',
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
    print(f"Testing if same-kernel additions (RBF+RBF, Matern+Matern) can specialize")
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
    with open('kernel_hyperparameter_results_2.pkl', 'wb') as f:
        pickle.dump({
            'metrics': all_results,
            'hyperparams': all_hyperparams,
            'kernel_names': kernel_names,
        }, f)

    print("\n" + "="*100)
    print("Results saved to kernel_hyperparameter_results_2.pkl")
    print("="*100)

    # Create analysis plots
    analyze_hyperparameters(all_results, all_hyperparams, kernel_names, test_functions)


def analyze_hyperparameters(all_results, all_hyperparams, kernel_names, test_functions):
    """Analyze hyperparameter patterns for same-kernel additions."""

    print("\nAnalyzing hyperparameter patterns...")

    # For each test function and kernel, compute statistics of length scales
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
            if kernel_type == 'rbf_rbf':
                rbf1_ls_all = [h['rbf1_length_scales'] for h in hyperparams]
                rbf2_ls_all = [h['rbf2_length_scales'] for h in hyperparams]

                rbf1_ls_mean = np.mean(rbf1_ls_all, axis=0)
                rbf2_ls_mean = np.mean(rbf2_ls_all, axis=0)

                rbf1_ls_global = np.mean([np.mean(ls) for ls in rbf1_ls_all])
                rbf2_ls_global = np.mean([np.mean(ls) for ls in rbf2_ls_all])

                print(f"  RBF1 length scales (mean across leaves): {rbf1_ls_mean}")
                print(f"  RBF2 length scales (mean across leaves): {rbf2_ls_mean}")
                print(f"  RBF1 global avg: {rbf1_ls_global:.4f}")
                print(f"  RBF2 global avg: {rbf2_ls_global:.4f}")
                print(f"  Ratio (RBF1/RBF2): {rbf1_ls_global/rbf2_ls_global:.4f}")

            elif kernel_type == 'matern_matern':
                matern1_ls_all = [h['matern1_length_scales'] for h in hyperparams]
                matern2_ls_all = [h['matern2_length_scales'] for h in hyperparams]

                matern1_ls_mean = np.mean(matern1_ls_all, axis=0)
                matern2_ls_mean = np.mean(matern2_ls_all, axis=0)

                matern1_ls_global = np.mean([np.mean(ls) for ls in matern1_ls_all])
                matern2_ls_global = np.mean([np.mean(ls) for ls in matern2_ls_all])

                print(f"  Matern1 length scales (mean across leaves): {matern1_ls_mean}")
                print(f"  Matern2 length scales (mean across leaves): {matern2_ls_mean}")
                print(f"  Matern1 global avg: {matern1_ls_global:.4f}")
                print(f"  Matern2 global avg: {matern2_ls_global:.4f}")
                print(f"  Ratio (Matern1/Matern2): {matern1_ls_global/matern2_ls_global:.4f}")

            elif kernel_type == 'anisrq_anisrq':
                anisrq1_ls_all = [h['anisrq1_length_scales'] for h in hyperparams]
                anisrq1_alpha_all = [h['anisrq1_alpha'] for h in hyperparams]
                anisrq2_ls_all = [h['anisrq2_length_scales'] for h in hyperparams]
                anisrq2_alpha_all = [h['anisrq2_alpha'] for h in hyperparams]

                anisrq1_ls_mean = np.mean(anisrq1_ls_all, axis=0)
                anisrq2_ls_mean = np.mean(anisrq2_ls_all, axis=0)
                anisrq1_alpha_mean = np.mean(anisrq1_alpha_all)
                anisrq2_alpha_mean = np.mean(anisrq2_alpha_all)

                anisrq1_ls_global = np.mean([np.mean(ls) for ls in anisrq1_ls_all])
                anisrq2_ls_global = np.mean([np.mean(ls) for ls in anisrq2_ls_all])

                print(f"  AnisRQ1 length scales (mean across leaves): {anisrq1_ls_mean}")
                print(f"  AnisRQ1 alpha (mean across leaves): {anisrq1_alpha_mean:.4f}")
                print(f"  AnisRQ2 length scales (mean across leaves): {anisrq2_ls_mean}")
                print(f"  AnisRQ2 alpha (mean across leaves): {anisrq2_alpha_mean:.4f}")
                print(f"  AnisRQ1 global avg: {anisrq1_ls_global:.4f}")
                print(f"  AnisRQ2 global avg: {anisrq2_ls_global:.4f}")
                print(f"  Ratio (AnisRQ1/AnisRQ2): {anisrq1_ls_global/anisrq2_ls_global:.4f}")

            elif kernel_type == 'rbf_matern':
                rbf_ls_all = [h['rbf_length_scales'] for h in hyperparams]
                matern_ls_all = [h['matern_length_scales'] for h in hyperparams]

                rbf_ls_mean = np.mean(rbf_ls_all, axis=0)
                matern_ls_mean = np.mean(matern_ls_all, axis=0)

                rbf_ls_global = np.mean([np.mean(ls) for ls in rbf_ls_all])
                matern_ls_global = np.mean([np.mean(ls) for ls in matern_ls_all])

                print(f"  RBF length scales (mean across leaves): {rbf_ls_mean}")
                print(f"  Matern length scales (mean across leaves): {matern_ls_mean}")
                print(f"  RBF global avg: {rbf_ls_global:.4f}")
                print(f"  Matern global avg: {matern_ls_global:.4f}")
                print(f"  Ratio (RBF/Matern): {rbf_ls_global/matern_ls_global:.4f}")

            elif kernel_type == 'anisrq_matern':
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
                print(f"  AnisRQ global avg: {anisrq_ls_global:.4f}")
                print(f"  Matern global avg: {matern_ls_global:.4f}")
                print(f"  Ratio (AnisRQ/Matern): {anisrq_ls_global/matern_ls_global:.4f}")

            # Print performance for reference
            final_metrics = all_results[target_name][kernel_type][-1]
            print(f"  Performance: NRMSE={final_metrics['nrmse']:.4f}, within_4%={final_metrics['within_4']:.3f}")

    # Create visualization
    create_hyperparameter_plots(all_results, all_hyperparams, kernel_names, test_functions)


def create_hyperparameter_plots(all_results, all_hyperparams, kernel_names, test_functions):
    """Create plots showing hyperparameter evolution for same-kernel additions."""

    n_funcs = len(test_functions)
    fig, axs = plt.subplots(n_funcs, 3, figsize=(18, 5*n_funcs))
    if n_funcs == 1:
        axs = axs.reshape(1, -1)

    fig.suptitle('Same-Kernel Addition: Can Two Identical Kernels Specialize to Different Scales?', fontsize=16)

    colors = {
        'rbf_rbf': 'blue',
        'matern_matern': 'green',
        'anisrq_anisrq': 'red',
        'rbf_matern': 'purple',
        'anisrq_matern': 'orange',
    }

    for i, target_name in enumerate(test_functions):
        # Column 1: Length scale ratios for all kernels
        ax1 = axs[i, 0]

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

                if kernel_type == 'rbf_rbf':
                    rbf1_ls = np.mean([np.mean(h['rbf1_length_scales']) for h in hyperparams])
                    rbf2_ls = np.mean([np.mean(h['rbf2_length_scales']) for h in hyperparams])
                    ratios.append(rbf1_ls / rbf2_ls)
                elif kernel_type == 'matern_matern':
                    m1_ls = np.mean([np.mean(h['matern1_length_scales']) for h in hyperparams])
                    m2_ls = np.mean([np.mean(h['matern2_length_scales']) for h in hyperparams])
                    ratios.append(m1_ls / m2_ls)
                elif kernel_type == 'anisrq_anisrq':
                    a1_ls = np.mean([np.mean(h['anisrq1_length_scales']) for h in hyperparams])
                    a2_ls = np.mean([np.mean(h['anisrq2_length_scales']) for h in hyperparams])
                    ratios.append(a1_ls / a2_ls)
                elif kernel_type == 'rbf_matern':
                    rbf_ls = np.mean([np.mean(h['rbf_length_scales']) for h in hyperparams])
                    matern_ls = np.mean([np.mean(h['matern_length_scales']) for h in hyperparams])
                    ratios.append(rbf_ls / matern_ls)
                elif kernel_type == 'anisrq_matern':
                    anisrq_ls = np.mean([np.mean(h['anisrq_length_scales']) for h in hyperparams])
                    matern_ls = np.mean([np.mean(h['matern_length_scales']) for h in hyperparams])
                    ratios.append(anisrq_ls / matern_ls)

            ax1.plot(points, ratios, label=kernel_names[kernel_type],
                    color=colors[kernel_type], linewidth=2)

        ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Ratio=1')
        ax1.set_xlabel('Points Processed')
        ax1.set_ylabel('Length Scale Ratio (Component1/Component2)')
        ax1.set_title(f'{target_name.title()}: Length Scale Specialization')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Column 2: Average length scales over time
        ax2 = axs[i, 1]

        for kernel_type in kernel_names.keys():
            if kernel_type not in all_hyperparams[target_name]:
                continue

            batches = all_hyperparams[target_name][kernel_type]
            points = [b['points'] for b in batches]
            avg_ls = []

            for batch in batches:
                hyperparams = batch['hyperparams']
                if not hyperparams:
                    avg_ls.append(np.nan)
                    continue

                # Average all length scales
                all_ls = []
                for h in hyperparams:
                    if 'rbf1_length_scales' in h:
                        all_ls.extend(h['rbf1_length_scales'])
                        all_ls.extend(h['rbf2_length_scales'])
                    elif 'matern1_length_scales' in h:
                        all_ls.extend(h['matern1_length_scales'])
                        all_ls.extend(h['matern2_length_scales'])
                    elif 'anisrq1_length_scales' in h:
                        all_ls.extend(h['anisrq1_length_scales'])
                        all_ls.extend(h['anisrq2_length_scales'])
                    elif 'rbf_length_scales' in h:
                        all_ls.extend(h['rbf_length_scales'])
                        all_ls.extend(h['matern_length_scales'])
                    elif 'anisrq_length_scales' in h:
                        all_ls.extend(h['anisrq_length_scales'])
                        all_ls.extend(h['matern_length_scales'])

                avg_ls.append(np.mean(all_ls) if all_ls else np.nan)

            ax2.plot(points, avg_ls, label=kernel_names[kernel_type],
                    color=colors[kernel_type], linewidth=2)

        ax2.set_xlabel('Points Processed')
        ax2.set_ylabel('Average Length Scale')
        ax2.set_title(f'{target_name.title()}: Average Length Scale Evolution')
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

            ax3.plot(points, within_4, label=kernel_names[kernel_type],
                    color=colors[kernel_type], linewidth=2)

        ax3.set_xlabel('Points Processed')
        ax3.set_ylabel('Fraction Within 4%')
        ax3.set_title(f'{target_name.title()}: Performance Comparison')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('kernel_hyperparameter_analysis_2.png', dpi=150)
    print("\nHyperparameter analysis plot saved to kernel_hyperparameter_analysis_2.png")


if __name__ == '__main__':
    main()
