"""Quick performance test for first 8000 points"""
import numpy as np
from pygptreeo import GPTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
import sys

from examples.target_functions import Eggholder

def is_within_percentage(value, target, percentage):
    if target == 0:
        return value == 0
    return (np.abs(value - target) / np.abs(target)) <= (0.01 * percentage)

np.random.seed(512312)

target = Eggholder
n_dims = 3
n_pts = 8000  # Only test first 8000 points

# Parameters from performance_test.py
Nbar = 200
theta = 1e-4
retrain_step = 200
x_min = 0.0
x_max = 1.0

X_input = np.random.uniform(x_min, x_max, n_dims * n_pts).reshape(n_pts, n_dims)
y_input = target(X_input.T)


class my_GPR_class(GaussianProcessRegressor):
    def __init__(self, kernel=None, *, alpha=1e-6, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=1, normalize_y=False, copy_X_train=True, n_targets=None, random_state=None):
        super().__init__()

        from pygptreeo.kernels import AnisotropicRationalQuadratic
        self.kernel = ConstantKernel(
            constant_value=1.0,
            constant_value_bounds=(1e-3,1e8)
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

        self.min_length_scale = 0.001
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = 3
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.n_targets = n_targets
        self.random_state = random_state


# Construct GPTree with configuration from command line args or defaults
config = {
    'Nbar': Nbar,
    'theta': theta,
    'split_position_method': 'median',
    'split_dimension_criteria': 'max_uncertainty',
    'retrain_every_n_points': retrain_step,
    'use_calibrated_sigma': True,
    'splitting_strategy': 'gradual',
    'max_n_pred_leaves': 3,
    'aggregation': 'moe',
    'use_hyperparameter_inheritance': False,
    'use_standard_scaling': True,
    'enable_point_rejection': False,
    'rejection_threshold': 1e-2,
    'min_points_before_rejection': 25,
    'enable_point_merging': False,
    'merge_distance_threshold': 0.01,
    'min_points_before_merging': 10,
    'enable_split_evaluation': True,
    'n_split_candidates': 4,
    'split_eval_train_fraction': 0.4,
    'split_eval_min_points': 20,
}

# Allow overriding specific configs from command line
if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, val = arg.split('=', 1)
            if key in config:
                # Try to parse as int, float, bool, or keep as string
                try:
                    if val.lower() in ('true', 'false'):
                        config[key] = val.lower() == 'true'
                    elif '.' in val:
                        config[key] = float(val)
                    else:
                        config[key] = int(val)
                except ValueError:
                    config[key] = val

print("Configuration:")
for k, v in config.items():
    print(f"  {k}: {v}")
print()

gpt = GPTree(GPR=my_GPR_class(), **config)

# Track statistics
within_2_percent_count = 0
predicted_values = []
actual_values = []
errors = []

# Run through points
for point_i in range(n_pts):
    x = X_input[point_i].reshape((1, n_dims))
    y = y_input[point_i].reshape((1, 1))

    # Predict
    y_pred, y_pred_std, _ = gpt.predict(x, show_progress=False, return_leaf_names=True)

    # Update tree
    gpt.update_tree(x, y, 0.001 * np.abs(y))

    # Track accuracy
    abs_err = np.abs(y_pred[0][0] - y[0][0])
    rel_err = abs_err / np.max([np.abs(y[0][0]), 1e-10])

    if is_within_percentage(y_pred[0][0], y[0][0], 2):
        within_2_percent_count += 1

    predicted_values.append(y_pred[0][0])
    actual_values.append(y[0][0])
    errors.append(abs_err)

    if (point_i + 1) % 1000 == 0:
        accuracy_so_far = within_2_percent_count / (point_i + 1)
        print(f"Point {point_i + 1}: Within 2% accuracy = {accuracy_so_far:.4f} ({within_2_percent_count}/{point_i + 1})")

# Final results
accuracy = within_2_percent_count / n_pts
avg_error = np.mean(errors)
median_error = np.median(errors)

print()
print("="*60)
print(f"FINAL RESULTS (first {n_pts} points):")
print(f"  Points within 2% error: {within_2_percent_count}/{n_pts} ({accuracy:.4f})")
print(f"  Average absolute error: {avg_error:.6f}")
print(f"  Median absolute error: {median_error:.6f}")
print("="*60)
