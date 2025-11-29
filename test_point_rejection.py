"""Test script for intelligent point rejection feature.

This script demonstrates the point rejection functionality by:
1. Creating a GPTree with point rejection enabled
2. Training online (one point at a time) on a test function
3. Monitoring how many points are rejected during training
4. Comparing with a tree without rejection
"""
import numpy as np
from pygptreeo import GPTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

# Simple test function
def test_function(X):
    """Simple 2D test function"""
    return np.sin(3 * X[:, 0]) + np.cos(2 * X[:, 1])

# Set random seed for reproducibility
np.random.seed(12345)

# Test settings
n_dims = 2
n_pts = 2000  # Train online with many points

# GPR configuration
class TestGPR(GaussianProcessRegressor):
    def __init__(self):
        super().__init__()
        self.kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=1.5, length_scale=[1.0]*n_dims, length_scale_bounds=[(1e-3, 1e3)]*n_dims)
        self.alpha = 1e-6
        self.n_restarts_optimizer = 0  # Fast for testing

# Generate test data
X_input = np.random.uniform(0, 1, n_dims * n_pts).reshape(n_pts, n_dims)
y_input = test_function(X_input).reshape(-1, 1)

# Test set for evaluation
X_test = np.random.uniform(0, 1, (100, n_dims))
y_test_true = test_function(X_test).reshape(-1, 1)

print("=" * 70)
print("Testing Intelligent Point Rejection (Online Learning)")
print("=" * 70)
print(f"Total training points: {n_pts}")
print(f"Dimensions: {n_dims}")
print()

# Create GPTree WITH point rejection
print("Creating GPTree WITH point rejection enabled...")
gpt_rejection = GPTree(
    GPR=TestGPR(),
    Nbar=100,
    theta=1e-4,
    retrain_every_n_points=20,
    enable_point_rejection=True,
    rejection_threshold=1e-2,  # Reject if relative error < 1%
    min_points_before_rejection=30,  # Start rejecting after 30 points
)

print("Training online WITH point rejection...")
print("-" * 70)
for i, (x, y) in enumerate(zip(X_input, y_input)):
    x_reshaped = x.reshape(1, -1)
    y_reshaped = y.reshape(1, 1)
    gpt_rejection.update_tree(x_reshaped, y_reshaped)

    if (i + 1) % 500 == 0:
        print(f"  Processed {i+1}/{n_pts} points...")
print("-" * 70)
print()

# Get leaf nodes and check sizes
def get_leaf_nodes(node):
    """Recursively collect all leaf nodes"""
    if node.is_leaf:
        return [node]
    leaves = []
    if node.left:
        leaves.extend(get_leaf_nodes(node.left))
    if node.right:
        leaves.extend(get_leaf_nodes(node.right))
    return leaves

leaves_rejection = get_leaf_nodes(gpt_rejection.root)
total_points_rejection = sum(leaf.n_points for leaf in leaves_rejection)

print("Results WITH rejection:")
print(f"  Number of leaf nodes: {len(leaves_rejection)}")
print(f"  Total points stored: {total_points_rejection} / {n_pts} offered")
print(f"  Points rejected: {n_pts - total_points_rejection} ({100*(n_pts - total_points_rejection)/n_pts:.1f}%)")
print(f"  Average points per leaf: {total_points_rejection / len(leaves_rejection):.1f}")
print()

# Test predictions
y_pred_rej, y_std_rej = gpt_rejection.predict(X_test)
rmse_rej = np.sqrt(np.mean((y_test_true - y_pred_rej.reshape(-1, 1))**2))
errors_rej = np.abs(y_test_true.flatten() - y_pred_rej)
coverage_rej = np.mean(errors_rej <= y_std_rej)

print(f"  RMSE on test set: {rmse_rej:.4f}")
print(f"  Empirical coverage: {coverage_rej:.2%}")
print()

# Now test WITHOUT rejection for comparison
print("=" * 70)
print("Comparison: Training WITHOUT point rejection...")
gpt_no_rejection = GPTree(
    GPR=TestGPR(),
    Nbar=100,
    theta=1e-4,
    retrain_every_n_points=20,
    enable_point_rejection=False,
)

print("Training online WITHOUT point rejection...")
print("-" * 70)
for i, (x, y) in enumerate(zip(X_input, y_input)):
    x_reshaped = x.reshape(1, -1)
    y_reshaped = y.reshape(1, 1)
    gpt_no_rejection.update_tree(x_reshaped, y_reshaped)

    if (i + 1) % 500 == 0:
        print(f"  Processed {i+1}/{n_pts} points...")
print("-" * 70)
print()

leaves_no_rejection = get_leaf_nodes(gpt_no_rejection.root)
total_points_no_rejection = sum(leaf.n_points for leaf in leaves_no_rejection)

print("Results WITHOUT rejection:")
print(f"  Number of leaf nodes: {len(leaves_no_rejection)}")
print(f"  Total points stored: {total_points_no_rejection} / {n_pts} offered")
print(f"  Average points per leaf: {total_points_no_rejection / len(leaves_no_rejection):.1f}")
print()

y_pred_no_rej, y_std_no_rej = gpt_no_rejection.predict(X_test)
rmse_no_rej = np.sqrt(np.mean((y_test_true - y_pred_no_rej.reshape(-1, 1))**2))
errors_no_rej = np.abs(y_test_true.flatten() - y_pred_no_rej)
coverage_no_rej = np.mean(errors_no_rej <= y_std_no_rej)

print(f"  RMSE on test set: {rmse_no_rej:.4f}")
print(f"  Empirical coverage: {coverage_no_rej:.2%}")
print()

# Summary comparison
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Memory reduction: {total_points_no_rejection} -> {total_points_rejection} points")
reduction_pct = 100 * (1 - total_points_rejection/total_points_no_rejection)
print(f"  ({reduction_pct:.1f}% reduction)")
print()
print(f"RMSE comparison:")
print(f"  With rejection:    {rmse_rej:.4f}")
print(f"  Without rejection: {rmse_no_rej:.4f}")
rmse_diff_pct = 100 * abs(rmse_rej - rmse_no_rej) / rmse_no_rej
print(f"  Difference:        {abs(rmse_rej - rmse_no_rej):.4f} ({rmse_diff_pct:.1f}%)")
print()
print(f"Coverage comparison:")
print(f"  With rejection:    {coverage_rej:.2%}")
print(f"  Without rejection: {coverage_no_rej:.2%}")
print()

if total_points_rejection < total_points_no_rejection * 0.8:
    print("✓ Point rejection successfully reduced memory usage!")
else:
    print("⚠ Point rejection did not significantly reduce memory")

if rmse_diff_pct < 10:
    print("✓ Prediction accuracy maintained!")
else:
    print("⚠ Prediction accuracy may have degraded")

print()
print("Test complete!")
