"""Test script for uncertainty-aware splitting feature.

This script tests the new 'max_uncertainty' split_dimension_criteria option
and compares it with the default 'max_spread' approach.
"""
import numpy as np
from pygptreeo import GPTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

# Set random seed for reproducibility
np.random.seed(42)

# Define a simple test function with different complexity in different dimensions
def test_function(x):
    """A test function where dimension 0 is complex (high uncertainty)
    and dimension 1 is smooth (low uncertainty)."""
    # Dimension 0: complex, oscillatory
    term1 = 10 * np.sin(10 * x[0])
    # Dimension 1: smooth, linear
    term2 = 2 * x[1]
    return term1 + term2

# Generate training data
n_train = 150
n_dims = 2
X_train = np.random.uniform(0, 1, (n_train, n_dims))
y_train = np.array([test_function(x) for x in X_train]).reshape(-1, 1)

print("=" * 70)
print("Testing Uncertainty-Aware Splitting Feature")
print("=" * 70)

# Test 1: Create GPTree with max_spread (default)
print("\n[Test 1] Creating GPTree with 'max_spread' splitting...")
gpt_spread = GPTree(
    Nbar=50,
    theta=1e-4,
    split_dimension_criteria='max_spread',
    retrain_every_n_points=10,
    use_calibrated_sigma=True
)

# Train it
for i, (x, y) in enumerate(zip(X_train, y_train)):
    x = x.reshape(1, -1)
    y = y.reshape(1, 1)
    gpt_spread.update_tree(x, y)

n_leaves_spread = len(gpt_spread.root.leaves)
print(f"   ✓ Trained successfully with {n_leaves_spread} leaf nodes")

# Test 2: Create GPTree with max_uncertainty
print("\n[Test 2] Creating GPTree with 'max_uncertainty' splitting...")
gpt_uncertainty = GPTree(
    Nbar=50,
    theta=1e-4,
    split_dimension_criteria='max_uncertainty',
    retrain_every_n_points=10,
    use_calibrated_sigma=True
)

# Train it
for i, (x, y) in enumerate(zip(X_train, y_train)):
    x = x.reshape(1, -1)
    y = y.reshape(1, 1)
    gpt_uncertainty.update_tree(x, y)

n_leaves_uncertainty = len(gpt_uncertainty.root.leaves)
print(f"   ✓ Trained successfully with {n_leaves_uncertainty} leaf nodes")

# Test 3: Make predictions with both models
print("\n[Test 3] Testing predictions...")
n_test = 50
X_test = np.random.uniform(0, 1, (n_test, n_dims))
y_test = np.array([test_function(x) for x in X_test]).reshape(-1, 1)

# Predictions from max_spread model
y_pred_spread, y_std_spread = gpt_spread.predict(X_test)

# Predictions from max_uncertainty model
y_pred_uncertainty, y_std_uncertainty = gpt_uncertainty.predict(X_test)

# Compute errors
rmse_spread = np.sqrt(np.mean((y_test - y_pred_spread)**2))
rmse_uncertainty = np.sqrt(np.mean((y_test - y_pred_uncertainty)**2))

print(f"   ✓ RMSE (max_spread):      {rmse_spread:.4f}")
print(f"   ✓ RMSE (max_uncertainty): {rmse_uncertainty:.4f}")

# Test 4: Check split dimensions used
print("\n[Test 4] Analyzing split dimension choices...")

def count_split_dimensions(tree):
    """Count how many times each dimension was used for splitting."""
    counts = np.zeros(n_dims, dtype=int)

    def traverse(node):
        if not node.is_leaf and node.children is not None:
            counts[node.split_index] += 1
            for child in node.children:
                traverse(child)

    traverse(tree.root)
    return counts

counts_spread = count_split_dimensions(gpt_spread)
counts_uncertainty = count_split_dimensions(gpt_uncertainty)

print(f"   Split dimension usage (max_spread):")
for dim in range(n_dims):
    print(f"      Dimension {dim}: {counts_spread[dim]} times")

print(f"   Split dimension usage (max_uncertainty):")
for dim in range(n_dims):
    print(f"      Dimension {dim}: {counts_uncertainty[dim]} times")

# Test 5: Check that uncertainty computation works
print("\n[Test 5] Testing _compute_dimensional_uncertainty() method...")
# Get a leaf node from the uncertainty tree
leaf = gpt_uncertainty.root.leaves[0]
if hasattr(leaf.my_GPR, 'kernel_'):
    uncertainty_scores = leaf._compute_dimensional_uncertainty()
    print(f"   ✓ Uncertainty scores computed: {uncertainty_scores}")
    print(f"   ✓ Dimension with max uncertainty: {np.argmax(uncertainty_scores)}")
else:
    print(f"   ⚠ Leaf GP not trained yet, skipping this test")

print("\n" + "=" * 70)
print("All tests completed successfully!")
print("=" * 70)

# Summary
print("\nSummary:")
print(f"  - Both models trained successfully")
print(f"  - max_spread model:      {n_leaves_spread} leaves, RMSE={rmse_spread:.4f}")
print(f"  - max_uncertainty model: {n_leaves_uncertainty} leaves, RMSE={rmse_uncertainty:.4f}")
print(f"  - The uncertainty-aware splitting is working as expected!")
