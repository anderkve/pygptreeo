"""Test script for automatic kernel selection feature."""

import numpy as np
from pygptreeo import GPTree

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_points = 500
X = np.random.uniform(-5, 5, (n_points, 2))
y = np.sin(X[:, 0]) + 0.5 * X[:, 1]**2 + np.random.normal(0, 0.1, n_points)
sigma = np.ones(n_points) * 0.1

print("Testing automatic kernel selection...")
print(f"Generated {n_points} training points")
print()

# Create GPTree with automatic kernel selection enabled
gpt = GPTree(
    Nbar=50,  # Small Nbar to create multiple splits
    theta=0.001,
    enable_kernel_selection=True,  # Enable automatic kernel selection
    use_calibrated_sigma=True,
    split_dimension_criteria='max_spread',
    use_standard_scaling=False
)

print("Created GPTree with enable_kernel_selection=True")
print(f"Root node kernel_type_idx: {gpt.root.kernel_type_idx}")
print()

# Fit the tree
print("Fitting the tree...")
gpt.fit(X, y, sigma, show_progress=False, shuffle=True)
print(f"Tree has {len(gpt.root.leaves)} leaf nodes")
print()

# Collect kernel type statistics
kernel_types = {}
for leaf in gpt.root.leaves:
    kernel_idx = leaf.kernel_type_idx
    if kernel_idx is not None:
        kernel_types[kernel_idx] = kernel_types.get(kernel_idx, 0) + 1

print("Kernel type distribution across leaves:")
kernel_names = {
    0: "Const*(RBF + Matern(nu=1.5))",
    1: "Const*(RQ + Matern(nu=1.5))",
    2: "Const*(RQ + RBF)",
    3: "Const*RQ",
    4: "Const*Matern(nu=1.5)",
    5: "Const*RBF"
}

for idx in sorted(kernel_types.keys()):
    count = kernel_types[idx]
    name = kernel_names.get(idx, f"Unknown ({idx})")
    print(f"  Kernel {idx} ({name}): {count} leaves")

print()

# Test prediction
X_test = np.random.uniform(-5, 5, (10, 2))
y_pred, y_std = gpt.predict(X_test, mode='recursive', show_progress=False)
print(f"Successfully predicted on {X_test.shape[0]} test points")
print(f"Prediction shape: {y_pred.shape}")
print(f"Std shape: {y_std.shape}")

print()
print("Test completed successfully!")
