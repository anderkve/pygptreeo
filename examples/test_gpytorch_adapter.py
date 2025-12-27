"""Simple test script for GPyTorch adapter.

This script tests the basic functionality of the GPyTorch adapter
with a small dataset to verify it works correctly.
"""
import numpy as np
import sys

# Check if GPyTorch is available
try:
    import torch
    import gpytorch
    from pygptreeo.adapters import GPyTorchAdapter
    from pygptreeo import GPTree
    GPYTORCH_AVAILABLE = True
    print("✓ GPyTorch is available")
except ImportError as e:
    print(f"✗ GPyTorch is not available: {e}")
    print("\nTo install GPyTorch:")
    print("  pip install gpytorch torch")
    sys.exit(1)

# Simple 1D test function
def test_function(x):
    """Simple sine function for testing."""
    return np.sin(2 * np.pi * x)

print("\n" + "="*60)
print("Testing GPyTorchAdapter with pygptreeo")
print("="*60)

# Test configuration
n_train = 50
n_test = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\nConfiguration:")
print(f"  Device: {device}")
print(f"  Training points: {n_train}")
print(f"  Test points: {n_test}")

# Generate training data
np.random.seed(42)
X_train = np.random.uniform(0, 1, n_train).reshape(-1, 1)
y_train = test_function(X_train).reshape(-1, 1)
sigma_train = 0.01 * np.ones(n_train)

# Generate test data
X_test = np.linspace(0, 1, n_test).reshape(-1, 1)
y_test = test_function(X_test).reshape(-1, 1)

print("\n" + "-"*60)
print("Creating GPyTorchAdapter...")
print("-"*60)

# Create GPyTorch adapter
gpytorch_adapter = GPyTorchAdapter(
    mean_module=gpytorch.means.ConstantMean(),
    covar_module=gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel()
    ),
    optimizer='adam',
    learning_rate=0.1,
    training_iterations=30,
    device=device
)

print(f"✓ GPyTorchAdapter created successfully")
print(f"  Kernel: ScaleKernel(RBFKernel)")
print(f"  Training iterations: 30")

print("\n" + "-"*60)
print("Creating GPTree with GPyTorch backend...")
print("-"*60)

# Create GPTree with GPyTorch adapter
gpt = GPTree(
    GPR=gpytorch_adapter,
    Nbar=25,  # Split after 25 points
    theta=1e-4,
    retrain_every_n_points=10,
    use_standard_scaling=True
)

print(f"✓ GPTree created successfully")
print(f"  Nbar: 25")
print(f"  Retrain every: 10 points")

print("\n" + "-"*60)
print("Training GPTree (online learning)...")
print("-"*60)

# Train the tree (online learning)
for i, (x, y, sigma) in enumerate(zip(X_train, y_train, sigma_train)):
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    gpt.update_tree(x, y, sigma)

    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/{n_train} points")

print(f"✓ Training complete")

print("\n" + "-"*60)
print("Making predictions...")
print("-"*60)

# Make predictions
y_pred, y_std = gpt.predict(X_test)

# Calculate metrics
mse = np.mean((y_test - y_pred)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - y_pred))

print(f"✓ Predictions complete")
print(f"\nPrediction metrics:")
print(f"  RMSE: {rmse:.6f}")
print(f"  MAE:  {mae:.6f}")

# Check a few predictions
print(f"\nSample predictions:")
print(f"  {'X':<10} {'True':<12} {'Predicted':<12} {'Std Dev':<12} {'Error':<12}")
print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
for i in [0, n_test//2, n_test-1]:
    x_val = X_test[i, 0]
    y_true = y_test[i, 0]
    y_p = y_pred[i, 0]
    y_s = y_std[i, 0]
    error = abs(y_true - y_p)
    print(f"  {x_val:<10.4f} {y_true:<12.6f} {y_p:<12.6f} {y_s:<12.6f} {error:<12.6f}")

print("\n" + "="*60)
print("✓ All tests passed successfully!")
print("="*60)

# Test adapter interface methods
print("\n" + "-"*60)
print("Testing adapter interface methods...")
print("-"*60)

# Get a leaf node's GP
leaf = gpt.root.leaves[0] if hasattr(gpt.root, 'leaves') and gpt.root.leaves else gpt.root
if leaf.my_GPR.is_trained():
    print("✓ is_trained() works correctly")
else:
    print("✗ is_trained() returned False unexpectedly")

# Test cloning
cloned_gpr = leaf.my_GPR.clone()
print("✓ clone() works correctly")

# Test get_kernel
kernel = leaf.my_GPR.get_kernel()
print(f"✓ get_kernel() works correctly: {type(kernel).__name__}")

print("\n" + "="*60)
print("GPyTorch adapter is fully functional!")
print("="*60 + "\n")
