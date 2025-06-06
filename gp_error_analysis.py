# This file is for Gaussian Process training and validation data generation.

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Define the 4D function to emulate
def target_function(x):
    """
    The function to be emulated by the Gaussian Process.
    f(x1, x2, x3, x4) = sin(x1) + x2^2 - cos(x3) + x4
    """
    if x.ndim == 1: # Handle single input array
        x = x.reshape(1, -1)
    x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    return np.sin(x1) + x2**2 - np.cos(x3) + x4

# Generate training data
N_train = 800
# Ensure results are reproducible for X_train, X_val
np.random.seed(42)
X_train = np.random.rand(N_train, 4)
y_train = target_function(X_train)

# Define the kernel for the GP
# Using a default RBF kernel. Length scale can be tuned.
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))

# Create and train a GaussianProcessRegressor model
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42, alpha=1e-5) # Added alpha for numerical stability
print("Training Gaussian Process model...")
gp.fit(X_train, y_train)
print("Training complete.")
print(f"Optimized kernel parameters: {gp.kernel_}")

# Generate validation data
N_val = 200
X_val = np.random.rand(N_val, 4)
y_val = target_function(X_val)

print(f"Generated {N_train} training samples and {N_val} validation samples.")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# The trained GP model is available as `gp`.
# Validation inputs are `X_val` and validation true responses are `y_val`.

# --- Appended Logic for Error Analysis ---

# 1. Compute predictions and errors
y_pred, y_std = gp.predict(X_val, return_std=True)
e = y_pred - y_val

# 2. Form weights w_i = e_i^2
w = e**2

# 3. Standardize each column of X_val:
X_val_std = X_val.std(axis=0)
X_val_std[X_val_std == 0] = 1e-9
X = (X_val - X_val.mean(axis=0)) / X_val_std

# 4. Compute weighted mean of X:
W_total = np.sum(w)
epsilon = 1e-9
xw_mean = np.sum(X * w[:,None], axis=0) / (W_total + epsilon)

# 5. Center X around xw_mean:
X_centered = X - xw_mean[None,:]

# 6. Build the weighted covariance matrix C:
C = (X_centered.T * w) @ X_centered / (W_total + epsilon)

# 7. Eigen‐decompose:
eigvals, eigvecs = np.linalg.eigh(C)
idx_desc = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx_desc]
eigvecs = eigvecs[:, idx_desc]

# 8. Now:
v_worst = eigvecs[:, 0]
lambda_worst = eigvals[0]
v_best = eigvecs[:, -1]
lambda_best = eigvals[-1]

# 9. Print results
print("\n--- Error Analysis Results ---")
print("Eigenvalues (high→low):", eigvals)
print("Worst‐direction (v1 - eigenvector for largest eigenvalue):", v_worst)
print("Variance along worst-direction (lambda1 - largest eigenvalue):", lambda_worst)
print("Best‐direction (v4 - eigenvector for smallest eigenvalue):", v_best)
print("Variance along best-direction (lambda4 - smallest eigenvalue):", lambda_best)


if __name__ == "__main__":
    # Basic check to see if the script runs and objects are created
    print("\n--- Script executed as main ---")
    print(f"GP object: {gp}")
    print("Variables v_worst, lambda_worst, v_best, lambda_best are computed and results printed.")
