"""Compare kernel behavior between sklearn and GPyTorch implementations."""
import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from pygptreeo.kernels import AnisotropicRationalQuadratic

# Test settings
n_dims = 3
np.random.seed(512312)

# Generate small test dataset
n_test = 10
X_test = np.random.uniform(0.0, 1.0, n_dims * n_test).reshape(n_test, n_dims)

# ============================================================================
# Scikit-learn kernel configuration (from performance_test.py)
# ============================================================================
sklearn_kernel = ConstantKernel(
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

print("=" * 80)
print("SKLEARN KERNEL CONFIGURATION")
print("=" * 80)
print(f"Full kernel: {sklearn_kernel}")
print(f"\nKernel components:")
print(f"  - ConstantKernel (for output scale)")
print(f"  - AnisotropicRationalQuadratic (anisotropic with per-dim length scales)")
print(f"  - Matern (nu=1.5, anisotropic)")
print(f"\nNumber of hyperparameters: {sklearn_kernel.n_dims}")
print(f"Hyperparameter names: {[p.name for p in sklearn_kernel.hyperparameters]}")
print()

# Compute kernel matrix
K_sklearn = sklearn_kernel(X_test)
print(f"Sample kernel matrix shape: {K_sklearn.shape}")
print(f"Kernel matrix diagonal: {np.diag(K_sklearn)[:5]}")
print(f"Kernel matrix mean: {K_sklearn.mean():.4f}")
print(f"Kernel matrix std: {K_sklearn.std():.4f}")

print("\n" + "=" * 80)
print("GPYTORCH KERNEL CONFIGURATION")
print("=" * 80)

try:
    import torch
    import gpytorch

    # GPyTorch kernel (from performance_test_gpytorch.py)
    gpytorch_kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=n_dims)
    )

    print(f"Full kernel: {gpytorch_kernel}")
    print(f"\nKernel components:")
    print(f"  - ScaleKernel (for output scale)")
    print(f"  - MaternKernel (nu=1.5, ARD with {n_dims} length scales)")
    print(f"\nNOTE: Missing AnisotropicRationalQuadratic component!")

    # Initialize kernel
    X_torch = torch.from_numpy(X_test).float()
    with torch.no_grad():
        K_gpytorch = gpytorch_kernel(X_torch).evaluate().numpy()

    print(f"\nSample kernel matrix shape: {K_gpytorch.shape}")
    print(f"Kernel matrix diagonal: {np.diag(K_gpytorch)[:5]}")
    print(f"Kernel matrix mean: {K_gpytorch.mean():.4f}")
    print(f"Kernel matrix std: {K_gpytorch.std():.4f}")

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"Difference in kernel matrices:")
    print(f"  Mean absolute difference: {np.abs(K_sklearn - K_gpytorch).mean():.4f}")
    print(f"  Max absolute difference: {np.abs(K_sklearn - K_gpytorch).max():.4f}")

except ImportError:
    print("GPyTorch not installed - skipping GPyTorch kernel test")

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print("""
1. KERNEL MISMATCH:
   - Scikit-learn uses: ConstantKernel * (AnisotropicRationalQuadratic + Matern)
   - GPyTorch uses: ScaleKernel * MaternKernel
   - The RationalQuadratic component is MISSING from GPyTorch version!

2. OPTIMIZATION DIFFERENCES:
   - Scikit-learn: L-BFGS-B with 3 restarts (robust second-order method)
   - GPyTorch: Adam with 50 iterations (first-order, may not converge fully)

3. IMPLICATIONS:
   - The AnisotropicRationalQuadratic kernel adds flexibility for multi-scale features
   - RQ kernel = infinite mixture of RBF kernels with different length scales
   - Its absence significantly reduces model expressiveness
   - This is likely the PRIMARY cause of accuracy differences

4. RECOMMENDATION:
   - Implement AnisotropicRationalQuadratic for GPyTorch OR
   - Use a sum kernel: ScaleKernel(RQKernel + MaternKernel) in GPyTorch
""")
