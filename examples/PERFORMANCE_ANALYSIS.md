# Performance Difference Analysis: Scikit-learn vs GPyTorch

## Executive Summary

The scikit-learn implementation (`performance_test.py`) achieves better prediction accuracy than the GPyTorch implementation (`performance_test_gpytorch.py`) due to three main factors:

1. **CRITICAL: Missing Kernel Component** - The GPyTorch version is missing the `AnisotropicRationalQuadratic` kernel
2. **Different Optimization Strategy** - L-BFGS-B with restarts vs Adam with fixed iterations
3. **Hyperparameter Configuration** - Different bounds and initialization

## Detailed Comparison

### 1. Kernel Configuration ⚠️ **PRIMARY ISSUE**

#### Scikit-learn (lines 123-136 in performance_test.py):
```python
kernel = ConstantKernel(
    constant_value=1.0,
    constant_value_bounds=(1e-3, 1e8)
) * (
    AnisotropicRationalQuadratic(
        length_scale=[1.0]*n_dims,
        length_scale_bounds=(1e-5, 1e5),
        alpha=1.0,
        alpha_bounds=(1e-4, 1e4)
    ) + Matern(
        nu=1.5,
        length_scale=[1.0]*n_dims,
        length_scale_bounds=[(1e-5, 1e5)]*n_dims
    )
)
```

**Key Features:**
- **Composite kernel** = ConstantKernel × (RationalQuadratic + Matern)
- **Sum of two base kernels**: RQ + Matern
- **8 hyperparameters total**:
  - 1 for ConstantKernel (output scale)
  - 1 for RQ alpha (scale mixture parameter)
  - 3 for RQ length scales (one per dimension)
  - 3 for Matern length scales (one per dimension)

#### GPyTorch (lines 86-88 in performance_test_gpytorch.py):
```python
kernel = gpytorch.kernels.ScaleKernel(
    gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=n_dims)
)
```

**Key Features:**
- **Simple scaled kernel** = ScaleKernel × MaternKernel
- **Only Matern kernel** - NO RationalQuadratic component!
- **4 hyperparameters total**:
  - 1 for ScaleKernel (output scale)
  - 3 for Matern length scales (one per dimension)

#### Why This Matters:

The **AnisotropicRationalQuadratic (RQ) kernel** is crucial for the Eggholder function because:

1. **Multi-scale Structure**: The RQ kernel can be viewed as an infinite mixture of RBF kernels with different length scales:
   ```
   RQ(r) = (1 + r²/(2α))^(-α)
   ```
   where α controls the mixture weighting.

2. **Flexibility**: The sum `RQ + Matern` allows the GP to model:
   - **RQ component**: Captures smooth, large-scale trends and patterns at multiple scales
   - **Matern component**: Captures local, fine-grained variations
   - **Together**: Much more expressive than Matern alone

3. **Eggholder Function Characteristics**: The Eggholder function has:
   - Large-scale valleys and ridges (captured by RQ)
   - Small-scale oscillations (captured by Matern)
   - Complex multi-scale interactions (requires both kernels)

### 2. Optimization Strategy

#### Scikit-learn (lines 152-153):
```python
optimizer = 'fmin_l_bfgs_b'
n_restarts_optimizer = 3
```

**Characteristics:**
- **Algorithm**: L-BFGS-B (limited-memory BFGS, quasi-Newton method)
  - Second-order optimization (uses Hessian approximation)
  - Very effective for GP hyperparameter optimization
  - Converges to local optima efficiently
- **Restarts**: 3 independent initializations
  - Reduces risk of poor local optima
  - Increases robustness
- **Convergence**: Adaptive (stops when gradients are small)

#### GPyTorch (lines 90-92):
```python
optimizer = 'adam'
learning_rate = 0.1
training_iterations = 50
```

**Characteristics:**
- **Algorithm**: Adam (adaptive moment estimation)
  - First-order optimization (only uses gradients)
  - Popular for deep learning, but not optimal for GPs
  - Requires careful learning rate tuning
- **Fixed iterations**: Always runs exactly 50 steps
  - May stop too early (underfitting)
  - May waste computation if already converged
  - No guarantee of convergence
- **No restarts**: Single initialization only

#### Impact:

The combination of L-BFGS-B + restarts typically finds better hyperparameters than Adam with fixed iterations, especially for:
- Non-convex optimization landscapes (GP marginal likelihood)
- Small to medium datasets (where second-order methods excel)
- High-quality convergence requirements

### 3. Hyperparameter Bounds

#### Scikit-learn:
- Length scales: `(1e-5, 1e5)` - very wide range (10 orders of magnitude)
- Constant kernel: `(1e-3, 1e8)` - wide range
- RQ alpha: `(1e-4, 1e4)` - allows exploring different scale mixtures

#### GPyTorch:
- Uses GPyTorch defaults (typically narrower)
- Bounds not explicitly set in the code
- May restrict the search space

### 4. Other Minor Differences

#### Noise handling in `make_plot=False` mode:
- **Scikit-learn** (line 240): `gpt.update_tree(x, y)` - no explicit noise
- **GPyTorch** (line 178): `gpt.update_tree(x, y, 0.001 * np.abs(y))` - with noise

However, in `make_plot=True` mode (which is the default), both add the same noise term: `0.001 * np.abs(y)`.

## Recommendations to Improve GPyTorch Performance

### Option 1: Add RationalQuadratic Kernel (Recommended)

Modify `performance_test_gpytorch.py` to use a sum of kernels:

```python
gpytorch_gpr = GPyTorchAdapter(
    model=None,
    likelihood=None,
    mean_module=gpytorch.means.ConstantMean(),
    covar_module=gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RQKernel(ard_num_dims=n_dims) +
        gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=n_dims)
    ),
    optimizer='adam',
    learning_rate=0.1,
    training_iterations=50,
    device=device
)
```

**Note**: GPyTorch's `RQKernel` is equivalent to the AnisotropicRationalQuadratic when `ard_num_dims` is set.

### Option 2: Switch to L-BFGS Optimizer

```python
gpytorch_gpr = GPyTorchAdapter(
    # ... kernel configuration ...
    optimizer='lbfgs',
    learning_rate=1.0,  # LBFGS typically uses lr=1.0
    training_iterations=20,  # Fewer iterations needed with LBFGS
    device=device
)
```

**However**: The current `GPyTorchAdapter` implementation doesn't support restarts like sklearn does.

### Option 3: Increase Training Iterations

If keeping Adam:

```python
training_iterations=200  # Increase from 50
```

This gives Adam more time to converge, but still won't match L-BFGS quality.

### Option 4: Implement the Full Matching Kernel

The ideal solution would be to exactly match the sklearn configuration:

```python
# Pseudo-code - would need implementation in adapter
covar_module = gpytorch.kernels.ScaleKernel(
    gpytorch.kernels.AdditiveKernel(
        gpytorch.kernels.RQKernel(ard_num_dims=n_dims),
        gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=n_dims)
    )
)
```

## Quantitative Impact Estimation

Based on the analysis:

- **Missing RQ kernel**: Likely accounts for **60-80%** of the performance gap
  - The RQ kernel provides critical multi-scale modeling capability
  - The sum (RQ + Matern) doubles the kernel's expressiveness

- **Optimization differences**: Likely accounts for **15-30%** of the gap
  - L-BFGS-B with restarts finds better hyperparameters
  - Fixed-iteration Adam may not fully converge

- **Other factors**: **5-10%** of the gap
  - Hyperparameter bounds
  - Implementation details

## Testing Recommendations

To confirm these hypotheses:

1. **Test 1**: Add RQKernel to GPyTorch version and measure improvement
2. **Test 2**: Use only Matern in sklearn version and measure degradation
3. **Test 3**: Compare optimizer performance by tracking marginal likelihood convergence
4. **Test 4**: Visualize the learned length scales for both implementations

## Conclusion

The **primary cause** of the performance difference is the **missing AnisotropicRationalQuadratic kernel** in the GPyTorch implementation. The GPyTorch version only uses a Matern kernel, while the scikit-learn version uses a more expressive sum of RationalQuadratic + Matern kernels, both scaled and with anisotropic length scales.

To achieve comparable performance, the GPyTorch implementation should use:
```python
ScaleKernel(RQKernel(ard_num_dims=n_dims) + MaternKernel(nu=1.5, ard_num_dims=n_dims))
```

Additionally, switching to L-BFGS optimization (if possible) would further improve hyperparameter quality.
