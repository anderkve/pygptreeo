# Using pygptreeo with GPyTorch

This directory contains examples demonstrating how to use pygptreeo with the GPyTorch backend for GPU-accelerated Gaussian Process regression.

## Installation

To use the GPyTorch adapter, you need to install GPyTorch and PyTorch:

```bash
# Install GPyTorch and PyTorch
pip install gpytorch torch

# Or install with CUDA support for GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gpytorch
```

## Examples

### 1. `test_gpytorch_adapter.py`

A simple test script that verifies the GPyTorch adapter works correctly with pygptreeo.

**Usage:**
```bash
python test_gpytorch_adapter.py
```

**What it does:**
- Creates a GPyTorchAdapter with a simple RBF kernel
- Builds a GPTree using the GPyTorch backend
- Trains on a 1D sine function
- Makes predictions and evaluates accuracy
- Tests all adapter interface methods

**Expected output:**
```
✓ GPyTorch is available
Using device: cpu (or cuda if GPU available)

✓ GPyTorchAdapter created successfully
✓ GPTree created successfully
✓ Training complete
✓ Predictions complete

Prediction metrics:
  RMSE: 0.012345
  MAE:  0.009876

✓ All tests passed successfully!
```

### 2. `performance_test_gpytorch.py`

A comprehensive performance test comparing GPyTorch backend performance on various benchmark functions.

**Usage:**
```bash
python performance_test_gpytorch.py
```

**Configuration:**
- **Target functions**: Eggholder, Himmelblau, Rosenbrock, Rastrigin, Levy
- **Dimensionality**: 3D (configurable via `n_dims`)
- **Data points**: 10,000 (configurable via `n_pts`)
- **Device**: Automatically uses CUDA if available, otherwise CPU

**Features:**
- Real-time performance tracking and visualization
- Metrics tracked:
  - Average prediction time per point
  - Average tree update time per point
  - Normalized RMSE (NRMSE)
  - Accuracy (fraction within 1%, 2%, 4%, 8%, 16% error)
  - Empirical coverage of uncertainty estimates
- Generates plots saved as `plot_gpytorch_{device}.png`

**Performance comparison:**

On CPU:
- GPyTorch may be slower than scikit-learn for small datasets
- Training iterations are configurable (default: 50)

On GPU (with CUDA):
- Significantly faster for larger datasets (>1000 points)
- Parallel processing of predictions
- Efficient kernel computations

## Key Differences from scikit-learn

### Kernel Configuration

**scikit-learn:**
```python
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
kernel = ConstantKernel(1.0) * RBF(1.0)
```

**GPyTorch:**
```python
import gpytorch
covar_module = gpytorch.kernels.ScaleKernel(
    gpytorch.kernels.RBFKernel()
)
```

### Adapter Creation

**scikit-learn (default):**
```python
from pygptreeo import GPTree, Default_GPR
gpt = GPTree()  # Uses sklearn by default
```

**GPyTorch:**
```python
from pygptreeo.adapters import GPyTorchAdapter
import gpytorch

adapter = GPyTorchAdapter(
    mean_module=gpytorch.means.ConstantMean(),
    covar_module=gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=1.5)
    ),
    optimizer='adam',
    learning_rate=0.1,
    training_iterations=50,
    device='cuda'  # or 'cpu'
)

gpt = GPTree(GPR=adapter)
```

### Training Configuration

GPyTorch uses iterative optimization instead of direct hyperparameter optimization:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `optimizer` | Optimizer type ('adam' or 'lbfgs') | 'adam' |
| `learning_rate` | Learning rate for optimizer | 0.1 |
| `training_iterations` | Number of training iterations | 50 |
| `device` | Computation device ('cpu' or 'cuda') | 'cpu' |

**Note:** More iterations = better fit but slower training. Adjust based on your needs.

## When to Use GPyTorch

### ✅ Use GPyTorch when:
- You have a GPU available (CUDA)
- Working with large datasets (>5,000 points)
- Need scalability and modern GP algorithms
- Want to leverage PyTorch ecosystem
- Require custom likelihood functions

### ❌ Stick with scikit-learn when:
- Working with small datasets (<1,000 points)
- CPU-only environment
- Need simple, well-tested implementation
- Prefer minimal dependencies
- Want faster prototyping

## Performance Tips

### For CPU:
```python
adapter = GPyTorchAdapter(
    training_iterations=30,  # Reduce for speed
    optimizer='lbfgs',       # Often faster than adam
    device='cpu'
)
```

### For GPU:
```python
adapter = GPyTorchAdapter(
    training_iterations=100,  # Can afford more iterations
    optimizer='adam',
    learning_rate=0.1,
    device='cuda'
)

# Ensure your tree doesn't retrain too frequently
gpt = GPTree(
    GPR=adapter,
    retrain_every_n_points=200  # Larger value = less frequent retraining
)
```

## Troubleshooting

### "No module named 'torch'"
Install PyTorch:
```bash
pip install torch
```

### "CUDA out of memory"
Reduce the number of training iterations or switch to CPU:
```python
adapter = GPyTorchAdapter(
    training_iterations=20,  # Reduce
    device='cpu'             # Use CPU instead
)
```

### Slow training
- Reduce `training_iterations` (try 20-30)
- Increase `retrain_every_n_points` in GPTree
- Disable `enable_split_evaluation` in GPTree
- Use simpler kernels (RBF instead of Matern)

### Poor predictions
- Increase `training_iterations` (try 100-200)
- Use more sophisticated kernels (Matern instead of RBF)
- Enable `use_standard_scaling` in GPTree
- Check if data is normalized

## Advanced Usage

### Custom GPyTorch Model

You can provide your own GPyTorch model:

```python
import torch
import gpytorch
from pygptreeo.adapters import GPyTorchAdapter

class CustomGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Initialize with dummy data (will be updated during training)
likelihood = gpytorch.likelihoods.GaussianLikelihood()
train_x = torch.zeros(1, 3)  # 3D input
train_y = torch.zeros(1)
model = CustomGPModel(train_x, train_y, likelihood)

adapter = GPyTorchAdapter(
    model=model,
    likelihood=likelihood,
    training_iterations=50
)

gpt = GPTree(GPR=adapter)
```

## References

- [GPyTorch Documentation](https://gpytorch.ai/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [pygptreeo GitHub Repository](https://github.com/anderkve/pygptreeo)

## Support

For issues specific to:
- **GPyTorch adapter**: Open an issue on pygptreeo GitHub
- **GPyTorch library**: See GPyTorch documentation or GitHub
- **CUDA/GPU**: See PyTorch CUDA documentation
