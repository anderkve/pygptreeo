# pygptreeo Examples

This directory contains example scripts demonstrating the usage of the pygptreeo library for online/continual regression tasks.

## Files

### Core Examples

- **`example.py`**: Basic example demonstrating GPTree usage
  - Shows how to set up and use a `GPTree` for online regression
  - Implements a custom GPR class with specific kernel configuration
  - Processes points one at a time, making predictions and updating the tree
  - Good starting point for understanding the basic workflow

- **`performance_test.py`**: Comprehensive performance evaluation script
  - Tests GPTree on various benchmark functions (Eggholder, Himmelblau, etc.)
  - Tracks and plots multiple performance metrics over time:
    - Average prediction time per point
    - Average update time per point
    - Batch NRMSE (Normalized Root Mean Square Error)
    - Accuracy within different error thresholds (1%, 2%, 4%, 8%, 16%)
    - Empirical coverage of prediction uncertainty
  - Generates performance plots saved as `plot.png`
  - Can process large numbers of points (default: 300,000)

- **`test_animated.py`**: Animated visualization of GPTree learning (2D only)
  - Creates animated GIFs showing how the tree learns the target function
  - Displays the tree structure, leaf boundaries, and prediction surface
  - Requires command-line arguments for configuration

### Utilities

- **`target_functions.py`**: Collection of benchmark test functions
  - Eggholder: Complex landscape with many local minima
  - Himmelblau: Classic optimization benchmark with 4 local minima
  - Rosenbrock: Narrow parabolic valley (challenging for optimization)
  - Rastrigin: Regular grid of local minima
  - Levy: Many local minima
  - Custom: Weighted combination of multiple functions

- **`plot_performance_metrics.py`**: Post-processing script for performance analysis
  - Reads results from CSV files
  - Generates detailed performance plots
  - Useful for analyzing results from long runs

## Running the Examples

### Basic Example

```bash
cd examples
python example.py
```

This will run a basic test with 1000 points on the Eggholder function in 2D.

### Performance Test

```bash
cd examples
OMP_NUM_THREADS=1 python performance_test.py
```

**Note**: Setting `OMP_NUM_THREADS=1` is recommended to get consistent timing results and avoid overhead from parallel processing in the underlying linear algebra libraries.

The performance test generates a plot (`plot.png`) showing various metrics over time.

### Split-dimension criteria benchmark

```bash
cd examples
OMP_NUM_THREADS=1 python benchmark_split_direction.py [target] [n_points]
```

Compares every available `split_dimension_criteria` on the same streaming data
set and tracks batch metrics as a function of the number of processed points:
`max_spread`, `max_variance`, `max_uncertainty`, `min_lengthscale`, and
`random`. The `min_lengthscale` criterion splits the dimension with the smallest
fitted ARD length scale (i.e. where the GP says the function varies fastest),
reusing the GP's already-optimized hyperparameters.

`target` is `aniso_chirp` (default) or `eggholder`; `n_points` defaults to 20000.
The `aniso_chirp` target is anisotropic and heterogeneous (rough/chirped along
`x0`, but smooth along a wider `x1` that misleads the spread-based criteria),
which is where the choice of split dimension matters most; `eggholder` is a
roughly-isotropic, uniformly-rough reference where all criteria behave similarly.
The script writes a comparison figure and a CSV of batch metrics (NRMSE,
accuracy, coverage, leaf count, predict/update times) per criterion.

### Animated Visualization

```bash
cd examples
python test_animated.py <target_function> <n_points> <Nbar> <retrain_step> <update_step> <live_update>
```

**Parameters**:
- `target_function`: One of 'eggholder', 'himmelblau', 'rosenbrock', 'rastrigin', 'levy', 'custom'
- `n_points`: Total number of training points
- `Nbar`: Maximum points per leaf before splitting
- `retrain_step`: Number of new points before retraining a GP
- `update_step`: How often to update the animation (in points)
- `live_update`: 1 for live display, 0 for saving only

**Example**:
```bash
python test_animated.py eggholder 10000 200 200 10 1
```

This creates an animated GIF showing the tree learning the 2D Eggholder function.

## Customizing the Examples

### Changing the Target Function

Edit the `target_name` variable in the script:
```python
target_name = "rastrigin"  # or 'eggholder', 'himmelblau', etc.
```

### Adjusting GPTree Parameters

Key parameters to experiment with:
- `Nbar`: Maximum points per leaf (default: 100-200)
  - Smaller values: more leaf nodes, more localized GPs
  - Larger values: fewer leaf nodes, each covering more of the input space

- `theta`: Overlap parameter for node splitting (default: 1e-4)
  - Controls the size of overlapping regions between sibling nodes

- `retrain_step`: How often to retrain GPs (default: 20-200)
  - Smaller values: more frequent retraining (higher accuracy, slower)
  - Larger values: less frequent retraining (faster, may sacrifice accuracy)

- `splitting_strategy`: How nodes split their data (default: 'gradual')
  - `'standard'`: Hard splits of training data
  - `'gradual'`: Nodes share training data during transition

- `use_calibrated_sigma`: Enable uncertainty calibration (default: True)
  - Adjusts prediction uncertainties to achieve target coverage

- `split_dimension_criteria`: How a node picks its split dimension (default: 'max_spread')
  - `'max_spread'` / `'max_variance'`: split the dimension with the largest data
    range / variance
  - `'max_uncertainty'`: split where the GP is most uncertain (grid-based, costly)
  - `'min_lengthscale'`: split the dimension with the smallest fitted ARD length
    scale, i.e. where the GP says the function varies fastest. Requires an
    anisotropic (ARD) kernel and a trained GP; falls back to `max_spread`
    otherwise.

### Custom Kernel Configuration

The examples define a custom `my_GPR_class` that you can modify:

```python
class my_GPR_class(GaussianProcessRegressor):
    def __init__(self, kernel=None, *, alpha=1e-6, ...):
        super().__init__()
        # Customize the kernel
        self.kernel = ConstantKernel(...) * Matern(nu=1.5, ...)
        self.min_length_scale = 0.001
        self.alpha = alpha
        # ... other parameters
```

You can change:
- The kernel type (Matern, RBF, etc.)
- Kernel hyperparameters (nu, length_scale, etc.)
- Alpha value (noise level)
- Optimizer settings

## Tips

1. **For fast testing**: Use fewer points (n_pts = 1000) and smaller dimensions (n_dims = 2)

2. **For performance benchmarking**: Set `OMP_NUM_THREADS=1` and use consistent settings across runs

3. **Memory usage**: Large values of Nbar with many dimensions can require significant memory

4. **Visualization**: The animated examples work best in 2D; higher dimensions are harder to visualize

5. **Debugging**: Set `make_plot = False` in performance_test.py to disable plotting and see detailed per-point output
