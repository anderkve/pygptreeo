# pyGPTreeO: a Gaussian process tree for online regression

(Work in progress.)

## Introduction
pyGPTreeO is a Python tool designed for online/continual regression tasks. It implements a dynamically growing tree where each leaf node is a local Gaussian Process (GP) regressor. This structure makes it particularly well-suited for learning from data streams where data points arrive sequentially. It builds on the DLGP approach by Lederer et al (https://arxiv.org/abs/2006.09446) and the R package GPTreeO (https://arxiv.org/abs/2410.01024).

## Features
*   **Dynamic Tree Structure**: The tree adaptively changes its structure based on the incoming data, growing by splitting nodes as more data is observed in specific regions.
*   **Local GP Models**: Utilizes Gaussian Process regressors at the leaf nodes to perform regression, capturing local data characteristics.
*   **Continual Learning**: Designed to learn from data points one by one, allowing the model to evolve over time.
*   **Online Prediction**: Capable of making predictions at any point during the learning process.
*   **Ensemble Method**: Includes `GPForest` for running an ensemble of multiple GPTrees, which can improve prediction stability and accuracy.
*   **Customizable GPRs**: Allows users to define and use their own scikit-learn compatible Gaussian Process Regressor models within the tree nodes.
*   **Additive Leaf Kernels**: An optional low-order *additive* leaf kernel (`make_additive_kernel`) that improves sample efficiency and scaling to higher-dimensional problems, with a built-in full-dimensional "rescue" term that prevents any degradation on non-additive targets. See "Improving sample efficiency in higher dimensions" below.
*   **Interaction Pruning**: With `prune_interactions=True`, each node discovers from its own region's data which input pairs actually interact and gives its children an additive kernel pruned to those pairs — collapsing toward main-effects on separable regions, for cheaper, often more accurate leaf GPs. See "Pruning unused interactions as the tree grows" below.

## How it Works (Briefly)
GPTreeO builds a binary tree where each node represents a specific region of the input space.
- Leaf nodes contain their own Gaussian Process (GP) model, which is trained on the data points that fall into that node's defined region.
- When a leaf node accumulates a sufficient number of data points (determined by the `Nbar` parameter), it splits into two children. This process creates more specialized models for subregions of the data space.
- Predictions are typically made by the GP model in the leaf node into which a new data point falls. For overlapping regions (due to the `theta` parameter) or when using `GPForest`, predictions can be a weighted average from multiple relevant GPs.

## Installation
Details on installing via pip will be added soon. For now, you can clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/pygptreeo.git # Replace with the actual repository URL
cd pygptreeo
pip install numpy scikit-learn binarytree tqdm joblib
```

## Usage Example
Here's a simple example of how to use GPTreeO:

```python
import numpy as np
from pygptreeo import GPTree, Default_GPR

# 1. Initialize GPTree
# You can use the Default_GPR or define your own scikit-learn compatible GPR
# Nbar is the maximum number of points per leaf before it considers splitting.
gpt = GPTree(Nbar=50)

# 2. Prepare some data
# Let's create some 1D data for simplicity
X_train = np.linspace(0, 10, 100).reshape(-1, 1)
y_train = np.sin(X_train).ravel().reshape(-1, 1)

# 3. Feed data points to the tree sequentially
print("Training the GPTree...")
for i in range(len(X_train)):
    # GPTree expects 2D input for X and y for a single sample
    x_sample = X_train[i:i+1, :]
    y_sample = y_train[i:i+1, :]
    gpt.update_tree(x_sample, y_sample)
    if (i + 1) % 20 == 0:
        print(f"Processed {i+1}/{len(X_train)} points.")

# 4. Make predictions
print("\nMaking predictions...")
X_test = np.array([[0.5], [2.5], [5.5], [7.5], [9.5]])
y_pred, y_std = gpt.predict(X_test)

for i in range(len(X_test)):
    print(f"Input: {X_test[i,0]}, Prediction: {y_pred[i,0]:.4f}, StdDev: {y_std[i,0]:.4f}")

# The tree structure can be printed (optional, can be very large for many points)
# print("\nGPTree structure:")
# print(gpt.root)
```

## Improving sample efficiency in higher dimensions
Each leaf GP is trained on at most `Nbar` points, so a full-dimensional kernel
has to learn the leaf's function over *all* input dimensions from a small sample —
sample complexity that grows quickly with dimension. For targets with additive or
low-order interaction structure (very common in practice), a low-order **additive
kernel** learns the same function far more sample-efficiently, because each
observation constrains every one- and two-dimensional piece of the model.

`make_additive_kernel` builds such a kernel, plus a full-dimensional Matérn
"rescue" term whose learnable amplitude lets the model fall back to the ordinary
kernel on non-additive targets — so it cannot do worse than the default:

```python
import numpy as np
from pygptreeo import GPTree, Default_GPR, make_additive_kernel

n_features = 6
kernel = make_additive_kernel(n_features, interaction_depth=2, rescue=True)
gpt = GPTree(GPR=Default_GPR(kernel=kernel, alpha=1e-6), Nbar=80,
             use_standard_scaling=True, splitting_strategy="gradual",
             split_dimension_criteria="min_lengthscale")
# ... then stream data with gpt.update_tree(x, y, sigma) as usual.
```

On the standard benchmark functions this gives large held-out NRMSE reductions
(e.g. up to ~80% on `rosenbrock`/`levy`) with no degradation on any target. See
`examples/BENCHMARK_RESULTS_additive_kernel.md` and reproduce with
`examples/benchmark_additive_kernel.py`.

### A cheaper additive kernel: `make_order_additive_kernel`

`make_additive_kernel` builds the additive structure by enumerating every
interaction term explicitly, so a depth-`D` kernel over `d` inputs assembles
`O(d^D)` product matrices on each evaluation. `make_order_additive_kernel`
(`OrderAdditiveKernel`) is a drop-in alternative that uses the Duvenaud/OAK
construction: it gives each interaction *order* a single variance and assembles
all orders from the per-dimension kernels via the Newton–Girard recursion in
`O(d·D)`, independent of the number of terms. It keeps the same rescue term and
the same sample efficiency (it adds essentially no hyperparameters), but is much
cheaper to evaluate as `d` or the interaction order grows — making higher
`max_order` practical. Its gradients are analytic.

```python
from pygptreeo import GPTree, Default_GPR, make_order_additive_kernel
kernel = make_order_additive_kernel(n_features, max_order=2, rescue=True)
gpt = GPTree(GPR=Default_GPR(kernel=kernel, alpha=1e-6), Nbar=100,
             use_standard_scaling=True, splitting_strategy="gradual")
```

Compare it against the baseline and the explicit additive kernel at different
leaf sizes with `examples/benchmark_order_additive_nbar.py`.

### Pruning unused interactions as the tree grows

A depth-2 additive kernel carries a pairwise term for *every* pair of inputs
(`d + C(d,2)` terms), but most targets only couple a few pairs — and an
additively separable target couples none. `prune_interactions=True` makes each
node discover, from its own region's data, which pairs actually interact, and
gives the children it spawns an additive kernel restricted to those pairs:

```python
gpt = GPTree(GPR=Default_GPR(kernel=make_additive_kernel(n_features, interaction_depth=2),
                             alpha=1e-6),
             Nbar=80, use_standard_scaling=True, splitting_strategy="gradual",
             prune_interactions=True)
```

Discovery is **region-local and hierarchical**: every node carries a model-free
2-way ANOVA interaction screen, and when it splits each child inherits a copy of
the accumulated statistics and keeps refining it on its own points — so evidence
both accumulates down the tree and localises as leaves deepen. Children are warm
clones of their parent (they predict immediately) whose *next* fit uses the
pruned kernel, so pruning compounds down the tree with no extra GP fits. The
full-dimensional rescue term remains the safety net, which is what lets the
default settings prune aggressively. On the benchmark functions this leaves
accuracy neutral-to-better — large gains on separable targets (e.g. `levy`) — at
equal or better speed; the one cost is a small accuracy hit on targets whose
pairwise coupling is essential (e.g. `rosenbrock`), bounded by the rescue term.
Reproduce with `examples/benchmark_interaction_pruning.py`.

## Running Examples
For more detailed demonstrations, see the example scripts in the `examples/` directory:

*   `examples/example.py`: Shows a basic workflow of training and predicting with `GPTree`.
    ```bash
    python examples/example.py
    ```
*   `examples/performance_test.py`: Demonstrates performance metrics tracking and visualization during online learning.
    ```bash
    OMP_NUM_THREADS=1 python examples/performance_test.py
    ```
*   `examples/test_animated.py`: Provides an animated visualization of the `GPTree` learning process for 2D data.
    It requires command-line arguments:
    ```bash
    python examples/test_animated.py <target_function_name> <n_points> <Nbar> <retrain_step> <update_step> <live_update_bool>
    ```
    For example:
    ```bash
    python examples/test_animated.py eggholder 10000 200 200 10 1
    ```
    (Available target functions: `eggholder`, `himmelblau`, `rosenbrock`, `rastrigin`, `levy`, `custom`)

## Contributing
Contributions are welcome! Please feel free to submit issues or pull requests.

## License
This project is licensed under the terms of the LICENSE file.
