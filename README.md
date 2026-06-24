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
*   **Sample-efficient additive kernels**: `NewtonGirardAdditiveKernel` exploits low interaction-order structure with only `O(d)` hyperparameters, improving sample efficiency on functions that decompose into low-dimensional terms.

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

## Sample-efficient additive kernels

For functions with low *interaction order* — i.e. well approximated by a sum of
low-dimensional terms (main effects + pairwise interactions) rather than a fully joint
`d`-dimensional surface — the leaf GPs can be made substantially more sample-efficient
with an **additive kernel**. `NewtonGirardAdditiveKernel` models the covariance as a sum
over interaction orders,

```
k(x, x') = Σ_q  σ_q² · e_q(z_1, …, z_d),     z_i = exp(-(x_i - x'_i)² / 2ℓ_i²),
```

where `e_q` is the order-`q` elementary symmetric polynomial of the per-dimension RBFs
`z_i`. The `e_q` are evaluated from power sums via the Newton–Girard identities in
`O(d·Q)` time, so the kernel has only `d` length scales + `Q` order variances (`O(d)`
hyperparameters) instead of the `C(d, q)` terms a naive additive kernel would enumerate.
A small maximum order `Q` (e.g. 2: main effects + pairwise interactions) breaks the curse
of dimensionality for such functions while remaining cheap to fit.

The recommended leaf kernel pairs an order-2 additive component with a *separate* Matérn
"catch-all" (with its own length scales) that absorbs any higher-order or rougher
residual:

```python
import numpy as np
from pygptreeo import GPTree, Default_GPR, NewtonGirardAdditiveKernel
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

d = 4  # input dimensionality

# order-2 additive (main effects + pairwise) + a separate Matern catch-all
kernel = (NewtonGirardAdditiveKernel(length_scale=[1.0] * d, order_std=[1.0, 1.0])
          + ConstantKernel() * Matern(nu=1.5, length_scale=[1.0] * d))

gpt = GPTree(GPR=Default_GPR(kernel=kernel, alpha=1e-6), Nbar=100)
# then feed data and predict exactly as in the Usage Example above
```

`order_std` has one entry per active interaction order, and its length sets the maximum
order `Q` (`[1.0, 1.0]` ⇒ orders 1 and 2). All hyperparameters — the `d` shared length
scales, the per-order variances, and the catch-all — are tuned per leaf by
marginal-likelihood maximisation. When a function has no low-order structure to exploit,
the additive amplitudes are simply down-weighted and the kernel degrades gracefully to
the plain Matérn catch-all, so it is safe to use as a general-purpose leaf kernel.

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
