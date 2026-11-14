"""GPyTorch as the backend of the leaf GPs.

Streams points of the 2D Himmelblau function into a GPTree whose leaves are
GPyTorch models, and into a tree with the default scikit-learn backend for
comparison. Needs GPyTorch (``pip install pygptreeo[gpytorch]``).

    python examples/gpytorch_backend.py
"""
import time

import numpy as np
import gpytorch

from pygptreeo import GPTree
from pygptreeo.adapters import GPyTorchAdapter
from target_functions import Himmelblau

rng = np.random.RandomState(1)
X_train, X_test = rng.rand(600, 2), rng.rand(500, 2)
y_train, y_test = Himmelblau(X_train.T), Himmelblau(X_test.T)
sigma = np.full((1, 1), 1e-3)                # observation-noise std of each point

# The mean and the kernel are GPyTorch modules. The adapter optimises the
# hyperparameters of each leaf with 'adam' or 'lbfgs'; device='cuda' uses a GPU.
adapter = GPyTorchAdapter(
    mean_module=gpytorch.means.ConstantMean(),
    covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2)),
    optimizer='adam', learning_rate=0.1, training_iterations=50, device='cpu')

trees = {'GPyTorch': GPTree(GPR=adapter, Nbar=100), 'scikit-learn': GPTree(Nbar=100)}

for name, gpt in trees.items():
    t0 = time.perf_counter()
    for i in range(len(X_train)):
        gpt.update_tree(X_train[i:i + 1], y_train[i:i + 1, None], sigma)
    y_pred, y_std = gpt.predict(X_test)
    err = y_pred[:, 0] - y_test
    print(f"{name:13s} RMSE {np.sqrt(np.mean(err ** 2)):.4f}  1-sigma coverage {np.mean(np.abs(err) <= y_std[:, 0]):.2f}  "
          f"{len(gpt.root.leaves)} leaves  {time.perf_counter() - t0:.1f} s")
