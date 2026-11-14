"""Basic usage: online regression with a GPTree.

Streams points of the 2D Himmelblau function into a GPTree. Every point is
predicted before the tree is updated with it, so the reported errors are those
of predictions on points the tree has not seen.

    python examples/example.py
"""
import numpy as np

from pygptreeo import GPTree
from target_functions import Himmelblau

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

n_dims, n_pts, batch = 2, 2000, 250
rng = np.random.RandomState(1)
X = rng.rand(n_pts, n_dims)                  # inputs in the unit square
y = Himmelblau(X.T)
sigma = 1e-3 * np.abs(y)                     # observation-noise std of each point

# The defaults: Matern 3/2 leaf GPs with one length scale per input dimension,
# at most Nbar points per leaf, and calibrated predictive uncertainties.
gpt = GPTree(Nbar=100)

err, std = np.empty(n_pts), np.empty(n_pts)
for i in range(n_pts):
    x_i = X[i:i + 1]                         # 2D arrays, one row per point
    y_pred, y_std = gpt.predict(x_i)
    gpt.update_tree(x_i, y[i:i + 1, None], sigma[i:i + 1, None])
    err[i], std[i] = y_pred[0, 0] - y[i], y_std[0, 0]
    if (i + 1) % batch == 0:
        e, s = err[i + 1 - batch:i + 1], std[i + 1 - batch:i + 1]
        print(f"{i + 1:5d} points: NRMSE of the last {batch} predictions {np.sqrt(np.mean(e ** 2)) / np.ptp(y):.3f}, "
              f"1-sigma coverage {np.mean(np.abs(e) <= s):.2f}, {len(gpt.root.leaves)} leaves")

# Predictions for many points at once
X_test = rng.rand(1000, n_dims)
y_pred, y_std = gpt.predict(X_test)
e = y_pred[:, 0] - Himmelblau(X_test.T)
print(f"\nTest set: NRMSE {np.sqrt(np.mean(e ** 2)) / np.ptp(y):.3f}, 1-sigma coverage {np.mean(np.abs(e) <= y_std[:, 0]):.2f}")
