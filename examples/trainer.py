"""Training a GPTree on points that a Trainer chooses.

The tree is to be accurate where the 4D GaussianPeaks function is below 3,
which is a tenth of the unit cube. A Trainer suggests the points at which the
function is evaluated, and a second tree gets as many uniform random points.
Both are tested on points where the function is below 3.

    OMP_NUM_THREADS=1 python examples/trainer.py
"""
import numpy as np

from pygptreeo import GPTree, Trainer
from target_functions import GaussianPeaks

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

n_dims, n_pts, batch = 4, 3000, 50
highest, atol = 3.0, 0.05                    # the range of interest, and the accuracy wanted there
rng = np.random.RandomState(1)
np.random.seed(1)


def evaluate(X):
    y = GaussianPeaks(X.T)
    return y, 1e-3 * np.abs(y)               # the target and its observation-noise std


X_test = rng.rand(100000, n_dims)
X_test = X_test[GaussianPeaks(X_test.T) < highest]
y_test = GaussianPeaks(X_test.T)


def rmse(tree):
    y_pred, _ = tree.predict(X_test)
    return np.sqrt(np.mean((y_pred[:, 0] - y_test) ** 2))


gpt = GPTree(Nbar=100)
trainer = Trainer(gpt, bounds=(np.zeros(n_dims), np.ones(n_dims)), atol=atol, target_range=(-np.inf, highest), seed=1)
gpt_random = GPTree(Nbar=100)

while gpt.n_points_seen < n_pts:
    X = trainer.suggest(batch)               # the points to evaluate next
    y, sigma = evaluate(X)
    trainer.add(X, y, sigma)

    X = rng.rand(batch, n_dims)
    y, sigma = evaluate(X)
    gpt_random.batch_update_tree(X, y, sigma)

    if gpt.n_points_seen % 500 == 0:
        report = trainer.report()
        print(f"{gpt.n_points_seen:5d} points: RMSE in the range {rmse(gpt):.3f} with the Trainer, {rmse(gpt_random):.3f} "
              f"with random points. Report: sigma above the tolerance in {100 * report['share_above_tolerance']:.0f}% "
              f"of the range, {100 * report['share_within_tolerance']:.0f}% of {report['n_checks']} checks within the tolerance")
