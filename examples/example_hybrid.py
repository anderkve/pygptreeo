"""The hybrid tree: GP leaves on the residual of a tree-wide feature network.

Streams points of a 6D target into a plain GPTree and into the hybrid
(``GPTree(global_mean='net')``), predicting each point before it is added, and
prints the prequential error, the 1-sigma coverage and the time per point of
both. Needs PyTorch (``pip install pygptreeo[neural]``).

    OMP_NUM_THREADS=1 python examples/example_hybrid.py
"""
import time

import numpy as np

from pygptreeo import GPTree
from target_functions import GaussianPeaks

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

d, N = 6, 6000
rng = np.random.RandomState(1)
X = rng.rand(N, d)
y = GaussianPeaks(X.T)
sigma = 1e-3 * np.abs(y) + 1e-6          # observation-noise std of each point

trees = {
    'GP tree': GPTree(Nbar=100),
    # the leaves model y - h(x), h the feature network trained on the stream;
    # global_mean_kwargs takes FeatureNetLearner arguments, here an amortised
    # refit (one L-BFGS iteration per update) instead of a latency spike per refit
    'hybrid': GPTree(Nbar=100, global_mean='net', global_mean_kwargs=dict(steps_per_update=1, random_state=1)),
}

for name, gpt in trees.items():
    err, cover, t0 = [], [], time.perf_counter()
    for i in range(N):
        mu, sd = gpt.predict(X[i:i + 1])
        err.append(mu[0, 0] - y[i]); cover.append(abs(err[-1]) <= sd[0, 0])
        gpt.update_tree(X[i:i + 1], y[i:i + 1, None], sigma[i:i + 1, None])
    err = np.array(err[N // 2:]); cover = np.array(cover[N // 2:])   # the second half of the stream
    nrmse = np.sqrt(np.mean(err ** 2)) / (y.max() - y.min())
    print(f"{name:8s} NRMSE {nrmse:.4f}  coverage {cover.mean():.2f}  {1e3 * (time.perf_counter() - t0) / N:.1f} ms per point  "
          f"{len(gpt.root.leaves)} leaves")
