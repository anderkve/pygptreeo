"""Per-query diagnostics of nearest-neighbour conditioning against the leaf's own GP.

Replays a stream through the baseline tree and logs, for each query of its second
half: the truth, the leaf GP's prediction, and what a GP with the leaf's kernel
gives when conditioned on the query's nearest stored points, for every size of
the neighbour set at once. With the neighbours ordered nearest first, the
Cholesky factor of the k nearest is the leading block of the factor of all of
them, so one factorisation gives every prefix: with v = L^-1 y and u = L^-1 k*,
the mean from the k nearest is sum(u[:k] v[:k]), its variance k** - sum(u[:k]^2),
and v[i] is the error of predicting neighbour i from the nearer ones, in units
of its predicted sigma.
"""
import os
import sys

import numpy as np
from scipy.linalg import cholesky, solve_triangular

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench                                                    # noqa: E402
from pygptreeo import GPTree                                     # noqa: E402
from variants import BASE                                        # noqa: E402

KS = np.array([10, 15, 20, 30, 40, 50, 60, 80, 100, 130, 160, 200])


def replay(target, stream, N, seed, k_max=200, start=0.5, tree_kwargs=None):
    data = bench.make_data(target, stream, N, seed)
    X, y, y_obs, sigma = data['X'], data['y'], data['y_obs'], data['sigma']
    np.random.seed(seed)
    kw = dict(BASE)
    kw.update(tree_kwargs or {})
    gpt = GPTree(**kw)
    out = dict(y=[], leaf=[], leaf_sd=[], leaf_scaler=[], knn=[], knn_sd=[], v=[], far=[], index=[])
    for i in range(N):
        xi = X[i:i + 1]
        if i >= int(start * N) and i > k_max:
            node = gpt.root
            while not node.is_leaf:
                node = node.children[int(node.prob_func(xi)[0, 0] > 0.5)]
            if node._is_fitted() and node.X_scaler is not None:
                mu_l, sd_l = node.predict(xi, return_std=True, use_calibrated_sigma=False)
                kernel = node.my_GPRs[0].sklearn_gpr.kernel_
                xm, xs = node.X_scaler.mean_, node.X_scaler.scale_
                ym, ysc = node.y_scalers[0].mean_[0], node.y_scalers[0].scale_[0]
                d2 = (((X[:i] - xi[0]) / xs) ** 2).sum(axis=1)
                idx = np.argpartition(d2, k_max - 1)[:k_max]
                idx = idx[np.argsort(d2[idx])]
                Xn = (X[idx] - xm) / xs
                qs = (xi - xm) / xs
                K = kernel(Xn)
                K[np.diag_indices_from(K)] += (sigma[idx] / ysc) ** 2
                try:
                    L = cholesky(K, lower=True, check_finite=False)
                except np.linalg.LinAlgError:
                    K[np.diag_indices_from(K)] += 1e-8 * kernel.diag(qs)[0]
                    L = cholesky(K, lower=True, check_finite=False)
                v = solve_triangular(L, (y_obs[idx] - ym) / ysc, lower=True, check_finite=False)
                u = solve_triangular(L, kernel(Xn, qs)[:, 0], lower=True, check_finite=False)
                mean = np.cumsum(u * v)[KS - 1] * ysc + ym
                var = np.maximum(kernel.diag(qs)[0] - np.cumsum(u ** 2)[KS - 1], 0.0)
                out['y'].append(y[i]); out['leaf'].append(mu_l[0, 0]); out['leaf_sd'].append(sd_l[0, 0])
                out['leaf_scaler'].append(node.sigma_scaler); out['knn'].append(mean); out['knn_sd'].append(np.sqrt(var) * ysc)
                out['v'].append(v.astype(np.float32)); out['far'].append(node.distance_from_fit(xi)[0]); out['index'].append(i)
        gpt.update_tree(xi, y_obs[i:i + 1, None], sigma[i:i + 1, None])
    return {k: np.array(v) for k, v in out.items()}


if __name__ == '__main__':
    target, stream, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
    d = replay(target, stream, N, 1)
    el = np.abs(d['leaf'] - d['y'])
    print(f"{target} {stream}: rmse leaf {np.sqrt(np.mean(el ** 2)):.4g}  knn by k: "
          + "  ".join(f"{k}:{np.sqrt(np.mean((d['knn'][:, j] - d['y']) ** 2)):.4g}" for j, k in enumerate(KS)))
