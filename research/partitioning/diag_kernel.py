"""Replay: which leaf's kernel should a neighbour-conditioned prediction use?

For each query of the second half of a stream, the neighbours are found in the
metric of the query's own leaf, and the prediction is made with the kernel of
(a) the own leaf and (b) the leaf that holds most of the neighbours. Logged:
both predictions, their consistency and their log marginal likelihood of the
neighbours' targets in the targets' own units.

    python diag_kernel.py dev 5000
"""
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.linalg.lapack import dpotrf, dtrtrs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench                                                    # noqa: E402
from pygptreeo import GPTree                                     # noqa: E402
from variants import BASE                                        # noqa: E402
from run_matrix import SETS                                      # noqa: E402


def conditioned(leaf, Xn, yn, sn, xq):
    """Prediction at xq of the leaf's kernel given the points, with its
    consistency and the log marginal likelihood per point in raw target units."""
    kernel = leaf.my_GPRs[0].sklearn_gpr.kernel_
    xm, xs = leaf.X_scaler.mean_, leaf.X_scaler.scale_
    ym, ysc = leaf.y_scalers[0].mean_[0], leaf.y_scalers[0].scale_[0]
    n = len(yn)
    K_all = kernel(np.vstack(((Xn - xm) / xs, (xq - xm) / xs)))
    K = K_all[:n, :n].copy()
    K.flat[::n + 1] += (sn / ysc) ** 2
    L, info = dpotrf(K, lower=True, overwrite_a=True)
    if info != 0:
        return np.nan, np.nan, np.inf, -np.inf
    solved, _ = dtrtrs(L, np.column_stack(((yn - ym) / ysc, K_all[:n, n:])), lower=True)
    v, u = solved[:, 0], solved[:, 1]
    chi2 = v @ v / n
    loglik = -0.5 * chi2 - np.log(np.diagonal(L)).sum() / n - np.log(ysc)
    return u @ v * ysc + ym, np.sqrt(max(K_all[n, n] - u @ u, 0.0)) * ysc, chi2, loglik


def replay(args):
    target, stream, N, seed, k = args
    path = os.path.join(bench.RESULTS, '_diagk', f"{target}__{stream}__N{N}__s{seed}.npz")
    if os.path.exists(path):
        return args
    data = bench.make_data(target, stream, N, seed)
    X, y, y_obs, sigma = data['X'], data['y'], data['y_obs'], data['sigma']
    np.random.seed(seed)
    gpt = GPTree(**BASE)
    owner = {}                         # stream index of a stored point -> its leaf, via the leaves' arrays
    rows = []
    for i in range(N):
        xi = X[i:i + 1]
        if i >= N // 2:
            leaf = gpt.root
            while not leaf.is_leaf:
                leaf = leaf.children[int(leaf.prob_func(xi)[0, 0] > 0.5)]
            if leaf._is_fitted() and leaf.X_scaler is not None:
                mu_l, sd_l = leaf.predict(xi, return_std=True, use_calibrated_sigma=False)
                # neighbours in the own leaf's metric, with the leaf each lies in
                leaves = [node for node in gpt.root.leaves if node.n_points > 0]
                Xall = np.concatenate([node.my_X_data for node in leaves])
                yall = np.concatenate([node.my_y_data[:, 0] for node in leaves])
                sall = np.concatenate([node.my_sigma_data[:, 0] for node in leaves])
                which = np.repeat(np.arange(len(leaves)), [node.n_points for node in leaves])
                d2 = (((Xall - xi[0]) / leaf.fit_std) ** 2).sum(axis=1)
                idx = np.argpartition(d2, k - 1)[:k]
                counts = np.bincount(which[idx], minlength=len(leaves))
                major = leaves[int(np.argmax(counts))]
                own = conditioned(leaf, Xall[idx], yall[idx], sall[idx], xi)
                if major is leaf or not major._is_fitted() or major.X_scaler is None:
                    other, same = own, 1.0
                else:
                    other, same = conditioned(major, Xall[idx], yall[idx], sall[idx], xi), 0.0
                rows.append((y[i], mu_l[0, 0], sd_l[0, 0]) + own + other
                            + (same, counts[leaves.index(leaf)] / k, counts.max() / k))
        gpt.update_tree(xi, y_obs[i:i + 1, None], sigma[i:i + 1, None])
    fields = ('y', 'leaf', 'leaf_sd', 'own', 'own_sd', 'own_chi2', 'own_ll', 'maj', 'maj_sd', 'maj_chi2', 'maj_ll',
              'same', 'share_own', 'share_major')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{f: np.array([r[j] for r in rows]) for j, f in enumerate(fields)})
    return args


if __name__ == '__main__':
    targets = SETS[sys.argv[1]] if sys.argv[1] in SETS else sys.argv[1].split(',')
    N = int(sys.argv[2])
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    jobs = [(t, s, N, 1, 100) for t, s in itertools.product(targets, ('uniform', 'walker', 'sweeping'))]
    with Pool(procs) as pool:
        for n, a in enumerate(pool.imap_unordered(replay, jobs), 1):
            print(n, len(jobs), a[:2], flush=True)
