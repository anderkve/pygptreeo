"""Predictions away from the data: the neighbours against the leaf's own GP, by how
far a query lies outside its neighbours and how well they fit the leaf's kernel.

    python extrapolation_diag.py target stream N out.npz [noise] [kernel]

For uniform test points after N points of the stream, with the kernel of the
query's leaf alone: the two predictions, the consistency of the neighbours,
measures of how far the query lies outside the data, and the prediction with
the neighbours' noise raised until they are consistent with the kernel.
`noise` is the declared noise as a share of the target value (analytic targets
only), `kernel` is 'default' or 'additive'.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # before numpy is loaded
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ['PYGPTREEO_LIVE'] = '1'
sys.path.insert(0, HERE)
warnings.simplefilter('ignore')
import numpy as np                                              # noqa: E402
import bench                                                    # noqa: E402
from variants import BASE                                       # noqa: E402
from pygptreeo import GPTree, Default_GPR, AdditiveMaternKernel  # noqa: E402

bench.ANALYTIC['himmelblau'] = (lambda x: 10.0 * (bench.tf.Himmelblau(x) + 1.0), 2)
bench.ANALYTIC['himmelblau4'] = (lambda x: 10.0 * (bench.tf.Himmelblau(x) + 1.0), 4)
target, stream, N, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
noise_share = float(sys.argv[5]) if len(sys.argv) > 5 else None
kernel = sys.argv[6] if len(sys.argv) > 6 else 'default'
data = bench.make_data(target, stream, N, 1)
X, y_obs, sigma = data['X'], data['y_obs'], data['sigma']
if noise_share is not None:
    sigma = np.maximum(noise_share * np.abs(y_obs), 1e-12)
np.random.seed(1)
options = dict(BASE, predict_from='neighbours')
if kernel == 'additive':
    options['GPR'] = Default_GPR(kernel=AdditiveMaternKernel(d=data['d'], order=2, nu=1.5), alpha=1e-6)
m = GPTree(**options)
for i in range(N):
    m.update_tree(X[i:i + 1], y_obs[i:i + 1, None], sigma[i:i + 1, None])
search = m.neighbours


def consistent(gp, Z, t, noise, zq, first):
    """Conditioning with the smallest added noise at which the neighbours fit the kernel."""
    if first[2][0] <= 1.0:
        return first
    lo, hi = np.log(max(noise.min(), 1e-14)), 0.0
    result = first
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        trial = gp.condition(Z, t, noise + np.exp(mid), zq)
        if trial is None or trial[2][0] > 1.0:
            lo = mid
        else:
            hi, result = mid, trial
    return result


rows = []
for q, truth in zip(data['X_uni'], data['y_uni']):
    x = q[None, :]
    leaf = m.leaf_index.locate(q)
    if leaf is None or not leaf._can_condition():
        continue
    Xn, yn, sn, growth = search.find(q, leaf.fit_std, leaf=leaf)
    ys = leaf.y_scalers[0]
    noise = sn[:, 0] ** 2
    if growth is not None:
        unit = leaf._predicted_variance()
        noise = noise + growth * (noise + (unit[0] if unit is not None else 0.01 * yn[:, 0].var()))
    Z, t, zq = leaf._scaled_inputs(Xn), (yn[:, 0] - ys.mean_[0]) / ys.scale_[0], leaf._scaled_inputs(x)
    gp = leaf.my_GPRs[0]
    result = gp.condition(Z, t, noise / ys.scale_[0] ** 2, zq)
    if result is None:
        continue
    eased = consistent(gp, Z, t, noise / ys.scale_[0] ** 2, zq, result)
    own = leaf._predict_own(x)
    shipped = m.predict(x)
    spread = Xn.std(axis=0)
    distance = np.sqrt((((Xn - q) / leaf.fit_std) ** 2).sum(axis=1))
    mean = result[0][0, 0] * ys.scale_[0] + ys.mean_[0]
    low, high = yn[:, 0].min(), yn[:, 0].max()
    rows.append((truth, mean, result[1][0, 0] * ys.scale_[0], result[2][0], own[0][0, 0], own[1][0, 0],
                 leaf.distance_from_fit(x)[0],                                           # from the leaf's fitted points, in their spreads
                 np.abs((q - Xn.mean(axis=0)) / np.where(spread > 0, spread, 1.0)).max(),  # from the centre of the neighbours, in their spreads
                 distance.min() / distance.max(),                                        # nearest neighbour against the furthest
                 float(leaf.prefers_own), leaf.sigma_scalers[0],
                 distance.min(),                                                         # nearest neighbour, in the leaf's spreads
                 shipped[0][0, 0], shipped[1][0, 0],
                 eased[0][0, 0] * ys.scale_[0] + ys.mean_[0], eased[1][0, 0] * ys.scale_[0],
                 np.abs(t).max(),                                                        # neighbours' targets, in the leaf's scale
                 (mean - np.clip(mean, low, high)) / max(high - low, 1e-300)))            # mean outside the neighbours' targets, in their range
np.savez(out, rows=np.array(rows), range=np.ptp(data['y_uni']), target=target, stream=stream)
print(target, stream, len(rows), 'queries', flush=True)
