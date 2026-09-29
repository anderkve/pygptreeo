"""Per-prediction attribution for the residual tree on the sweeping stream.

Runs the plain tree and the residual tree (global model as in
``benchmark_global_mean_streams.py``) on one seed of the sweeping stream and, for
every prequential prediction, records the leaf's number of own points, how often
it has been retrained, its depth, the error of the global snapshot the leaf was
fitted against, the error the *current* global snapshot would have made, and how
many versions the leaf's snapshot is behind. The summary bins the error by these
quantities, which shows whether leaves with outdated snapshots drive the error.

Usage:
    OMP_NUM_THREADS=1 python examples/diagnose_global_mean_sweep.py [target] [seed]
"""
import sys, io, contextlib, numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warnings; warnings.filterwarnings("ignore")
import benchmark_global_mean_streams as bm
from pygptreeo import GPTree, Default_GPR
from pygptreeo.gpnode import GPNode

tname = sys.argv[1] if len(sys.argv) > 1 else 'rotated_rosenbrock'
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
target = bm.TARGETS[tname]; d, N = 6, 4000
rng = np.random.RandomState(seed)
X, X_focus = bm.make_stream('sweeping', target, d, N, rng); y = target(X.T)
sig = np.maximum(1e-3 * np.abs(y), 1e-6)
yrange = float(np.ptp(target(rng.rand(3000, d).T)))

# count fits per node
_fit = GPNode.fit_my_GPR
def counted_fit(self, force_training=False):
    did = _fit(self, force_training)
    if did: self._n_fits = getattr(self, '_n_fits', 0) + 1
    return did
GPNode.fit_my_GPR = counted_fit

def route(gpt, x):
    node = gpt.root
    while not node.is_leaf:
        node = node.children[int(node.prob_func(x)[0][0] >= 0.5)]
    return node

def run(config):
    model = bm.GlobalMeanModel(d, seed) if config == 'global' else None
    bm._ACTIVE['model'] = model
    np.random.seed(seed)
    gpt = GPTree(GPR=Default_GPR(n_restarts_optimizer=1), Nbar=100, theta=1e-4, retrain_every_n_points=25,
                 splitting_strategy='gradual', use_calibrated_sigma=False)
    rec = []
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(N):
            xi = X[i:i+1]
            if model is not None: model.observe(xi, y[i])
            leaf = route(gpt, xi) if i > 0 else gpt.root
            n_own = leaf.n_points; n_fits = getattr(leaf, '_n_fits', 0); depth = len(leaf.name) - 1
            snap = getattr(leaf, '_fitted_mean', None)
            m_err = (snap.predict(xi)[0] - y[i]) if snap is not None else np.nan
            cur_err = (model.current.predict(xi)[0] - y[i]) if (model is not None and model.current is not None) else np.nan
            stale = (model.current.version - snap.version) if (snap is not None and model is not None and model.current is not None) else -1
            mu, _ = gpt.predict(xi)
            rec.append((i, mu[0, 0] - y[i], n_own, n_fits, depth, m_err, cur_err, stale))
            gpt.update_tree(xi, np.array([[y[i]]]), np.array([[sig[i]]]))
    bm._ACTIVE['model'] = None
    return np.array(rec, dtype=float), gpt

out = {}
for cfg in ['tree', 'global']:
    rec, gpt = run(cfg); out[cfg] = rec
    i, err, n_own, n_fits, depth, m_err, cur_err, stale = rec.T
    after = i >= 666
    print(f"\n{tname} seed {seed} sweeping, {cfg}: leaves {len(gpt.root.leaves)}, prequential NRMSE {np.sqrt(np.mean(err[after]**2))/yrange:.4f}")
    bins = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 10**9)]
    print("   leaf own points at prediction | share of pts | RMSE/range | share of SSE")
    sse = np.sum(err[after]**2)
    for a, b in bins:
        m = after & (n_own >= a) & (n_own < b)
        if m.sum() == 0: continue
        print(f"   [{a:>3d},{b if b < 10**9 else 'inf':>4}) {m.sum()/after.sum():>12.2f} {np.sqrt(np.mean(err[m]**2))/yrange:>11.4f} {np.sum(err[m]**2)/sse:>13.2f}")
    m0 = after & (n_fits == 0); m1 = after & (n_fits >= 1)
    print(f"   leaf never retrained since creation: {m0.sum()/after.sum():.2f} of pts, RMSE/range {np.sqrt(np.mean(err[m0]**2))/yrange:.4f} | retrained: RMSE/range {np.sqrt(np.mean(err[m1]**2))/yrange:.4f}")
    if cfg == 'global':
        ok = after & np.isfinite(m_err)
        print(f"   global model alone (snapshot used by the leaf) RMSE/range on the stream: {np.sqrt(np.mean(m_err[ok]**2))/yrange:.4f}; "
              f"leaf residual error = total - global: RMSE/range {np.sqrt(np.mean((err[ok]-m_err[ok])**2))/yrange:.4f}")
        print(f"   CURRENT global snapshot RMSE/range on the same points: {np.sqrt(np.mean(cur_err[ok]**2))/yrange:.4f}")
        for k in [0, 1, 2, 3]:
            mk = ok & (stale == k) if k < 3 else ok & (stale >= 3)
            if mk.sum(): print(f"   leaf snapshot {k if k < 3 else '>=3'} versions behind: {mk.sum()/ok.sum():.2f} of pts | total RMSE/range {np.sqrt(np.mean(err[mk]**2))/yrange:.4f} | leaf-snapshot global err {np.sqrt(np.mean(m_err[mk]**2))/yrange:.4f} | current-snapshot err {np.sqrt(np.mean(cur_err[mk]**2))/yrange:.4f}")
        # top-10 worst predictions: what happened there?
        worst = np.argsort(-np.abs(err * after))[:8]
        print("   worst predictions (i, err/range, own pts, fits, depth, global err/range):")
        for j in worst: print(f"      {int(i[j]):5d} {err[j]/yrange:+.4f} {int(n_own[j]):3d} {int(n_fits[j]):2d} {int(depth[j]):2d} {m_err[j]/yrange:+.4f}")
# paired comparison: same stream, per-batch
et, eg = out['tree'][:, 1], out['global'][:, 1]
print("\nper-500-point batch RMSE/range, tree vs global:")
for a in range(666, N, 500):
    print(f"   [{a},{a+500}) tree {np.sqrt(np.mean(et[a:a+500]**2))/yrange:.4f}  global {np.sqrt(np.mean(eg[a:a+500]**2))/yrange:.4f}")
