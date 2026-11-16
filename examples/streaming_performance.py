"""Accuracy, coverage and cost per point of the tree over a stream.

Streams N points of a target into one tree per configuration, predicting every
point before the tree is updated with it, and draws the metrics batch by batch:
prediction and update time, NRMSE, the fraction of predictions within 1 to 16%
of the true value, the 1-sigma coverage and the number of leaves.

    OMP_NUM_THREADS=1 python examples/streaming_performance.py --target eggholder --d 3 --N 20000
    OMP_NUM_THREADS=1 python examples/streaming_performance.py --target relic_density --configs tree,hybrid,global_gp
    OMP_NUM_THREADS=1 python examples/streaming_performance.py --target gaussian_peaks --d 6 --stream walker --configs tree,neighbours

Configurations: ``tree`` (the plain GPTree), ``neighbours`` (predictions from
each query's nearest stored points, ``predict_from='neighbours'``), ``hybrid``
(GP leaves on the residual of the feature network; needs PyTorch) and
``global_gp`` (GP leaves on the residual of the additive global GP). Targets: the functions of
``target_functions.py`` in ``--d`` dimensions and those of
``target_functions_physics.py`` in their own dimension. ``--stream`` is a kind
of ``target_functions.make_stream``; ``--noise`` adds Gaussian observation
noise, in units of the standard deviation of the target, and declares it to the
tree. Errors are always against the noise-free target. ``--max_memory_mb``
limits the memory of every tree, which then puts leaves to sleep and removes
the ones the stream has left:

    OMP_NUM_THREADS=1 python examples/streaming_performance.py --target gaussian_peaks --d 6 --stream walker --configs tree --max_memory_mb 10
"""
import argparse
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pygptreeo import GPTree
import target_functions as tf
from target_functions_physics import TARGETS as PHYSICS_TARGETS

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

ANALYTIC_TARGETS = {
    'eggholder': tf.Eggholder, 'himmelblau': tf.Himmelblau, 'rosenbrock': tf.Rosenbrock, 'rastrigin': tf.Rastrigin,
    'levy': tf.Levy, 'rotated_rosenbrock': tf.RotatedRosenbrock, 'gaussian_peaks': tf.GaussianPeaks,
    'active_peaks': tf.ActiveSubspacePeaks, 'michalewicz': tf.Michalewicz, 'ackley': tf.Ackley,
    'griewank': tf.Griewank, 'step_ridge': tf.StepRidge, 'chirp': tf.Chirp,
}
LABELS = {'tree': 'GP tree', 'neighbours': 'GP tree, predictions from neighbours',
          'hybrid': 'hybrid (GP leaves on network residual)', 'global_gp': 'GP tree + additive global GP'}
COLORS = {'tree': 'tab:blue', 'neighbours': 'tab:red', 'hybrid': 'tab:green', 'global_gp': 'tab:orange'}
TOLERANCES = (16, 8, 4, 2, 1)   # percent


def make_data(a, rng):
    """The stream inputs, the noise-free target values, the observations and their declared sigma."""
    if a.target in PHYSICS_TARGETS:
        target = PHYSICS_TARGETS[a.target]

        def by_column(x):        # make_stream's convention: one column per point, a float for one point
            x = np.asarray(x)
            return target(x.T) if x.ndim > 1 else float(target(x)[0])

        X = tf.make_stream(a.stream, by_column, target.d, a.N, rng)
        y_obs, sigma, y = target.sample(X, rng)
    else:
        target = ANALYTIC_TARGETS[a.target]
        X = tf.make_stream(a.stream, target, a.d, a.N, rng)
        y = target(X.T); y_obs = y; sigma = np.maximum(1e-3 * np.abs(y), 1e-6)
    if a.noise > 0:
        extra = a.noise * y.std()
        y_obs = y_obs + extra * rng.randn(a.N); sigma = np.hypot(sigma, extra)
    return X, y, y_obs, sigma


def make_tree(config, seed, max_memory_mb=None):
    if config == 'tree':
        return GPTree(max_memory_mb=max_memory_mb)
    if config == 'neighbours':
        return GPTree(max_memory_mb=max_memory_mb, predict_from='neighbours')
    if config == 'hybrid':
        # steps_per_update=1 spreads each refit of the network over the following updates
        return GPTree(max_memory_mb=max_memory_mb, global_mean='net',
                      global_mean_kwargs=dict(steps_per_update=1, random_state=seed))
    if config == 'global_gp':
        return GPTree(max_memory_mb=max_memory_mb, global_mean='additive_gp', global_mean_kwargs=dict(random_state=seed))
    raise ValueError(f"unknown configuration '{config}'")


def run(config, X, y, y_obs, sigma, batch, seed, max_memory_mb=None):
    """Stream the points through one tree; the metrics per batch of ``batch`` points."""
    np.random.seed(seed)
    N = (len(X) // batch) * batch
    pred, std, t_pred, t_upd, leaves = (np.empty(N) for _ in range(5))
    gpt = make_tree(config, seed, max_memory_mb)
    for i in range(N):
        xi = X[i:i + 1]
        t0 = time.perf_counter(); mu, sd = gpt.predict(xi); t1 = time.perf_counter()
        gpt.update_tree(xi, y_obs[i:i + 1, None], sigma[i:i + 1, None]); t2 = time.perf_counter()
        pred[i], std[i], t_pred[i], t_upd[i], leaves[i] = mu[0, 0], sd[0, 0], t1 - t0, t2 - t1, gpt.n_leaves
    err = (pred - y[:N]).reshape(-1, batch)
    y_b = y[:N].reshape(-1, batch)
    rel = np.abs(err) / np.maximum(np.abs(y_b), 1e-10)
    m = {'points': (np.arange(N // batch) + 1) * batch,
         'predict_time': t_pred.reshape(-1, batch).mean(1), 'update_time': t_upd.reshape(-1, batch).mean(1),
         'update_time_max': t_upd.reshape(-1, batch).max(1),
         'nrmse': np.sqrt((err ** 2).mean(1)) / np.ptp(y_b, axis=1),
         'coverage': (np.abs(err) <= std.reshape(-1, batch)).mean(1),
         'leaves': leaves.reshape(-1, batch)[:, -1]}
    for p in TOLERANCES:
        m[f'within_{p}'] = (rel <= p / 100.0).mean(1)
    return m


def plot(runs, a, d, path):
    fig, axs = plt.subplots(6, 1, figsize=(15, 19), sharex=True)
    fig.suptitle(f"pyGPTreeO performance metrics\n{a.target}, d = {d}, {a.stream} stream, {a.N} points, "
                 f"metrics per batch of {a.batch}", fontsize=16)
    for cfg, m in runs.items():
        lab, c = LABELS[cfg], COLORS[cfg]
        axs[0].plot(m['points'], m['predict_time'], label=f"{lab}: mean", linewidth=2.0, color=c)
        axs[1].plot(m['points'], m['update_time'], label=f"{lab}: mean", linewidth=2.0, color=c)
        axs[1].plot(m['points'], m['update_time_max'], label=f"{lab}: maximum in batch", linewidth=1.0, alpha=0.5, color=c)
        axs[2].plot(m['points'], m['nrmse'], label=lab, linewidth=2.0, color=c)
        for p, alpha in zip(TOLERANCES, (1.0, 0.85, 0.7, 0.55, 0.4)):
            axs[3].plot(m['points'], m[f'within_{p}'], label=f"{lab}: < {p}% error", linewidth=2.0, color=c, alpha=alpha)
        axs[4].plot(m['points'], m['coverage'], label=lab, linewidth=2.0, color=c)
        axs[5].plot(m['points'], m['leaves'], label=lab, linewidth=2.0, color=c)
    axs[0].set_ylabel('Time (s)'); axs[0].set_title('Prediction time per point'); axs[0].set_yscale('log')
    axs[1].set_ylabel('Time (s)'); axs[1].set_title('Tree update time per point'); axs[1].set_yscale('log')
    axs[2].set_ylabel('NRMSE'); axs[2].set_title('NRMSE of the predictions'); axs[2].set_yscale('log')
    axs[3].set_ylabel('Fraction'); axs[3].set_title('Fraction of predictions within x% of the true value'); axs[3].set_ylim([0, 1])
    axs[4].set_ylabel('Fraction'); axs[4].set_title('Empirical 1-sigma coverage'); axs[4].set_ylim([0, 1])
    axs[4].axhline(0.68, ls='--', color='black', linewidth=2.0)
    axs[5].set_ylabel('Leaves'); axs[5].set_title('Number of leaves'); axs[5].set_xlabel('Points processed')
    for ax in axs:
        ax.set_xlim([0, a.N]); ax.grid(True)
        ax.legend(loc='lower right' if ax is axs[3] else 'upper left', ncol=4, fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='eggholder', choices=sorted(ANALYTIC_TARGETS) + sorted(PHYSICS_TARGETS))
    ap.add_argument('--d', type=int, default=3, help='input dimension (the targets of target_functions.py only)')
    ap.add_argument('--stream', default='uniform', choices=('uniform', 'focusing', 'sweeping', 'walker'))
    ap.add_argument('--N', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=1000)
    ap.add_argument('--configs', default='tree,hybrid', help='comma-separated: tree, neighbours, hybrid, global_gp')
    ap.add_argument('--noise', type=float, default=0.0, help='noise std in units of the standard deviation of the target')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--max_memory_mb', type=float, default=None,
                    help="memory limit of a tree in MB, global model included (default: no limit)")
    ap.add_argument('--out', default=None, help='figure file (default: streaming_performance_<target>.png)')
    a = ap.parse_args()

    X, y, y_obs, sigma = make_data(a, np.random.RandomState(a.seed))
    runs = {}
    for cfg in a.configs.split(','):
        t0 = time.time()
        runs[cfg] = m = run(cfg, X, y, y_obs, sigma, a.batch, a.seed, a.max_memory_mb)
        print(f"{cfg:10s} {time.time() - t0:6.0f} s   last batch: predict {1e3 * m['predict_time'][-1]:.2f} ms, "
              f"update {1e3 * m['update_time'][-1]:.2f} ms (max {1e3 * m['update_time_max'][-1]:.0f} ms), "
              f"NRMSE {m['nrmse'][-1]:.4f}, within 1% {m['within_1'][-1]:.2f}, coverage {m['coverage'][-1]:.2f}, "
              f"leaves {int(m['leaves'][-1])}", flush=True)
    out = a.out or f"streaming_performance_{a.target}.png"
    plot(runs, a, X.shape[1], out)
    print(f"figure saved to {out}")


if __name__ == '__main__':
    main()
