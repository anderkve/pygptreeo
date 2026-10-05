"""Compare the plain GP tree with the neural-linear tree on one stream, point by point.

Both trees receive the same stream; every point is predicted before it is given
to the tree (prequential), and the prediction, its sigma, the prediction and
update times and the number of leaves are written per point to a CSV in the
layout of ``performance_test.py``. ``--plot`` draws the usual performance
figure, batch by batch, with both configurations overlaid: prediction time,
update time, NRMSE, the fraction of predictions within 1 to 16% of the true
value, the empirical 1-sigma coverage, and the number of leaves.

    # one configuration per process (so the timings are clean), then the figure
    OMP_NUM_THREADS=1 python examples/compare_tree_vs_neural.py --target eggholder --d 3 --N 40000 --config tree
    OMP_NUM_THREADS=1 python examples/compare_tree_vs_neural.py --target eggholder --d 3 --N 40000 --config neural
    python examples/compare_tree_vs_neural.py --target eggholder --d 3 --N 40000 --plot

Configurations: ``tree`` (``Default_GPR(n_restarts_optimizer=1)``, ARD Matern
leaves), ``neural`` (``NeuralLinearGPR`` on a ``FeatureNetLearner`` with
refits spread over the stream, 8 Adam steps per update) and ``hybrid`` (the GP
leaves on the residual of that network, ``global_mean=NetGlobalMean(...)``) and ``global_gp``
(the GP leaves on the residual of the additive-GP global model of
``BENCHMARK_RESULTS_global_mean_streams.md``); all with
``Nbar = 100``, ``theta = 1e-4``, a retrain every 25 points, gradual splitting
and calibrated sigma. Streams: ``uniform`` (the default), ``focusing``,
``sweeping`` and ``walker`` from ``benchmark_global_mean_streams.py``.
"""

import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'compare_tree_vs_neural')
BATCH = 2000
TOLERANCES = (16, 8, 4, 2, 1)   # percent


def tag(a):
    return f"{a.target}_d{a.d}_{a.stream}_N{a.N}_seed{a.seed}"


def run(a):
    from pygptreeo import GPTree, Default_GPR, FeatureNetLearner, NeuralLinearGPR
    from benchmark_global_mean_streams import make_stream
    import target_functions as tf
    targets = {'eggholder': tf.Eggholder, 'himmelblau': tf.Himmelblau, 'rosenbrock': tf.Rosenbrock,
               'rastrigin': tf.Rastrigin, 'levy': tf.Levy, 'rotated_rosenbrock': tf.RotatedRosenbrock,
               'gaussian_peaks': tf.GaussianPeaks}
    target = targets[a.target]
    rng = np.random.RandomState(a.seed)
    X, _ = make_stream(a.stream, target, a.d, a.N, rng)
    y = target(X.T); sig = np.maximum(1e-3 * np.abs(y), 1e-6)
    np.random.seed(a.seed)
    common = dict(Nbar=100, theta=1e-4, retrain_every_n_points=25, splitting_strategy='gradual',
                  use_calibrated_sigma=True)
    if a.config == 'tree':
        gpt = GPTree(GPR=Default_GPR(n_restarts_optimizer=1), **common)
    elif a.config == 'neural':
        gpt = GPTree(GPR=NeuralLinearGPR(FeatureNetLearner(steps_per_update=8, random_state=a.seed)), **common)
    elif a.config == 'hybrid':
        from pygptreeo import NetGlobalMean
        gpt = GPTree(GPR=Default_GPR(n_restarts_optimizer=1),
                     global_mean=NetGlobalMean(steps_per_update=8, random_state=a.seed), **common)
    elif a.config == 'global_gp':
        # the additive-GP global model of BENCHMARK_RESULTS_global_mean_streams.md, GP leaves on its residual
        from pygptreeo import AdditiveGPGlobalMean
        gpt = GPTree(GPR=Default_GPR(n_restarts_optimizer=1),
                     global_mean=AdditiveGPGlobalMean(reservoir_size=500, min_points=200, min_turnover=0.25,
                                                      n_restarts_optimizer=2, random_state=a.seed), **common)
    else:
        raise ValueError(a.config)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{tag(a)}_{a.config}.csv")
    t_start = time.time()
    with open(path, 'w') as f, contextlib.redirect_stdout(io.StringIO()):
        f.write("true_y,predicted_y,prediction_uncertainty,predict_time_s,update_tree_time_s,n_leaves\n")
        for i in range(a.N):
            xi = X[i:i + 1]
            t0 = time.perf_counter(); mu, sd = gpt.predict(xi); t1 = time.perf_counter()
            gpt.update_tree(xi, np.array([[y[i]]]), np.array([[sig[i]]])); t2 = time.perf_counter()
            f.write(f"{y[i]:.10e},{mu[0, 0]:.10e},{sd[0, 0]:.6e},{t1 - t0:.3e},{t2 - t1:.3e},{len(gpt.root.leaves)}\n")
            if (i + 1) % BATCH == 0:
                print(f"{a.config}: {i + 1} points, {time.time() - t_start:.0f} s", file=sys.__stdout__, flush=True)
    print(f"wrote {path} in {time.time() - t_start:.0f} s")


def batch_metrics(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    n = (len(df) // BATCH) * BATCH
    df = df.iloc[:n]
    g = np.arange(n) // BATCH
    out = {'points': (np.arange(n // BATCH) + 1) * BATCH}
    err = (df['predicted_y'] - df['true_y']).values
    rng_y = df['true_y'].groupby(g).transform(lambda s: s.max() - s.min()).values
    out['predict_time'] = df['predict_time_s'].groupby(g).mean().values
    out['update_time'] = df['update_tree_time_s'].groupby(g).mean().values
    out['update_time_max'] = df['update_tree_time_s'].groupby(g).max().values
    out['nrmse'] = np.array([np.sqrt(np.mean(err[b * BATCH:(b + 1) * BATCH] ** 2)) / rng_y[b * BATCH]
                             for b in range(n // BATCH)])
    rel = np.abs(err) / np.maximum(np.abs(df['true_y'].values), 1e-10)
    for p in TOLERANCES:
        out[f'within_{p}'] = (rel <= p / 100.0).reshape(-1, BATCH).mean(1)
    out['coverage'] = (np.abs(err) <= df['prediction_uncertainty'].values).reshape(-1, BATCH).mean(1)
    out['leaves'] = df['n_leaves'].values.reshape(-1, BATCH)[:, -1]
    return out


def plot(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs = {}
    for cfg in ('tree', 'global_gp', 'neural', 'hybrid'):
        path = os.path.join(RESULTS_DIR, f"{tag(a)}_{cfg}.csv")
        if os.path.exists(path):
            runs[cfg] = batch_metrics(path)
    if not runs:
        sys.exit(f"no CSV files for {tag(a)} in {RESULTS_DIR}")
    style = {'tree': dict(color='tab:blue', ls='-'), 'neural': dict(color='tab:red', ls='-'),
             'hybrid': dict(color='tab:green', ls='-'), 'global_gp': dict(color='tab:orange', ls='-')}
    name = {'tree': 'GP tree', 'neural': 'neural-linear tree', 'hybrid': 'hybrid (GP leaves on network residual)',
            'global_gp': 'GP tree + additive global GP'}
    fig, axs = plt.subplots(6, 1, figsize=(15, 19), sharex=True)
    fig.suptitle(f"PyGPTreeo performance metrics: GP tree vs neural-linear tree\n"
                 f"{a.target}, d = {a.d}, {a.stream} stream, {a.N} points, metrics per batch of {BATCH}", fontsize=16)
    for cfg, m in runs.items():
        s = style[cfg]; lab = name[cfg]
        axs[0].plot(m['points'], m['predict_time'], label=f"{lab}: avg. predict time", linewidth=2.0, **s)
        axs[1].plot(m['points'], m['update_time'], label=f"{lab}: avg. update time", linewidth=2.0, **s)
        axs[1].plot(m['points'], m['update_time_max'], label=f"{lab}: max update time in batch", linewidth=1.0, alpha=0.5, **s)
        axs[2].plot(m['points'], m['nrmse'], label=f"{lab}: NRMSE", linewidth=2.0, **s)
        for p, alpha in zip(TOLERANCES, (1.0, 0.85, 0.7, 0.55, 0.4)):
            axs[3].plot(m['points'], m[f'within_{p}'], label=f"{lab}: fraction < {p}% error", linewidth=2.0,
                        color=s['color'], ls=s['ls'], alpha=alpha)
        axs[4].plot(m['points'], m['coverage'], label=f"{lab}: empirical coverage", linewidth=2.0, **s)
        axs[5].plot(m['points'], m['leaves'], label=f"{lab}: leaves", linewidth=2.0, **s)
    axs[0].set_ylabel('Time (s)'); axs[0].set_title('Average prediction time per point'); axs[0].set_yscale('log')
    axs[1].set_ylabel('Time (s)'); axs[1].set_title('Average (and maximum) tree update time per point'); axs[1].set_yscale('log')
    axs[2].set_ylabel('NRMSE'); axs[2].set_title('NRMSE for predictions'); axs[2].set_yscale('log')
    axs[3].set_ylabel('Fraction'); axs[3].set_title('Fraction of predictions within x% of true value'); axs[3].set_ylim([0, 1])
    axs[4].set_ylabel('Fraction'); axs[4].set_title('Empirical coverage of prediction uncertainty'); axs[4].set_ylim([0, 1])
    axs[4].axhline(0.68, ls='--', color='black', linewidth=2.0)
    axs[5].set_ylabel('Leaves'); axs[5].set_title('Number of leaves'); axs[5].set_xlabel('Total points processed')
    for ax in axs:
        ax.set_xlim([0, a.N]); ax.grid(True); ax.legend(loc='upper left', ncol=4, fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(RESULTS_DIR, f"{tag(a)}_compare.png")
    plt.savefig(out, dpi=150)
    print(f"figure saved to {out}")
    # the per-batch metrics of both configurations, the record kept in the repository
    import pandas as pd
    frames = []
    for cfg, m in runs.items():
        df = pd.DataFrame({k: v for k, v in m.items()}); df.insert(0, 'config', cfg); frames.append(df)
    pd.concat(frames).to_csv(os.path.join(RESULTS_DIR, f"{tag(a)}_batches.csv"), index=False, float_format='%.6g')
    # a compact summary of the last batch
    for cfg, m in runs.items():
        print(f"{cfg:7s} last batch: predict {1e3 * m['predict_time'][-1]:.2f} ms, update {1e3 * m['update_time'][-1]:.2f} ms "
              f"(max {1e3 * m['update_time_max'][-1]:.0f} ms), NRMSE {m['nrmse'][-1]:.4f}, within 1% {m['within_1'][-1]:.2f}, "
              f"coverage {m['coverage'][-1]:.2f}, leaves {m['leaves'][-1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='eggholder'); ap.add_argument('--d', type=int, default=3)
    ap.add_argument('--stream', default='uniform'); ap.add_argument('--N', type=int, default=40000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--config', default='tree', help='tree or neural')
    ap.add_argument('--plot', action='store_true', help='draw the figure from the CSV files of both configurations')
    a = ap.parse_args()
    if a.plot:
        plot(a)
    else:
        run(a)


if __name__ == '__main__':
    main()
