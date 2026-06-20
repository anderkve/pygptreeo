"""Benchmark for tree-global hyperparameter pooling (GPTree(pool_hyperparameters=...)).

This script measures the effect of the `pool_hyperparameters` option on online
learning. For a stream of points it trains two otherwise-identical GPTrees --
one with pooling enabled and one without -- and compares them at a sequence of
checkpoints on a fixed held-out test set.

Because node routing and tree growth do not depend on the GP hyperparameters,
using the same random seed gives the two trees an identical structure and an
identical data ordering. The only difference is how each leaf's GP is
hyperparameter-fitted, which isolates the effect of pooling.

Metrics reported as a function of N (number of points ingested):
    - RMSE  : root-mean-square prediction error (lower is better)
    - NLPD  : negative log predictive density, i.e. quality of the predicted
              mean *and* uncertainty (lower is better)
    - time  : cumulative wall-clock training time (lower is better)

Usage:
    python benchmarks/benchmark_hp_pooling.py                 # isotropic kernel
    python benchmarks/benchmark_hp_pooling.py --kernel ard    # anisotropic + noise
"""

import argparse
import contextlib
import io
import time
import warnings

import numpy as np

from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from pygptreeo import GPTree, Default_GPR

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Target function and data
# --------------------------------------------------------------------------- #
def target_function(X, kind="mixed"):
    """A 2D target function on [0, 1]^2.

    kind="mixed": a fast + slow oscillation plus a mild interaction (length
        scales are similar in magnitude; close to spatially homogeneous).
    kind="aniso": strongly anisotropic but spatially homogeneous -- x0 varies
        much faster than x1 everywhere. This is the regime most favourable to
        hyperparameter pooling: every leaf wants the same (anisotropic) length
        scales, but each leaf has too few points to find them on its own.
    """
    x0, x1 = X[:, 0], X[:, 1]
    if kind == "aniso":
        return np.sin(8.0 * np.pi * x0) + np.sin(1.5 * np.pi * x1)
    return (np.sin(4.0 * np.pi * x0)
            + 0.5 * np.sin(2.0 * np.pi * x1)
            + 0.3 * x0 * x1)


def make_data(n_train, n_test, noise_sigma, seed, kind="mixed"):
    rng = np.random.RandomState(seed)
    X_train = rng.rand(n_train, 2)
    y_train = target_function(X_train, kind) + rng.normal(0.0, noise_sigma, n_train)
    X_test = rng.rand(n_test, 2)
    y_test = target_function(X_test, kind)  # noise-free targets for evaluation
    return X_train, y_train, X_test, y_test


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
def make_gpr(kernel_kind, n_restarts=0):
    """Build a fresh Default_GPR with the requested kernel."""
    if kernel_kind == "iso":
        kernel = ConstantKernel() * Matern(length_scale=1.0, nu=1.5)
    elif kernel_kind == "ard":
        # Anisotropic length scales + an explicit noise term -> 4 hyperparameters,
        # which are hard to identify from a young leaf's handful of points.
        kernel = (ConstantKernel()
                  * Matern(length_scale=[1.0, 1.0], nu=1.5)
                  + WhiteKernel(noise_level=1e-3))
    else:
        raise ValueError(f"Unknown kernel kind: {kernel_kind}")
    return Default_GPR(kernel=kernel, n_restarts_optimizer=n_restarts)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def rmse(mu, y):
    return float(np.sqrt(np.mean((mu - y) ** 2)))


def nlpd(mu, sd, y):
    """Mean negative log predictive density under a Gaussian."""
    sd = np.maximum(sd, 1e-9)
    return float(np.mean(0.5 * np.log(2.0 * np.pi * sd ** 2)
                         + (y - mu) ** 2 / (2.0 * sd ** 2)))


# --------------------------------------------------------------------------- #
# One training run
# --------------------------------------------------------------------------- #
def run_one(pool, kernel_kind, X_train, y_train, X_test, y_test,
            Nbar, theta, retrain_every, noise_sigma, checkpoints, seed, n_restarts=0):
    """Train a single GPTree, evaluating at the given checkpoints.

    Returns dict of arrays keyed by 'rmse', 'nlpd', 'time' (one value per
    checkpoint).
    """
    np.random.seed(seed)  # tree growth / routing RNG -> identical across configs
    gpt = GPTree(GPR=make_gpr(kernel_kind, n_restarts), Nbar=Nbar, theta=theta,
                 pool_hyperparameters=pool, retrain_every_n_points=retrain_every)

    sig = np.full(len(X_train), noise_sigma)

    out = {"rmse": [], "nlpd": [], "time": []}
    cum_time = 0.0
    ckpt_set = set(checkpoints)

    with contextlib.redirect_stdout(io.StringIO()):  # silence node prints
        for i in range(len(X_train)):
            t0 = time.perf_counter()
            gpt.update_tree(X_train[i:i + 1],
                            y_train[i:i + 1].reshape(1, 1),
                            sig[i:i + 1].reshape(1, 1))
            cum_time += time.perf_counter() - t0

            n = i + 1
            if n in ckpt_set:
                mu, sd = gpt.predict(X_test)
                out["rmse"].append(rmse(mu[:, 0], y_test))
                out["nlpd"].append(nlpd(mu[:, 0], sd[:, 0], y_test))
                out["time"].append(cum_time)

    return {k: np.array(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", choices=["iso", "ard"], default="iso")
    p.add_argument("--n-train", type=int, default=800)
    p.add_argument("--n-test", type=int, default=400)
    p.add_argument("--Nbar", type=int, default=50)
    p.add_argument("--theta", type=float, default=1e-4)
    p.add_argument("--retrain-every", type=int, default=15)
    p.add_argument("--noise", type=float, default=1e-2)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--restarts", type=int, default=0)
    p.add_argument("--target", choices=["mixed", "aniso"], default="mixed")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    checkpoints = list(range(args.Nbar, args.n_train + 1, max(args.Nbar, 50)))
    if checkpoints[-1] != args.n_train:
        checkpoints.append(args.n_train)

    print(f"Benchmark: hyperparameter pooling | kernel={args.kernel} "
          f"Nbar={args.Nbar} retrain_every={args.retrain_every} "
          f"noise={args.noise} seeds={args.seeds}")
    print(f"Checkpoints (N): {checkpoints}\n")

    results = {True: [], False: []}
    for seed in range(args.seeds):
        X_tr, y_tr, X_te, y_te = make_data(args.n_train, args.n_test,
                                           args.noise, seed, args.target)
        for pool in (False, True):
            res = run_one(pool, args.kernel, X_tr, y_tr, X_te, y_te,
                          args.Nbar, args.theta, args.retrain_every,
                          args.noise, checkpoints, seed, args.restarts)
            results[pool].append(res)
        print(f"  seed {seed} done")

    # Aggregate across seeds -> mean and standard error
    def agg(pool, key):
        stack = np.vstack([r[key] for r in results[pool]])
        mean = stack.mean(axis=0)
        sem = stack.std(axis=0) / np.sqrt(stack.shape[0])
        return mean, sem

    ckpt = np.array(checkpoints)
    print("\n{:>6} | {:>21} | {:>21} | {:>17}".format(
        "N", "RMSE off -> on", "NLPD off -> on", "time[s] off->on"))
    print("-" * 78)
    rmse_off, _ = agg(False, "rmse"); rmse_on, _ = agg(True, "rmse")
    nlpd_off, _ = agg(False, "nlpd"); nlpd_on, _ = agg(True, "nlpd")
    time_off, _ = agg(False, "time"); time_on, _ = agg(True, "time")
    for j, n in enumerate(ckpt):
        print("{:>6} | {:>9.4g} -> {:>8.4g} | {:>9.4g} -> {:>8.4g} | {:>7.2f} -> {:>6.2f}".format(
            n, rmse_off[j], rmse_on[j], nlpd_off[j], nlpd_on[j],
            time_off[j], time_on[j]))

    def pct(off, on):
        return 100.0 * (off - on) / abs(off) if off != 0 else 0.0

    print("\nFinal-N improvement (pooling vs off):")
    print(f"  RMSE : {pct(rmse_off[-1], rmse_on[-1]):+.1f}%  "
          f"({rmse_off[-1]:.4g} -> {rmse_on[-1]:.4g})")
    print(f"  NLPD : {nlpd_off[-1] - nlpd_on[-1]:+.4g} nats  "
          f"({nlpd_off[-1]:.4g} -> {nlpd_on[-1]:.4g})")
    print(f"  time : {pct(time_off[-1], time_on[-1]):+.1f}%  "
          f"({time_off[-1]:.2f}s -> {time_on[-1]:.2f}s)")
    # Sample efficiency: mean improvement averaged over the first half of checkpoints
    half = max(1, len(ckpt) // 2)
    print(f"  early-N mean RMSE improvement (first {half} checkpoints): "
          f"{np.mean([pct(rmse_off[j], rmse_on[j]) for j in range(half)]):+.1f}%")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
            for ax, key, label in zip(
                    axes, ["rmse", "nlpd", "time"],
                    ["Test RMSE", "Test NLPD", "Cumulative train time [s]"]):
                m_off, s_off = agg(False, key)
                m_on, s_on = agg(True, key)
                ax.plot(ckpt, m_off, "o-", label="pooling off", color="C1")
                ax.fill_between(ckpt, m_off - s_off, m_off + s_off,
                                color="C1", alpha=0.2)
                ax.plot(ckpt, m_on, "s-", label="pooling on", color="C0")
                ax.fill_between(ckpt, m_on - s_on, m_on + s_on,
                                color="C0", alpha=0.2)
                ax.set_xlabel("N points ingested")
                ax.set_ylabel(label)
                if key in ("rmse", "nlpd"):
                    ax.set_yscale("log")
                ax.grid(True, alpha=0.3)
                ax.legend()
            fig.suptitle(f"Hyperparameter pooling ({args.kernel} kernel, "
                         f"{args.seeds} seeds, Nbar={args.Nbar})")
            fig.tight_layout()
            out_png = f"benchmarks/hp_pooling_{args.kernel}.png"
            fig.savefig(out_png, dpi=120)
            print(f"\nSaved plot to {out_png}")
        except Exception as e:
            print(f"\n(Plotting skipped: {e})")


if __name__ == "__main__":
    main()
