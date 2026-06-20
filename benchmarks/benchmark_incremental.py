"""Benchmark for incremental rank-1 Cholesky updates (IncrementalGP).

A GPTree leaf normally only re-incorporates new points when its GP is fully
re-fitted, which happens every `retrain_every_n_points` points (an O(n^3)
operation). Between refits the posterior ignores the most recent points. The
`IncrementalGP` backend instead incorporates every new point immediately via an
exact rank-1 Cholesky update (O(n^2)) while still re-optimizing hyperparameters
only periodically.

This script compares three regimes on the *same* backend (so observation-noise
handling and hyperparameter optimization are identical), differing only in when
points enter the GP:

    full (R=1)   : retrain_every_n_points=1, no rank-1 updates
                   -> a full refit at every point. Accuracy gold standard, slow.
    lazy (R=R)   : retrain_every_n_points=R, no rank-1 updates
                   -> recent points ignored until the next full refit. Cheap.
    incremental  : retrain_every_n_points=R, rank-1 updates between refits
                   -> posterior always current; refits only every R points.

Metrics at checkpoints vs N (points ingested): test RMSE, test NLPD, and
cumulative training wall-clock time.

Usage:
    OMP_NUM_THREADS=1 python benchmarks/benchmark_incremental.py
"""

import argparse
import contextlib
import io
import time
import warnings

import numpy as np

from pygptreeo import GPTree, IncrementalGP

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def target_function(X):
    x0, x1 = X[:, 0], X[:, 1]
    return (np.sin(4.0 * np.pi * x0)
            + 0.5 * np.sin(2.0 * np.pi * x1)
            + 0.3 * x0 * x1)


def make_data(n_train, n_test, noise_sigma, seed):
    rng = np.random.RandomState(seed)
    X_train = rng.rand(n_train, 2)
    y_train = target_function(X_train) + rng.normal(0.0, noise_sigma, n_train)
    X_test = rng.rand(n_test, 2)
    y_test = target_function(X_test)
    return X_train, y_train, X_test, y_test


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def rmse(mu, y):
    return float(np.sqrt(np.mean((mu - y) ** 2)))


def nlpd(mu, sd, y):
    sd = np.maximum(sd, 1e-9)
    return float(np.mean(0.5 * np.log(2.0 * np.pi * sd ** 2)
                         + (y - mu) ** 2 / (2.0 * sd ** 2)))


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #
def run_one(incremental, retrain_every, X_train, y_train, X_test, y_test,
            Nbar, theta, noise_sigma, checkpoints, seed):
    np.random.seed(seed)
    gpt = GPTree(GPR=IncrementalGP(), Nbar=Nbar, theta=theta,
                 retrain_every_n_points=retrain_every,
                 incremental_updates=incremental)
    sig = np.full(len(X_train), noise_sigma)

    out = {"rmse": [], "nlpd": [], "time": []}
    cum_time = 0.0
    ckpt_set = set(checkpoints)

    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(len(X_train)):
            t0 = time.perf_counter()
            gpt.update_tree(X_train[i:i + 1], y_train[i:i + 1].reshape(1, 1),
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
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--n-test", type=int, default=400)
    p.add_argument("--Nbar", type=int, default=200)
    p.add_argument("--theta", type=float, default=1e-4)
    p.add_argument("--R", type=int, default=30, help="retrain_every_n_points")
    p.add_argument("--noise", type=float, default=1e-2)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--no-gold", action="store_true",
                   help="skip the expensive full-refit-every-point reference")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    checkpoints = list(range(40, args.n_train + 1, 40))
    if checkpoints[-1] != args.n_train:
        checkpoints.append(args.n_train)

    # (label, incremental, retrain_every)
    configs = [
        (f"lazy (R={args.R})", False, args.R),
        (f"incremental (R={args.R})", True, args.R),
    ]
    if not args.no_gold:
        configs.append(("full (R=1)", False, 1))

    print(f"Benchmark: incremental rank-1 updates | Nbar={args.Nbar} R={args.R} "
          f"noise={args.noise} seeds={args.seeds}")
    print(f"Checkpoints (N): {checkpoints}\n")

    results = {label: [] for label, _, _ in configs}
    for seed in range(args.seeds):
        X_tr, y_tr, X_te, y_te = make_data(args.n_train, args.n_test,
                                           args.noise, seed)
        for label, inc, R in configs:
            results[label].append(
                run_one(inc, R, X_tr, y_tr, X_te, y_te,
                        args.Nbar, args.theta, args.noise, checkpoints, seed))
        print(f"  seed {seed} done")

    def agg(label, key):
        stack = np.vstack([r[key] for r in results[label]])
        return stack.mean(axis=0), stack.std(axis=0) / np.sqrt(stack.shape[0])

    ckpt = np.array(checkpoints)
    labels = [c[0] for c in configs]

    print("\nFinal-N (N={}):".format(ckpt[-1]))
    print("  {:<22} {:>10} {:>10} {:>12}".format("config", "RMSE", "NLPD", "time[s]"))
    for label in labels:
        r, _ = agg(label, "rmse")
        nl, _ = agg(label, "nlpd")
        t, _ = agg(label, "time")
        print("  {:<22} {:>10.4g} {:>10.4g} {:>12.2f}".format(label, r[-1], nl[-1], t[-1]))

    # Headline comparison: incremental vs lazy and vs gold (if present).
    lazy = f"lazy (R={args.R})"
    inc = f"incremental (R={args.R})"
    r_lazy = agg(lazy, "rmse")[0]
    r_inc = agg(inc, "rmse")[0]
    print("\nIncremental vs lazy (mean over all checkpoints):")
    print("  RMSE improvement: {:+.1f}%".format(
        100.0 * np.mean((r_lazy - r_inc) / r_lazy)))
    if not args.no_gold:
        t_inc = agg(inc, "time")[0][-1]
        t_gold = agg("full (R=1)", "time")[0][-1]
        r_gold = agg("full (R=1)", "rmse")[0][-1]
        print("  speedup vs full(R=1): {:.1f}x  "
              "(final RMSE {:.4g} incr vs {:.4g} gold)".format(
                  t_gold / t_inc, r_inc[-1], r_gold))

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            colors = {lazy: "C1", inc: "C0", "full (R=1)": "C2"}
            markers = {lazy: "o-", inc: "s-", "full (R=1)": "^--"}
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
            for ax, key, ylab in zip(
                    axes, ["rmse", "nlpd", "time"],
                    ["Test RMSE", "Test NLPD", "Cumulative train time [s]"]):
                for label in labels:
                    m, s = agg(label, key)
                    ax.plot(ckpt, m, markers.get(label, "o-"),
                            label=label, color=colors.get(label))
                    ax.fill_between(ckpt, m - s, m + s, alpha=0.2,
                                    color=colors.get(label))
                ax.set_xlabel("N points ingested")
                ax.set_ylabel(ylab)
                if key in ("rmse", "nlpd"):
                    ax.set_yscale("log")
                ax.grid(True, alpha=0.3)
                ax.legend()
            fig.suptitle(f"Incremental rank-1 updates (Nbar={args.Nbar}, "
                         f"R={args.R}, {args.seeds} seeds)")
            fig.tight_layout()
            out_png = "benchmarks/incremental.png"
            fig.savefig(out_png, dpi=120)
            print(f"\nSaved plot to {out_png}")
        except Exception as e:
            print(f"\n(Plotting skipped: {e})")


if __name__ == "__main__":
    main()
