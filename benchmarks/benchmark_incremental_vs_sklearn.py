"""Long streaming comparison: IncrementalGP vs the default scikit-learn GP.

This runs a *prequential* (progressive-validation) evaluation: every streamed
point is first predicted (a genuine held-out test) and then learned. The stream
therefore provides 20000 test points. Prediction performance is summarised in
batches of 2000 points.

Both backends use the "lazy" retraining config: full hyperparameter
re-optimization only every Nbar points (retrain_every_n_points = Nbar). The
difference is that the IncrementalGP also incorporates each new point between
refits via an exact rank-1 Cholesky update (and re-fits each child on its own
data at a split so those updates can run).

For a fair comparison standard scaling is disabled and both backends use the
same observation-noise variance (the default sklearn adapter ignores the
per-point noise the tree computes, so we set it explicitly via Default_GPR's
alpha; without scaling that value is correct in the original units).

Usage:
    OMP_NUM_THREADS=1 python benchmarks/benchmark_incremental_vs_sklearn.py
"""

import contextlib
import io
import time
import warnings

import numpy as np

from pygptreeo import GPTree, Default_GPR, IncrementalGP

warnings.filterwarnings("ignore")

SEED = 0
N_STREAM = 20000
BATCH = 2000
NBARS = [100, 500]
NOISE = 0.05            # observation noise std dev
THETA = 1e-4


def target_function(X):
    x0, x1 = X[:, 0], X[:, 1]
    return (np.sin(4.0 * np.pi * x0)
            + 0.5 * np.sin(2.0 * np.pi * x1)
            + 0.3 * x0 * x1)


def make_stream(n, seed):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 2)
    y = target_function(X) + rng.normal(0.0, NOISE, n)
    return X, y


def make_backend(kind):
    if kind == "sklearn":
        # Fixed observation-noise variance matching the data (scaling disabled).
        return Default_GPR(alpha=NOISE ** 2)
    elif kind == "incremental":
        return IncrementalGP()
    raise ValueError(kind)


def run_stream(kind, Nbar, X, y):
    """Prequential run: predict each point, then learn it.

    Returns (sq_errors, nll_per_point, total_time).
    """
    np.random.seed(SEED)  # identical routing RNG across backends at a given Nbar
    gpt = GPTree(GPR=make_backend(kind), Nbar=Nbar, theta=THETA,
                 retrain_every_n_points=Nbar,        # "lazy"
                 incremental_updates=(kind == "incremental"),
                 use_standard_scaling=False)

    sig = NOISE
    n = X.shape[0]
    sq_err = np.empty(n)
    nll = np.empty(n)

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(n):
            xi = X[i:i + 1]
            yi = y[i]
            # Predict before learning (held-out).
            mu, sd = gpt.predict(xi)
            mu = float(mu[0, 0]); sd = max(float(sd[0, 0]), 1e-9)
            sq_err[i] = (mu - yi) ** 2
            nll[i] = 0.5 * np.log(2.0 * np.pi * sd ** 2) + (yi - mu) ** 2 / (2.0 * sd ** 2)
            # Learn it.
            gpt.update_tree(xi, np.array([[yi]]), np.array([[sig]]))
    dt = time.perf_counter() - t0
    return sq_err, nll, dt


def batched(arr, batch, reduce_rmse=False):
    m = (len(arr) // batch) * batch
    blocks = arr[:m].reshape(-1, batch)
    if reduce_rmse:
        return np.sqrt(blocks.mean(axis=1))
    return blocks.mean(axis=1)


def main():
    X, y = make_stream(N_STREAM, SEED)
    n_batches = N_STREAM // BATCH
    x_axis = (np.arange(n_batches) + 1) * BATCH  # points processed at batch end

    results = {}
    print(f"Prequential stream: N={N_STREAM}, batch={BATCH}, noise={NOISE}, "
          f"lazy (retrain_every=Nbar), scaling off\n")
    for Nbar in NBARS:
        for kind in ["sklearn", "incremental"]:
            sq, nll, dt = run_stream(kind, Nbar, X, y)
            results[(Nbar, kind)] = {
                "rmse": batched(sq, BATCH, reduce_rmse=True),
                "nlpd": batched(nll, BATCH),
                "time": dt,
                "rmse_overall": float(np.sqrt(sq.mean())),
            }
            print(f"  Nbar={Nbar:<4} {kind:<12} done in {dt:6.1f}s   "
                  f"overall RMSE={results[(Nbar, kind)]['rmse_overall']:.4f}")

    # Summary table
    print("\nFinal-batch (last 2000 points) RMSE and total runtime:")
    print("  {:<6} {:<12} {:>10} {:>12}".format("Nbar", "backend", "RMSE", "time[s]"))
    for Nbar in NBARS:
        for kind in ["sklearn", "incremental"]:
            r = results[(Nbar, kind)]
            print("  {:<6} {:<12} {:>10.4f} {:>12.1f}".format(
                Nbar, kind, r["rmse"][-1], r["time"]))

    # ----- Plots -----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"sklearn": "C1", "incremental": "C0"}
    markers = {"sklearn": "o-", "incremental": "s-"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, Nbar in enumerate(NBARS):
        for kind in ["sklearn", "incremental"]:
            r = results[(Nbar, kind)]
            lbl = f"{kind} ({r['time']:.0f}s)"
            axes[0, col].plot(x_axis, r["rmse"], markers[kind],
                              color=colors[kind], label=lbl)
            axes[1, col].plot(x_axis, r["nlpd"], markers[kind],
                              color=colors[kind], label=lbl)
        axes[0, col].set_title(f"Nbar = {Nbar}   (lazy: retrain every {Nbar})")
        axes[0, col].set_ylabel("Prequential RMSE (per 2000-pt batch)")
        axes[1, col].set_ylabel("Prequential NLPD (per 2000-pt batch)")
        for row in (0, 1):
            axes[row, col].set_xlabel("points processed")
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend()
        axes[0, col].set_yscale("log")
    fig.suptitle("IncrementalGP vs default scikit-learn GP — streaming "
                 f"prequential performance (N={N_STREAM}, noise={NOISE})")
    fig.tight_layout()
    fig.savefig("benchmarks/incremental_vs_sklearn.png", dpi=120)
    print("\nSaved plot to benchmarks/incremental_vs_sklearn.png")

    # Runtime bar chart
    fig2, ax = plt.subplots(figsize=(7, 4.2))
    labels = [f"Nbar={Nbar}\n{kind}" for Nbar in NBARS for kind in ["sklearn", "incremental"]]
    times = [results[(Nbar, kind)]["time"] for Nbar in NBARS for kind in ["sklearn", "incremental"]]
    bar_colors = [colors[kind] for Nbar in NBARS for kind in ["sklearn", "incremental"]]
    ax.bar(labels, times, color=bar_colors)
    ax.set_ylabel("total streaming time [s]")
    ax.set_title(f"Runtime over {N_STREAM}-point stream")
    for i, t in enumerate(times):
        ax.text(i, t, f"{t:.0f}s", ha="center", va="bottom")
    fig2.tight_layout()
    fig2.savefig("benchmarks/incremental_vs_sklearn_runtime.png", dpi=120)
    print("Saved plot to benchmarks/incremental_vs_sklearn_runtime.png")


if __name__ == "__main__":
    main()
