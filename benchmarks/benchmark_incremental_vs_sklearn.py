"""Long streaming comparison: IncrementalGP vs the default scikit-learn GP.

Presented in the pygptreeo house style (cf. examples/performance_test.py and
examples/plot_performance_metrics.py): a prequential (progressive-validation)
stream where each point is predicted before being learned, with metrics
summarised in 2000-point batches and shown as a 5-panel figure:

    1. average prediction time per point
    2. average tree-update time per point
    3. NRMSE (RMSE normalised by the batch value range), log scale
    4. fraction of predictions within {1,2,4,8,16}% of the true value
    5. empirical coverage of the 1-sigma uncertainty (dashed 0.68 reference)

One 5-panel figure is produced per Nbar, overlaying the two backends
(sklearn = solid, incremental = dashed). Per-run CSVs are also written in the
same column format as examples/plot_performance_metrics.py.

Both backends use the "lazy" config: full re-optimization only every Nbar
points (retrain_every_n_points = Nbar). The IncrementalGP additionally
incorporates each new point between refits via an exact rank-1 update.

Usage:
    OMP_NUM_THREADS=1 python benchmarks/benchmark_incremental_vs_sklearn.py
"""

import contextlib
import csv
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
THETA = 1e-4
BANDS = [1, 2, 4, 8, 16]  # percent-error accuracy bands


def target_function(X):
    x0, x1 = X[:, 0], X[:, 1]
    return (np.sin(4.0 * np.pi * x0)
            + 0.5 * np.sin(2.0 * np.pi * x1)
            + 0.3 * x0 * x1)


def make_stream(n, seed):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 2)
    y = target_function(X)                     # noiseless targets (as in performance_test.py)
    sigma = np.maximum(1e-3 * np.abs(y), 1e-6)  # small per-point uncertainty
    return X, y, sigma


def make_backend(kind):
    if kind == "sklearn":
        return Default_GPR()
    if kind == "incremental":
        return IncrementalGP()
    raise ValueError(kind)


def run_stream(kind, Nbar, X, y, sigma):
    """Prequential run. Returns per-point arrays: actual, pred, std, t_pred, t_upd."""
    np.random.seed(SEED)  # identical routing RNG across backends at a given Nbar
    gpt = GPTree(GPR=make_backend(kind), Nbar=Nbar, theta=THETA,
                 retrain_every_n_points=Nbar,                 # "lazy"
                 incremental_updates=(kind == "incremental"))
    n = X.shape[0]
    actual = np.empty(n); pred = np.empty(n); std = np.empty(n)
    t_pred = np.empty(n); t_upd = np.empty(n)

    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(n):
            xi = X[i:i + 1]
            t0 = time.perf_counter()
            mu, sd = gpt.predict(xi)
            t_pred[i] = time.perf_counter() - t0

            t0 = time.perf_counter()
            gpt.update_tree(xi, np.array([[y[i]]]), np.array([[sigma[i]]]))
            t_upd[i] = time.perf_counter() - t0

            actual[i] = y[i]; pred[i] = mu[0, 0]; std[i] = sd[0, 0]
    return {"actual": actual, "pred": pred, "std": std,
            "t_pred": t_pred, "t_upd": t_upd}


def batch_metrics(run):
    """Compute per-2000-point-batch metrics, matching performance_test.py."""
    n = len(run["actual"])
    nb = n // BATCH
    x_axis, predt, updt, nrmse, cov = [], [], [], [], []
    frac = {b: [] for b in BANDS}
    for k in range(nb):
        s = slice(k * BATCH, (k + 1) * BATCH)
        a = run["actual"][s]; p = run["pred"][s]; sd = run["std"][s]
        rng = np.max(a) - np.min(a)
        rng = rng if rng > 0 else 1.0
        x_axis.append((k + 1) * BATCH)
        predt.append(np.mean(run["t_pred"][s]))
        updt.append(np.mean(run["t_upd"][s]))
        nrmse.append(np.sqrt(np.mean((a - p) ** 2)) / rng)
        cov.append(np.mean(np.abs(a - p) <= sd))
        denom = np.maximum(np.abs(a), 1e-10)
        rel = np.abs(p - a) / denom
        for b in BANDS:
            frac[b].append(np.mean(rel <= 0.01 * b))
    return {"x": np.array(x_axis), "predt": np.array(predt),
            "updt": np.array(updt), "nrmse": np.array(nrmse),
            "cov": np.array(cov), "frac": {b: np.array(frac[b]) for b in BANDS}}


def write_csv(path, run):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_y", "predicted_y", "prediction_uncertainty",
                    "predict_time_s", "update_tree_time_s"])
        for i in range(len(run["actual"])):
            w.writerow([run["actual"][i], run["pred"][i], run["std"][i],
                        run["t_pred"][i], run["t_upd"][i]])


def main():
    X, y, sigma = make_stream(N_STREAM, SEED)
    runs, metrics = {}, {}
    print(f"Prequential stream: N={N_STREAM}, batch={BATCH}, "
          f"lazy (retrain_every=Nbar)\n")
    for Nbar in NBARS:
        for kind in ["sklearn", "incremental"]:
            t0 = time.perf_counter()
            runs[(Nbar, kind)] = run_stream(kind, Nbar, X, y, sigma)
            dt = time.perf_counter() - t0
            metrics[(Nbar, kind)] = batch_metrics(runs[(Nbar, kind)])
            write_csv(f"benchmarks/stream_{kind}_Nbar{Nbar}.csv", runs[(Nbar, kind)])
            m = metrics[(Nbar, kind)]
            print(f"  Nbar={Nbar:<4} {kind:<12} {dt:6.1f}s   "
                  f"final-batch NRMSE={m['nrmse'][-1]:.4f}  coverage={m['cov'][-1]:.2f}")

    # ----- Plots: one 5-panel figure per Nbar, sklearn (solid) vs incremental (dashed) -----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    style = {"sklearn": "-", "incremental": "--"}
    band_colors = plt.cm.viridis(np.linspace(0, 0.9, len(BANDS)))

    for Nbar in NBARS:
        fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)
        fig.suptitle(f"IncrementalGP vs scikit-learn GP — Nbar={Nbar} "
                     f"(lazy: retrain every {Nbar})", fontsize=16)
        for kind in ["sklearn", "incremental"]:
            m = metrics[(Nbar, kind)]
            ls = style[kind]
            axs[0].plot(m["x"], m["predt"], ls, linewidth=2.0, label=kind)
            axs[1].plot(m["x"], m["updt"], ls, linewidth=2.0, color="orange"
                        if kind == "sklearn" else "red", label=kind)
            axs[2].plot(m["x"], m["nrmse"], ls, linewidth=2.0, color="green"
                        if kind == "sklearn" else "darkgreen", label=kind)
            for b, c in zip(BANDS, band_colors):
                axs[3].plot(m["x"], m["frac"][b], ls, linewidth=2.0, color=c)
            axs[4].plot(m["x"], m["cov"], ls, linewidth=2.0, color="purple"
                        if kind == "sklearn" else "magenta", label=kind)

        axs[0].set_ylabel("Time (s)"); axs[0].set_title("Average prediction time per point"); axs[0].legend()
        axs[1].set_ylabel("Time (s)"); axs[1].set_title("Average tree update time per point"); axs[1].legend()
        axs[2].set_ylabel("NRMSE"); axs[2].set_title("NRMSE for predictions")
        axs[2].set_yscale("log"); axs[2].legend()
        axs[3].set_ylabel("Fraction"); axs[3].set_title("Fraction of predictions within x% of true value")
        axs[3].set_ylim([0, 1])
        band_handles = [Line2D([0], [0], color=c, lw=2, label=f"< {b}%")
                        for b, c in zip(BANDS, band_colors)]
        style_handles = [Line2D([0], [0], color="black", lw=2, ls="-", label="sklearn"),
                         Line2D([0], [0], color="black", lw=2, ls="--", label="incremental")]
        axs[3].legend(handles=band_handles + style_handles, ncol=2, fontsize=9)
        axs[4].set_ylabel("Fraction"); axs[4].set_title("Empirical coverage of prediction uncertainty")
        axs[4].plot([metrics[(Nbar, 'sklearn')]["x"][0], metrics[(Nbar, 'sklearn')]["x"][-1]],
                    [0.68, 0.68], "k--", linewidth=1.5)
        axs[4].set_ylim([0, 1]); axs[4].set_xlabel("Total points processed"); axs[4].legend()

        for ax in axs:
            ax.grid(True)
            ax.set_xlim([0, N_STREAM])
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out = f"benchmarks/incremental_vs_sklearn_Nbar{Nbar}.png"
        plt.savefig(out, dpi=120)
        print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
