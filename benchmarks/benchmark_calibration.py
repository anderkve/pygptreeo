"""Validate the closed-form sigma calibration ('quantile') against the legacy
root-finding calibration ('rootfind').

The two methods only differ in how the per-leaf uncertainty scaler is computed;
the predicted means (and hence NRMSE / accuracy bands) are identical, so the key
comparison is the empirical coverage of the predicted 1-sigma uncertainty, which
should track the 0.68 target equally well for both.

Presented in the pygptreeo house style (cf. examples/performance_test.py): a
prequential stream where each point is predicted before being learned, metrics
summarised in 2000-point batches, shown as a 5-panel figure:

    1. average prediction time per point
    2. average tree-update time per point   <- includes the calibration cost
    3. NRMSE (RMSE normalised by batch value range), log scale
    4. fraction of predictions within {1,2,4,8,16}% of the true value
    5. empirical coverage of the 1-sigma uncertainty (dashed 0.68 reference)

One figure per Nbar overlays the two calibration methods
(rootfind = solid, quantile = dashed). Per-run CSVs are also written.

Usage:
    OMP_NUM_THREADS=1 python benchmarks/benchmark_calibration.py
"""

import contextlib
import csv
import io
import time
import warnings

import numpy as np

from pygptreeo import GPTree, Default_GPR

warnings.filterwarnings("ignore")

SEED = 0
N_STREAM = 20000
BATCH = 2000
NBARS = [100, 500]
THETA = 1e-4
BANDS = [1, 2, 4, 8, 16]
METHODS = ["rootfind", "quantile"]   # legacy vs new closed-form


def target_function(X):
    x0, x1 = X[:, 0], X[:, 1]
    return (np.sin(4.0 * np.pi * x0)
            + 0.5 * np.sin(2.0 * np.pi * x1)
            + 0.3 * x0 * x1)


def make_stream(n, seed):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 2)
    y = target_function(X)
    sigma = np.maximum(1e-3 * np.abs(y), 1e-6)
    return X, y, sigma


def run_stream(method, Nbar, X, y, sigma):
    np.random.seed(SEED)  # identical routing/fits across methods at a given Nbar
    gpt = GPTree(GPR=Default_GPR(), Nbar=Nbar, theta=THETA,
                 retrain_every_n_points=Nbar,
                 calibration_method=method)
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
    n = len(run["actual"]); nb = n // BATCH
    x_axis, predt, updt, nrmse, cov = [], [], [], [], []
    frac = {b: [] for b in BANDS}
    for k in range(nb):
        s = slice(k * BATCH, (k + 1) * BATCH)
        a = run["actual"][s]; p = run["pred"][s]; sd = run["std"][s]
        rng = np.max(a) - np.min(a); rng = rng if rng > 0 else 1.0
        x_axis.append((k + 1) * BATCH)
        predt.append(np.mean(run["t_pred"][s]))
        updt.append(np.mean(run["t_upd"][s]))
        nrmse.append(np.sqrt(np.mean((a - p) ** 2)) / rng)
        cov.append(np.mean(np.abs(a - p) <= sd))
        rel = np.abs(p - a) / np.maximum(np.abs(a), 1e-10)
        for b in BANDS:
            frac[b].append(np.mean(rel <= 0.01 * b))
    return {"x": np.array(x_axis), "predt": np.array(predt), "updt": np.array(updt),
            "nrmse": np.array(nrmse), "cov": np.array(cov),
            "frac": {b: np.array(frac[b]) for b in BANDS}}


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
    metrics = {}
    print(f"Prequential stream: N={N_STREAM}, batch={BATCH}, "
          f"comparing calibration methods {METHODS}\n")
    for Nbar in NBARS:
        for method in METHODS:
            t0 = time.perf_counter()
            run = run_stream(method, Nbar, X, y, sigma)
            dt = time.perf_counter() - t0
            metrics[(Nbar, method)] = batch_metrics(run)
            write_csv(f"benchmarks/calib_{method}_Nbar{Nbar}.csv", run)
            m = metrics[(Nbar, method)]
            print(f"  Nbar={Nbar:<4} {method:<10} {dt:6.1f}s   "
                  f"mean coverage={np.mean(m['cov']):.3f}  "
                  f"final-batch coverage={m['cov'][-1]:.3f}  "
                  f"final NRMSE={m['nrmse'][-1]:.4f}")

    # Coverage agreement summary
    print("\nCoverage agreement (|quantile - rootfind| per batch):")
    for Nbar in NBARS:
        d = np.abs(metrics[(Nbar, "quantile")]["cov"] - metrics[(Nbar, "rootfind")]["cov"])
        print(f"  Nbar={Nbar:<4} mean |Δcoverage|={d.mean():.4f}  max |Δcoverage|={d.max():.4f}")

    # ----- Plots -----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    style = {"rootfind": "-", "quantile": "--"}
    band_colors = plt.cm.viridis(np.linspace(0, 0.9, len(BANDS)))

    for Nbar in NBARS:
        fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)
        fig.suptitle(f"Calibration: closed-form 'quantile' vs legacy 'rootfind' "
                     f"— Nbar={Nbar}", fontsize=16)
        for method in METHODS:
            m = metrics[(Nbar, method)]; ls = style[method]
            axs[0].plot(m["x"], m["predt"], ls, lw=2.0, label=method)
            axs[1].plot(m["x"], m["updt"], ls, lw=2.0,
                        color="orange" if method == "rootfind" else "red", label=method)
            axs[2].plot(m["x"], m["nrmse"], ls, lw=2.0,
                        color="green" if method == "rootfind" else "darkgreen", label=method)
            for b, c in zip(BANDS, band_colors):
                axs[3].plot(m["x"], m["frac"][b], ls, lw=2.0, color=c)
            axs[4].plot(m["x"], m["cov"], ls, lw=2.0,
                        color="purple" if method == "rootfind" else "magenta", label=method)

        axs[0].set_ylabel("Time (s)"); axs[0].set_title("Average prediction time per point"); axs[0].legend()
        axs[1].set_ylabel("Time (s)"); axs[1].set_title("Average tree update time per point (incl. calibration)"); axs[1].legend()
        axs[2].set_ylabel("NRMSE"); axs[2].set_title("NRMSE for predictions"); axs[2].set_yscale("log"); axs[2].legend()
        axs[3].set_ylabel("Fraction"); axs[3].set_title("Fraction of predictions within x% of true value"); axs[3].set_ylim([0, 1])
        band_handles = [Line2D([0], [0], color=c, lw=2, label=f"< {b}%") for b, c in zip(BANDS, band_colors)]
        style_handles = [Line2D([0], [0], color="black", lw=2, ls="-", label="rootfind"),
                         Line2D([0], [0], color="black", lw=2, ls="--", label="quantile")]
        axs[3].legend(handles=band_handles + style_handles, ncol=2, fontsize=9)
        axs[4].set_ylabel("Fraction"); axs[4].set_title("Empirical coverage of prediction uncertainty")
        axs[4].plot([metrics[(Nbar, 'rootfind')]["x"][0], metrics[(Nbar, 'rootfind')]["x"][-1]],
                    [0.68, 0.68], "k--", lw=1.5, label="0.68 target")
        axs[4].set_ylim([0, 1]); axs[4].set_xlabel("Total points processed"); axs[4].legend()
        for ax in axs:
            ax.grid(True); ax.set_xlim([0, N_STREAM])
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out = f"benchmarks/calibration_Nbar{Nbar}.png"
        plt.savefig(out, dpi=120)
        print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
