"""Paired trusted-error diagnostic.

Runs pygptreeo_A and sklearn_gp_A on an identical stream and records
per-step (mu, sigma, |mu - f|) for both methods at a single tau_sigma.
Points where both methods trust the prediction go on a scatter plot:
x = |mu_pygp - f|, y = |mu_sklearn - f|. Below diagonal = pygp more
accurate on that step; above = sklearn more accurate.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/user/pygptreeo")

from benchmarks.problems import PROBLEMS
from benchmarks.run_all import _make_pygptreeo_A, _make_sklearn_gp_A


def _run(method_factory, problem, seed, n_stream, tau_sigma_rel):
    rng_stream = np.random.default_rng(seed)
    rng_test = np.random.default_rng(seed + 10_000)
    X_stream, y_stream = problem.sample_schedule(
        n_stream, rng_stream, schedule="mcmc",
    )
    X_test, y_test = problem.sample(1000, rng_test)
    y_range = float(np.ptp(y_test))
    tau_abs = tau_sigma_rel * y_range

    method = method_factory(problem.dim)
    trusted_err = np.full(n_stream, np.nan)
    sigma_rec = np.full(n_stream, np.nan)
    trusted_flag = np.zeros(n_stream, dtype=bool)

    for i in range(n_stream):
        x = X_stream[i:i+1]
        y_true = float(y_stream[i])
        mean, std = method.predict(x)
        mu = float(np.asarray(mean).ravel()[0])
        sigma = float(np.asarray(std).ravel()[0])
        sigma_rec[i] = sigma
        if np.isfinite(sigma) and sigma <= tau_abs:
            trusted_flag[i] = True
            trusted_err[i] = abs(mu - y_true)
        else:
            method.update(x, np.array([[y_true]]))
    method.close()
    return {"trusted_flag": trusted_flag,
            "trusted_err": trusted_err,
            "sigma": sigma_rec}


print("Running pygptreeo_A...")
r_pygp = _run(_make_pygptreeo_A, PROBLEMS["rosenbrock_2d"],
              seed=0, n_stream=3000, tau_sigma_rel=1e-2)
print(f"  pygp: {int(r_pygp['trusted_flag'].sum())} trusted / 3000")
print("Running sklearn_gp_A...")
r_skl = _run(_make_sklearn_gp_A, PROBLEMS["rosenbrock_2d"],
             seed=0, n_stream=3000, tau_sigma_rel=1e-2)
print(f"  skl: {int(r_skl['trusted_flag'].sum())} trusted / 3000")

# Union: both methods trusted the same proposal?
both = r_pygp["trusted_flag"] & r_skl["trusted_flag"]
print(f"  both trusted: {int(both.sum())} steps")

if int(both.sum()) < 5:
    print("Too few paired steps; widening tau or re-running at tau=3e-2")
    r_pygp = _run(_make_pygptreeo_A, PROBLEMS["rosenbrock_2d"],
                  seed=0, n_stream=3000, tau_sigma_rel=3e-2)
    r_skl = _run(_make_sklearn_gp_A, PROBLEMS["rosenbrock_2d"],
                 seed=0, n_stream=3000, tau_sigma_rel=3e-2)
    both = r_pygp["trusted_flag"] & r_skl["trusted_flag"]
    print(f"  at tau=3e-2: both trusted = {int(both.sum())}")

# Scatter.
idx = np.where(both)[0]
x_err = r_pygp["trusted_err"][idx]
y_err = r_skl["trusted_err"][idx]

fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.scatter(x_err, y_err, s=10, alpha=0.5, color="#d7263d",
           edgecolor="black", linewidth=0.2)
mx = float(np.nanmax(np.concatenate([x_err, y_err])))
mn = max(1e-6, float(np.nanmin(np.concatenate([x_err, y_err]))))
ax.plot([mn, mx], [mn, mx], "k--", linewidth=1, alpha=0.7)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"|$\mu_\mathrm{pygp} - f$| on paired trusted steps")
ax.set_ylabel(r"|$\mu_\mathrm{sklearn} - f$| on paired trusted steps")
ax.set_title(
    "Paired trusted-prediction error — rosenbrock_2d MCMC\n"
    f"({int(both.sum())} paired steps; below diagonal = pygptreeo more accurate)"
)
ax.grid(True, which="both", alpha=0.3)
pct_below = float(np.mean(x_err < y_err))
ax.text(0.05, 0.95, f"{100*pct_below:.0f} % of paired steps below diagonal",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"))
out = "/home/user/pygptreeo/benchmarks/iterations/iteration_17/plots/paired_trusted_err.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")
print(f"fraction of paired steps where pygp error < sklearn error: {pct_below:.3f}")
print(f"median pygp err: {np.nanmedian(x_err):.3e}")
print(f"median sklearn err: {np.nanmedian(y_err):.3e}")
