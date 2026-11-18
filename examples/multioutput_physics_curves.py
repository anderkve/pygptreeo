"""
Curves f(x; theta) as multi-output targets: the output models compared

Learns one of the curve families of ``target_functions_curves.py`` from a stream of
curves: each curve is predicted from its parameters theta before the tree is updated
with it. The curve on its grid in x is the multi-output target. For each output model
the script prints the error and the coverage of the predictions along the stream, the
error on a set of test curves and the time per update. With matplotlib it also draws
three test curves and the errors of the last two models.

The families differ in what makes them hard: 'matter_power' and 'dust_sed' are smooth,
'z_lineshape', 'zprime_spectrum' and 'kinematic_edge' have a peak or an edge that moves
with the parameters, 'cmb_peaks' and 'ringdown' oscillate with a phase that depends on
them, and 'peak_counts_mc' has Monte Carlo noise.

Usage:
    python examples/multioutput_physics_curves.py [family] [n_train] [models] [global_mean]

    family       a key of FAMILIES (default 'z_lineshape')
    n_train      number of curves in the stream (default 400)
    models       comma-separated output models (default 'pca,shared,aligned';
                 'independent' is also possible, and slow)
    global_mean  'additive_gp' or 'net' gives every tree that global model
                 (default: none)
"""

import sys
import time
import warnings

import numpy as np

from pygptreeo import GPTree, Default_GPR
from target_functions_curves import FAMILIES

warnings.filterwarnings("ignore")
np.random.seed(0)

NU = 2.5       # smoothness of the Matern leaf kernel: these curves are smooth in theta, and 2.5 does better than the default 1.5

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOURS = ["#2a78d6", "#eb6834"]


def learn(family, model, U, Y, S, truth, U_test, global_mean=None):
    """Stream the curves through a tree. Returns the predictions along the stream,
    those for the test set, and the time per update in ms."""
    gpt = GPTree(GPR=Default_GPR(nu=NU), Nbar=100, retrain_every_n_points=25, n_outputs=family.n_outputs, output_model=model,
                 global_mean=global_mean)
    N = len(U)
    mean, sigma = np.zeros_like(Y), np.zeros_like(Y)
    t_update = 0.0
    for i in range(N):
        if i > 0:
            mean[i], sigma[i] = (a[0] for a in gpt.predict(U[i:i+1]))
        t0 = time.time()
        gpt.update_tree(U[i:i+1], Y[i:i+1], S[i:i+1])
        t_update += time.time() - t0
    test_mean, test_sigma = gpt.predict(U_test, mode='loop')
    half = slice(N // 2, N)
    error = mean[half] - truth[half]
    return dict(model=model, test_mean=test_mean, test_sigma=test_sigma, ms=1e3 * t_update / N,
                rmse=np.sqrt(np.mean(error ** 2)), coverage=np.mean(np.abs(error) < sigma[half]),
                gps=np.mean([len(leaf.my_GPRs) for leaf in gpt.root.leaves]),
                aligned=np.mean([leaf.alignment is not None for leaf in gpt.root.leaves]))


def plot(family, name, results, U_test, truth_test, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.ticker import LogLocator, NullFormatter, FormatStrFormatter

    shown = results[-2:]
    worst = np.argsort(np.sqrt(np.mean((shown[0]["test_mean"] - truth_test) ** 2, axis=1)))
    picks = [worst[len(worst) // 2], worst[int(0.9 * len(worst))], worst[-1]]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, facecolor=SURFACE)
    log_x = family.x[0] > 0 and np.allclose(np.diff(np.log(family.x)), np.diff(np.log(family.x))[0])
    for column, i in enumerate(picks):
        top, bottom = axes[0, column], axes[1, column]
        top.plot(family.x, truth_test[i], color=MUTED, lw=4, alpha=0.35, solid_capstyle="round", label="true curve")
        for r, colour in zip(shown, COLOURS):
            top.plot(family.x, r["test_mean"][i], color=colour, lw=2, label=f"'{r['model']}': prediction (top), error and its sigma (bottom)")
            error = r["test_mean"][i] - truth_test[i]
            bottom.fill_between(family.x, error - r["test_sigma"][i], error + r["test_sigma"][i], color=colour, alpha=0.10, lw=0)
            bottom.plot(family.x, error, color=colour, lw=2)
        bottom.axhline(0.0, color=MUTED, lw=1)
        top.set_title("theta = (" + ", ".join(f"{u:.2f}" for u in U_test[i]) + ")", color=INK, fontsize=10)
        bottom.set_xlabel(family.xlabel, color=MUTED)
        for ax in (top, bottom):
            ax.set_facecolor(SURFACE)
            ax.grid(color=GRID, lw=1)
            ax.tick_params(colors=MUTED, length=0, which="both")
            for spine in ax.spines.values():
                spine.set_visible(False)
            if log_x:
                ax.set_xscale("log")
                ax.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
                ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
                ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0, 0].set_ylabel(f"curve [{family.unit}]", color=MUTED)
    axes[1, 0].set_ylabel(f"prediction - true curve [{family.unit}]", color=MUTED)
    fig.suptitle(f"{name}: test curves with a median, a 90th-percentile and the largest error of '{shown[0]['model']}'",
                 color=INK, fontsize=12)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3,
               frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print(f"\nSaved {out}")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "z_lineshape"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    models = (sys.argv[3] if len(sys.argv) > 3 else "pca,shared,aligned").split(",")
    global_mean = sys.argv[4] if len(sys.argv) > 4 else None
    family = FAMILIES[name]

    rng = np.random.RandomState(0)
    U = rng.rand(n_train, family.d)
    Y, S, truth = family.sample(U, rng)
    U_test = rng.rand(300, family.d)
    truth_test = family(U_test)
    value_range = truth_test.max() - truth_test.min()

    print(f"{name}: {family.d} parameters, {family.n_outputs} grid points in {family.xlabel}, {n_train} curves ({family.kind})"
          + (f", global model '{global_mean}'" if global_mean else "") + "\n")
    print(f"{'output model':<14s} {'GPs/leaf':>8s} {'ms/update':>10s} {'stream NRMSE':>13s} {'coverage':>9s} {'test NRMSE':>11s}")
    results = []
    for model in models:
        r = learn(family, model, U, Y, S, truth, U_test, global_mean)
        test_rmse = np.sqrt(np.mean((r["test_mean"] - truth_test) ** 2))
        note = f"   {100 * r['aligned']:.0f}% of the leaves aligned" if model == "aligned" else ""
        print(f"{model:<14s} {r['gps']:>8.0f} {r['ms']:>10.1f} {r['rmse'] / value_range:>13.2e} {r['coverage']:>9.2f} "
              f"{test_rmse / value_range:>11.2e}{note}")
        results.append(r)

    try:
        plot(family, name, results, U_test, truth_test, f"multioutput_physics_curves_{name}.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
