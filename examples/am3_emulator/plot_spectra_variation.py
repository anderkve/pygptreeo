"""How much the AM3 spectra vary, and how well the tree follows them, in two regimes:
across the whole prior box (an emulator trained on 350 space-filling points, tested on
held-out points) and in the posterior region (the tree saved by the reference DE scan,
tested at points drawn around the best fit).

Usage: python plot_spectra_variation.py lhs_sample.npz ref_scan.npz ref_tree.joblib out.png
"""

import contextlib
import io
import os
import sys

import joblib
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from am3_model import AM3SSCModel, PARAM_NAMES  # noqa: E402
from pygptreeo import GPTree  # noqa: E402

FLOOR = 1e-2
N_SHOW = 25


def main():
    lhs_path, scan_path, tree_path, out = sys.argv[1:5]
    rng = np.random.RandomState(3)
    d = np.load(lhs_path)
    Theta, Y, E = d['Theta'], d['Y'], d['E']
    logY = np.log10(Y + FLOOR)

    # (a) whole box: train on 350, test on the rest
    idx = rng.permutation(Theta.shape[0])
    tr, te = idx[:350], idx[350:]
    tree_box = GPTree(Nbar=100, n_outputs=E.size, output_model='pca', output_basis_components=16, retrain_every_n_points=10)
    with contextlib.redirect_stdout(io.StringIO()):
        tree_box.fit(Theta[tr], logY[tr], np.full_like(logY[tr], 1e-2))
        mu_box, sd_box = tree_box.predict(Theta[te])
    show_box = te[:N_SHOW]
    err_box = (mu_box - logY[te])
    vis_box = Y[te] > 100 * FLOOR

    # (b) posterior region: perturb the reference scan's best point by the spread of its chi2 < 100 points
    s = np.load(scan_path, allow_pickle=True)
    good = np.isfinite(s['chi2_true']) & (s['chi2_true'] < 100)
    centre = s['best_x'][:6]
    width = s['theta'][good][:, :6].std(axis=0)
    tree_post = joblib.load(tree_path)
    model = AM3SSCModel()
    Theta_post = centre + width * rng.randn(N_SHOW, 6)
    Y_post = np.array([model.comoving_sed(t) for t in Theta_post])
    logY_post = np.log10(Y_post + FLOOR)
    with contextlib.redirect_stdout(io.StringIO()):
        mu_post, sd_post = tree_post.predict(Theta_post)
    err_post = mu_post - logY_post
    vis_post = Y_post > 100 * FLOOR

    # data energies in the comoving frame for the reference source (delta = 20, z = 0.1)
    E_data_lo, E_data_hi = 3e-4 * 1.1 / 20, 3e11 * 1.1 / 20

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    for col, (title, Yt, mu, err, vis, nshow) in enumerate([
            (f'whole prior box: {N_SHOW} held-out points, tree trained on 350 space-filling points', Y[show_box], mu_box[:N_SHOW], err_box[:N_SHOW], vis_box[:N_SHOW], N_SHOW),
            (f'posterior region: {N_SHOW} points around the best fit, tree from the 3443-point reference scan', Y_post, mu_post, err_post, vis_post, N_SHOW)]):
        ax, axr = axes[0, col], axes[1, col]
        colors = plt.cm.viridis(np.linspace(0, 1, nshow))
        for i in range(nshow):
            ax.loglog(E, Yt[i], color=colors[i], lw=1.0)
            ax.loglog(E, np.maximum(10 ** mu[i] - FLOOR, 1e-300), color=colors[i], lw=1.0, ls='--')
            e = np.where(vis[i], err[i], np.nan)
            axr.semilogx(E, e, color=colors[i], lw=0.8)
        ax.axvspan(E_data_lo, E_data_hi, color='k', alpha=0.06)
        axr.axvspan(E_data_lo, E_data_hi, color='k', alpha=0.06)
        axr.axhspan(-0.05, 0.05, color='C3', alpha=0.15, label='data error 0.05 dex')
        ax.set_title(title, fontsize=9)
        ax.set_ylim(1e-2, 1e13); ax.set_ylabel(r'comoving $E^2 n(E)$ [eV cm$^{-3}$]   (solid AM3, dashed emulator)')
        axr.set_ylabel('emulator - AM3 [dex]'); axr.set_xlabel('comoving photon energy [eV]')
        rms = np.sqrt(np.nanmean(np.where(vis, err, np.nan) ** 2)); med = np.nanmedian(np.abs(np.where(vis, err, np.nan)))
        axr.set_ylim(-1.0 if col == 0 else -0.25, 1.0 if col == 0 else 0.25)
        axr.text(0.02, 0.9, f'visible-bin error over all shown: rms {rms:.3f} dex, median {med:.3f} dex', transform=axr.transAxes, fontsize=8)
        axr.grid(alpha=0.3); ax.grid(alpha=0.3)
    axes[1, 0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Shaded: comoving energies the data cover for the reference source (delta = 20)', fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")
    print(f"whole box, all {te.size} held-out: rms {np.sqrt(np.mean(err_box[vis_box]**2)):.3f} dex, median {np.median(np.abs(err_box[vis_box])):.3f}, max {np.abs(err_box[vis_box]).max():.2f}")
    print(f"posterior region, {N_SHOW} points: rms {np.sqrt(np.mean(err_post[vis_post]**2)):.3f} dex, median {np.median(np.abs(err_post[vis_post])):.3f}, max {np.abs(err_post[vis_post]).max():.2f}")
    print("posterior-region spread used (std of the chi2<100 scan points):", dict(zip(PARAM_NAMES, np.round(width, 3))))
    pk = np.log10(Y.max(axis=1)); print(f"peak E^2 n across the box: log10 from {pk.min():.1f} to {pk.max():.1f}; in the posterior region: {np.log10(Y_post.max(axis=1)).min():.2f} to {np.log10(Y_post.max(axis=1)).max():.2f}")


if __name__ == '__main__':
    main()
