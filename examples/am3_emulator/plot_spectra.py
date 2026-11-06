"""Plot emulated against true AM3 spectra for a finished scan.

Needs the scan's result file and the tree it saved (``--save-tree``). Shows the
synthetic data, the AM3 spectrum and the emulator's prediction with its 1-sigma
band at the scan's best point and at a few random points of the last generations.

Usage: python plot_spectra.py results.npz tree.joblib out.png
"""

import os
import sys

import joblib
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import scan_emulator as se  # noqa: E402
from am3_model import AM3SSCModel, luminosity_distance_cm, observed_sed  # noqa: E402


def main():
    res_path, tree_path, out = sys.argv[1:4]
    d = np.load(res_path, allow_pickle=True)
    tree = joblib.load(tree_path)
    model = AM3SSCModel()
    d_L = luminosity_distance_cm(se.Z_SRC)
    floor = 1e-2
    rng = np.random.RandomState(0)
    n = d['theta'].shape[0]
    picks = [('best point', d['best_x'])]
    for i in sorted(rng.choice(np.arange(n - 5 * 84, n), 3, replace=False)):
        picks.append((f'scan point {i}, chi2={d["chi2_true"][i]:.0f}' if np.isfinite(d['chi2_true'][i])
                      else f'scan point {i} (emulated)', d['theta'][i]))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, (title, theta) in zip(axes.ravel(), picks):
        e2n = model.comoving_sed(theta[:6])
        E_o, f_true = observed_sed(model.E, e2n, theta[1], theta[6], se.Z_SRC, d_L)
        mu, sd = tree.predict(np.asarray(theta[:6])[None, :])
        e2n_emu = np.maximum(10.0 ** mu[0] - floor, 0.0)
        _, f_emu = observed_sed(model.E, e2n_emu, theta[1], theta[6], se.Z_SRC, d_L)
        _, f_lo = observed_sed(model.E, np.maximum(10.0 ** (mu[0] - sd[0]) - floor, 0.0), theta[1], theta[6], se.Z_SRC, d_L)
        _, f_hi = observed_sed(model.E, np.maximum(10.0 ** (mu[0] + sd[0]) - floor, 0.0), theta[1], theta[6], se.Z_SRC, d_L)
        ax.loglog(E_o, f_true, 'k-', lw=2, label='AM3')
        ax.loglog(E_o, f_emu, 'C3--', lw=1.5, label='GPTree emulator')
        ax.fill_between(E_o, f_lo, f_hi, color='C3', alpha=0.2, label='emulator 1 sigma')
        ax.errorbar(d['E_obs'], 10.0 ** d['log_f_obs'], yerr=np.log(10) * d['sig_obs'] * 10.0 ** d['log_f_obs'],
                    fmt='o', color='C0', ms=4, label='synthetic data')
        ax.set_title(f"{title}\n" + ', '.join(f'{nm}={v:.2f}' for nm, v in zip(se.SCAN_NAMES, theta)), fontsize=8)
        ax.set_ylim(1e-16, 1e-8); ax.set_xlim(1e-6, 1e13); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel('observed photon energy [eV]')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\nu F_\nu$ [erg cm$^{-2}$ s$^{-1}$]')
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == '__main__':
    main()
