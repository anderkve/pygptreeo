"""Spectral parts of the AM3 SSC spectrum: do they sum to the total, and is the
PCA rank of each part (raw, normalised by the injected energy density, or shifted
by the analytic synchrotron cutoff) lower than that of the total?

Usage: python components_check.py   (caches 150 AM3 spectra with parsed SED in cwd)
"""
import sys, time, os, numpy as np
sys.path.insert(0, '/home/user/pygptreeo/examples/am3_emulator')
from am3_model import AM3SSCModel, BOUNDS, _silence_fd, C_CGS, EV_TO_ERG
from scipy.stats import qmc
FLOOR = 1e-2
cache = 'am3_components_150.npz'
if os.path.exists(cache):
    d = np.load(cache); Theta, E, tot, syn, ic, pairs = d['Theta'], d['E'], d['tot'], d['syn'], d['ic'], d['pairs']
else:
    m = AM3SSCModel(); sim = m.sim
    with _silence_fd(): sim.set_process_parse_sed(1)
    E = m.E; lo, hi = np.array(BOUNDS).T
    Theta = lo + qmc.LatinHypercube(d=6, seed=5).random(150) * (hi - lo)
    tot, syn, ic, pairs = [], [], [], []
    for th in Theta:
        nx = m.photon_density(th); tot.append(E * nx)
        syn.append(E * np.asarray(sim.get_photons_injected_electrons_syn(), float))
        ic.append(E * np.asarray(sim.get_photons_injected_electrons_compton(), float))
        pairs.append(E * np.asarray(sim.get_photons_annihilation_pairs_syn_compton(), float))
    tot, syn, ic, pairs = map(np.array, (tot, syn, ic, pairs))
    np.savez(cache, Theta=Theta, E=E, tot=tot, syn=syn, ic=ic, pairs=pairs)
logE = np.log10(E); dlog = logE[1] - logE[0]

def rank(L, vis, label):
    mu = L.mean(0); _, S, Vt = np.linalg.svd(L - mu, full_matrices=False)
    cum = np.cumsum(S**2) / np.sum(S**2); ks = [int(np.searchsorted(cum, f)) + 1 for f in (0.99, 0.999, 0.9999)]
    W = Vt[:16].T; rec = mu + ((L - mu) @ W) @ W.T; err = np.abs(rec - L)[vis]
    print(f"  {label:52s} 99/99.9/99.99%: {ks[0]:3d} {ks[1]:3d} {ks[2]:3d};  k=16 rms {np.sqrt(np.mean(err**2)):.3f} dex, max {err.max():.2f}")

def shift_log(Y, shift_dex):
    """Shift each spectrum along log E by shift_dex (fractional bins, linear interpolation of log Y)."""
    L = np.log10(Y + FLOOR); out = np.empty_like(L)
    for i in range(L.shape[0]):
        out[i] = np.interp(logE - shift_dex[i], logE, L[i], left=np.log10(FLOOR), right=np.log10(FLOOR))
    return out

logB, logR, logLe, p, loggmin, loggmax = Theta.T
# injected energy density U = L_e t_esc / V = 3 L_e / (4 pi R^2 c)  [erg/cm3] -> eV/cm3
logU = np.log10(3 * 10**logLe / (4 * np.pi * (10**logR)**2 * C_CGS) / EV_TO_ERG)
# analytic synchrotron cutoff energy: E_c = 1.7e-8 eV * B[G] * gamma^2 (comoving, delta outside); IC (Thomson): ~ gamma^2 E_c
logEsyn = np.log10(1.7e-8) + logB + 2 * loggmax
logEic = logEsyn + 2 * loggmax
print("PCA rank, 150 whole-box spectra, 482 bins (visible = part > 100 floor after transformation):")
for name, Y in (('total', tot), ('syn', syn), ('ic', ic)):
    L = np.log10(Y + FLOOR); rank(L, Y > 100 * FLOOR, f'log {name}')
    # level normalisation by the injected energy density
    Ln = np.log10(Y / 10**logU[:, None] + 1e-14); rank(Ln, Y / 10**logU[:, None] > 1e-12, f'log {name} / U_inj')
    # energy registration by the analytic cutoff (relative to its median)
    sh = (logEsyn if name != 'ic' else logEic); sh = sh - np.median(sh)
    Ls = shift_log(Y, sh); rank(Ls, Ls > np.log10(100 * FLOOR), f'log {name}, shifted by analytic E_cut')
    Lns = shift_log(Y / 10**logU[:, None] * 1e8, sh); rank(Lns, Lns > np.log10(100 * FLOOR), f'log {name} / U_inj, shifted by analytic E_cut')
# how well does the analytic cutoff track the actual peak / cutoff?
for name, Y, le in (('syn', syn, logEsyn), ('ic', ic, logEic)):
    ipk = Y.argmax(1); lpk = logE[ipk]
    # cutoff: energy where the part falls to 1% of its peak above the peak
    lcut = np.array([logE[j + np.argmax(Y[i, j:] < 0.01 * Y[i, j])] if np.any(Y[i, j:] < 0.01 * Y[i, j]) else np.nan for i, j in enumerate(ipk)])
    r1 = lpk - le; r2 = lcut - le
    print(f"  {name}: log E_peak - log E_cut(analytic): std {np.nanstd(r1):.2f} dex; log E_1%cutoff - log E_cut: std {np.nanstd(r2):.2f} dex (median offset {np.nanmedian(r2):+.2f})")
