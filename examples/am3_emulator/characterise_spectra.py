"""Characterise AM3 SSC spectra over the scan box: dynamic range, PCA rank and a
first batch train/test of the GPTree emulator.

Usage: python characterise_spectra.py [n_sample] [npz_path]
The AM3 sample is cached in ``npz_path`` (default: am3_lhs_sample.npz in cwd).
"""

import os
import sys
import time
import warnings

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from am3_model import AM3SSCModel, BOUNDS, PARAM_NAMES   # noqa: E402

warnings.filterwarnings('ignore')


def lhs_sample(n, seed=1):
    lo, hi = np.array(BOUNDS).T
    U = qmc.LatinHypercube(d=len(BOUNDS), seed=seed).random(n)
    return lo + U * (hi - lo)


def load_or_run(n, path):
    if os.path.exists(path):
        d = np.load(path)
        if d['Theta'].shape[0] >= n:
            return d['Theta'][:n], d['Y'][:n], d['E']
    model = AM3SSCModel()
    Theta = lhs_sample(n)
    Y = np.empty((n, model.E.size))
    t0 = time.perf_counter()
    for i in range(n):
        Y[i] = model.comoving_sed(Theta[i])
    print(f"AM3: {n} spectra in {time.perf_counter() - t0:.1f}s ({model.t_total / n:.3f}s per call)")
    np.savez(path, Theta=Theta, Y=Y, E=model.E)
    return Theta, Y, model.E


def pca_report(name, Ytr, Yte, visible_tr, visible_te, ks=(4, 8, 12, 16, 24, 32, 48)):
    mu = Ytr.mean(axis=0)
    _, S, Vt = np.linalg.svd(Ytr - mu, full_matrices=False)
    var = S ** 2 / (Ytr.shape[0] - 1)
    cum = np.cumsum(var) / var.sum()
    print(f"\n[{name}] PCA on {Ytr.shape[0]} spectra x {Ytr.shape[1]} bins")
    for frac in (0.99, 0.999, 0.9999, 0.99999):
        print(f"  components for {frac:.5f} of variance: {int(np.searchsorted(cum, frac)) + 1}")
    print("  held-out reconstruction error (dex) on visible bins, by k:")
    for k in ks:
        W = Vt[:k].T
        rec = mu + ((Yte - mu) @ W) @ W.T
        err = np.abs(rec - Yte)[visible_te]
        rec_tr = mu + ((Ytr - mu) @ W) @ W.T
        err_tr = np.abs(rec_tr - Ytr)[visible_tr]
        print(f"    k={k:3d}: test rms {np.sqrt(np.mean(err**2)):.4f}  test max {err.max():.3f}   "
              f"(train rms {np.sqrt(np.mean(err_tr**2)):.4f})")


def gptree_batch_test(name, Theta_tr, Ytr, Theta_te, Yte, visible_te, sigma, configs):
    from pygptreeo import GPTree
    print(f"\n[{name}] GPTree batch train ({Theta_tr.shape[0]}) / test ({Theta_te.shape[0]}), sigma={sigma}")
    for cfg in configs:
        gpt = GPTree(n_outputs=Ytr.shape[1], **cfg)
        t0 = time.perf_counter()
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            gpt.fit(Theta_tr, Ytr, np.full_like(Ytr, sigma))
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        mu, sd = gpt.predict(Theta_te)
        t_pred = (time.perf_counter() - t0) / Theta_te.shape[0]
        err = (mu - Yte)[visible_te]
        z = np.abs(err) / sd[visible_te]
        basis = gpt.output_basis.current if gpt.output_basis is not None else None
        kinfo = f" k={basis.n_components}" if basis is not None else ""
        print(f"  {cfg}{kinfo}: rms {np.sqrt(np.mean(err**2)):.4f} dex, median|err| {np.median(np.abs(err)):.4f}, "
              f"max {np.abs(err).max():.3f}; |err|/sigma median {np.median(z):.2f}, 95% {np.percentile(z, 95):.2f}; "
              f"fit {t_fit:.1f}s, predict {1e3*t_pred:.1f} ms/pt, leaves {len(gpt.root.leaves)}")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    path = sys.argv[2] if len(sys.argv) > 2 else 'am3_lhs_sample.npz'
    Theta, Y, E = load_or_run(n, path)
    n_te = n // 4
    tr, te = slice(0, n - n_te), slice(n - n_te, n)

    peak = Y.max(axis=1)
    print(f"log10 peak E^2 n [eV/cm3]: min {np.log10(peak.min()):.2f}, median {np.log10(np.median(peak)):.2f}, "
          f"max {np.log10(peak.max()):.2f}")
    ipk = Y.argmax(axis=1)
    print(f"peak energy [eV]: min {E[ipk].min():.2e}, max {E[ipk].max():.2e}")

    # Representation A: absolute soft floor in log10(E^2 n + floor)
    for floor in (1e-2, 1e0):
        YA = np.log10(Y + floor)
        vis = Y > 100 * floor
        print(f"\n=== Representation A: log10(E^2 n + {floor:g}); visible bins (>100 floor) "
              f"{vis.mean():.2%}; range {YA.min():.1f}..{YA.max():.1f}")
        pca_report(f"A floor={floor:g}", YA[tr], YA[te], vis[tr], vis[te])

    # Representation B: shape relative to the peak (soft floor 1e-12 of the peak) + log10 peak
    rel_floor = 1e-12
    YB = np.hstack([np.log10(Y / peak[:, None] + rel_floor), np.log10(peak)[:, None]])
    visB = np.hstack([Y / peak[:, None] > 100 * rel_floor, np.ones((n, 1), bool)])
    print(f"\n=== Representation B: shape log10(E^2n/peak + 1e-12) plus log10 peak; visible {visB.mean():.2%}")
    pca_report("B", YB[tr], YB[te], visB[tr], visB[te])

    # GPTree batch baseline on representation A (floor 1e-2) and B
    floor = 1e-2
    YA = np.log10(Y + floor)
    visA = Y > 100 * floor
    configs = [
        dict(Nbar=100, output_model='pca', output_basis_components='noise'),
        dict(Nbar=100, output_model='pca', output_basis_components=8),
        dict(Nbar=100, output_model='pca', output_basis_components=16),
        dict(Nbar=100, output_model='pca', output_basis_components=32),
        dict(Nbar=200, output_model='pca', output_basis_components=16),
    ]
    gptree_batch_test("A floor=1e-2", Theta[tr], YA[tr], Theta[te], YA[te], visA[te], 1e-2, configs)
    gptree_batch_test("B", Theta[tr], YB[tr], Theta[te], YB[te], visB[te], 1e-2, configs[:3])
    # 'independent' on a 4x coarser grid as the brute-force reference
    sub = slice(0, None, 4)
    gptree_batch_test("A floor=1e-2, every 4th bin, independent", Theta[tr], YA[tr][:, sub], Theta[te],
                      YA[te][:, sub], visA[te][:, sub], 1e-2, [dict(Nbar=100, output_model='independent')])


if __name__ == '__main__':
    main()
