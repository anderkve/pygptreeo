# AM3 spectra emulated on the fly with pyGPTreeO

A test of `GPTree(output_model='pca')` as an on-the-fly emulator of the photon
spectra computed by [AM3](https://am3.readthedocs.io/) inside a parameter scan.
`RESULTS_am3_emulator.md` has the setup, the measurements and the conclusions.

Scripts (run from this directory; AM3 must be importable, see the results file
for how to build it):

* `am3_model.py`: the one-zone SSC blazar model as a function of six comoving
  parameters, returning the steady-state spectrum on AM3's log-energy grid, plus
  the analytic observer-frame conversion. `python am3_model.py` times a few calls.
* `characterise_spectra.py [n] [cache.npz]`: Latin-hypercube sample of AM3
  spectra over the scan box; dynamic range, PCA rank and a batch GPTree baseline.
* `scan_emulator.py --mode shadow|active [...]`: differential-evolution fit of a
  synthetic SED with a GPTree learning every AM3 spectrum the scan produces.
  `shadow` always calls AM3 and scores the emulator; `active` lets the emulator
  replace AM3 when its propagated chi^2 uncertainty passes the gate. Options set
  the tree (`--components`, `--nbar`, `--retrain-every`), the gate
  (`--chi2-tol-abs`, `--chi2-tol-rel`, `--verify-frac`), the synthetic source
  (`--theta-true`, `--obs-seed`) and tree persistence (`--save-tree`, `--load-tree`).
* `analyse_scan.py results.npz [...] [--plots dir]`: learning curves, chi^2
  errors, calibration, a what-if gate analysis for shadow runs, and timings.
* `plot_spectra.py results.npz tree.joblib out.png`: emulated against AM3 spectra
  at the best point and at late scan points.

Reproduce the main runs with

```bash
python scan_emulator.py --mode shadow --components 16 --out shadow.npz
python scan_emulator.py --mode active --components 16 --out active.npz --save-tree tree.joblib
python analyse_scan.py shadow.npz active.npz --plots plots
```

Do not run more than two AM3 scans at once on a small machine: four concurrent
runs made every AM3 call about nine times slower here.
