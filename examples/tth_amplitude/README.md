# Tree-level ttH amplitudes on the 5D phase space of "Interpolating amplitudes"

Data for `examples/tth_amplitude_benchmark.py`: the test functions f1
(q qbar -> t tbar H) and f3 (g g -> t tbar H) of Bresó-Pla, Heinrich, Magerya and
Olsson, arXiv:2412.09534, at tree level, on their five phase-space variables
(beta^2 in [0.1, 0.96], f_ttbar in [0, 1], theta_H, theta_t, phi_t), with the domain
folded by the amplitude's symmetries as in their code (theta_H, theta_t in [0, pi/2],
phi_t in [0, pi]) and m_H / m_t = sqrt(12/23).

1. MadGraph5_aMC@NLO (3.6.x), standalone C++ output, no Fortran needed:
   `python bin/mg5_aMC tth_cpp.mg5` (creates `PROC_gg_tth` and `PROC_uu_tth`).
2. In each `PROC_*/Cards/param_card.dat` set `MH = 124.9605` (173 sqrt(12/23)) and
   the widths `WT = 0`, `WH = 0`.
3. Build the driver in each `PROC_*/SubProcesses/P1_*/` directory:
   `g++ -O2 -I../../src -I. -o tth_driver tth_driver.cpp CPPProcess.cc -L../../lib -lmodel_sm`
   It reads lines of the five variables and writes the phase-space weight (the
   paper's, in m_t = 1 units) and the spin/colour-averaged |M|^2.
4. Samples (about 15 us per point):
   `python gen_tth.py ./tth_driver gg_train_sobol.npz sobol 131072 1`,
   `python gen_tth.py ./tth_driver gg_test_uniform.npz uniform 40000 2`,
   `python gen_unweighted.py ./tth_driver gg_test_unweighted.npz 5000 7 <1.3 * max(w |M|^2)>`,
   and the same with `uu`.
5. The paper's test functions carry a flux and pseudo-PDF factor ρ(β²) that the driver's
   weight does not; the benchmark applies it (`rho()` in `tth_amplitude_benchmark.py`).
   LO-unweighted points drawn with that factor, which `--train unweighted`/`mixed` read
   by default, are generated with the channel as sixth argument:
   `python gen_unweighted.py ./tth_driver gg_train_unweighted_rho.npz 65536 13 <1.3 * max(w |M|^2 rho)> gg`
   (and 5000 test points with another seed). For the full polar-angle domain
   (`--domain full`) pass `theta_max = pi` as the last argument of both generators and
   keep the files in a separate data directory.
6. Larger gg sets for the runs beyond 131 072 points (Section 2.6 of the results):
   `python gen_tth.py ./tth_driver gg_train_sobol.npz sobol 524288 1` (the first 131 072
   points are the earlier set) and 262 144 ρ-unweighted points, e.g. the earlier 65 536
   plus three blocks of 65 536 from `gen_unweighted.py` with seeds 21, 22, 23 and the same
   fmax, concatenated into `gg_train_unweighted_rho.npz` in their own data directory.

## One-loop test functions f2 and f4 (`gosam/`)

The paper's f2 (q qbar) and f4 (g g) are the finite parts of 2 Re <M0|M1> with the
Coulomb singularity subtracted (their eqs. 14 and 19), evaluated with GoSam. The files
in `gosam/` reproduce them with GoSam 2.0.4 and its contrib libraries:

1. Tools: gfortran, FORM (`apt install form`), QGRAF 3.4.2 (`gfortran -o qgraf qgraf-3.4.2.f`),
   gosam-contrib 2.0 (`./configure --prefix=<prefix>`, FFLAGS `-O2 -fallow-argument-mismatch
   -std=legacy`) and GoSam 2.0.4 (Python 2.7, `python setup.py install --prefix=<prefix>`),
   both from gosam.hepforge.org/downloads. Link `<prefix>/share/gosam-contrib/gosam.conf` to
   `$HOME/gosam.conf` and add `qgraf.bin`, `form.bin` and `python.bin` (Python 2.7) to it.
2. Generate each process in OLP mode, with the paths in `tth.in` adjusted:
   `gosam.py --olp --config=tth.in --destination=olp --ignore-unknown --name=tthqq tth_qq.lh`
   (`tthgg`, `tth_gg.lh` for g g). The order file asks for the loop amplitude and the
   colour-correlated Born (`ccTree`), couplings stripped off, DRED, model SMdiag with only
   m_t and m_H non-zero. In `olp/Makefile.conf` add `-fallow-invalid-boz` to FCFLAGS, then
   `make -j4` in `olp/`.
3. GoSam's precision check compares the single pole with its own IR formula, which for these
   processes differs from the amplitude by a constant times the Born (-40/3 for q qbar); add
   `irp(2) = irp(2) - 40.0_ki / 3.0_ki * ampdef(1)` after the `call ir_subtraction` in
   `samplitude` of `olp/<process>/matrix/matrix.f90` and rebuild that file, otherwise every
   point is reported as imprecise. The double pole, the Coulomb cancellation and the Born
   against the MadGraph driver are checked by `validate.py`.
4. Build the evaluator against all objects of `olp/` and the contrib libraries:
   `g++ -O2 -DPREFIX=tthqq_ -o tth_loop_eval_qq tth_loop_eval.cpp $(find olp -name '*.o') -Wl,--start-group <prefix>/lib/lib{avh_olo,qcdloop,ff,golem,ninja,samurai}.a -Wl,--end-group -lgfortran -lquadmath -lz`.
   It reads the five variables per line and writes weight, Born, finite part, the two pole
   coefficients, <M0|T_t.T_tbar|M0>, beta_ttbar and GoSam's accuracy, with m_t = 1,
   mu_R = sqrt(s)/2 and no alpha_s/(2 pi) factor; about 5 ms (q qbar) per point.
5. `python gen_loop.py tth_loop_eval_qq tth_qq.olc <tree-level file>.npz <out>.npz 4`
   evaluates the loop amplitude at the points of an existing data file (same X, P and w)
   and stores the paper's amplitude as `me` with the pieces it is made of; run it for the
   Sobol, rho-unweighted, uniform-test and unweighted-test files of each channel into a
   separate data directory. The benchmark learns `me / tree` with `--ratio`.
