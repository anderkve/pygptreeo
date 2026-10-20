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

## One-loop test functions f2 and f4 (`loop/`)

The paper's f2 (q qbar) and f4 (g g) are the finite parts of 2 Re <M0|M1> with the
Coulomb singularity subtracted (their eqs. 14 and 19), evaluated with GoSam. Here they
are evaluated with MadLoop (MadGraph5_aMC@NLO 3.6.x, needs gfortran), cross-checked
against OpenLoops 2.1.5 at test points; GoSam 2.0.4 built from source disagreed with
both in the finite parts and was not used.

1. `python bin/mg5_aMC loop/tth_loop_uu.mg5` (and `tth_loop_gg.mg5`): standalone MadLoop
   output of `u u~ > t t~ h [virt=QCD]` and `g g > t t~ h [virt=QCD]` in the model
   `loop_sm-no_b_mass` (five massless flavours). MadGraph compiles CutTools and Ninja
   from its own tarballs. In each `Cards/param_card.dat` set `MT = 173`, `MH = 124.9605`,
   `WT = 0`, `WH = 0`.
2. In `SubProcesses/P0_*`, copy `loop/tth_ml_driver.f`, run `make check` once to build the
   libraries, then compile and link the driver with the same command as `check` (take it
   from `make -n check`, replacing `check_sa.o` by `tth_ml_driver.o`). The driver reads one
   point per line (E px py pz of the five particles in GeV, process order) and prints
   `RES`, Born, finite part, 1/eps and 1/eps^2 coefficients (without the alpha_s/(2 pi)
   factor), MadLoop's accuracy estimate and return code, with mu_R = sqrt(s)/2. About
   2 ms (q qbar) and 20 ms (g g) per point.
3. `MG5=<MadGraph dir> python loop/gen_loop_ml.py <driver> <tree-level file>.npz <out>.npz 4`
   evaluates the loop amplitude at the points of an existing data file (same X, P and w)
   and stores the paper's amplitude as `me` with the pieces it is made of; the
   colour-correlated Born of the Coulomb term is Born/6 for q qbar (colour octet) and
   is taken from the full colour correlator for g g. Run it for the Sobol,
   rho-unweighted, uniform-test and unweighted-test files of each channel into a
   separate data directory; the benchmark learns `me / tree` with `--ratio`.
4. `loop/tth_ol_eval.cpp` evaluates the same points with OpenLoops (`./openloops
   libinstall pphtt`; `g++ tth_ol_eval.cpp -I<OL>/include -L<OL>/lib -lopenloops`) in the
   same layout as the drivers above, for cross-checks.
5. The g g colour correlator <M0|T_t.T_tbar|M0> / Born comes from GoSam's tree-level
   `ccTree` amplitude type (`gosam/`): GoSam 2.0.4 with gosam-contrib 2.0, QGRAF 3.4.2
   and FORM from gosam.hepforge.org and Spack's source mirror, Python 2.7; generate with
   `gosam.py --olp --config=tth.in --destination=olp --ignore-unknown --name=tthgg tth_gg.lh`
   (add `-fallow-invalid-boz` to FCFLAGS in `olp/Makefile.conf`, Python 2.7 as PYTHON),
   `make -j4` in `olp/`, link `tth_loop_eval.cpp` with `-DPREFIX=tthgg_ -DPREFIX_IS_GG=1`
   against all objects of `olp/` and the contrib libraries, then
   `python gen_cc.py tth_loop_eval_gg tth_gg.olc <tree-level file>.npz <cc file>.npz`
   (0.1 ms per point) and `python loop/fix_gg_cc.py <MadLoop g g file>.npz <cc file>.npz`,
   or pass the cc file as the last argument of `gen_loop_ml.py`. GoSam's own one-loop
   finite parts, from the same build, disagreed with MadLoop and OpenLoops and were
   not used.
6. `loop/tth_loop_driver.py <uu|gg> <tth_ml_driver> [--cc <tth_loop_eval_gg> <contract>] [--ratio]`
   wraps the MadLoop driver (and the GoSam colour correlator for g g) in the tree-level
   driver's interface, for the leaf-driven acquisition of the benchmark:
   `tth_amplitude_benchmark.py --configs leafacq --driver "python3 loop/tth_loop_driver.py ..."`
   asks it for 75 fresh points at a time. With `--ratio` it returns a / a_LO and the
   benchmark needs `--born-driver <tree-level driver>` for the leaf score w a_LO sigma.
