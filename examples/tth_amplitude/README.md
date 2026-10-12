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
