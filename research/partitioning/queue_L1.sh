#!/bin/bash
# long streams: 20k points, 19 targets, four stream types, two seeds; and how many leaves the search needs
until ! pgrep -f "queue_B3.sh" > /dev/null; do sleep 5; done
until ! pgrep -f "run_matrix.py" > /dev/null; do sleep 5; done
T=gaussian_peaks,rotated_rosenbrock,eggholder,chirp,diagonal_ridge,step_ridge,active_peaks,michalewicz,resonance_xsec,dilepton_spectrum,jet_spectrum,oscillation_probability,branching_ratio,loop_decay_width,relic_density,neutralino_mass,eft_signal_strength,signal_yield_mc,global_fit_loglike
python run_matrix.py --variants "base;nb" --targets $T --streams uniform,walker,sweeping,focusing --N 20000 --seeds 1 > logs_L1_seed1.txt 2>&1
python run_matrix.py --variants "nb:neighbour_leaves=4;nb:neighbour_leaves=8;nb:neighbour_leaves=16;nb:neighbour_leaves=32" --targets gaussian_peaks,eggholder,resonance_xsec,eft_signal_strength,global_fit_loglike,active_peaks --streams uniform,walker,sweeping --N 20000 --seeds 1 > logs_B4.txt 2>&1
python run_matrix.py --variants "base;nb" --targets $T --streams uniform,walker,sweeping,focusing --N 20000 --seeds 2 > logs_L1_seed2.txt 2>&1
