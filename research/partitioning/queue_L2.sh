#!/bin/bash
# after the 20k matrix: 100k streams, the hybrid with and without neighbours, the best oblique cuts at 20k
until ! pgrep -f "queue_L1.sh" > /dev/null; do sleep 5; done
until ! pgrep -f "run_matrix.py" > /dev/null; do sleep 5; done
python run_matrix.py --variants "base;nb" --targets gaussian_peaks,eggholder,resonance_xsec,eft_signal_strength,global_fit_loglike --streams uniform,walker,sweeping --N 100000 --seeds 1 > logs_L2.txt 2>&1
python run_matrix.py --variants "hybrid;hybrid:nb=True" --targets gaussian_peaks,rotated_rosenbrock,eggholder,resonance_xsec,eft_signal_strength,relic_density --streams uniform,walker,sweeping --N 20000 --seeds 1 > logs_H1.txt 2>&1
python run_matrix.py --variants "oblique:dir=curv,pos=gsplit;oblique:dir=curv,pos=gsplit,whiten=True" --set dev --N 20000 --seeds 1 > logs_L1b.txt 2>&1
