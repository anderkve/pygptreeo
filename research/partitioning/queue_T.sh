#!/bin/bash
# timings with one process at a time, after everything else
until ! pgrep -f "queue_L2.sh" > /dev/null; do sleep 5; done
until ! pgrep -f "run_matrix.py" > /dev/null; do sleep 5; done
V="base;nb:neighbour_leaves=None;nb:neighbour_leaves=8;nb:neighbour_leaves=16;nb:neighbour_leaves=8,n_neighbours=50;nb:neighbour_leaves=16,n_neighbours=200"
for ts in "gaussian_peaks uniform" "gaussian_peaks walker" "resonance_xsec uniform" "global_fit_loglike uniform" "eggholder sweeping"; do
    python time_predict.py snapshot $ts 20000 "$V" 2>&1 | grep -v Warn >> logs_timing.txt
done
python drift_bench.py snapshot.old leaf 12000 >> logs_drift.txt 2>&1
python drift_bench.py snapshot leaf 12000 >> logs_drift.txt 2>&1
python drift_bench.py snapshot neighbours 12000 >> logs_drift.txt 2>&1
