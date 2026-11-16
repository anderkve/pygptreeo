#!/bin/bash
# round B3: leaf size, neighbour count and guard threshold under the package's neighbour conditioning
until ! pgrep -f "queue_B2.sh" > /dev/null; do sleep 5; done
until ! pgrep -f "run_matrix.py" > /dev/null; do sleep 5; done
python run_matrix.py --variants "nb;nb:Nbar=50,n_neighbours=100;nb:Nbar=50,n_neighbours=100,retrain_every_n_points=12;nb:Nbar=200,retrain_every_n_points=50,n_neighbours=100;nb:n_neighbours=150;nb:neighbour_consistency=30;nb:neighbour_consistency=3;nb:retrain_every_n_points=50" --set dev --N 5000 --seeds 1 > logs_roundB3.txt 2>&1
