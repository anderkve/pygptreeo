#!/bin/bash
# round B2 on the dev set, then the baseline on the long-stream matrix
until ! pgrep -f "run_matrix.py" > /dev/null; do sleep 5; done
python run_matrix.py --variants "knn:metric=std,fallback=10,dir=curv;knn:metric=std,fallback=10,dir=curv,pos=gsplit;oblique:dir=axis,whiten=True;oblique:dir=ard,whiten=True;oblique:dir=curv,whiten=True;oblique:dir=curv,pos=gsplit,whiten=True;fisheye:fallback=10;knn:metric=std,fallback=10,retrain_every_n_points=100;knn:metric=std,fallback=10,search=tree,max_leaves=10;knn:metric=ard,fallback=10" --set dev --N 5000 --seeds 1 > logs_roundB2.txt 2>&1
python run_matrix.py --variants "base;knn:metric=std,fallback=10" --set dev --N 5000 --seeds 2 > logs_roundB2_seed2.txt 2>&1
