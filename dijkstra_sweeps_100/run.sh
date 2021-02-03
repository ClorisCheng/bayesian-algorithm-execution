#!/bin/bash
for seed in 1 2 3 4 5 do
do
    for acq_func in "bax" "rand" "uncert"
    do
        python sweep_dijkstras.py --acq_func $acq_func --seed $seed --plot --can_requery --n_path 30 --n_iter 200 --grid_size 10
    done
done
