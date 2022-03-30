#!/bin/bash


# alpha=( 10 1 0.1 0.01)
# seed=( 1 2 3 4 5)

# for a in "${alpha[@]}"
#     do
#         for s in "${seed[@]}"
#         do
#             python3 initialize.py --seed $s --dataset mnist --model_name mnist_cnn1 --n_clients 10 --alpha $a
#         done
#     done

paths=./settings/mnist*/
settings=()
for path in $paths
do  
    set=${path%*/}
    set=${set//"./settings/"}
    settings+=($set)
done

for set in ${settings[@]}
do
    echo "$set"
done