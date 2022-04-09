#!/bin/bash


alpha=( 10 1 0.1 0.01)
seed=( 1 2 3 4 5)

# for a in "${alpha[@]}"
#     do
#         for s in "${seed[@]}"
#         do
#             python3 initialize.py --seed $s --dataset mnist --model_name mnist_cnn1 --n_clients 10 --alpha $a
#         done
#     done

# for a in "${alpha[@]}"
#     do
#         for s in "${seed[@]}"
#         do
#             python3 initialize.py --seed $s --dataset emnist --model_name emnist_cnn1 --n_clients 10 --alpha $a
#         done
#     done

for a in "${alpha[@]}"
    do
        for s in "${seed[@]}"
        do
            python3 initialize.py --seed $s --dataset cifar10 --model_name cifar10_resnet18 --n_clients 10 --alpha $a
        done
    done