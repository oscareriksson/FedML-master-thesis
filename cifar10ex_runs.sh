#!/bin/bash

dataset="cifar10ex"
n_clients=10
public_fraction=0.5
n_rounds=100
local_epochs_ensemble=20
student_epochs=10
student_epochs_w2=50
student_lr_w2=1e-5
autoencoder_epochs=50
public_data_sizes="50000"
local_model="cifar10_resnet18"

seeds=(0 2 3 4 5 6 7 8 10)
alphas=(10.0 1.0 0.1 0.01)
student_models="cifar10_resnet18"
weight_schemes="0 1 2"

seeds=(0)
alphas=(10.0)
student_models="cifar10_resnet18"
weight_schemes="2"

# n_rounds=1
# local_epochs_ensemble=1
# student_epochs=1
# student_epochs_w2=1
# autoencoder_epochs=1
# public_data_sizes="1000"

settings_summary="--dataset $dataset --n_clients $n_clients --public_fraction $public_fraction --distribution niid --local_model $local_model --client_sample_fraction 1.0 --train_batch_size 80"

settings_ensemble="--local_epochs_ensemble $local_epochs_ensemble --student_epochs $student_epochs --student_epochs_w2 $student_epochs_w2 --student_lr_w2 $student_lr_w2  --autoencoder_epochs $autoencoder_epochs"

# FEDED w0 w1 w2
for seed in ${seeds[@]}
do
    for alpha in ${alphas[@]}
    do  
        python3 main.py $settings_summary $settings_ensemble --student_models="$student_models" --public_data_sizes="$public_data_sizes" --algorithm feded --student_epochs $student_epochs --seed $seed --alpha $alpha --weight_schemes="$weight_schemes"
    done
done