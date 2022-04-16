#!/bin/bash

dataset="mnist"
n_clients=10
public_fraction=0.5
n_rounds=100
local_epochs_ensemble=20
student_epochs=30
student_epochs_w2=100
student_lr_w2=1e-5
autoencoder_epochs=30
public_data_sizes="500 1000 5000 15000 30000"
local_model="mnist_cnn1"

seeds=(8 9 10)
alphas=(10.0 1.0 0.1 0.01)
student_models="mnist_cnn1 mnist_cnn2 mnist_cnn3"
weight_schemes="0 1 2"

# n_rounds=1
# local_epochs_ensemble=1
# student_epochs=1
# student_epochs_w2=1
# autoencoder_epochs=1
# public_data_sizes="500 1000"
# seeds=(20)
# alphas=(10.0 1.0)

settings_summary="--dataset $dataset --n_clients $n_clients --public_fraction $public_fraction --distribution niid --local_model $local_model --client_sample_fraction 1.0"

# FEDAVG
for seed in ${seeds[@]}
do
    for alpha in ${alphas[@]}
    do  
        python3 main.py $settings_summary --algorithm fedavg --seed $seed --alpha $alpha --local_epochs 1 --n_rounds $n_rounds
    done
done

# # FEDPROX
# for seed in ${seeds[@]}
# do
#     for alpha in ${alphas[@]}
#     do  
#         python3 main.py $settings_summary --algorithm fedprox --seed $seed --alpha $alpha --local_epochs 1 --n_rounds $n_rounds --mu 1
#     done
# done

# settings_ensemble="--local_epochs_ensemble $local_epochs_ensemble --student_epochs $student_epochs --student_epochs_w2 $student_epochs_w2 --student_lr_w2 $student_lr_w2  --autoencoder_epochs $autoencoder_epochs"

# # FEDED w0 w1 w2
# for seed in ${seeds[@]}
# do
#     for alpha in ${alphas[@]}
#     do  
#         python3 main.py $settings_summary $settings_ensemble --student_models="$student_models" --public_data_sizes="$public_data_sizes" --algorithm feded --student_epochs $student_epochs --seed $seed --alpha $alpha --weight_schemes="$weight_schemes"
#     done
# done