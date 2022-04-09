#!/bin/bash

paths=./settings/cifar10*/
settings=()
for path in $paths
do  
    set=${path%*/}
    set=${set//"./settings/"}
    settings+=($set)
done

n_rounds=100
local_epochs_ensemble=20
student_epochs=30
student_epochs_w2=100
autoencoder_epochs=50
public_data_sizes="500 1000 5000 10000 25000"

# n_rounds=1
# local_epochs_ensemble=1
# student_epochs=1
# student_epochs_w2=1
# autoencoder_epochs=1
# public_data_sizes="500 1000"

student_models=("cifar10_resnet18" "cifar10_resnet34")
loss_functions=("mse" "ce")

# FEDAVG
for set in ${settings[@]}
do  
    python3 main.py --settings_file $set --algorithm fedavg --local_epochs 1 --n_rounds $n_rounds --train_batch_size 80
done

# FEDPROX
for set in ${settings[@]}
do  
    python3 main.py --settings_file $set --algorithm fedprox --mu 0.01 --local_epochs 1 --n_rounds $n_rounds --train_batch_size 80
done

# FEDED
for set in ${settings[@]}
do  
    for model in ${student_models[@]}
    do
        python3 main.py --settings_file $set --algorithm feded --train_batch_size 80 --local_epochs_ensemble $local_epochs_ensemble --student_model $model --public_data_sizes="$public_data_sizes" --client_sample_fraction 1.0 --student_epochs $student_epochs --weight_scheme 0
    done
done

# FEDED, w1
for set in ${settings[@]}
do  
    for model in ${student_models[@]}
    do
        python3 main.py --settings_file $set --algorithm feded --train_batch_size 80 --local_epochs_ensemble $local_epochs_ensemble --student_model $model --public_data_sizes="$public_data_sizes" --client_sample_fraction 1.0 --student_epochs $student_epochs --weight_scheme 1
    done
done

# FEDED, w2
for set in ${settings[@]}
do  
    for model in ${student_models[@]}
    do
        python3 main.py --settings_file $set --algorithm feded --train_batch_size 80 --local_epochs_ensemble $local_epochs_ensemble --student_model $model --public_data_sizes="$public_data_sizes" --client_sample_fraction 1.0 --student_epochs $student_epochs --weight_scheme 2 --student_loss ce --autoencoder_epochs $autoencoder_epochs
    done
done