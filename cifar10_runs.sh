#!/bin/sh

# python3 main.py --settings_file cifar10_resnet_c10_iid_qre --algorithm fedavg --local_epochs 1 --n_rounds 100
# python3 main.py --settings_file cifar10_resnet_c10_iid_vvr --algorithm fedavg --local_epochs 1 --n_rounds 100
# python3 main.py --settings_file cifar10_resnet_c10_iid_yzu --algorithm fedavg --local_epochs 1 --n_rounds 100
# python3 main.py --settings_file cifar10_resnet_c10_iid_dpp --algorithm fedavg --local_epochs 1 --n_rounds 100
# python3 main.py --settings_file cifar10_resnet_c10_iid_xsf --algorithm fedavg --local_epochs 1 --n_rounds 100

# python3 main.py --settings_file cifar10_resnet_c10_niid0.1_got --algorithm fedavg --local_epochs 1 --n_rounds 100
# python3 main.py --settings_file cifar10_resnet_c10_niid0.1_hhk --algorithm fedavg --local_epochs 1 --n_rounds 100
# python3 main.py --settings_file cifar10_resnet_c10_niid0.1_zkc --algorithm fedavg --local_epochs 1 --n_rounds 100
# python3 main.py --settings_file cifar10_resnet_c10_niid0.1_kih --algorithm fedavg --local_epochs 1 --n_rounds 100
# python3 main.py --settings_file cifar10_resnet_c10_niid0.1_sgk --algorithm fedavg --local_epochs 1 --n_rounds 100

# python3 main.py --settings_file cifar10_resnet_c10_iid_qre --algorithm feded --local_epochs_ensemble 50 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 15 --weight_scheme 0
# python3 main.py --settings_file cifar10_resnet_c10_iid_vvr --algorithm feded --local_epochs_ensemble 50 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 15 --weight_scheme 0
# python3 main.py --settings_file cifar10_resnet_c10_iid_yzu --algorithm feded --local_epochs_ensemble 50 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 15 --weight_scheme 0
# python3 main.py --settings_file cifar10_resnet_c10_iid_dpp --algorithm feded --local_epochs_ensemble 50 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 15 --weight_scheme 0
# python3 main.py --settings_file cifar10_resnet_c10_iid_xsf --algorithm feded --local_epochs_ensemble 50 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 15 --weight_scheme 0

# python3 main.py --settings_file cifar10_resnet_c10_niid0.1_got --algorithm feded --local_epochs_ensemble 50 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 15 --weight_scheme 0
# python3 main.py --settings_file cifar10_resnet_c10_niid0.1_hhk --algorithm feded --local_epochs_ensemble 50 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 15 --weight_scheme 0
# python3 main.py --settings_file cifar10_resnet_c10_niid0.1_zkc --algorithm feded --local_epochs_ensemble 50 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 15 --weight_scheme 0
# python3 main.py --settings_file cifar10_resnet_c10_niid0.1_kih --algorithm feded --local_epochs_ensemble 50 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 15 --weight_scheme 0
# python3 main.py --settings_file cifar10_resnet_c10_niid0.1_sgk --algorithm feded --local_epochs_ensemble 50 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 15 --weight_scheme 0

# python3 main.py --settings_file cifar10_resnet_c10_niid0.1_got --algorithm feded --local_epochs_ensemble 50 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 15 --weight_scheme 1
# python3 main.py --settings_file cifar10_resnet_c10_niid0.1_hhk --algorithm feded --local_epochs_ensemble 50 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 15 --weight_scheme 1
# python3 main.py --settings_file cifar10_resnet_c10_niid0.1_zkc --algorithm feded --local_epochs_ensemble 50 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 15 --weight_scheme 1
# python3 main.py --settings_file cifar10_resnet_c10_niid0.1_kih --algorithm feded --local_epochs_ensemble 50 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 15 --weight_scheme 1
# python3 main.py --settings_file cifar10_resnet_c10_niid0.1_sgk --algorithm feded --local_epochs_ensemble 50 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 15 --weight_scheme 1

python3 main.py --settings_file cifar10_resnet_c10_niid0.1_got --algorithm feded --local_epochs_ensemble 50 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 100 --weight_scheme 2 --autoencoder_epochs 30 --student_lr 0.00001 --student_loss ce
python3 main.py --settings_file cifar10_resnet_c10_niid0.1_hhk --algorithm feded --local_epochs_ensemble 50 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 100 --weight_scheme 2 --autoencoder_epochs 30 --student_lr 0.00001 --student_loss ce
python3 main.py --settings_file cifar10_resnet_c10_niid0.1_zkc --algorithm feded --local_epochs_ensemble 50 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 100 --weight_scheme 2 --autoencoder_epochs 30 --student_lr 0.00001 --student_loss ce
python3 main.py --settings_file cifar10_resnet_c10_niid0.1_kih --algorithm feded --local_epochs_ensemble 50 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 100 --weight_scheme 2 --autoencoder_epochs 30 --student_lr 0.00001 --student_loss ce
python3 main.py --settings_file cifar10_resnet_c10_niid0.1_sgk --algorithm feded --local_epochs_ensemble 50 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 100 --weight_scheme 2 --autoencoder_epochs 30 --student_lr 0.00001 --student_loss ce
