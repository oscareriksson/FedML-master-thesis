#!/bin/sh

# python3 main.py --settings_file cifar10_resnet_c10_iid_qre --algorithm fedavg --local_epochs 1 --n_rounds 100
# python3 main.py --settings_file cifar10_resnet_c10_iid_vvr --algorithm fedavg --local_epochs 1 --n_rounds 100
# python3 main.py --settings_file cifar10_resnet_c10_iid_yzu --algorithm fedavg --local_epochs 1 --n_rounds 100
# python3 main.py --settings_file cifar10_resnet_c10_iid_dpp --algorithm fedavg --local_epochs 1 --n_rounds 100
# python3 main.py --settings_file cifar10_resnet_c10_iid_xsf --algorithm fedavg --local_epochs 1 --n_rounds 100

# python3 main.py --settings_file cifar10_resnet_c10_niid0.1_got --algorithm fedavg --local_epochs 1 --n_rounds 100 --train_batch_size 80
# python3 main.py --settings_file cifar10_resnet_c10_niid0.1_hhk --algorithm fedavg --local_epochs 1 --n_rounds 100 --train_batch_size 80
# python3 main.py --settings_file cifar10_resnet_c10_niid0.1_zkc --algorithm fedavg --local_epochs 1 --n_rounds 100 --train_batch_size 80
# python3 main.py --settings_file cifar10_resnet_c10_niid0.1_kih --algorithm fedavg --local_epochs 1 --n_rounds 100 --train_batch_size 80
# python3 main.py --settings_file cifar10_resnet_c10_niid0.1_sgk --algorithm fedavg --local_epochs 1 --n_rounds 100 --train_batch_size 80

python3 main.py --settings_file cifar10_resnet_c10_iid_qre --algorithm feded --local_epochs_ensemble 20 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 0 --train_batch_size 80
python3 main.py --settings_file cifar10_resnet_c10_iid_vvr --algorithm feded --local_epochs_ensemble 20 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 0 --train_batch_size 80
python3 main.py --settings_file cifar10_resnet_c10_iid_yzu --algorithm feded --local_epochs_ensemble 20 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 0 --train_batch_size 80
python3 main.py --settings_file cifar10_resnet_c10_iid_dpp --algorithm feded --local_epochs_ensemble 20 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 0 --train_batch_size 80
python3 main.py --settings_file cifar10_resnet_c10_iid_xsf --algorithm feded --local_epochs_ensemble 20 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 0 --train_batch_size 80

python3 main.py --settings_file cifar10_resnet_c10_niid0.1_got --algorithm feded --local_epochs_ensemble 5 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 0 --train_batch_size 80
python3 main.py --settings_file cifar10_resnet_c10_niid0.1_hhk --algorithm feded --local_epochs_ensemble 5 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 0 --train_batch_size 80
python3 main.py --settings_file cifar10_resnet_c10_niid0.1_zkc --algorithm feded --local_epochs_ensemble 5 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 0 --train_batch_size 80
python3 main.py --settings_file cifar10_resnet_c10_niid0.1_kih --algorithm feded --local_epochs_ensemble 5 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 0 --train_batch_size 80
python3 main.py --settings_file cifar10_resnet_c10_niid0.1_sgk --algorithm feded --local_epochs_ensemble 5 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 0 --train_batch_size 80

python3 main.py --settings_file cifar10_resnet_c10_niid0.1_got --algorithm feded --local_epochs_ensemble 5 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 1 --train_batch_size 80
python3 main.py --settings_file cifar10_resnet_c10_niid0.1_hhk --algorithm feded --local_epochs_ensemble 5 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 1 --train_batch_size 80
python3 main.py --settings_file cifar10_resnet_c10_niid0.1_zkc --algorithm feded --local_epochs_ensemble 5 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 1 --train_batch_size 80
python3 main.py --settings_file cifar10_resnet_c10_niid0.1_kih --algorithm feded --local_epochs_ensemble 5 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 1 --train_batch_size 80
python3 main.py --settings_file cifar10_resnet_c10_niid0.1_sgk --algorithm feded --local_epochs_ensemble 5 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 1 --train_batch_size 80

python3 main.py --settings_file cifar10_resnet_c10_niid0.1_got --algorithm feded --local_epochs_ensemble 5 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 2 --autoencoder_epochs 100 --student_loss ce
python3 main.py --settings_file cifar10_resnet_c10_niid0.1_hhk --algorithm feded --local_epochs_ensemble 5 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 2 --autoencoder_epochs 100 --student_loss ce
python3 main.py --settings_file cifar10_resnet_c10_niid0.1_zkc --algorithm feded --local_epochs_ensemble 5 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 2 --autoencoder_epochs 100 --student_loss ce
python3 main.py --settings_file cifar10_resnet_c10_niid0.1_kih --algorithm feded --local_epochs_ensemble 5 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 2 --autoencoder_epochs 100 --student_loss ce
python3 main.py --settings_file cifar10_resnet_c10_niid0.1_sgk --algorithm feded --local_epochs_ensemble 5 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 2 --autoencoder_epochs 100 --student_loss ce

python3 main.py --settings_file cifar10_resnet_c10_niid0.1_got --algorithm feded --local_epochs_ensemble 5 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 2 --autoencoder_epochs 100 --student_loss mse
python3 main.py --settings_file cifar10_resnet_c10_niid0.1_hhk --algorithm feded --local_epochs_ensemble 5 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 2 --autoencoder_epochs 100 --student_loss mse
python3 main.py --settings_file cifar10_resnet_c10_niid0.1_zkc --algorithm feded --local_epochs_ensemble 5 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 2 --autoencoder_epochs 100 --student_loss mse
python3 main.py --settings_file cifar10_resnet_c10_niid0.1_kih --algorithm feded --local_epochs_ensemble 5 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 2 --autoencoder_epochs 100 --student_loss mse
python3 main.py --settings_file cifar10_resnet_c10_niid0.1_sgk --algorithm feded --local_epochs_ensemble 5 --student_model cifar10_resnet --public_data_sizes="500 1000 5000 10000 25000" --client_sample_fraction 0.8 --student_epochs 10 --weight_scheme 2 --autoencoder_epochs 100 --student_loss mse
