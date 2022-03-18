#!/bin/sh

python3 main.py --settings_file mnist_cnn1_c10_iid_ljh --algorithm fedavg --local_epochs 1 --n_rounds 100
python3 main.py --settings_file mnist_cnn1_c10_iid_llw --algorithm fedavg --local_epochs 1 --n_rounds 100
python3 main.py --settings_file mnist_cnn1_c10_iid_uto --algorithm fedavg --local_epochs 1 --n_rounds 100

python3 main.py --settings_file mnist_cnn1_c10_niid0.1_anr --algorithm fedavg --local_epochs 1 --n_rounds 100
python3 main.py --settings_file mnist_cnn1_c10_niid0.1_bud --algorithm fedavg --local_epochs 1 --n_rounds 100
python3 main.py --settings_file mnist_cnn1_c10_niid0.1_bxf --algorithm fedavg --local_epochs 1 --n_rounds 100

python3 main.py --settings_file mnist_cnn1_c10_iid_ljh --algorithm feded --local_epochs_ensemble 20 --student_model mnist_cnn1 --public_data_sizes="500 1000 5000 15000 30000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
python3 main.py --settings_file mnist_cnn1_c10_iid_llw --algorithm feded --local_epochs_ensemble 20 --student_model mnist_cnn1 --public_data_sizes="500 1000 5000 15000 30000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
python3 main.py --settings_file mnist_cnn1_c10_iid_uto --algorithm feded --local_epochs_ensemble 20 --student_model mnist_cnn1 --public_data_sizes="500 1000 5000 15000 30000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0

python3 main.py --settings_file mnist_cnn1_c10_iid_ljh --algorithm feded --local_epochs_ensemble 20 --student_model mnist_cnn2 --public_data_sizes="500 1000 5000 15000 30000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
python3 main.py --settings_file mnist_cnn1_c10_iid_llw --algorithm feded --local_epochs_ensemble 20 --student_model mnist_cnn2 --public_data_sizes="500 1000 5000 15000 30000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
python3 main.py --settings_file mnist_cnn1_c10_iid_uto --algorithm feded --local_epochs_ensemble 20 --student_model mnist_cnn2 --public_data_sizes="500 1000 5000 15000 30000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0

python3 main.py --settings_file mnist_cnn1_c10_niid0.1_anr --algorithm feded --local_epochs_ensemble 20 --student_model mnist_cnn1 --public_data_sizes="500 1000 5000 15000 30000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
python3 main.py --settings_file mnist_cnn1_c10_niid0.1_bud --algorithm feded --local_epochs_ensemble 20 --student_model mnist_cnn1 --public_data_sizes="500 1000 5000 15000 30000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
python3 main.py --settings_file mnist_cnn1_c10_niid0.1_bxf --algorithm feded --local_epochs_ensemble 20 --student_model mnist_cnn1 --public_data_sizes="500 1000 5000 15000 30000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0

python3 main.py --settings_file mnist_cnn1_c10_niid0.1_anr --algorithm feded --local_epochs_ensemble 20 --student_model mnist_cnn2 --public_data_sizes="500 1000 5000 15000 30000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
python3 main.py --settings_file mnist_cnn1_c10_niid0.1_bud --algorithm feded --local_epochs_ensemble 20 --student_model mnist_cnn2 --public_data_sizes="500 1000 5000 15000 30000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
python3 main.py --settings_file mnist_cnn1_c10_niid0.1_bxf --algorithm feded --local_epochs_ensemble 20 --student_model mnist_cnn2 --public_data_sizes="500 1000 5000 15000 30000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0

python3 main.py --settings_file mnist_cnn1_c10_niid0.1_anr --algorithm feded --local_epochs_ensemble 20 --student_model mnist_cnn2 --public_data_sizes="500 1000 5000 15000 30000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 1
python3 main.py --settings_file mnist_cnn1_c10_niid0.1_bud --algorithm feded --local_epochs_ensemble 20 --student_model mnist_cnn2 --public_data_sizes="500 1000 5000 15000 30000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 1
python3 main.py --settings_file mnist_cnn1_c10_niid0.1_bxf --algorithm feded --local_epochs_ensemble 20 --student_model mnist_cnn2 --public_data_sizes="500 1000 5000 15000 30000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 1
