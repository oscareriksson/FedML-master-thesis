#!/bin/sh

python3 main.py --settings_file mnist_c10_iid_a0.1_kfd --algorithm fedavg --local_epochs 1 --n_rounds 100
python3 main.py --settings_file mnist_c10_iid_a0.1_odl --algorithm fedavg --local_epochs 1 --n_rounds 100
python3 main.py --settings_file mnist_c10_iid_a0.1_rac --algorithm fedavg --local_epochs 1 --n_rounds 100
python3 main.py --settings_file mnist_c10_niid_a0.1_kbj --algorithm fedavg --local_epochs 1 --n_rounds 100
python3 main.py --settings_file mnist_c10_niid_a0.1_wwl --algorithm fedavg --local_epochs 1 --n_rounds 100
python3 main.py --settings_file mnist_c10_niid_a0.1_zja --algorithm fedavg --local_epochs 1 --n_rounds 100

python3 main.py --settings_file mnist_c10_iid_a0.1_kfd --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 20000 30000
python3 main.py --settings_file mnist_c10_iid_a0.1_odl --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 20000 30000
python3 main.py --settings_file mnist_c10_iid_a0.1_rac --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 20000 30000
python3 main.py --settings_file mnist_c10_niid_a0.1_kbj --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 20000 30000
python3 main.py --settings_file mnist_c10_niid_a0.1_wwl --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 20000 30000
python3 main.py --settings_file mnist_c10_niid_a0.1_zja --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 20000 30000

python3 main.py --settings_file mnist_c10_iid_a0.1_kfd --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 20000 30000 --weight_scheme w1
python3 main.py --settings_file mnist_c10_iid_a0.1_odl --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 20000 30000 --weight_scheme w1
python3 main.py --settings_file mnist_c10_iid_a0.1_rac --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 20000 30000 --weight_scheme w1
python3 main.py --settings_file mnist_c10_niid_a0.1_kbj --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 20000 30000 --weight_scheme w1
python3 main.py --settings_file mnist_c10_niid_a0.1_wwl --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 20000 30000 --weight_scheme w1
python3 main.py --settings_file mnist_c10_niid_a0.1_zja --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 20000 30000 --weight_scheme w1

python3 main.py --settings_file cifar10_c10_iid_a0.1_cpp --algorithm fedavg --local_epochs 1 --n_rounds 100
python3 main.py --settings_file cifar10_c10_iid_a0.1_mwj --algorithm fedavg --local_epochs 1 --n_rounds 100
python3 main.py --settings_file cifar10_c10_iid_a0.1_zcv --algorithm fedavg --local_epochs 1 --n_rounds 100
python3 main.py --settings_file cifar10_c10_niid_a0.1_hop --algorithm fedavg --local_epochs 1 --n_rounds 100
python3 main.py --settings_file cifar10_c10_niid_a0.1_mpf --algorithm fedavg --local_epochs 1 --n_rounds 100
python3 main.py --settings_file cifar10_c10_niid_a0.1_tug --algorithm fedavg --local_epochs 1 --n_rounds 100

python3 main.py --settings_file cifar10_c10_iid_a0.1_cpp --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 25000
python3 main.py --settings_file cifar10_c10_iid_a0.1_mwj --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 25000
python3 main.py --settings_file cifar10_c10_iid_a0.1_zcv --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 25000
python3 main.py --settings_file cifar10_c10_niid_a0.1_hop --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 25000
python3 main.py --settings_file cifar10_c10_niid_a0.1_mpf --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 25000
python3 main.py --settings_file cifar10_c10_niid_a0.1_tug --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 25000

python3 main.py --settings_file cifar10_c10_iid_a0.1_cpp --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 25000 --weight_scheme w1
python3 main.py --settings_file cifar10_c10_iid_a0.1_mwj --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 25000 --weight_scheme w1
python3 main.py --settings_file cifar10_c10_iid_a0.1_zcv --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 25000 --weight_scheme w1
python3 main.py --settings_file cifar10_c10_niid_a0.1_hop --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 25000 --weight_scheme w1
python3 main.py --settings_file cifar10_c10_niid_a0.1_mpf --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 25000 --weight_scheme w1
python3 main.py --settings_file cifar10_c10_niid_a0.1_tug --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 100 --public_data_sizes 1000 5000 10000 25000 --weight_scheme w1
