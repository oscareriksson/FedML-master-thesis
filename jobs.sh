#!/bin/sh

python3 main.py --settings_file mnist_c10_iid_a0.1_hrq --algorithm fedavg --local_epochs 1 --n_rounds 100
python3 main.py --settings_file mnist_c10_iid_a0.1_suu --algorithm fedavg --local_epochs 1 --n_rounds 100
python3 main.py --settings_file mnist_c10_iid_a0.1_zdr --algorithm fedavg --local_epochs 1 --n_rounds 100

python3 main.py --settings_file mnist_c10_niid_a0.1_jsc --algorithm fedavg --local_epochs 1 --n_rounds 100
python3 main.py --settings_file mnist_c10_niid_a0.1_exf --algorithm fedavg --local_epochs 1 --n_rounds 100
python3 main.py --settings_file mnist_c10_niid_a0.1_mjm --algorithm fedavg --local_epochs 1 --n_rounds 100

# python3 main.py --settings_file mnist_c10_iid_a0.1_hrq --algorithm feded --local_epochs 20 --n_rounds 1 --public_data_sizes="500 1000 5000 15000 30000" --client_sample_fraction 0.4 --student_epochs 30
# python3 main.py --settings_file mnist_c10_iid_a0.1_suu --algorithm feded --local_epochs 20 --n_rounds 1 --public_data_sizes="500 1000 5000 15000 30000" --client_sample_fraction 0.4 --student_epochs 30
# python3 main.py --settings_file mnist_c10_iid_a0.1_zdr --algorithm feded --local_epochs 20 --n_rounds 1 --public_data_sizes="500 1000 5000 15000 30000" --client_sample_fraction 0.4 --student_epochs 30

# python3 main.py --settings_file mnist_c10_niid_a0.1_jsc --algorithm feded --local_epochs 20 --n_rounds 1 --public_data_sizes="500 1000 5000 15000 30000" --client_sample_fraction 0.4 --student_epochs 30
# python3 main.py --settings_file mnist_c10_niid_a0.1_exf --algorithm feded --local_epochs 20 --n_rounds 1 --public_data_sizes="500 1000 5000 15000 30000" --client_sample_fraction 0.4 --student_epochs 30
# python3 main.py --settings_file mnist_c10_niid_a0.1_mjm --algorithm feded --local_epochs 20 --n_rounds 1 --public_data_sizes="500 1000 5000 15000 30000" --client_sample_fraction 0.4 --student_epochs 30
