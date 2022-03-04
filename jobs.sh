#!/bin/sh
python3 main.py --settings_file mnist_c10_f1.0_iid_a0.1_npub7000_tnx --algorithm fedavg --local_epochs 1 --n_rounds 30

python3 main.py --settings_file mnist_c10_f1.0_niid_a0.1_npub1000_yjc --algorithm fedavg --local_epochs 1 --n_rounds 30
python3 main.py --settings_file mnist_c10_f1.0_niid_a0.1_npub3000_pbh --algorithm fedavg --local_epochs 1 --n_rounds 30
python3 main.py --settings_file mnist_c10_f1.0_niid_a0.1_npub5000_zjv --algorithm fedavg --local_epochs 1 --n_rounds 30
python3 main.py --settings_file mnist_c10_f1.0_niid_a0.1_npub7000_lex --algorithm fedavg --local_epochs 1 --n_rounds 30

python3 main.py --settings_file mnist_c10_f1.0_iid_a0.1_npub1000_bze --algorithm feded --local_epochs_ensemble 15 --student_epochs 150 --student_batch_size 64
python3 main.py --settings_file mnist_c10_f1.0_iid_a0.1_npub3000_hbe --algorithm feded --local_epochs_ensemble 15 --student_epochs 150 --student_batch_size 64
python3 main.py --settings_file mnist_c10_f1.0_iid_a0.1_npub5000_eqt --algorithm feded --local_epochs_ensemble 15 --student_epochs 150 --student_batch_size 64
python3 main.py --settings_file mnist_c10_f1.0_iid_a0.1_npub7000_tnx --algorithm feded --local_epochs_ensemble 15 --student_epochs 150 --student_batch_size 64

python3 main.py --settings_file mnist_c10_f1.0_niid_a0.1_npub1000_yjc --algorithm feded --local_epochs_ensemble 15 --student_epochs 150 --student_batch_size 64
python3 main.py --settings_file mnist_c10_f1.0_niid_a0.1_npub3000_pbh --algorithm feded --local_epochs_ensemble 15 --student_epochs 150 --student_batch_size 64
python3 main.py --settings_file mnist_c10_f1.0_niid_a0.1_npub5000_zjv --algorithm feded --local_epochs_ensemble 15 --student_epochs 150 --student_batch_size 64
python3 main.py --settings_file mnist_c10_f1.0_niid_a0.1_npub7000_lex --algorithm feded --local_epochs_ensemble 15 --student_epochs 150 --student_batch_size 64