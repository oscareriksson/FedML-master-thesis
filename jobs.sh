#!/bin/sh

python3 main.py --settings_file mnist_c10_f1.0_niid_a0.1_yft --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 150
python3 main.py --settings_file mnist_c10_f1.0_iid_a0.1_hjp --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 150
python3 main.py --settings_file mnist_c10_f1.0_iid_a0.1_ppi --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 150
python3 main.py --settings_file mnist_c10_f1.0_iid_a0.1_qrc --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 150