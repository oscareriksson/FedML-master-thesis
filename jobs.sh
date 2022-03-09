#!/bin/sh

# python3 main.py --settings_file mnist_c10_f1.0_iid_a0.1_ooy --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 150
# python3 main.py --settings_file mnist_c10_f1.0_iid_a0.1_oup --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 150
# python3 main.py --settings_file mnist_c10_f1.0_iid_a0.1_tqr --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 150
# python3 main.py --settings_file mnist_c10_f1.0_iid_a0.1_txx --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 150
# python3 main.py --settings_file mnist_c10_f1.0_iid_a0.1_udi --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 150

# python3 main.py --settings_file mnist_c10_f1.0_niid_a0.1_fyd --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 150
# python3 main.py --settings_file mnist_c10_f1.0_niid_a0.1_kne --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 150
# python3 main.py --settings_file mnist_c10_f1.0_niid_a0.1_mhg --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 150
# python3 main.py --settings_file mnist_c10_f1.0_niid_a0.1_yhm --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 150
# python3 main.py --settings_file mnist_c10_f1.0_niid_a0.1_yjr --algorithm feded --local_epochs_ensemble 30 --n_rounds 1 --student_epochs 150

# python3 main.py --settings_file mnist_c10_f1.0_iid_a0.1_ooy --algorithm fedavg --local_epochs 1 --n_rounds 50
# python3 main.py --settings_file mnist_c10_f1.0_iid_a0.1_oup --algorithm fedavg --local_epochs 1 --n_rounds 50
# python3 main.py --settings_file mnist_c10_f1.0_iid_a0.1_tqr --algorithm fedavg --local_epochs 1 --n_rounds 50
# python3 main.py --settings_file mnist_c10_f1.0_iid_a0.1_txx --algorithm fedavg --local_epochs 1 --n_rounds 50
# python3 main.py --settings_file mnist_c10_f1.0_iid_a0.1_udi --algorithm fedavg --local_epochs 1 --n_rounds 50

# python3 main.py --settings_file mnist_c10_f1.0_niid_a0.1_fyd --algorithm fedavg --local_epochs 1 --n_rounds 50
# python3 main.py --settings_file mnist_c10_f1.0_niid_a0.1_kne --algorithm fedavg --local_epochs 1 --n_rounds 50
# python3 main.py --settings_file mnist_c10_f1.0_niid_a0.1_mhg --algorithm fedavg --local_epochs 1 --n_rounds 50
# python3 main.py --settings_file mnist_c10_f1.0_niid_a0.1_yhm --algorithm fedavg --local_epochs 1 --n_rounds 50
# python3 main.py --settings_file mnist_c10_f1.0_niid_a0.1_yjr --algorithm fedavg --local_epochs 1 --n_rounds 50

python3 main.py --settings_file cifar10_c10_f1.0_iid_a0.1_eft --algorithm fedavg --local_epochs 1 --n_rounds 150
python3 main.py --settings_file cifar10_c10_f1.0_iid_a0.1_gmg --algorithm fedavg --local_epochs 1 --n_rounds 150
python3 main.py --settings_file cifar10_c10_f1.0_iid_a0.1_mfb --algorithm fedavg --local_epochs 1 --n_rounds 150
python3 main.py --settings_file cifar10_c10_f1.0_iid_a0.1_xmo --algorithm fedavg --local_epochs 1 --n_rounds 150
python3 main.py --settings_file cifar10_c10_f1.0_iid_a0.1_ygi --algorithm fedavg --local_epochs 1 --n_rounds 150