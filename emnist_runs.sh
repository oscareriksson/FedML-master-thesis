#!/bin/sh

# python3 main.py --settings_file emnist_cnn1_c10_iid_amu --algorithm fedavg --local_epochs 1 --n_rounds 100
# python3 main.py --settings_file emnist_cnn1_c10_iid_giu --algorithm fedavg --local_epochs 1 --n_rounds 100
# python3 main.py --settings_file emnist_cnn1_c10_iid_qvt --algorithm fedavg --local_epochs 1 --n_rounds 100
# python3 main.py --settings_file emnist_cnn1_c10_iid_wdz --algorithm fedavg --local_epochs 1 --n_rounds 100
# python3 main.py --settings_file emnist_cnn1_c10_iid_kfa --algorithm fedavg --local_epochs 1 --n_rounds 100

# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_xjr --algorithm fedavg --local_epochs 1 --n_rounds 100
# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_zex --algorithm fedavg --local_epochs 1 --n_rounds 100
# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_zft --algorithm fedavg --local_epochs 1 --n_rounds 100
# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_nod --algorithm fedavg --local_epochs 1 --n_rounds 100
# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_aln --algorithm fedavg --local_epochs 1 --n_rounds 100

# python3 main.py --settings_file emnist_cnn1_c10_iid_amu --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn1 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
# python3 main.py --settings_file emnist_cnn1_c10_iid_giu --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn1 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
# python3 main.py --settings_file emnist_cnn1_c10_iid_qvt --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn1 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
# python3 main.py --settings_file emnist_cnn1_c10_iid_wdz --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn1 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
# python3 main.py --settings_file emnist_cnn1_c10_iid_kfa --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn1 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0

# python3 main.py --settings_file emnist_cnn1_c10_iid_amu --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
# python3 main.py --settings_file emnist_cnn1_c10_iid_giu --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
# python3 main.py --settings_file emnist_cnn1_c10_iid_qvt --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
# python3 main.py --settings_file emnist_cnn1_c10_iid_wdz --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
# python3 main.py --settings_file emnist_cnn1_c10_iid_kfa --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0

# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_xjr --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn1 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_zex --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn1 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_zft --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn1 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_nod --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn1 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_aln --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn1 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0

# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_xjr --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_zex --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_zft --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_nod --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0
# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_aln --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 0

# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_xjr --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 1
# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_zex --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 1
# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_zft --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 1
# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_nod --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 1
# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_aln --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 30 --weight_scheme 1

# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_xjr --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 100 --weight_scheme 2 --autoencoder_epochs 30 --student_lr 0.00001
# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_zex --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 100 --weight_scheme 2 --autoencoder_epochs 30 --student_lr 0.00001
# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_zft --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 100 --weight_scheme 2 --autoencoder_epochs 30 --student_lr 0.00001
# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_nod --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 100 --weight_scheme 2 --autoencoder_epochs 30 --student_lr 0.00001
# python3 main.py --settings_file emnist_cnn1_c10_niid0.1_aln --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 100 --weight_scheme 2 --autoencoder_epochs 30 --student_lr 0.00001

python3 main.py --settings_file emnist_cnn1_c10_niid0.1_xjr --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 100 --weight_scheme 2 --autoencoder_epochs 30 --student_lr 0.00001 --student_loss ce
python3 main.py --settings_file emnist_cnn1_c10_niid0.1_zex --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 100 --weight_scheme 2 --autoencoder_epochs 30 --student_lr 0.00001 --student_loss ce
python3 main.py --settings_file emnist_cnn1_c10_niid0.1_zft --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 100 --weight_scheme 2 --autoencoder_epochs 30 --student_lr 0.00001 --student_loss ce
python3 main.py --settings_file emnist_cnn1_c10_niid0.1_nod --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 100 --weight_scheme 2 --autoencoder_epochs 30 --student_lr 0.00001 --student_loss ce
python3 main.py --settings_file emnist_cnn1_c10_niid0.1_aln --algorithm feded --local_epochs_ensemble 50 --student_model emnist_cnn2 --public_data_sizes="500 1000 5000 30000 60000" --client_sample_fraction 0.8 --student_epochs 100 --weight_scheme 2 --autoencoder_epochs 30 --student_lr 0.00001 --student_loss ce
