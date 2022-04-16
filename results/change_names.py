import os
from pathlib import Path
import shutil

dir = '/home/oscar/CAS/Exjobb/FedML-master-thesis/results/mnist/feded/temp'

for res_dir in os.listdir(dir):
    info = res_dir.split('_')
    student = f"mnist_{info[5]}"
    loss = info[6]
    scheme = info[7]

    new_dir = f"/home/oscar/CAS/Exjobb/FedML-master-thesis/results/mnist/feded/{'_'.join(info[:5])}"
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    res_dir = os.path.join(dir, res_dir)

    for filename in os.listdir(res_dir):
        file_path = os.path.join(res_dir, filename)

        if filename in ["client_accuracy.npy", "client_loss.npy", "run_settings"]:
            shutil.copyfile(file_path, os.path.join(new_dir, filename))
        
        if filename == "ensemble_test_acc.npy":
            shutil.copyfile(file_path, os.path.join(new_dir, f"{scheme}_ensemble_test_acc.npy"))
        
        if filename[:12] == "student_test":
            npub = filename.split('_')[3]
            new_filename = f"{scheme}_student_{student}_{loss}_test_results_{npub}"
            shutil.copyfile(file_path, os.path.join(new_dir, new_filename))

        if filename[:12] == "student_trai":
            npub = filename[21:]
            new_filename = f"{scheme}_student_{student}_{loss}_train_results_{npub}"
            shutil.copyfile(file_path, os.path.join(new_dir, new_filename))

