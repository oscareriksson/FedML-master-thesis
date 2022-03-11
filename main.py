import argparse
from cProfile import run
import importlib
import sys
import os
import torch
import numpy as np
import string
import random

def prepare_run_folder(args):
    run_folder = f"./results/run_{args.weight_scheme}_{args.algorithm}_{args.settings_id}"
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
    return run_folder

def save_run_settings(args, run_folder):
    with open(f"{run_folder}/run_settings", "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg} {value}")
            f.write("\n")


def load_data_loaders(args, local_indices, public_indices):
    assert args.dataset in ["mnist", "cifar10", "cifar100"], f"Chosen dataset is not available."
    module = importlib.import_module(f"src.datasets.{args.dataset}")
    data_class_ = getattr(module, args.dataset.title())

    dataset = data_class_(args.num_workers)

    dataset.set_local_sets_indices(local_indices)
    dataset.set_public_indices(public_indices)

    client_data_loaders = dataset.get_train_data_loaders(args.train_batch_size)
    test_data_loader = dataset.get_test_data_loader(args.test_batch_size)
    public_data_loader = dataset.get_public_data_loader(args.public_batch_size)

    return client_data_loaders, test_data_loader, public_data_loader


def create_server(args, model, client_loaders, test_loader, public_loader, run_folder):
    if args.algorithm == "fedavg":
        from src.algorithms.fedavg import FedAvgServer
        server = FedAvgServer(args, model, run_folder, client_loaders, test_loader)

    elif args.algorithm == "fedprox":
        from src.algorithms.fedprox import FedProxServer
        server = FedProxServer(args, model, run_folder, client_loaders, test_loader)

    elif args.algorithm == "feded":
        from src.algorithms.feded import FedEdServer
        server = FedEdServer(args, model, run_folder, client_loaders, test_loader, public_loader)
        
    else:
        print("Chosen algorithm is not supported.")
        sys.exit()

    return server


def main(args):
    run_folder = prepare_run_folder(args)
    save_run_settings(args, run_folder)

    settings_path = f"./settings/{args.settings_file}"
    model = torch.load(f"{settings_path}/model")
    
    local_indices = []
    with open(f'{settings_path}/data_splits.npy', 'rb') as f:
        public_indices = np.load(f)
        try:
            while True:
                local_indices.append(np.load(f))
        except:
            print("")

    client_data_loaders, test_data_loader, public_data_loader = load_data_loaders(args, local_indices, public_indices)

    server = create_server(args, model, client_data_loaders, test_data_loader, public_data_loader, run_folder)
    server.run()

    print(f"Results saved in: {run_folder}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings_file", type=str, default="mnist_c10_iid_a0.1_dyx")
    parser.add_argument("--algorithm", type=str, default="feded")
    parser.add_argument("--n_rounds", type=int, default=1)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Local learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Local momentum")
    parser.add_argument("--num_workers", type=int, default=4)

    # Ensemble parameters
    parser.add_argument("--local_epochs_ensemble", type=int, default=1)
    parser.add_argument("--public_batch_size", type=int, default=64)
    parser.add_argument("--student_batch_size", type=int, default=64)
    parser.add_argument("--student_epochs", type=int, default=1)
    parser.add_argument("--public_data_sizes", type=str, default="1000 3000 5000 7000")
    parser.add_argument("--weight_scheme", type=str, default="w0")


    args = parser.parse_args()

    init_data = args.settings_file.split("_")
    dargs = vars(args)
    dargs['dataset'] = init_data[0]
    dargs['n_clients'] = int(init_data[1][1:])
    dargs['distribution'] = init_data[2]
    dargs['alpha'] = float(init_data[3][1:])
    dargs['settings_id'] = init_data[4]

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm:                   {}".format(args.algorithm))
    print("Dataset:                     {}".format(args.dataset))
    print("Number of clients:           {}".format(args.n_clients))
    print("Number of global rounds:     {}".format(args.n_rounds))
    print("=" * 80)

    main(args)
