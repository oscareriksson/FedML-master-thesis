import argparse
import importlib
import sys
import os
import torch
import numpy as np
import random
from src.models.models import create_model


def prepare_run_folder(args):
    folder = f"{args.local_model}_c{args.n_clients}_{args.distribution}{args.alpha}_s{args.seed}"
    run_folder = f"./results/{args.dataset}/{args.algorithm}/{folder}"
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
    return run_folder

def save_run_settings(args, run_folder):
    with open(f"{run_folder}/run_settings", "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg} {value}")
            f.write("\n")


def initialize_data(args):
    assert args.dataset in ["mnist", "cifar10", "emnist"], f"Chosen dataset is not available."
    module = importlib.import_module(f"src.datasets.{args.dataset}")
    data_class_ = getattr(module, args.dataset.title())

    dataset = data_class_(num_workers=0, public_fraction=args.public_fraction)
    dataset.generate_client_data(args.n_clients, args.distribution, args.alpha)

    local_indices, public_indices = dataset.get_local_sets_indices(), dataset.get_public_indices()

    return local_indices, public_indices


def load_data_loaders(args, local_indices, public_indices):
    assert args.dataset in ["mnist", "cifar10", "emnist"], f"Chosen dataset is not available."
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model = create_model(args.local_model)
    
    print("Initializing client data ...", flush=True)
    local_indices, public_indices = initialize_data(args)
    print("Done.", flush=True)

    client_data_loaders, test_data_loader, public_data_loader = load_data_loaders(args, local_indices, public_indices)

    run_folder = prepare_run_folder(args)
    save_run_settings(args, run_folder)

    print("Creating server ...", flush=True)
    server = create_server(args, model, client_data_loaders, test_data_loader, public_data_loader, run_folder)
    print("Done.", flush=True)

    print("Starting federated training")
    server.run()

    print(f"Results saved in: {run_folder}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Federated settings
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--n_clients", type=int, default=2, help="Number of clients per round")
    parser.add_argument("--public_fraction", type=float, default=0.8)
    parser.add_argument("--distribution", type=str, default="niid")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--algorithm", type=str, default="feded")

    # Training settings
    parser.add_argument("--local_model", type=str, default="mnist_cnn1")
    parser.add_argument("--n_rounds", type=int, default=1)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Local learning rate")
    parser.add_argument("--num_workers", type=int, default=4)

    # FedED settings
    parser.add_argument("--student_models", type=str, default="mnist_cnn1 mnist_cnn2")
    parser.add_argument("--local_epochs_ensemble", type=int, default=1)
    parser.add_argument("--momentum", type=float, default=0.9, help="Local momentum")
    parser.add_argument("--client_sample_fraction", type=float, default=0.4)
    parser.add_argument("--public_batch_size", type=int, default=64)
    parser.add_argument("--student_batch_size", type=int, default=50)
    parser.add_argument("--student_epochs", type=int, default=1)
    parser.add_argument("--student_epochs_w2", type=int, default=1)
    parser.add_argument("--public_data_sizes", type=str, default="1000 1000")
    parser.add_argument("--weight_schemes", type=str, default="0")
    parser.add_argument("--autoencoder_epochs", type=int, default=1)
    parser.add_argument("--student_lr", type=float, default=1e-3)
    parser.add_argument("--student_lr_w2", type=float, default=1e-3)
    parser.add_argument("--student_loss", type=str, default="mse")


    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm:                   {}".format(args.algorithm))
    print("Seed:                        {}".format(args.algorithm))
    print("Dataset:                     {}".format(args.dataset))
    print("Number of clients:           {}".format(args.n_clients))
    print("Alpha:                       {}".format(args.alpha))
    print("=" * 80)

    main(args)
