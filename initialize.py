import argparse
import importlib
import os
from src.models.models import create_model
import torch
import numpy as np
import string
import random


def prepare_settings_folder(path_name):
    while True:
        settings_folder = f"{path_name}_{''.join(random.choice(string.ascii_lowercase) for _ in range(3))}"
        if not os.path.exists(settings_folder):
            os.makedirs(settings_folder)
            break
    return settings_folder

def initialize_data(args):
    assert args.dataset in ["mnist", "cifar10", "cifar100"], f"Chosen dataset is not available."
    module = importlib.import_module(f"src.datasets.{args.dataset}")
    data_class_ = getattr(module, args.dataset.title())

    dataset = data_class_(num_workers=0, train_fraction=args.train_fraction)
    dataset.generate_client_data(args.n_clients, args.distribution, args.alpha)
    local_indices = dataset.get_local_sets_indices()

    dataset.split_test_public()
    test_indices, public_indices = dataset.get_test_indices(), dataset.get_public_indices()

    return local_indices, test_indices, public_indices



def main(args):
    path_name = f"./settings/{args.dataset}_c{args.n_clients}_f{args.train_fraction}_{args.distribution}_a{args.alpha}"

    settings_folder = prepare_settings_folder(path_name)
    
    init_model = create_model(args.model_name)
    torch.save(init_model, f"{settings_folder}/model")
    
    local_indices, test_indices, public_indices = initialize_data(args)

    with open(f'{settings_folder}/data_splits.npy', 'wb') as f:
        np.save(f, test_indices)
        np.save(f, public_indices)
        for indices in local_indices:
            np.save(f, indices)
    
    print(f"Saved settings in: {settings_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--model_name", type=str, default="mnist_cnn")
    parser.add_argument("--n_clients", type=int, default=10, help="Number of clients per round")
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--distribution", type=str, default="niid")
    parser.add_argument("--alpha", type=float, default=0.1)

    args = parser.parse_args()

    main(args)