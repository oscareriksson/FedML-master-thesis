import argparse
import importlib
import sys
import torch
from src.models.models import create_model


def load_data_loaders(args):
    assert args.dataset in ["mnist", "cifar10", "cifar100"], f"Chosen dataset is not available."
    module = importlib.import_module(f"src.datasets.{args.dataset}")
    data_class_ = getattr(module, args.dataset.title())

    dataset = data_class_(args.train_fraction)
    dataset.generate_client_data(args.n_clients, args.distribution, args.alpha)

    public_data_loader = None
    if args.algorithm in ["feded"]:
        dataset.split_test_public(args.n_samples_public)
        public_data_loader = dataset.get_public_data_loader(args.public_batch_size)

    client_data_loaders = dataset.get_train_data_loaders(args.train_batch_size)
    test_data_loader = dataset.get_test_data_loader(args.test_batch_size)

    return client_data_loaders, test_data_loader, public_data_loader

def create_server(alg, args, model, client_loaders, test_loader, public_loader):
    if alg == "fedavg":
        from src.algorithms.fedavg import FedAvgServer
        server = FedAvgServer(args, model, client_loaders, test_loader)

    elif alg == "fedprox":
        from src.algorithms.fedprox import FedProxServer
        server = FedProxServer(args, model, client_loaders, test_loader)

    elif alg == "feded":
        from src.algorithms.feded import FedEdServer
        server = FedProxServer(args, model, client_loaders, test_loader, public_loader)
        
    else:
        print("Chosen algorithm is not supported.")
        sys.exit()

    return server


def run_job(args, i):
    torch.manual_seed(i)
    client_data_loaders, test_data_loader, public_data_loader = load_data_loaders(args)
    model = create_model(args.model_name)

    algorithms = args.algorithm.split("+")
    for alg in algorithms:
        print(alg.upper(), flush=True)
        server = create_server(alg, args, model, client_data_loaders, test_data_loader, public_data_loader)

        server.run()



def main(args):
    for i in range(args.n_times):
        print(f"_________ Iteration {i+1} _________ ", flush=True)
        run_job(args, i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--algorithm", type=str, default="fedavg")
    parser.add_argument("--n_clients", type=int, default=2, help="Number of clients per round")
    parser.add_argument("--n_times", type=int, default=1)
    parser.add_argument("--n_rounds", type=int, default=1)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--train_fraction", type=float, default=0.1)
    parser.add_argument("--distribution", type=str, default="iid")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--model_name", type=str, default="mnist_cnn")
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Local learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Local momentum")
    parser.add_argument("--evaluate_train", type=bool, default=True, help="Do evaluation of local training")

    parser.add_argument("--n_samples_public", type=int, default=400)
    parser.add_argument("--public_batch_size", type=int, default=64)

    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm:                   {}".format(args.algorithm))
    print("Dataset:                     {}".format(args.dataset))
    print("Number of clients:           {}".format(args.n_clients))
    print("Number of global rounds:     {}".format(args.n_rounds))
    print("Number of local epochs:      {}".format(args.local_epochs))
    print("=" * 80)

    main(args)
