import argparse
import importlib
from src.algorithms.fedavg import FedAvgServer
from src.algorithms.fedprox import FedProxServer


def load_data(dataset, train_fraction):
    assert dataset in ["mnist", "cifar10"], f"Chosen dataset is not available."
    module = importlib.import_module(f"src.datasets.{dataset}")
    data_class_ = getattr(module, dataset.title())
    return data_class_(train_fraction)

def create_server(args, client_loaders, test_loader):
    if "fedavg" in args.algorithm:
        server = FedAvgServer(args, client_loaders, test_loader)
    elif "fedprox" in args.algorithm:
        server = FedProxServer(args, client_loaders, test_loader, mu=args.mu)
    return server


def main(args):
    dataset = load_data(args.dataset, args.train_fraction)
    partition_matrix, client_data_loaders = dataset.get_train_data_loaders(args.n_clients, args.distribution, args.alpha, args.train_batch_size)
    test_data_loader = dataset.get_test_data_loader(args.test_batch_size)

    server = create_server(args, client_data_loaders, test_data_loader)

    for i in range(args.n_rounds):
        print("-- Round {} --".format(i+1), flush=True)
        server.run()
        server.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--algorithm", type=str, default="fedavg")
    parser.add_argument("--n_clients", type=int, default=5, help="Number of clients per round")
    parser.add_argument("--n_rounds", type=int, default=3)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--train_fraction", type=float, default=0.1)
    parser.add_argument("--distribution", type=str, default="niid")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--model_name", type=str, default="mnist_cnn")
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Local learning rate")

    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm:                   {}".format(args.algorithm))
    print("Dataset:                     {}".format(args.dataset))
    print("Number of clients:           {}".format(args.n_clients))
    print("Number of global rounds:     {}".format(args.n_rounds))
    print("Number of local epochs:      {}".format(args.local_epochs))
    print("Training batch size:         {}".format(args.train_batch_size))
    print("Learning rate:               {}".format(args.learning_rate))
    print("=" * 80)

    main(args)
