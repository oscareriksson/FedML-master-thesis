import argparse
from datasets.mnist import Mnist
from algorithms.fedavg_server import FedAvgServer


def load_data(dataset, train_fraction):
    if "mnist" in dataset:
        dataset = Mnist(train_fraction)
    elif "cifar10" in dataset:
        pass
    return dataset


def create_server(args, client_loaders, test_loader):
    if "fedavg" in args.algorithm:
        server = FedAvgServer(args, client_loaders, test_loader)
    elif "fedprox" in args.algorithm:
        pass
    return server


def main(args):
    dataset = load_data(args.dataset, args.train_fraction)
    _, client_data_loaders = dataset.get_train_data_loaders(args.n_clients, args.distribution, args.alpha, args.train_batch_size)
    test_data_loader = dataset.get_test_data_loader(args.test_batch_size)

    server = create_server(args, client_data_loaders, test_data_loader)

    for i in range(args.n_rounds):
        print("-- Round {} --".format(i+1))
        server.run()
        server.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--train_fraction", type=float, default=0.1)
    parser.add_argument("--distribution", type=str, default="iid")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--model_name", type=str, default="cnn")
    parser.add_argument("--algorithm", type=str, default="fedavg")
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Local learning rate")
    parser.add_argument("--n_clients", type=int, default=5, help="Number of clients per round")
    parser.add_argument("--n_rounds", type=int, default=2)
    parser.add_argument("--local_epochs", type=int, default=1)

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
