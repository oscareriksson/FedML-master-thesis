from .dataset_base import DatasetBase
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np


class Cifar10(DatasetBase):
    """ Mnist dataset class.
    """
    def __init__(self, train_fraction):
        """ Constructor method.

            Parameters:
            train_fraction (float): Fraction of training data to use.
        """

        transform = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.test_data = CIFAR10(
            root='data', 
            train=False, 
            transform=transform, 
            download=True)

        self.train_data = self._sample_train_data(train_fraction, transform)
        self.n_samples = len(self.train_data)

    def _sample_train_data(self, train_fraction, transform):
        """ Sample a chosen fraction of the training dataset to use.

            Parameters:
            train_fraction  (float): Fraction of training data to use.
            transform       (torchvision.transforms.Compose): Collection of transforms to apply on data.

            Returns:
            torch.utils.data.Subset: Subset of training data.
        """
        train_data = CIFAR10(
            root='data', 
            train=True, 
            transform=transform, 
            download=True)
        n_samples = len(train_data.targets)
        index_limit = int(train_fraction * n_samples)
        chosen_indices = np.random.choice(torch.arange(n_samples), size=index_limit, replace=False)
        print(f"\nUsing {index_limit} training samples\n", flush=True)

        return Subset(train_data, chosen_indices)

    def get_train_data_loaders(self, n_clients, distribution, alpha, batch_size):
        """ Get list of client training data loaders sampled from dirichlet distribution.

            Parameters:
            n_clients       (int): Number of clients.
            distribution    (str): iid/non-iid distributed data.
            alpha           (float): Concentration parameter for dirichlet distribution.
            batch_size      (int): Batch size for loading training data.

            Returns List[torch.utils.data.DataLoader]
        """
        labels = [y for (_, y) in self.train_data]
        n_classes = len(np.unique(labels))
        partition_matrix = np.ones((n_classes, n_clients))

        # iid
        if distribution == "iid":
            partition_matrix /= n_clients
            size = int(np.floor(len(labels)/n_clients))
            client_data_loaders = []
            client_indices = np.random.choice(torch.arange(self.n_samples), size=(n_clients, size), replace=False)
            for i in range(n_clients):
                client_data_loaders.append(DataLoader(Subset(self.train_data, client_indices[i]), batch_size))
        # non-iid
        else:
            class_indices = []
            for i in range(10):
                class_indices.append(np.array(range(len(labels)))[labels == i])
            valid_pm = False
            while not valid_pm:
                partition_matrix = np.random.dirichlet((alpha, )*n_clients, n_classes)
                valid_pm = all(np.sum(partition_matrix, axis=0) > 0.01)

            local_sets_indices = [[] for _ in range(n_clients)]
            for each_class in range(n_classes):
                sample_size = len(class_indices[each_class])
                for client in range(n_clients):
                    np.random.shuffle(class_indices[each_class])
                    local_size = int(np.floor(partition_matrix[each_class, client] * sample_size))
                    local_sets_indices[client] += list(class_indices[each_class][:local_size])
                    class_indices[each_class] = class_indices[each_class][local_size:]

            client_data_loaders = []
            for client_indices in local_sets_indices:
                np.random.shuffle(client_indices)
                client_data_loaders.append(DataLoader(Subset(self.train_data, client_indices), batch_size))

        return partition_matrix, client_data_loaders
    
    def get_test_data_loader(self, batch_size):
        """ Get test data loader.

            Parameters:
            batch_size      (int): Batch size for loading test data.
        """
        return DataLoader(self.test_data, batch_size)