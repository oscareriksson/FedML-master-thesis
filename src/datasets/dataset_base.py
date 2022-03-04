import importlib
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Subset

class PytorchDataset:
    """Base class defining pytorch datasets."""
    def __init__(self, dataset_name) -> None:
        """ Constructor method.

            Parameters:
            dataset_name    (str): Name of dataset to use.
        """
        module = importlib.import_module(f"torchvision.datasets")
        self.data_class = getattr(module, dataset_name)
        self.client_data_indices = []
        self.train_data = None
        self.test_data = None
        self.public_train_data = None
        self.public_val_data = None


    def _sample_train_data(self, train_fraction, transform):
        """ Sample a chosen fraction of the training dataset to use.

            Parameters:
            train_fraction  (float): Fraction of training data to use.
            transform       (torchvision.transforms.Compose): Collection of transforms to apply on data.

            Returns:
            torch.utils.data.Subset: Subset of training data.
        """
        train_data = self.data_class(
            root='data', 
            train=True, 
            transform=transform, 
            download=True)
        if train_fraction is None:
            return train_data
        else:
            n_samples = len(train_data.targets)
            index_limit = int(train_fraction * n_samples)
            chosen_indices = np.random.choice(torch.arange(n_samples), size=index_limit, replace=False)
            print(f"\nUsing {index_limit} training samples\n", flush=True)
            return Subset(train_data, chosen_indices)
        

    def generate_client_data(self, n_clients, distribution, alpha):
        """ Generate iid client data or non-iid by sampling from Dirichlet distribution.

            Parameters:
            n_clients       (int): Number of clients.
            distribution    (str): Indicator to sample iid or non-iid.
            alpha           (float): Concentration parameter for Dirichlet distribution.
        """
        labels = np.array([self.train_data[i][1] for i in range(len(self.train_data))])
        n_classes = len(np.unique(labels))
        partition_matrix = np.ones((n_classes, n_clients))

        # iid: Sample from each class until no samples left.
        if distribution == "iid":
            partition_matrix /= n_clients
            local_sets_indices = [np.array([], dtype=int) for _ in range(n_clients)]
            clients_iter = np.arange(n_clients)

            for i in range(n_classes):
                class_indices = np.where(labels == i)[0]
                
                clients_iter = clients_iter[::-1]
                samples_left = True
                while samples_left:
                    for j in clients_iter:
                        if len(class_indices) == 0:
                            samples_left = False
                            break
                        else:
                            sample_idx = np.random.choice(len(class_indices))
                            local_sets_indices[j] = np.append(local_sets_indices[j], self.train_data.indices[class_indices[sample_idx]])
                            class_indices = np.delete(class_indices, sample_idx)

        # non-iid: Sample from dirichlet distribution.
        else:
            class_indices = []
            for i in range(n_classes):
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
                    local_size = int(partition_matrix[each_class, client] * sample_size)
                    local_sets_indices[client] += list(self.train_data.indices[class_indices[each_class][:local_size]])
                    class_indices[each_class] = class_indices[each_class][local_size:]

        self.local_sets_indices = local_sets_indices

    def split_test_public(self, n_samples, val_fraction):
        labels = np.array([y for (_, y) in self.test_data])
        n_classes = len(np.unique(labels))
        n_samples_per_class = int(n_samples / n_classes)
        all_indices = np.arange(len(labels))

        public_set_indices = []

        for i in range(n_classes):
            class_indices = np.where(labels == i)[0]
            chosen_indices = list(np.random.choice(class_indices, n_samples_per_class, replace=False))
            public_set_indices.extend(chosen_indices)

        # Fill up to n_samples.
        samples_left = n_samples - len(public_set_indices)
        indices_left = [x for x in all_indices if x not in public_set_indices]
        chosen_indices = list(np.random.choice(indices_left, samples_left, replace=False))
        public_set_indices.extend(chosen_indices)

        # Split into train and validation.
        idx_limit = int(val_fraction * n_samples)
        random.shuffle(public_set_indices)
        public_train_indices = public_set_indices[idx_limit:]
        public_val_indices = public_set_indices[:idx_limit]

        test_set_indices = [x for x in all_indices if x not in public_set_indices]

        self.public_train_data = Subset(self.test_data, public_train_indices)
        self.public_val_data = Subset(self.test_data, public_val_indices)
        self.test_data = Subset(self.test_data, test_set_indices)


    def get_train_data_loaders(self, batch_size):
        """ Get list of client training data loaders.

            Parameters:
            n_clients       (int): Number of clients.
            distribution    (str): iid/non-iid distributed data.
            alpha           (float): Concentration parameter for dirichlet distribution.
            batch_size      (int): Batch size for loading training data.

            Returns List[torch.utils.data.DataLoader]
        """
        client_data_loaders = []
        for client_indices in self.local_sets_indices:
                np.random.shuffle(client_indices)
                client_data_loaders.append(DataLoader(Subset(self.train_data.dataset, client_indices), batch_size))
        return client_data_loaders
    
    def get_test_data_loader(self, batch_size):
        """ Get test data loader.

            Parameters:
            batch_size      (int): Batch size for loading test data.
        """
        return DataLoader(self.test_data, batch_size)

    def get_public_data_loaders(self, batch_size):
        """ Get test data loader.

            Parameters:
            batch_size      (int): Batch size for loading test data.
        """
        return DataLoader(self.public_train_data, batch_size), DataLoader(self.public_val_data, batch_size)
    
    def get_local_sets_indices(self):
        """
        """
        return self.local_sets_indices

    def get_test_indices(self):
        """
        """
        return self.test_data.indices

    def get_public_indices(self):
        """
        """
        return self.public_train_data.indices, self.public_val_data.indices

    def set_local_sets_indices(self, local_sets_indices):
        """
        """
        self.local_sets_indices = local_sets_indices
    
    def set_test_indices(self, test_indices):
        """
        """
        self.test_data.indices = test_indices

    def set_public_indices(self, train_indices, val_indices):
        """
        """
        self.public_train_data.indices = train_indices
        self.public_val_data.indices = val_indices
    