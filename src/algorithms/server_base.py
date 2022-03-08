from types import new_class
from ..models.models import create_model
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import copy
import numpy as np

class ServerBase(ABC):
    """Abstract base class defining datasets."""
    def __init__(self, args, model, run_folder, train_loaders, test_loader, public_loader=None) -> None:
        """ Constructor method.

            Parameters:
            args            (dict): Settings for federated setup and local training.
            model           (torch.nn.Module): Initialized global model.
            train_loaders   (list): List of client data loaders.
            test_loader     (torch.util.data.DataLoader): Data loader for test data.
            public_loader     (torch.util.data.DataLoader): Data loader for public data.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = copy.deepcopy(model).to(self.device)
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.public_loader = public_loader
        self.local_model = None
        self.dataset_name = args.dataset
        self.n_classes = len(np.unique(self.test_loader.dataset.targets))
        self.n_clients = args.n_clients
        self.n_rounds = args.n_rounds
        self.lr_rate = args.learning_rate
        self.momentum = args.momentum
        self.num_workers = args.num_workers
        self.local_epochs = args.local_epochs
        self.loss_function = nn.CrossEntropyLoss()
        self.local_sets_indices = [self.train_loaders[i].dataset.indices for i in range(self.n_clients)]
        self.label_count_matrix = np.array([[torch.sum(self.train_loaders[0].dataset.dataset.targets[self.local_sets_indices[i]] == c) for c in range(self.n_classes)] for i in range(self.n_clients)])
        self.n_samples_client = [len(data_loader.dataset) for data_loader in train_loaders]
        #self.n_samples_total = sum(self.n_samples_client)
        self.round_nr = 0
        self.run_folder = run_folder

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def _local_training(self, client_nr):
        pass

    def evaluate(self, model, data_loader):
        """ Evaluate global model on test dataset.
        
        """
        model.eval()
        loss = []
        correct = 0
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = model(x)
                error = self.loss_function(output, y)
                loss.append(error.item())
                _, pred = torch.max(output.data, 1)
                correct += (pred == y).sum().item()
        avg_loss = sum(loss) / len(loss)
        accuracy = 100. * correct / len(data_loader.dataset)

        return accuracy, avg_loss
        
    def set_round(self, round_nr):
        """ Set round number attribute.

            Parameters:
            round_nr    (int): Current round number.
        """
        self.round_nr = round_nr
    
    def _save_results(self, results, name):
        """
        """
        with open(f"{self.run_folder}/{name}.npy", "wb") as f:
            if isinstance(results[0], list):
                for i in range(len(results)):
                    np.save(f, results[i])
            else:
                np.save(f, results)