from ..models.models import create_model
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import copy
import numpy as np

class ServerBase(ABC):
    """Abstract base class defining datasets."""
    def __init__(self, args, model, train_loaders, test_loader, public_loader=None, run_folder=None) -> None:
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
        self.local_epochs = args.local_epochs
        self.loss_function = nn.CrossEntropyLoss()
        self.n_samples_client = [len(data_loader.dataset) for data_loader in train_loaders]
        self.n_samples_total = sum(self.n_samples_client)
        self.n_samples_public = args.n_samples_public
        self.evaluate_train = args.evaluate_train
        self.round_nr = 0
        self.run_folder = run_folder
        self.test_acc = []
        self.test_loss = []

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def _local_training(self, client_nr):
        pass

    def test(self):
        """ Evaluate global model on test dataset.
        
        """
        self.global_model.eval()
        test_loss = []
        correct = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.global_model(x)
                error = self.loss_function(output, y)
                test_loss.append(error.item())
                _, pred = torch.max(output.data, 1)
                correct += (pred == y).sum().item()
        avg_loss = sum(test_loss) / len(test_loss)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        self.test_loss.append(avg_loss)
        self.test_acc.append(accuracy)
        print('\nGlobal Model Test: Avg. loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
            avg_loss,
            accuracy))
        
    def set_round(self, round_nr):
        """ Set round number attribute.

            Parameters:
            round_nr    (int): Current round number.
        """
        self.round_nr = round_nr
    
    def _save_results(self):
        with open(f"{self.run_folder}/test_accuracy.npy", "wb") as f:
            np.save(f, np.array(self.test_acc))
        
        with open(f"{self.run_folder}/test_loss.npy", "wb") as f:
            np.save(f, np.array(self.test_loss))