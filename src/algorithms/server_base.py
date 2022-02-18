from ..models.models import create_model
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import copy

class ServerBase(ABC):
    """Abstract base class defining datasets."""
    def __init__(self, args, model, train_loaders, test_loader) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = copy.deepcopy(model).to(self.device)
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.local_model = None
        self.n_clients = args.n_clients
        self.n_rounds = args.n_rounds
        self.lr_rate = args.learning_rate
        self.momentum = args.momentum
        self.local_epochs = args.local_epochs
        self.loss_function = nn.CrossEntropyLoss()
        self.n_samples_client = [len(data_loader.dataset) for data_loader in train_loaders]
        self.n_samples_total = sum(self.n_samples_client)
        self.evaluate_train = args.evaluate_train
        self.round_nr = 0

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def _local_training(self, client_nr):
        pass

    def test(self):
        self.global_model.eval()
        test_losses = []
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.global_model(x)
                test_loss += self.loss_function(output, y).item()
                _, pred = torch.max(output.data, 1)
                correct += (pred == y).sum().item()
        test_loss /= len(self.test_loader.dataset)
        test_losses.append(test_loss)
        print('\nGlobal Model Test: Avg. loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
            test_loss,
            100. * correct / len(self.test_loader.dataset)))
        
    def set_round(self, round_nr):
        self.round_nr = round_nr