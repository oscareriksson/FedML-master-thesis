from ..models.models import create_model
from abc import ABC, abstractmethod
import torch

class ServerBase(ABC):
    """Abstract base class defining datasets."""
    def __init__(self, args, train_loaders, test_loader) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = create_model(args.model_name).to(self.device)
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def _local_training(self, client_nr):
        pass

    @abstractmethod
    def _loss_function(output, labels, epoch):
        pass

    def evaluate(self):
        self.global_model.eval()
        test_losses = []
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.global_model(x)
                test_loss += self._loss_function(output, y, 1).item()
                _, pred = torch.max(output.data, 1)
                correct += (pred == y).sum().item()
        test_loss /= len(self.test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(self.test_loader.dataset),
        100. * correct / len(self.test_loader.dataset)))