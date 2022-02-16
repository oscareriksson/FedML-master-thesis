from ..models.models import create_model
from abc import ABC, abstractmethod
import torch

class ServerBase(ABC):
    """Abstract base class defining servers."""
    def __init__(self, args, train_loaders, test_loader) -> None:
        self.global_model = create_model(args.model_name)
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
            for data, target in self.test_loader:
                output = self.global_model(data)
                test_loss += self._loss_function(output, target, 1).item()
                _, pred = torch.max(output.data, 1)
                correct += (pred == target).sum().item()
        test_loss /= len(self.test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(self.test_loader.dataset),
        100. * correct / len(self.test_loader.dataset)))