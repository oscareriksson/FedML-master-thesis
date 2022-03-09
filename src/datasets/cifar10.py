from .dataset_base import PytorchDataset
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Subset
import torch


class Cifar10(PytorchDataset):
    """ Cifar10 dataset class.
    """
    def __init__(self, num_workers, train_fraction=None):
        super().__init__("CIFAR10")
        """ Constructor method.

            Parameters:
            train_fraction (float): Fraction of training data to use.
        """

        transform = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.test_data = self.data_class(
            root='data', 
            train=False, 
            transform=transform, 
            download=True)
        
        self.test_data.targets = torch.tensor(self.test_data.targets)

        self.public_data = Subset(self.test_data, [])
        self.num_workers = num_workers

        self.train_data = self._sample_train_data(train_fraction, transform)
        self.n_samples = len(self.train_data)