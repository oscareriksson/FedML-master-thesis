from .dataset_base import PytorchDataset
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR100
from torch.utils.data import Subset
import torch
import numpy as np


class Cifar10ex(PytorchDataset):
    """ Cifar10 dataset class.
    """
    def __init__(self, num_workers, public_fraction=0.5):
        super().__init__("CIFAR10")
        """ Constructor method.

            Parameters:
            train_fraction (float): Fraction of training data to use.
        """

        transform_100 = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        transform_10 = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.test_data = self.data_class(
            root='data', 
            train=False, 
            transform=transform_10, 
            download=True)
        
        self.test_data.targets = torch.tensor(self.test_data.targets)

        _, self.train_data = self._split_train_public(public_fraction, transform_10)

        public_dataset = CIFAR100(root='data', train=True, transform=transform_100, download=True)
        self.public_data = Subset(public_dataset, np.arange(len(public_dataset)))

        self.num_workers = num_workers
        self.n_samples = len(self.train_data)