from .dataset_base import PytorchDataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import torch


class Cifar10(PytorchDataset):
    """ Cifar10 dataset class.
    """
    def __init__(self, num_workers, public_fraction=0.5):
        super().__init__("CIFAR10")
        """ Constructor method.

            Parameters:
            train_fraction (float): Fraction of training data to use.
        """

        #transform = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        transform = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.test_data = self.data_class(
            root='data', 
            train=False, 
            transform=transform, 
            download=True)
        
        self.test_data.targets = torch.tensor(self.test_data.targets)

        self.public_data, self.train_data = self._split_train_public(public_fraction, transform)

        self.num_workers = num_workers
        self.n_samples = len(self.train_data)