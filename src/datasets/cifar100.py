from .dataset_base import PytorchDataset
from torchvision.transforms import Compose, ToTensor, Normalize


class Cifar100(PytorchDataset):
    """ Cifar100 dataset class.
    """
    def __init__(self, train_fraction):
        super().__init__("CIFAR100")
        """ Constructor method.

            Parameters:
            train_fraction (float): Fraction of training data to use.
        """

        transform = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        self.test_data = self.data_class(
            root='data', 
            train=False, 
            transform=transform, 
            download=True)

        self.train_data = self._sample_train_data(train_fraction, transform)
        self.n_samples = len(self.train_data)
  