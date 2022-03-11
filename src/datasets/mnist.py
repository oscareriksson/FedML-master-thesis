from .dataset_base import PytorchDataset
from torchvision.transforms import Compose, ToTensor, Normalize

class Mnist(PytorchDataset):
    """ Mnist dataset class.
    """
    def __init__(self, num_workers, public_fraction=0.5):
        super().__init__("MNIST")
        """ Constructor method.

            Parameters:
            train_fraction (float): Fraction of training data to use.
        """

        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        self.test_data = self.data_class(
            root='data', 
            train=False, 
            transform=transform, 
            download=True)
        
        self.public_data, self.train_data = self._split_train_public(public_fraction, transform)

        self.num_workers = num_workers
        self.n_samples = len(self.train_data)