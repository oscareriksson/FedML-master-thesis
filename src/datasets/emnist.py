from .dataset_base import PytorchDataset
from torchvision.transforms import Compose, ToTensor, Normalize

class Emnist(PytorchDataset):
    """ Mnist dataset class.
    """
    def __init__(self, num_workers, public_fraction=0.5):
        super().__init__("EMNIST")
        """ Constructor method.

            Parameters:
            train_fraction (float): Fraction of training data to use.
        """

        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        split = 'letters'
        self.test_data = self.data_class(
            root='data', 
            train=False, 
            transform=transform, 
            download=True,
            split=split)
        
        self.test_data.targets = self.test_data.targets-1
        
        self.public_data, self.train_data = self._split_train_public(public_fraction, transform, split)

        self.num_workers = num_workers
        self.n_samples = len(self.train_data)