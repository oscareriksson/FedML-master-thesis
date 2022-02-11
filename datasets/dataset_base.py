from abc import ABC, abstractmethod
import numpy as np


class DatasetBase(ABC):
    """Abstract class defining datasets."""


    @abstractmethod
    def get_train_data_loaders(self, n_clients, distribution, alpha, batch_size):
        """ Returns a list of local training data loaders.
        """
        pass

    @abstractmethod
    def get_test_data_loader(self, batch_size):
        """ Returns data loader for test data.
        """
        pass

    