from abc import ABC, abstractmethod

class DatasetBase(ABC):
    """Abstract class defining datasets."""

    @abstractmethod
    def get_train_data_loaders(self, n_clients, distribution, alpha, batch_size):
        pass

    @abstractmethod
    def get_test_data_loader(self, batch_size):
        pass

    