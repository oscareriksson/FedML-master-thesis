from .fedavg import FedAvgServer
import torch.nn as nn
import torch
import torch.nn.functional as F


class FedProxServer(FedAvgServer):
    """ Class defining server for federated learning with the FedProx algorithm.
    """
    def __init__(self, args, model, run_folder, train_loaders, test_loader):
        super().__init__(args, model, run_folder, train_loaders, test_loader)

        self.mu = args.mu
        self.loss_function = self._fedprox_loss()
    
    def _fedprox_loss(self):
        """ Get FedProx loss function.

            Returns:
            torch.nn.CrossEntropyLoss
        """
        loss = nn.CrossEntropyLoss()

        if self.round_nr > 0:
            for w, w_t in zip(self.local_model.parameters(), self.global_model.parameters()):
                loss += (self.mu / 2.) * (w.data - w_t.data).norm(2)

        return loss
