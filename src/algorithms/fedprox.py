from .fedavg import FedAvgServer
import torch.nn as nn
import torch
import torch.nn.functional as F


class FedProxServer(FedAvgServer):
    def __init__(self, args, train_loaders, test_loader, mu):
        super().__init__(args, train_loaders, test_loader)

        self.mu = mu
    
    def _loss_function(self, output, labels, epoch):
        error = F.cross_entropy(output, labels)
        if epoch > 0:
            for w, w_t in zip(self.local_model.parameters(), self.global_model.parameters()):
                error += self.mu / 2. * torch.pow(torch.norm(w.data - w_t.data), 2)
                w.grad.data += self.mu * (w.data - w_t.data)
        return error
