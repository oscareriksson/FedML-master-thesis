from .fedavg import FedAvgServer
import torch.nn as nn
import torch
import torch.nn.functional as F


class FedProxServer(FedAvgServer):
    def __init__(self, args, model, train_loaders, test_loader):
        super().__init__(args, model, train_loaders, test_loader)

        self.mu = args.mu
        self.loss_function = self._fedprox_loss()
    
    def _fedprox_loss(self):
        loss = nn.CrossEntropyLoss()

        if self.round_nr > 0:
            for w, w_t in zip(self.local_model.parameters(), self.global_model.parameters()):
                loss += self.mu / 2. * torch.pow(torch.norm(w.data - w_t.data), 2)
                w.grad.data += self.mu * (w.data - w_t.data)
        return loss
