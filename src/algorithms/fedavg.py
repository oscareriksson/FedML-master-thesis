from .server_base import ServerBase
import copy
import torch.optim as optim
from collections import OrderedDict
import torch.nn.functional as F
import torch


class FedAvgServer(ServerBase):
    """ Server for FedAvg algorithm.
    """
    def __init__(self, args, train_loaders, test_loader) -> None:
        super().__init__(args, train_loaders, test_loader)
        """ Constructor method.

            Parameters:
            args            (dict): Dictionary of server parameters to use.
            train_loaders   (list): List of training data loaders.
            test_loader     (torch.util.data.DataLoader): Data loader for test data.
        """

        self.local_model = None
        self.n_clients = args.n_clients
        self.n_rounds = args.n_rounds
        self.lr_rate = args.learning_rate
        self.local_epochs = args.local_epochs
        self.n_samples_client = [len(data_loader.dataset) for data_loader in train_loaders]
        self.n_samples_total = sum(self.n_samples_client)

    
    def _loss_function(self, output, labels, epoch):
        """ Loss function for local training..

            Parameters:
            output  (tensor): Model output.
            labels  (tensor): True labels.

            Returns:
            tensor: Loss evaluated at given model output and labels.
        """
        return F.cross_entropy(output, labels)

    def run(self):
        
        # Initialize average weights to zero.
        avg_weights = OrderedDict()
        for param_name in self.global_model.state_dict().keys():
            avg_weights[param_name] = torch.zeros(self.global_model.state_dict()[param_name].shape)

        for j in range(self.n_clients):
            print("-- Training client nr {} --".format(j+1), flush=True)
            self._local_training(j)

            avg_weights = self._increment_weighted_average(avg_weights, self.local_model.state_dict(), j)

        self.global_model.load_state_dict(avg_weights)
    
    def _local_training(self, client_nr):
        self.local_model = copy.deepcopy(self.global_model)
        self.local_model.train()
        optimizer = optim.SGD(self.local_model.parameters(), lr=self.lr_rate)

        for i in range(self.local_epochs):
            for x, y in self.train_loaders[client_nr]:
                optimizer.zero_grad()
                output = self.local_model(x)
                error = self._loss_function(output, y, i)
                error.backward()
                optimizer.step()

    def _increment_weighted_average(self, model, model_next, client_nr):
        """ Update an weighted average """
        w = OrderedDict()
        for name in model.keys():
            #tensorDiff = model_next[name] - model[name]
            w[name] = model[name] + model_next[name] * self.get_scaling_factor(client_nr)
        return w
    
    def get_scaling_factor(self, client_nr):
        return self.n_samples_client[client_nr] / self.n_samples_total
