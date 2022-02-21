from .server_base import ServerBase
import copy
import torch.optim as optim
from collections import OrderedDict
import torch.nn.functional as F
import torch
from tqdm import tqdm


class FedAvgServer(ServerBase):
    """ Class defining server for federated averaging (FedAVG).
    """
    def __init__(self, args, model, train_loaders, test_loader) -> None:
        super().__init__(args, model, train_loaders, test_loader)

    def run(self, round_nr):
        """ Do one round of federated training and aggregation.

            Parameters:
            round_nr    (int): Current round number.
        """

        self.set_round(round_nr) # This simplifies FedProx inheritance.

        avg_weights = OrderedDict()
        for param_name in self.global_model.state_dict().keys():
            avg_weights[param_name] = torch.zeros(self.global_model.state_dict()[param_name].shape)
        
        for j in range(self.n_clients):
            print("-- Training client nr {} --".format(j+1))
            self._local_training(j)

            avg_weights = self._increment_weighted_average(avg_weights, self.local_model.state_dict(), j)

        self.global_model.load_state_dict(avg_weights)    
    
    def _local_training(self, client_nr):        
        """ Complete local training at client.

            Parameters:
            client_nr   (int): ID for the client to do local training at.
        """
        self.local_model = copy.deepcopy(self.global_model).to(self.device)
        self.local_model.train()
        optimizer = optim.SGD(self.local_model.parameters(), lr=self.lr_rate, momentum=self.momentum)

        for i in range(self.local_epochs):
            for x, y in tqdm(
                self.train_loaders[client_nr],
                leave=False,
                desc=f"Epoch {i+1}/{self.local_epochs}", 
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.local_model(x)
                error = self.loss_function(output, y)
                error.backward()
                optimizer.step()
        print("Training completed")
        if self.evaluate_train:
            print("Training Accuracy: {:.0f}%\n".format(self._evaluate_train(client_nr)), flush=True)
        else:
            print("")
    
    def _evaluate_train(self, client_nr):
        """ Evaluate local model on its private data.

            Parameters:
            client_nr   (int): ID for the client evaluate.
        """
        self.local_model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in self.train_loaders[client_nr]:
                x, y = x.to(self.device), y.to(self.device)
                output = self.local_model(x)
                _, pred = torch.max(output.data, 1)
                correct += (pred == y).sum().item()
        return 100. * correct / len(self.train_loaders[client_nr].dataset)

    def _increment_weighted_average(self, weights, weights_next, client_nr):
        """ Update an incremental average.
        
            Parameters:
            weights         (OrderedDict): Current running average of weights.
            weights_next    (OrderedDict): New weights from client.
            client_nr       (int): ID for contributing client.
        """
        w = OrderedDict()
        for name in weights.keys():
            w[name] = weights[name] + weights_next[name] * self.get_scaling_factor(client_nr)
        return w
    
    def get_scaling_factor(self, client_nr):
        """ Get scaling factor for FedAVG algorithm.

            Parameters:
            client_nr   (int): ID for client.
        """
        return self.n_samples_client[client_nr] / self.n_samples_total
