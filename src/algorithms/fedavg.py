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
    def __init__(self, args, model, run_folder, train_loaders, test_loader) -> None:
        super().__init__(args, model, run_folder, train_loaders, test_loader)

    def run(self, test=True):
        """ Do one round of federated training and aggregation.

            Parameters:
            round_nr    (int): Current round number.
        """
        local_accs, local_losses = [[] for _ in range(self.n_clients)], [[] for _ in range(self.n_clients)]
        test_accs, test_losses = [], []
        for i in range(self.n_rounds):
            self.set_round(i) # This simplifies FedProx inheritance.

            avg_weights = OrderedDict()
            for param_name in self.global_model.state_dict().keys():
                avg_weights[param_name] = torch.zeros(self.global_model.state_dict()[param_name].shape).to(self.device)
            
            for j in range(self.n_clients):
                print("Round {} : Training client nr {} ".format(i+1, j+1))
                acc, loss = self._local_training(j)
                local_accs[j].append(acc)
                local_losses[j].append(loss)

                avg_weights = self._increment_weighted_average(avg_weights, self.local_model.state_dict(), j)

            self.global_model.load_state_dict(avg_weights)

            test_acc, test_loss = self.evaluate(self.global_model, self.test_loader)
            print('\nGlobal Model Test: Avg. loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
                test_loss,
                test_acc))

            test_accs.append(test_acc)
            test_losses.append(test_loss)
        self._save_results(local_accs, "client_accuracy")
        self._save_results(local_losses, "client_loss")    
        self._save_results([test_accs, test_losses], f"fedavg_test_results")
    
    def _local_training(self, client_nr):        
        """ Complete local training at client.

            Parameters:
            client_nr   (int): ID for the client to do local training at.
        """
        self.local_model = copy.deepcopy(self.global_model).to(self.device)
        self.local_model.train()
        optimizer = optim.SGD(self.local_model.parameters(), lr=self.lr_rate, momentum=self.momentum)

        for i in range(self.local_epochs):
            for x, y in self.train_loaders[client_nr]:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.local_model(x)
                error = self.loss_function(output, y)
                error.backward()
                optimizer.step()

            train_acc, train_loss = self.evaluate(self.local_model, self.train_loaders[client_nr])
            print("Epoch {}/{} Train accuracy: {:.0f}%  Train loss: {:.4f}".format(
                i+1, self.local_epochs, train_acc, train_loss), end="\r", flush=True)

        return train_acc, train_loss
    
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
        return self.n_samples_client[client_nr] / sum(self.n_samples_client)
