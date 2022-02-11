from models import create_model
import copy
import torch
import torch.optim as optim
from collections import OrderedDict
import torch.nn.functional as F


class FedAvgServer():
    def __init__(self, args, train_loaders, test_loader):
        self.global_model = create_model(args.model_name)
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.n_clients = args.n_clients
        self.n_rounds = args.n_rounds
        self.lr_rate = args.learning_rate
        self.local_epochs = args.local_epochs
    

    def run(self):
        avg_weights = None
        
        for j in range(self.n_clients):
            print("-- Client {} --".format(j+1))

            local_model = self._local_training(j)
            if not avg_weights:
                avg_weights = local_model.state_dict()
            else:
                avg_weights = self._increment_average(avg_weights, local_model.state_dict(), j)
            # K.clear_session()

            self.set_weights(avg_weights)    
    
    def _local_training(self, client_nr):
        model = copy.deepcopy(self.global_model)
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.lr_rate)

        for i in range(self.local_epochs):
            for x, y in self.train_loaders[client_nr]:
                optimizer.zero_grad()
                output = model(x)
                error = F.nll_loss(output, y)
                error.backward()
                optimizer.step()
        
        return model

    def evaluate(self):
        self.global_model.eval()
        test_losses = []
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.global_model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(self.test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(self.test_loader.dataset),
        100. * correct / len(self.test_loader.dataset)))

    def _increment_average(self, model, model_next, n):
        """ Update an incremental average. """
        w = OrderedDict()
        for name in model.keys():
            tensorDiff = model_next[name] - model[name]
            w[name] = model[name] + tensorDiff / n
        return w
    
    def get_scaling_factor(self, tot_samples, client_samples):
        return client_samples / tot_samples  

    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.global_model.load_state_dict(weights)
