from .server_base import ServerBase
import copy
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..models.models import Mnist_Student
from ..datasets.dataset_student import StudentData
import sys


class FedEdServer(ServerBase):
    """ Class defining server for federated ensemble distillation.
    """
    def __init__(self, args, model, train_loaders, test_loader, public_loader) -> None:
        super().__init__(args, model, train_loaders, test_loader, public_loader)

        self.logits_local = None
        self.local_epochs_ensemble = args.local_epochs_ensemble
        self.student_batch_size = args.student_batch_size
        self.student_epochs = args.student_epochs

    def run(self):
        """ Execute federated training and distillation.

            Parameters:
            round_nr    (int): Current round number.
        """

        logits_ensemble = torch.zeros(self.n_samples_public, 10)
        
        for j in range(self.n_clients):
            print("-- Training client nr {} --".format(j+1))

            self._local_training(j)
            logits_local = self._get_local_logits()

            logits_ensemble = self._increment_logits_ensemble(logits_ensemble, logits_local, j)

        student_loader = self._get_student_data_loader(logits_ensemble)
        self._train_student(student_loader)
        self.test()
    
    def _local_training(self, client_nr):        
        """ Complete local training at client.

            Parameters:
            client_nr   (int): ID for the client to do local training at.
        """
        self.local_model = copy.deepcopy(self.global_model).to(self.device)
        self.local_model.train()
        optimizer = optim.SGD(self.local_model.parameters(), lr=self.lr_rate, momentum=self.momentum)

        for i in range(self.local_epochs_ensemble):
            for x, y in tqdm(
                self.train_loaders[client_nr],
                leave=False,
                desc=f"Epoch {i+1}/{self.local_epochs_ensemble}", 
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
    
    def _train_student(self, student_loader):
        print("-- Training student model --", flush=True)
        model = Mnist_Student()
        loss_function = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        model.train()
        for epoch in range(self.student_epochs):
            train_loss = []
            for x, y in tqdm(student_loader,
                            leave=False,
                            desc=f"Epoch {epoch+1}/{self.student_epochs}", 
                            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
                optimizer.zero_grad()
                output = model(x)
                loss = loss_function(output, y)
                loss.backward()
                optimizer.step()
        
        self.global_model = model

    def _increment_logits_ensemble(self, logits_ensemble, logits_local, client_nr):
        """ Update the ensembled logits on public dataset.
        
            Parameters:
            logits_ensemble
            logits_local
            client_nr
        """
        return logits_ensemble + logits_local * self._get_scaling_factor(client_nr)

    def _get_student_data_loader(self, logits_ensemble):
        """
        """
        student_targets = self._get_student_targets(logits_ensemble)
        student_dataset = StudentData(self.public_loader.dataset, student_targets)
        return DataLoader(student_dataset, self.student_batch_size)

    def _get_student_model(self):
        if self.dataset_name == "mnist":
            from ..models.models import Mnist_Student
            return Mnist_Student()
        elif self.dataset_name == "cifar10":
            from ..models.models import Cifar10_Student
            return Cifar10_Student()
        else:
            print("Student model not implemented for this dataset.", flush=True)
            sys.exit()

    
    def _get_scaling_factor(self, client_nr):
        """ Get scaling factor for FedAVG algorithm.

            Parameters:
            client_nr   (int): ID for client.
        """
        return self.n_samples_client[client_nr] / self.n_samples_total

    def _get_local_logits(self):
        """
        """
        self.local_model.eval()
        logits_local = None
        with torch.no_grad():
            for x, _ in self.public_loader:
                if logits_local is None:
                    logits_local = F.softmax(self.local_model(x), dim=1)
                else:
                    logits_local = torch.cat((logits_local, F.softmax(self.local_model(x), dim=1)))

        return logits_local
    
    def _get_student_targets(self, logits_ensemble):
        """
        """
        n_samples_public_test = len(self.public_loader.dataset) + len(self.test_loader.dataset)
        targets = torch.zeros(n_samples_public_test, 10)
        for i in range(self.n_samples_public):
            idx_public = self.public_loader.dataset.indices[i]
            targets[idx_public] = logits_ensemble[i]
        
        return targets
