from zmq import device
from .server_base import ServerBase
import copy
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..models.models import create_model
from ..datasets.dataset_student import StudentData
import numpy as np


class FedEdServer(ServerBase):
    """ Class defining server for federated ensemble distillation.
    """
    def __init__(self, args, model, run_folder, train_loaders, test_loader, public_train_loader, public_val_loader) -> None:
        super().__init__(args, model, run_folder, train_loaders, test_loader, public_train_loader, public_val_loader)

        self.logits_local = None
        self.local_epochs_ensemble = args.local_epochs_ensemble
        self.student_batch_size = args.student_batch_size
        self.student_epochs = args.student_epochs
        self.n_samples_train_public = len(public_train_loader.dataset.indices)

    def run(self):
        """ Execute federated training and distillation.

            Parameters:
            round_nr    (int): Current round number.
        """
        logits_ensemble = torch.zeros(self.n_samples_train_public, self.n_classes, device=self.device)
        local_accs, local_losses = [], []
        for j in range(self.n_clients):
            print("-- Training client nr {} --".format(j+1))

            accs, losses = self._local_training(j)
            local_accs.extend([accs])
            local_losses.extend([losses])
            logits_local = self._get_local_logits()

            logits_ensemble = self._increment_logits_ensemble(logits_ensemble, logits_local, j)

        self._save_results(local_accs, "client_accuracy")
        self._save_results(local_losses, "client_loss")

        student_loader = self._get_student_data_loader(logits_ensemble)
        self._train_student(student_loader)

        test_acc, test_loss = self.evaluate(self.global_model, self.test_loader)
    
        self._save_results([test_acc, test_loss], f"student_test_results")

        print('\nStudent Model Test: Avg. loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss,
        test_acc))
    
    def _local_training(self, client_nr):        
        """ Complete local training at client.

            Parameters:
            client_nr   (int): ID for the client to do local training at.
        """
        self.local_model = copy.deepcopy(self.global_model).to(self.device)
        self.local_model.train()
        optimizer = optim.SGD(self.local_model.parameters(), lr=self.lr_rate, momentum=self.momentum)
        train_accs, train_losses = [], []
        for i in range(self.local_epochs_ensemble):
            for x, y in tqdm(
                self.train_loaders[client_nr],
                leave=False,
                desc=f"Epoch {i+1}/{self.local_epochs_ensemble}"):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.local_model(x)
                error = self.loss_function(output, y)
                error.backward()
                optimizer.step()
            train_acc, train_loss = self.evaluate(self.local_model, self.train_loaders[client_nr])
            train_accs.append(train_acc)
            train_losses.append(train_loss)

        print("Training completed")
        print("Train accuracy: {:.0f}%  Train loss: {:.4f}\n".format(train_acc, train_loss), flush=True)

        return train_accs, train_losses
    
    def _train_student(self, student_loader):
        print("-- Training student model --", flush=True)
        model = create_model(self.dataset_name, student=True)
        model.to(self.device)
        loss_function = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        train_accs, train_losses, val_accs, val_losses = [], [], [], []
        model.train()
        for epoch in range(self.student_epochs):
            for x, y in student_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = model(x)
                loss = loss_function(output, y)
                loss.backward()
                optimizer.step()
            train_acc, train_loss = self.evaluate(model, self.public_train_loader)
            val_acc, val_loss = self.evaluate(model, self.public_val_loader)

            train_accs.append(train_acc)
            train_losses.append(train_loss)
            val_accs.append(val_acc)
            val_losses.append(val_loss)

            print("Epoch {}/{} Train accuracy: {:.0f}%  Train loss: {:.4f} Val accuracy: {:.0f}%  Val loss: {:.4f}".format(
                epoch+1, self.student_epochs, train_acc, train_loss, val_acc, val_loss), end="\r", flush=True)

        self.global_model = model
        self._save_results([train_accs, train_losses, val_accs, val_losses], 'student_train_results')

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
        student_dataset = StudentData(self.public_train_loader.dataset, student_targets)
        return DataLoader(student_dataset, self.student_batch_size)

    
    def _get_scaling_factor(self, client_nr):
        """ Get scaling factor for FedAVG algorithm.

            Parameters:
            client_nr   (int): ID for client.
        """
        return self.n_samples_client[client_nr] / sum(self.n_samples_client)

    def _get_local_logits(self):
        """
        """
        self.local_model.eval()
        logits_local = None
        with torch.no_grad():
            for x, _ in self.public_train_loader:
                x = x.to(self.device)
                if logits_local is None:
                    logits_local = F.softmax(self.local_model(x), dim=1)
                else:
                    logits_local = torch.cat((logits_local, F.softmax(self.local_model(x), dim=1)))

        return logits_local.to(self.device)
    
    def _get_student_targets(self, logits_ensemble):
        """
        """
        n_total_samples = len(self.test_loader.dataset.data)
        targets = torch.zeros(n_total_samples, self.n_classes)
        for i in range(self.n_samples_train_public):
            idx_public = self.public_train_loader.dataset.indices[i]
            targets[idx_public] = logits_ensemble[i]
        
        return targets
