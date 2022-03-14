from heapq import merge
from pydoc import cli
import sys
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
    def __init__(self, args, model, run_folder, train_loaders, test_loader, public_loader) -> None:
        super().__init__(args, model, run_folder, train_loaders, test_loader, public_loader)

        self.logits_local = None
        self.local_epochs_ensemble = args.local_epochs_ensemble
        self.student_batch_size = args.student_batch_size
        self.public_batch_size = args.public_batch_size
        self.student_epochs = args.student_epochs
        self.public_data_sizes = [int(x) for x in args.public_data_sizes.split(' ')]
        self.n_samples_train_public = len(public_loader.dataset.indices)
        self.weight_scheme = args.weight_scheme

        self.student_loader = DataLoader(StudentData(public_loader.dataset), self.student_batch_size, shuffle=True, num_workers=self.num_workers)

    def run(self):
        """ Execute federated training and distillation.

            Parameters:
            round_nr    (int): Current round number.
        """
        logits_ensemble = []
        local_accs, local_losses = [], []
        for j in range(self.n_clients):
            print("-- Training client nr {} --".format(j+1))

            accs, losses = self._local_training(j)
            local_accs.extend([accs])
            local_losses.extend([losses])
            logits_local = self._get_local_logits()

            logits_ensemble.append(logits_local)

        self._save_results(local_accs, "client_accuracy")
        self._save_results(local_losses, "client_loss")

        for public_size in self.public_data_sizes:
            print(f"Public dataset size: {public_size}")
            public_train_loader, public_val_loader = self._get_student_data_loaders(public_size)
            self._train_student(logits_ensemble, public_train_loader, public_val_loader, public_size)

            test_acc, test_loss = self.evaluate(self.global_model, self.test_loader)
        
            self._save_results([test_acc, test_loss], f"student_test_results_{public_size}")

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
            for x, y in self.train_loaders[client_nr]:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.local_model(x)
                error = self.loss_function(output, y)
                error.backward()
                optimizer.step()
            train_acc, train_loss = self.evaluate(self.local_model, self.train_loaders[client_nr])
            train_accs.append(train_acc)
            train_losses.append(train_loss)
            print("Epoch {}/{} Train accuracy: {:.0f}%  Train loss: {:.4f}".format(
                i+1, self.local_epochs_ensemble, train_acc, train_loss), end="\r", flush=True)

        print("Training completed")
        print("Train accuracy: {:.0f}%  Train loss: {:.4f}\n".format(train_acc, train_loss), flush=True)

        return train_accs, train_losses
    
    def _train_student(self, logits_ensemble, public_train_loader, public_val_loader, public_size):
        print("-- Training student model --", flush=True)
        model = create_model(self.dataset_name, student=True)
        model.to(self.device)
        loss_function = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        train_accs, train_losses, val_accs, val_losses = [], [], [], []
        for epoch in range(self.student_epochs):
            model.train()   
            for x, idx in self.student_loader:
                x = x.to(self.device)
                active_clients = np.random.choice(np.arange(self.n_clients), int(0.4 * self.n_clients), replace=False)
                merged_logits = torch.zeros(self.student_batch_size, self.n_classes, device=self.device)

                for c in active_clients:
                    merged_logits += logits_ensemble[c][idx] * torch.sum(self.label_count_matrix[c]) / torch.sum(torch.sum(self.label_count_matrix[active_clients]))
                optimizer.zero_grad()
                output = model(x)
                loss = loss_function(output, merged_logits)
                loss.backward()
                optimizer.step()
            train_acc, train_loss = self.evaluate(model, public_train_loader)
            val_acc, val_loss = self.evaluate(model, public_val_loader)

            train_accs.append(train_acc)
            train_losses.append(train_loss)
            val_accs.append(val_acc)
            val_losses.append(val_loss)

            print("Epoch {}/{} Train accuracy: {:.0f}%  Train loss: {:.4f} Val accuracy: {:.0f}%  Val loss: {:.4f}".format(
                epoch+1, self.student_epochs, train_acc, train_loss, val_acc, val_loss), end="\r", flush=True)

        self.global_model = model
        self._save_results([train_accs, train_losses, val_accs, val_losses], f'student_train_results{public_size}')

    def _increment_logits_ensemble(self, logits_ensemble, logits_local, client_nr):
        """ Update the ensembled logits on public dataset.
        
            Parameters:
            logits_ensemble
            logits_local
            client_nr
        """
        return logits_ensemble + logits_local * self._get_scaling_factor(client_nr)

    def _get_student_data_loaders(self, data_size):
        """
        """
        train_size = int(0.8 * data_size)
        train_indices, val_indices = np.arange(train_size), np.arange(train_size, data_size)
        public_train_data, public_val_data = copy.deepcopy(self.public_loader.dataset), copy.deepcopy(self.public_loader.dataset)
        public_train_data.indices, public_val_data.indices = train_indices, val_indices

        public_train_loader = DataLoader(public_train_data, batch_size=self.public_batch_size, num_workers=self.num_workers)
        public_val_loader = DataLoader(public_val_data, batch_size=self.public_batch_size, num_workers=self.num_workers)

        return public_train_loader, public_val_loader

    def _get_scaling_factor(self, client_nr):
        """ Weight client contributions.

            Parameters:
            client_nr   (int): ID for client.
        """
        if self.weight_scheme == "w0":
            return self.n_samples_client[client_nr] / sum(self.n_samples_client)
        elif self.weight_scheme == "w1":
            return torch.true_divide(self.label_count_matrix[client_nr], torch.sum(self.label_count_matrix, axis=0))
        elif self.weight_scheme == "w2":
            return self.label_count_matrix[client_nr]
        else:
            print("Chosen weight scheme is not implemented.")
            sys.exit(0)

    def _get_local_logits(self):
        """
        """
        self.local_model.eval()
        logits_local = None
        with torch.no_grad():
            for x, _ in self.public_loader:
                x = x.to(self.device)
                if logits_local is None:
                    logits_local = F.softmax(self.local_model(x), dim=1)
                else:
                    logits_local = torch.cat((logits_local, F.softmax(self.local_model(x), dim=1)))

        return logits_local.to(self.device)
    
    def _get_student_targets(self, ensemble_output, public_size):
        """
        """
        #_, ensemble_output = torch.max(ensemble_output, 1)
        targets = torch.zeros(ensemble_output.shape)
        for i in range(public_size):
            idx_public = self.public_loader.dataset.indices[i]
            targets[idx_public] = ensemble_output[i]
        
        return targets
