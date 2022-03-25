from .server_base import ServerBase
import copy
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..models.models import create_model
from ..datasets.dataset_student import StudentData
import numpy as np
import sys

class FedEdServer(ServerBase):
    """ Class defining server for federated ensemble distillation.
    """
    def __init__(self, args, model, run_folder, train_loaders, test_loader, public_loader) -> None:
        super().__init__(args, model, run_folder, train_loaders, test_loader, public_loader)

        self.student_model = args.student_model
        self.logits_local = None
        self.local_epochs_ensemble = args.local_epochs_ensemble
        self.student_batch_size = args.student_batch_size
        self.public_batch_size = args.public_batch_size
        self.student_epochs = args.student_epochs
        self.public_data_sizes = [int(x) for x in args.public_data_sizes.split(' ')]
        self.weight_scheme = args.weight_scheme
        self.client_sample_fraction = args.client_sample_fraction
        self.ae_public_weights = []
        self.ae_test_weights = []
        self.autoencoder_epochs = args.autoencoder_epochs
        self.student_lr = args.student_lr
        self.student_loss = args.student_loss

    def run(self):
        """ Execute federated training and distillation.

            Parameters:
            round_nr    (int): Current round number.
        """
        ensemble_public_logits, ensemble_test_logits = [], []
        local_accs, local_losses = [], []
        for j in range(self.n_clients):
            print("-- Training client nr {} --".format(j+1))

            accs, losses = self._local_training(j)
            local_accs.extend([accs])
            local_losses.extend([losses])
            local_public_logits, local_test_logits  = self._get_local_logits()
            if self.weight_scheme == 2:
                public_weights, test_weights = self._get_autoencoder_weights(client_nr=j)
                self.ae_public_weights.append(public_weights)
                self.ae_test_weights.append(test_weights)

            ensemble_public_logits.append(local_public_logits)
            ensemble_test_logits.append(local_test_logits)

        ensemble_test_acc = self._ensemble_accuracy(ensemble_test_logits)
        print("Ensemble test accuracy: {:.0f}%".format(ensemble_test_acc))

        self._save_results(local_accs, "client_accuracy")
        self._save_results(local_losses, "client_loss")
        self._save_results([ensemble_test_acc], "ensemble_test_acc")

        for public_size in self.public_data_sizes:
            print(f"Public dataset size: {public_size}")
            student_loader, public_train_loader, public_val_loader = self._get_student_data_loaders(public_size, ensemble_public_logits)

            self._train_student(ensemble_public_logits, student_loader, public_train_loader, public_val_loader, public_size)

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
    
    def _train_student(self, ensemble_logits, student_loader, public_train_loader, public_val_loader, public_size):
        print("-- Training student model --", flush=True)
        model = create_model(self.student_model).to(self.device)
        loss_function = nn.MSELoss() if self.student_loss == "mse" else nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.student_lr)

        train_accs, train_losses, val_accs, val_losses = [], [], [], []
        for epoch in range(self.student_epochs):
            model.train()   
            for x, idx in student_loader:
                x = x.to(self.device)
                idx = idx.to(self.device)
                active_clients = np.random.choice(np.arange(self.n_clients), int(self.client_sample_fraction * self.n_clients), replace=False)
                merged_logits = torch.zeros(self.student_batch_size, self.n_classes, device=self.device)

                for c in active_clients:
                    if len(idx) != self.student_batch_size:
                        selected_logits = torch.zeros(self.student_batch_size, self.n_classes, device=self.device)
                        selected_logits[:len(idx), self.n_classes-1] = ensemble_logits[c][idx]
                    else:
                        selected_logits = ensemble_logits[c][idx]

                    merged_logits += selected_logits * self._ensemble_weight(client_nr=c, active_clients=active_clients, sample_indices=idx)

                if self.student_loss == "ce":
                    _, merged_logits = torch.max(merged_logits, 1)
                elif self.weight_scheme == 2:
                    merged_logits = (merged_logits.T / torch.sum(merged_logits, 1)).T

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

    def _get_autoencoder_weights(self, client_nr):
        """
        """
        autoencoder = create_model(self.dataset_name + "_autoencoder").to(self.device)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=1e-05)
        loss_fn = nn.MSELoss()

        print("Training autoencoder")
        for epoch in range(self.autoencoder_epochs):
            autoencoder.train()
            train_loss = []

            for img_batch, _ in self.train_loaders[client_nr]: 
                img_batch = img_batch.to(self.device)
                output = autoencoder(img_batch)
                loss = loss_fn(output, img_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.detach().cpu().numpy())

            train_loss = np.mean(train_loss)
            print('Epoch {}/{} \t train loss {}'.format(epoch + 1, self.autoencoder_epochs, train_loss), end="\r")
        print("")

        autoencoder.eval()

        public_samples_loss = []
        with torch.no_grad():
            for img_batch, _ in self.public_loader:
                img_batch = img_batch.to(self.device)
                output = autoencoder(img_batch)
                sample_loss = []
                for j in range(len(img_batch)):
                    sample_loss.append(torch.mean((output[j]-img_batch[j])*(output[j]-img_batch[j])))
                public_samples_loss.extend(sample_loss)
        
        test_samples_loss = []
        with torch.no_grad():
            for img_batch, _ in self.test_loader:
                img_batch = img_batch.to(self.device)
                output = autoencoder(img_batch)
                sample_loss = []
                for j in range(len(img_batch)):
                    sample_loss.append(torch.mean((output[j]-img_batch[j])*(output[j]-img_batch[j])))
                test_samples_loss.extend(sample_loss)
        
        ae_public_weights = torch.tensor([1/sample_loss**2 for sample_loss in public_samples_loss], device=self.device)
        ae_test_weights = torch.tensor([1/sample_loss**2 for sample_loss in test_samples_loss], device=self.device)

        return ae_public_weights, ae_test_weights

    def _get_student_data_loaders(self, data_size, ensemble_logits):
        """
        """
        merged_logits = torch.zeros(ensemble_logits[0].shape, device=self.device)
        for c in range(self.n_clients):
            merged_logits += ensemble_logits[c] * self._ensemble_weight(client_nr=c, active_clients=np.arange(self.n_clients))
        _, targets = torch.max(merged_logits, 1)
    
        train_size = int(0.8 * data_size)
        train_indices, val_indices = np.arange(train_size), np.arange(train_size, data_size)
        public_train_data = copy.deepcopy(self.public_loader.dataset)
        public_train_data.dataset.targets = targets.to('cpu')
        public_val_data = copy.deepcopy(public_train_data)
        public_train_data.indices, public_val_data.indices = train_indices, val_indices

        student_data = copy.deepcopy(self.public_loader.dataset)
        student_data.indices = train_indices

        public_train_loader = DataLoader(public_train_data, batch_size=self.public_batch_size, num_workers=self.num_workers)
        public_val_loader = DataLoader(public_val_data, batch_size=self.public_batch_size, num_workers=self.num_workers)
        student_loader = DataLoader(StudentData(student_data), self.student_batch_size, shuffle=True, num_workers=self.num_workers)

        return student_loader, public_train_loader, public_val_loader

    def _ensemble_weight(self, client_nr, active_clients, sample_indices=None, test=False):
        """ Weight client contributions.

            Parameters:
            client_nr   (int): ID for client.
        """
        if self.weight_scheme == 0:
            return self.n_samples_client[client_nr] / sum([self.n_samples_client[c] for c in active_clients])
        elif self.weight_scheme == 1:
            return torch.true_divide(self.label_count_matrix[client_nr], torch.sum(self.label_count_matrix[active_clients], axis=0)+0.001)
        elif self.weight_scheme == 2:
            weights = self.ae_test_weights if test else self.ae_public_weights
            if sample_indices is None:
                return weights[client_nr][:, None] * self.n_samples_client[client_nr] / sum([self.n_samples_client[c] for c in active_clients])
            else:
                return weights[client_nr][sample_indices, None] * self.n_samples_client[client_nr] / sum([self.n_samples_client[c] for c in active_clients])
        else:
            print("Chosen weight scheme is not implemented.")
            sys.exit(0)

    def _get_local_logits(self):
        """
        """
        self.local_model.eval()
        public_logits = None
        with torch.no_grad():
            for x, _ in self.public_loader:
                x = x.to(self.device)
                if public_logits is None:
                    public_logits = F.softmax(self.local_model(x), dim=1)
                else:
                    public_logits = torch.cat((public_logits, F.softmax(self.local_model(x), dim=1)))

        test_logits = None
        with torch.no_grad():
            for x, _ in self.test_loader:
                x = x.to(self.device)
                if test_logits is None:
                    test_logits = F.softmax(self.local_model(x), dim=1)
                else:
                    test_logits = torch.cat((test_logits, F.softmax(self.local_model(x), dim=1)))

        return public_logits.to(self.device), test_logits.to(self.device)
    
    def _get_student_targets(self, ensemble_output, public_size):
        """
        """
        #_, ensemble_output = torch.max(ensemble_output, 1)
        targets = torch.zeros(ensemble_output.shape)
        for i in range(public_size):
            idx_public = self.public_loader.dataset.indices[i]
            targets[idx_public] = ensemble_output[i]
        
        return targets

    def _ensemble_accuracy(self, ensemble_logits):
        merged_logits = torch.zeros(ensemble_logits[0].shape, device=self.device)

        for c in range(self.n_clients):
            merged_logits += ensemble_logits[c] * self._ensemble_weight(client_nr=c, active_clients=np.arange(self.n_clients), test=True)
        
        #targets = self.public_loader.dataset.dataset.targets[self.public_loader.dataset.indices].to(self.device)
        targets = self.test_loader.dataset.targets.to(self.device)
        _, preds = torch.max(merged_logits, 1)
        correct = (preds == targets).sum().item()

        return 100. * correct / len(self.test_loader.dataset)
