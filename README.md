# Weighted Ensemble Distillation in Federated Learning with Non-IID Data
This repo contains the code implementation for my master's thesis project where I investigated the use of ensemble distillation in federated learning in non-IID scenarios. The thesis can be found [here](Master_s_thesis.pdf).

**Abstract**

Federated distillation (FD) is a novel algorithmic idea for federated learning (FL)
that allows clients to use heterogeneous model architectures. This is achieved by
distilling aggregated local model predictions on an unlabeled auxiliary dataset into
a global model. While standard FL algorithms are often based on averaging local
parameter updates over multiple communication rounds, FD can be performed with
only one communication round, giving favorable communication properties when
local models are large and the auxiliary dataset is small. However, both FD and
standard FL algorithms experience a significant performance loss when training
data is not independently and identically distributed (non-IID) over the clients.
This thesis investigates weighting schemes to improve the performance with FD
in non-IID scenarios. In particular, the sample-wise weighting scheme FedED-
w2 is proposed, where client predictions on auxiliary data are weighted based on
the similarity with local data. Data similarity is measured with the reconstruction
loss on auxiliary samples when passed through an autoencoder (AE) model that is
trained on local data. Image classification experiments with convolutional neural
networks performed in this study show that FedED-w2 exceeds the test accuracy
of FL baseline algorithms with up to 15 % on the MNIST and EMNIST datasets for
varying degrees of non-IID data over 10 clients. The performance of FedED-w2 is
lower than FL baselines on the CIFAR-10 dataset, where the experiments display
up to 5 % lower test accuracy.

# Reproduce results


## Install requirements

```
pip install -r requirements.txt
```
## Run experiments
Results have been generated with the algorithms

- FedAvg ([paper](https://arxiv.org/abs/1602.05629))
- FedProx ([paper](https://arxiv.org/abs/1812.06127))
- FedED (Federated Ensemble Distillation)

using datasets MNIST, EMNIST and CIFAR-10. Reproduce the results by running

```
bash mnist_runs.sh; bash emnist_runs.sh; bash cifar10_runs.sh
```
Plot the results and figures using the notebooks provided in [results_analysis](results_analysis/)

## General usage

```
usage: main.py [-h] [--seed SEED] [--dataset DATASET] [--n_clients N_CLIENTS] [--public_fraction PUBLIC_FRACTION] [--distribution DISTRIBUTION] [--alpha ALPHA] [--algorithm ALGORITHM] [--local_model LOCAL_MODEL] [--n_rounds N_ROUNDS] [--local_epochs LOCAL_EPOCHS] [--mu MU] [--train_batch_size TRAIN_BATCH_SIZE] [--test_batch_size TEST_BATCH_SIZE] [--learning_rate LEARNING_RATE] [--num_workers NUM_WORKERS] [--student_models STUDENT_MODELS] [--local_epochs_ensemble LOCAL_EPOCHS_ENSEMBLE [--momentum MOMENTUM] [--client_sample_fraction CLIENT_SAMPLE_FRACTION] [--public_batch_size PUBLIC_BATCH_SIZE] [--student_batch_size STUDENT_BATCH_SIZE] [--student_epochs STUDENT_EPOCHS] [--student_epochs_w2 STUDENT_EPOCHS_W2] [--public_data_sizes PUBLIC_DATA_SIZES] [--weight_schemes WEIGHT_SCHEMES] [--autoencoder_epochs AUTOENCODER_EPOCHS] [--student_lr STUDENT_LR] [--student_lr_w2 STUDENT_LR_W2] [--student_loss STUDENT_LOSS]

optional arguments:
  -h, --help                                        Show this help message and exit.
  --seed SEED                                       Seed number.
  --dataset DATASET                                 Dataset to use. Choose from mnist, emnist and cifar10.
  --n_clients N_CLIENTS                             Number of clients in the federated network.
  --public_fraction PUBLIC_FRACTION                 Fraction of public dataset to use.
  --distribution DISTRIBUTION                       IID or non-IID distribution of data.
  --alpha ALPHA                                     Concentration parameter for Latent Dirichlet Allocation.
  --algorithm ALGORITHM                             Aggregation algorithm.
  --local_model LOCAL_MODEL                         Local model name for clients.
  --n_rounds N_ROUNDS                               Number of communication rounds.
  --local_epochs LOCAL_EPOCHS                       Number of local epochs.
  --mu MU                                           FedProx hyperparameter.
  --train_batch_size TRAIN_BATCH_SIZE               Training batch size.
  --test_batch_size TEST_BATCH_SIZE                 Test batch size.
  --learning_rate LEARNING_RATE                     Local learning rate
  --num_workers NUM_WORKERS                         Number of workers for Pytorch dataloaders.
  --student_models STUDENT_MODELS                   List of student models to include.
  --local_epochs_ensemble LOCAL_EPOCHS_ENSEMBLE     Local epochs for FedED.
  --momentum MOMENTUM                               Local momentum.
  --client_sample_fraction CLIENT_SAMPLE_FRACTION   Fraction of participating clients each round.
  --public_batch_size PUBLIC_BATCH_SIZE             Auxiliary data batch size.
  --student_batch_size STUDENT_BATCH_SIZE           Batch size student data.
  --student_epochs STUDENT_EPOCHS                   Epochs for student model.
  --student_epochs_w2 STUDENT_EPOCHS_W2             Epochs for student model with scheme w2.
  --public_data_sizes PUBLIC_DATA_SIZES             List of auxiliary data sizes to include.
  --weight_schemes WEIGHT_SCHEMES                   List of weighting schemes to include.
  --autoencoder_epochs AUTOENCODER_EPOCHS           Epochs for each local autoencoder
  --student_lr STUDENT_LR                           Student learning rate.
  --student_lr_w2 STUDENT_LR_W2                     Student learning rate for scheme w2.
  --student_loss STUDENT_LOSS                       Student loss function. Choose from mse and ce.
```


...
