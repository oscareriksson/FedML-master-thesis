# FedML-master-thesis
My master's thesis project on ensemble distillation for federated learning.

## Install requirements

```
pip install -r requirements.txt
```
## Run experiments
Support is currently implemented for

- FedAvg
- FedProx

with datasets

- Mnist
- Cifar10
- Cifar100


**Experiments on Mnist**
```
python main.py --dataset mnist --algorithm fedavg --n_clients 5 --n_rounds 2 --local_epochs 1 --train_fraction 0.1 --distribution iid --model_name mnist_cnn --train_batch_size 64 --test_batch_size 64 --learning_rate 0.001
```
```
python main.py --dataset mnist --algorithm fedprox --n_clients 5 --n_rounds 2 --local_epochs 1 --train_fraction 0.1 --distribution iid --model_name mnist_cnn --train_batch_size 64 --test_batch_size 64 --learning_rate 0.001
```

**Experiments on Cifar10**
```
python main.py --dataset cifar10 --algorithm fedavg --n_clients 5 --n_rounds 2 --local_epochs 1 --train_fraction 0.1 --distribution iid --model_name cifar10_cnn --train_batch_size 64 --test_batch_size 64 --learning_rate 0.001
```
```
python main.py --dataset cifar10 --algorithm fedprox --n_clients 5 --n_rounds 2 --local_epochs 1 --train_fraction 0.1 --distribution iid --model_name cifar10_cnn --train_batch_size 64 --test_batch_size 64 --learning_rate 0.001
```

## Project plan
**Week 1-3**

[Planning report](Planning_report.pdf)

**Week 4**
- [X] Setup of test framework.
- [X] Mnist data.
- [X] FedAvg.

**Week 5**
- [X] Cifar10.
- [X] Cifar100.
- [X] FedProx.
- [X] Possible to run the main script with multiple algorithms over a specified number of random seeds.
- [X] Possible to evaluate and display local training accuracy.

**Week 6**
- [] Ensemble distillation methods ...
