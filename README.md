# FedML-master-thesis
My master's thesis project on ensemble distillation for federated learning.

## Install requirements

```
pip install -r requirements.txt
```
## Run experiments

**Experiments on Mnist**
```
python main.py --dataset mnist --algorithm fedavg --n_clients 5 --n_rounds 2 --local_epochs 1 --train_fraction 0.1 --distribution iid --model_name cnn --train_batch_size 64 --test_batch_size 64 --learning_rate 0.001
```
## Project plan
**Week 1-3**

[Planning report](Planning_report.pdf)

**Week 4**
- [X] Setup of test framework
- [X] Mnist data
- [X] FedAvg

**Week 5**
- [ ] Cifar10/100
- [ ] FedProx
