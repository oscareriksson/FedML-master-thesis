import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

def create_model(model_name, student=False):
    if not student:
        if model_name == "mnist_cnn":
            return Mnist_Cnn()
        elif model_name == "cifar10_cnn":
            return Cifar_Cnn(10)
        elif model_name == "cifar100_cnn":
            return Cifar_Cnn(100)
        else:
            print("Model name is not supported.")
            sys.exit()
    else:
        if model_name == "mnist":
            return Mnist_Student()
        elif model_name == "cifar10":
            return Cifar_Student(10)
        elif model_name == "cifar100":
            return Cifar_Student(100)
        else:
            print("Model name is not supported.")
            sys.exit()



class Mnist_Cnn(nn.Module):
    def __init__(self):
        super(Mnist_Cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 5, 1, 2)
        self.pool = nn.MaxPool2d(4)
        self.fc1 = nn.Linear(2 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class Cifar_Cnn(nn.Module):
    def __init__(self, n_classes):
        super(Cifar_Cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, 1, 2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5, 1, 2)
        self.conv4 = nn.Conv2d(64, 64, 5, 1, 2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(64, 128, 5, 1, 2)
        self.conv6 = nn.Conv2d(128, 128, 5, 1, 2)
        self.pool3 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 4 * 4 , 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.pool3(F.relu(self.conv6(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Mnist_Student(nn.Module):
    def __init__(self):
        super(Mnist_Student, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 2)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class Cifar_Student(nn.Module):
    def __init__(self, n_classes):
        super(Cifar_Student, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.pool3 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x