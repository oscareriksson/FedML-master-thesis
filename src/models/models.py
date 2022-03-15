import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torchvision.models import resnet18, vgg16

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
        base = resnet18(pretrained=False)
        self.base = nn.Sequential(*list(base.children())[:-1])
        in_features = base.fc.in_features
        self.drop = nn.Dropout()
        self.final = nn.Linear(in_features, n_classes)
    
    def forward(self,x):
        x = self.base(x)
        x = self.drop(x.view(-1,self.final.in_features))
        return self.final(x)

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
        base = resnet18(pretrained=False)
        self.base = nn.Sequential(*list(base.children())[:-1])
        in_features = base.fc.in_features
        self.drop = nn.Dropout()
        self.final = nn.Linear(in_features, n_classes)
    
    def forward(self,x):
        x = self.base(x)
        x = self.drop(x.view(-1,self.final.in_features))
        return self.final(x)

