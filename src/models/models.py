import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torchvision.models import resnet18, resnet34

def create_model(model_name):
    if model_name == "mnist_cnn1":
        return Mnist_Cnn1()
    elif model_name == "mnist_cnn2":
        return Mnist_Cnn2()
    elif model_name == "mnist_cnn3":
        return Mnist_Cnn2()
    elif model_name == "emnist_cnn1":
        return Emnist_Cnn1()
    elif model_name == "emnist_cnn2":
        return Emnist_Cnn2()
    elif model_name == "emnist_cnn3":
        return Emnist_Cnn3()
    elif model_name == "cifar10_resnet18":
        return Cifar10_Resnet18()
    elif model_name == "cifar10_resnet34":
        return Cifar10_Resnet34()
    elif model_name == "mnist_autoencoder" or model_name == "emnist_autoencoder":
        return Mnist_Autoencoder()
    elif model_name == "cifar10_autoencoder":
        return Cifar10_Autoencoder()
    elif model_name == "cifar10ex_autoencoder":
        return Cifar10_Autoencoder()
    else:
        print("Model name is not supported.")
        sys.exit()

class Mnist_Cnn1(nn.Module):
    def __init__(self):
        super(Mnist_Cnn1, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 5, 1, 2)
        self.pool = nn.MaxPool2d(4)
        self.fc1 = nn.Linear(2 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

class Mnist_Cnn2(nn.Module):
    def __init__(self):
        super(Mnist_Cnn2, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, 1, 2)
        self.pool = nn.MaxPool2d(4)
        self.fc1 = nn.Linear(8 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

class Mnist_Cnn3(nn.Module):
    def __init__(self):
        super(Mnist_Cnn3, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 1, 2)
        self.pool = nn.MaxPool2d(4)
        self.fc1 = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

# class Mnist_Cnn1(nn.Module):
#     def __init__(self):
#         super(Mnist_Cnn1, self).__init__()
#         self.conv1 = nn.Conv2d(1, 2, 5, 1, 2)
#         self.pool = nn.MaxPool2d(4)
#         self.fc1 = nn.Linear(2 * 7 * 7, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         return x

# class Mnist_Cnn2(nn.Module):
#     def __init__(self):
#         super(Mnist_Cnn2, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, 5, 1, 2)
#         self.pool = nn.MaxPool2d(4)
#         self.fc1 = nn.Linear(16 * 7 * 7, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         return x

# class Mnist_Cnn3(nn.Module):
#     def __init__(self):
#         super(Mnist_Cnn3, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, 5, 1, 2)
#         self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
#         self.fc1 = nn.Linear(32 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 10)
#         self.pool1 = nn.MaxPool2d(2)
#         self.pool2 = nn.MaxPool2d(2)

#     def forward(self, x):
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


class Emnist_Cnn1(nn.Module):
    def __init__(self):
        super(Emnist_Cnn1, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 5, 1, 2)
        self.pool = nn.MaxPool2d(4)
        self.fc1 = nn.Linear(2 * 7 * 7, 26)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

class Emnist_Cnn2(nn.Module):
    def __init__(self):
        super(Emnist_Cnn2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 1, 2)
        self.pool = nn.MaxPool2d(4)
        self.fc1 = nn.Linear(16 * 7 * 7, 26)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

class Emnist_Cnn3(nn.Module):
    def __init__(self):
        super(Emnist_Cnn3, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, 1, 2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 5, 1, 2)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 7 * 7, 26)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

# class Emnist_Cnn2(nn.Module):
#     def __init__(self):
#         super(Emnist_Cnn2, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, 5, 1, 2)
#         self.pool = nn.MaxPool2d(4)
#         self.fc1 = nn.Linear(16 * 7 * 7, 26)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         return x

# class Emnist_Cnn3(nn.Module):
#     def __init__(self):
#         super(Emnist_Cnn3, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, 5, 1, 2)
#         self.pool1 = nn.MaxPool2d(2)
#         self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
#         self.pool2 = nn.MaxPool2d(2)
#         self.fc1 = nn.Linear(32 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 26)

#     def forward(self, x):
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class Cifar10_Cnn(nn.Module):
    def __init__(self):
        super(Cifar10_Cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 26)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Cifar10_Resnet18(nn.Module):
    def __init__(self):
        super(Cifar10_Resnet18, self).__init__()
        base = resnet18(pretrained=False)
        self.base = nn.Sequential(*list(base.children())[:-1])
        in_features = base.fc.in_features
        self.drop = nn.Dropout()
        self.final = nn.Linear(in_features, 10)
    
    def forward(self,x):
        x = self.base(x)
        x = self.drop(x.view(-1,self.final.in_features))
        return self.final(x)

class Cifar10_Resnet34(nn.Module):
    def __init__(self):
        super(Cifar10_Resnet34, self).__init__()
        base = resnet34(pretrained=False)
        self.base = nn.Sequential(*list(base.children())[:-1])
        in_features = base.fc.in_features
        self.drop = nn.Dropout()
        self.final = nn.Linear(in_features, 10)
    
    def forward(self,x):
        x = self.base(x)
        x = self.drop(x.view(-1,self.final.in_features))
        return self.final(x)


class Mnist_Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, 4)
        )
        
        self.decoder_lin = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )
        
        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, 
            padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class Cifar10_Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 128, 128),
            nn.ReLU(True),
            nn.Linear(128, 64)
        )
        
        self.decoder_lin = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 128),
            nn.ReLU(True)
        )
        
        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(128, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, 
            padding=0, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, 
            padding=0, output_padding=1)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x