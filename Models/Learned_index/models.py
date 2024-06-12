import torch
import torch.nn as nn
import torch.nn.functional as F
#from auto_LiRPA import PerturbationLpNorm, BoundedParameter


class LINN(nn.Module):

    def __init__(self, input_dim, output_dim, width=128, dtype=None):
        super(LINN, self).__init__()
        self.fc1 = nn.Linear(input_dim, width)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(width, width)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(width, width)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(width, output_dim)



    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

class DeepLINN(nn.Module):

    def __init__(self, input_dim, output_dim, width=128, dtype=None):
        super(DeepLINN, self).__init__()
        self.fc1 = nn.Linear(input_dim, width)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(width, width)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(width, width)
        self.relu3 = nn.ReLU()
        self.fc3a = nn.Linear(width, width)
        self.relu3a = nn.ReLU()
        self.fc3b = nn.Linear(width, width)
        self.relu3b = nn.ReLU()
        self.fc4 = nn.Linear(width, output_dim)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu3a(self.fc3a(x))
        x = self.relu3b(self.fc3b(x))
        x = self.fc4(x)
        return x


