import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 512)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of the MLP model
model = MLPModel()


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc = nn.Linear(17, 64)
        self.relu = nn.ReLU(inplace=True)
        self.middle = nn.Linear(64, 128)
        self.dec = nn.Linear(128, 17)

    def forward(self, x):
        enc = self.relu(self.enc(x))
        mid = self.relu(self.middle(enc))
        dec = self.dec(mid)
        return dec
