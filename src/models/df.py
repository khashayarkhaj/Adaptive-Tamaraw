import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torchsummaryX import summary
from .sub_modules import MyConv1dPadSame, MyMaxPool1dPadSame



class DF(nn.Module):
    def __init__(self, length, num_classes=100):
        super(DF, self).__init__()
        self.length = length
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(
            # nn.Conv1d(1, 32, kernel_size=9, stride=1, padding=4),
            MyConv1dPadSame(1, 32, 8, 1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            # nn.Conv1d(32, 32, kernel_size=9, stride=1, padding=4),
            MyConv1dPadSame(32, 32, 8, 1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            # nn.MaxPool1d(kernel_size=4, stride=4),
            MyMaxPool1dPadSame(8, 1),
            nn.Dropout(0.1)
        )
        self.layer2 = nn.Sequential(
            # nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4),
            MyConv1dPadSame(32, 64, 8, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Conv1d(64, 64, kernel_size=9, stride=1, padding=4),
            MyConv1dPadSame(64, 64, 8, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=4, stride=4),
            MyMaxPool1dPadSame(8, 1),
            nn.Dropout(0.1)
        )
        self.layer3 = nn.Sequential(
            # nn.Conv1d(64, 128, kernel_size=9, stride=1, padding=4),
            MyConv1dPadSame(64, 128, 8, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Conv1d(128, 128, kernel_size=9, stride=1, padding=4),
            MyConv1dPadSame(128, 128, 8, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=4, stride=4),
            MyMaxPool1dPadSame(8, 1),
            nn.Dropout(0.1)
        )
        self.layer4 = nn.Sequential(
            MyConv1dPadSame(128, 256, 8, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            MyMaxPool1dPadSame(8, 1),
            nn.Dropout(0.1)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(256 * self.linear_input(), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(512, self.num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer5(out)
        out = self.fc(out)
        return out
    
    def get_layer4_output(self, x):
        # Replicate the forward logic up to layer4
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        layer4_output = self.layer4(out)  # Here is where you get the layer4 output
        return layer4_output

    def linear_input(self):
        res = self.length
        for i in range(4):
            res = int(np.ceil(res / 8))
        return res