from torch import nn
import torch
import src.ctes.str_ctes as sctes

class CNN_L_BIG_KERNEL(nn.Module):
    LABELS = [sctes.L]
    def __init__(self):
        super(CNN_L_BIG_KERNEL, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=120, stride=10)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=64, stride=5)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=30, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x
