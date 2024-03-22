from torch import nn
import torch
import torch.nn.functional as F
import src.ctes.str_ctes as sctes

class CNN_L_MORE_LAYERS(nn.Module):
    LABELS = [sctes.L]
    def __init__(self):
        super(CNN_L_MORE_LAYERS, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=10)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=10)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=10)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=10)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
