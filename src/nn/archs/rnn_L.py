from torch import nn
import torch


class RNN_L(nn.Module):
    def __init__(self, input_size):
        super(RNN_L, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=128, batch_first=True)
        self.dropout1 = nn.Dropout(0.4)  # Dropout layer with 50% probability
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.dropout2 = nn.Dropout(0.5)  # Dropout layer with 50% probability
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)  # Applying dropout after the first LSTM layer
        out, _ = self.lstm2(out)
        out = self.dropout2(out)  # Applying dropout after the second LSTM layer
        out = self.fc1(out[:, -1, :])  # Taking the last time step's output
        out = self.fc2(out)
        return out
