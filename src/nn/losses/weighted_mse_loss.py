import torch.nn as nn
import torch

class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        self.weights = weights
        super(WeightedMSELoss, self).__init__()

    def forward(self, inputs, targets):
        return torch.mean(self.weights * (inputs - targets)**2)

    def __str__(self):
        return f"{self.__class__.__name__}(weights={self.weights})"
    
    def __repr__(self):
        return str(self)