# -*- coding: utf-8 -*-
from torch import nn

class CNN(nn.Module):
    """
    CNN Model for predicting the following 4 turbulence statistiques:
    * L: Large scale delimiter 
    * eta: Small scale delimiter
        Dissipative range | eta | Inertial range | L | Integral range
    * H: Hurst exponent
    * c1 : Gamma
    """
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool1d(2, ceil_mode=False)
        self.avgpoolc = nn.AvgPool1d(2, ceil_mode=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear')

        self.cnn1 = nn.Sequential( 
            nn.Conv1d(1, 16, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size = 2, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            )
        self.cnn8 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size = 8, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            )
        self.cnn16 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size = 16, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            )
        self.cnn32 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size = 32, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            )
        self.cnn64 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size = 64, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            )
        
        self.flatten = nn.Flatten()

        self.dense = nn.LazyLinear(4)
        
    def forward(self, z):    
        residual1  = self.cnn1(z)
        out = residual1 #= out->  size
        
        residual2  = self.cnn2(out)
        out = residual2 #= out -> size/2
        
        out = self.avgpoolc(out)
        residual4  = self.cnn4(out)
        out = residual4 #= out ->size/4
        
        out = self.avgpoolc(out)
        residual8  = self.cnn8(out)
        out = residual8 #= out -> size=8
        
        out = self.avgpoolc(out)
        residual16  = self.cnn16(out)
        out = residual16 #= out -> size/16
        
        out = self.avgpoolc(out)
        residual32  = self.cnn32(out)
        out = residual32 #= out size/32
        
        out = self.avgpoolc(out)
        out  = self.cnn64(out)

        out = self.flatten(out)

        out = self.dense(out)
        
        return out
