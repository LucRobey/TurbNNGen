from torch import nn
from torch.nn import functional as F
from src.nn.archs.utils import ConvBlockBuilder, ConvTransBlockBuilder, AvgPoolBuilder
import numpy as np
import src.ctes.str_ctes as sctes

class CNN_ALL(nn.Module):
    """
    CNN Model for computing turbulence flow velocity statistics: 
    (c1, c2, L, epsilon).
        c1 : Parameter of long-range dependance
            In [0.2 0.4 0.6 0.8]
        c2 : Parameter of intermittency
            In [0.02 0.04 0.06 0.08]
        epsilon : Size of the small-scale regularization
            In [0.5 1.5 2.5 3.5 4.5]
        L : Size of the integral scale.
            In [1000 2000 3000 4000 5000] 
    """
    OUTPUT_SIZE = 4
    LABELS = [sctes.C1, sctes.C2, sctes.L, sctes.EPSILON]
    def __init__(self, input_size, dropout_probs=None):
        super().__init__()

        self.cnn2, len2 = ConvBlockBuilder.build(input_size, 1, 128, 2)
        self.pool2, len2 = AvgPoolBuilder.build(len2, 4, ceil_mode=True)

        self.cnn4, len4 = ConvBlockBuilder.build(len2, 128, 64, 4)
        self.pool4, len4 = AvgPoolBuilder.build(len4, 4, ceil_mode=True)

        self.cnn8, len8 = ConvBlockBuilder.build(len4, 64, 32, 8)
        self.pool8, len8 = AvgPoolBuilder.build(len8, 4, ceil_mode=True)

        self.cnn16, len16 = ConvBlockBuilder.build(len8, 32, 16, 16)
        self.pool16, len16 = AvgPoolBuilder.build(len16, 4, ceil_mode=True)
        self.cnn32, len32 = ConvBlockBuilder.build(len16, 16, 8, 32)
        self.pool32, len32 = AvgPoolBuilder.build(len32, 4, ceil_mode=True)

        self.flatten = nn.Flatten()
        len_flatten  = int(len32 * 8)

        len_dense1  = len_flatten // 2
        self.dense1 = nn.Linear(len_flatten, len_dense1)

        len_dense2  = len_dense1 // 2
        self.dense2 = nn.Linear(len_dense1, len_dense2)

        len_dense3  = self.OUTPUT_SIZE
        self.dense3 = nn.Linear(len_dense2, len_dense3)

        self.softplus = nn.Softplus()
        
        
    def forward(self, z):
        out = z

        out = self.cnn2(out)
        out = self.pool2(out)
        
        out = self.cnn4(out)
        out = self.pool4(out)
        
        out = self.cnn8(out)
        out = self.pool8(out)
        
        out = self.cnn16(out)
        out = self.pool16(out)
        
        out = self.cnn32(out)
        out = self.pool32(out)

        # out = self.cnn64(out)
        # out = self.pool64(out)

        out = self.flatten(out)

        out = self.dense1(out)
        out = self.dense2(out)
        out = self.dense3(out)

        out = self.softplus(out)
        
        return out
