from torch import nn
from torch.nn import functional as F
from src.nn.archs.utils import ConvBlockBuilder, ConvTransBlockBuilder, AvgPoolBuilder
import numpy as np

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
    def __init__(self, input_size, dropout_probs):
        super().__init__()
        self.avgpool  = nn.AvgPool1d(2, ceil_mode=False)
        self.avgpoolc = nn.AvgPool1d(2, ceil_mode=True)
        if dropout_probs is None:
            dropout_probs = [0.5] * 14  # Adjust the length based on the number of layers

        self.dropout_probs = dropout_probs
        self.dropout       = nn.Dropout()

        self.cnn1, len1  = ConvBlockBuilder.build(input_size, 1, 16, 1)

        self.cnn2, len2 = ConvBlockBuilder.build(len1, 16, 32, 2)
        self.pool2, len2 = AvgPoolBuilder.build(len2, 2, ceil_mode=True)

        self.cnn4, len4 = ConvBlockBuilder.build(len2, 32, 64, 4)
        self.pool4, len4 = AvgPoolBuilder.build(len4, 2, ceil_mode=True)

        self.cnn8, len8 = ConvBlockBuilder.build(len4, 64, 128, 8)
        self.pool8, len8 = AvgPoolBuilder.build(len8, 2, ceil_mode=True)

        self.cnn16, len16 = ConvBlockBuilder.build(len8, 128, 256, 16)
        self.pool16, len16 = AvgPoolBuilder.build(len16, 2, ceil_mode=True)

        self.cnn32, len32 = ConvBlockBuilder.build(len16, 256, 256, 32)
        self.pool32, len32 = AvgPoolBuilder.build(len32, 2, ceil_mode=True)

        self.cnn64, len64 = ConvBlockBuilder.build(len32, 256, 256, 64)

        self.cnntrans256, lenTrans256 = ConvTransBlockBuilder.build(len64, 256, 256)

        self.cnntrans128, lenTrans128 = ConvTransBlockBuilder.build(lenTrans256, 256, 128)

        self.cnntrans64, lenTrans64 = ConvTransBlockBuilder.build(lenTrans128, 128, 64)

        self.cnntrans32, lenTrans32 = ConvTransBlockBuilder.build(lenTrans64, 64, 32)
        
        self.cnntrans16, lenTrans16 = ConvTransBlockBuilder.build(lenTrans32, 32, 16)

        self.cnntrans8, lenTrans8 = ConvTransBlockBuilder.build(lenTrans16, 16, 8)
        
        self.cnntrans4, lenTrans4 = ConvTransBlockBuilder.build(lenTrans8, 8, 4)

        self.flatten = nn.Flatten()

        self.dense = nn.Linear(int(lenTrans4*4), self.OUTPUT_SIZE)

        self.softplus = nn.Softplus()
        
        
    def forward(self, z):
        out  = self.cnn1(z)
        out  = self.dropout(F.dropout(out, p=self.dropout_probs[0], training=self.training)) 

        out = self.cnn2(out)
        out = self.pool2(out)
        out  = self.dropout(F.dropout(out, p=self.dropout_probs[1], training=self.training)) 

        out = self.cnn4(out)
        out = self.pool4(out)
        out  = self.dropout(F.dropout(out, p=self.dropout_probs[2], training=self.training)) 

        out = self.cnn8(out)
        out = self.pool8(out)
        out  = self.dropout(F.dropout(out, p=self.dropout_probs[3], training=self.training)) 

        out = self.cnn16(out)
        out = self.pool16(out)
        out  = self.dropout(F.dropout(out, p=self.dropout_probs[4], training=self.training)) 

        out = self.cnn32(out)
        out = self.pool32(out)
        out  = self.dropout(F.dropout(out, p=self.dropout_probs[5], training=self.training)) 

        out = self.cnn64(out)
        out  = self.dropout(F.dropout(out, p=self.dropout_probs[6], training=self.training)) 

        out = self.cnntrans256(out)
        out  = self.dropout(F.dropout(out, p=self.dropout_probs[7], training=self.training)) 

        out = self.cnntrans128(out)
        out  = self.dropout(F.dropout(out, p=self.dropout_probs[8], training=self.training)) 

        out = self.cnntrans64(out)
        out  = self.dropout(F.dropout(out, p=self.dropout_probs[9], training=self.training)) 

        out = self.cnntrans32(out)
        out  = self.dropout(F.dropout(out, p=self.dropout_probs[10], training=self.training)) 

        out = self.cnntrans16(out)
        out  = self.dropout(F.dropout(out, p=self.dropout_probs[11], training=self.training)) 

        out = self.cnntrans8(out)
        out  = self.dropout(F.dropout(out, p=self.dropout_probs[12], training=self.training)) 

        out = self.cnntrans4(out)
        out  = self.dropout(F.dropout(out, p=self.dropout_probs[13], training=self.training)) 

        out = self.flatten(out)

        out = self.dense(out)

        out = self.softplus(out)
        
        return out
