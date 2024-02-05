from torch import nn
from torch.nn import functional as F
from src.nn.archs.utils import ConvBlockBuilder, ConvTransBlockBuilder
import numpy as np

class CNN_EPSILON(nn.Module):
    """
    CNN Model for estimating turbulence flow velocity statistics: 
        epsilon : Size of the small-scale regularization
            In [0.5 1.5 2.5 3.5 4.5]
    """
    OUTPUT_SIZE = 1
    def __init__(self, input_size, dropout_probs):
        super().__init__()

        self.avgpool  = nn.AvgPool1d(2, ceil_mode=False)
        self.avgpoolc = nn.AvgPool1d(2, ceil_mode=True)

        
        if dropout_probs is not None:
            self.dropout_layers = nn.ModuleList([nn.Dropout(p=p) for p in dropout_probs])
        else:
            self.dropout_layers = nn.ModuleList([])

        
        self.dropout  = nn.Dropout()

        self.cnn1, len1 = ConvBlockBuilder.build(input_size, 1, 16, 1)

        self.cnn2, len2 = ConvBlockBuilder.build(len1, 16, 32, 2)
        len2 = np.ceil((len2 + 2*0 - 2)/2 + 1)  # avgpoolc

        self.cnn4, len4 = ConvBlockBuilder.build(len2, 32, 64, 4)
        len4 = np.ceil((len4 + 2*0 - 2)/2 + 1)  # avgpoolc 

        self.cnn8, len8 = ConvBlockBuilder.build(len4, 64, 128, 8)
        len8 = np.ceil((len8 + 2*0 - 2)/2 + 1)  # avgpoolc

        self.cnn16, len16 = ConvBlockBuilder.build(len8, 128, 256, 16)
        len16 = np.ceil((len16 + 2*0 - 2)/2 + 1)  # avgpoolc

        self.cnn32, len32 = ConvBlockBuilder.build(len16, 256, 256, 32)
        len32 = np.ceil((len32 + 2*0 - 2)/2 + 1)  # avgpoolc

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

    
    def apply_dropout(self, x, index):
        if not self.dropout_layers:
            return x

        if self.dropout_layers[index] is not None:
            return self.dropout_layers[index](F.dropout(x, p=self.dropout_layers[index].p, training=self.training))
        return x 
   
        
    def forward(self, z):
        out  = self.cnn1(z)
        out = self.apply_dropout(out, 0)

        out = self.cnn2(out)
        out = self.avgpoolc(out)
        out = self.apply_dropout(out, 1) 

        out = self.cnn4(out)
        out = self.avgpoolc(out)
        out = self.apply_dropout(out, 2)
        
        out = self.cnn8(out)
        out = self.avgpoolc(out)
        out = self.apply_dropout(out, 3)
        
        out = self.cnn16(out)
        out = self.avgpoolc(out)
        out = self.apply_dropout(out, 4)
        
        out = self.cnn32(out)
        out = self.avgpoolc(out)
        out = self.apply_dropout(out, 5)
        
        out = self.cnn64(out)
        out = self.apply_dropout(out, 6)
        
        out = self.cnntrans256(out)
        out = self.apply_dropout(out, 7)
        
        out = self.cnntrans128(out)
        out = self.apply_dropout(out, 8)
        
        out = self.cnntrans64(out)
        out = self.apply_dropout(out, 9)
        
        out = self.cnntrans32(out)
        out = self.apply_dropout(out, 10)
        
        out = self.cnntrans16(out)
        out = self.apply_dropout(out, 11)
        
        out = self.cnntrans8(out)
        out = self.apply_dropout(out, 12)
        
        out = self.cnntrans4(out)
        out = self.apply_dropout(out, 13)
            
        out = self.flatten(out)

        out = self.dense(out)

        out = self.softplus(out)
        
        return out 
