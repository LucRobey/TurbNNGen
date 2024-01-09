from torch import nn
from torch.nn import functional as F
from src.nn.archs.utils import ConvBlockBuilder, ConvTransBlockBuilder
import numpy as np

class CNN_C1(nn.Module):
    """
    CNN Model for estimating turbulence flow velocity statistics:
        c1 : Parameter of long-range dependance
            In   [0.2 0.4 0.6 0.8]
    """
    OUTPUT_SIZE = 1
    def __init__(self, input_size, dropout_probs):
        super().__init__()
        self.avgpool  = nn.AvgPool1d(2, ceil_mode=False)
        self.avgpoolc = nn.AvgPool1d(2, ceil_mode=True)
        if dropout_probs is None:
            dropout_probs = [0.5] * 14  # Adjust the length based on the number of layers

        self.dropout_probs = dropout_probs
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
        
        
    def forward(self, z):
        out  = self.cnn1(z)
        out  = self.dropout(F.dropout(out, p=self.dropout_probs[0], training=self.training)) 

        out = self.cnn2(out)
        out = self.avgpoolc(out)
        out  = self.dropout(F.dropout(out, p=self.dropout_probs[1], training=self.training)) 

        out = self.cnn4(out)
        out = self.avgpoolc(out)
        out  = self.dropout(F.dropout(out, p=self.dropout_probs[2], training=self.training)) 

        out = self.cnn8(out)
        out = self.avgpoolc(out)
        out  = self.dropout(F.dropout(out, p=self.dropout_probs[3], training=self.training)) 

        out = self.cnn16(out)
        out = self.avgpoolc(out)
        out  = self.dropout(F.dropout(out, p=self.dropout_probs[4], training=self.training)) 

        out = self.cnn32(out)
        out = self.avgpoolc(out)
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
