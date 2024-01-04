from torch import nn
import numpy as np

class ConvBlockBuilder():
    @classmethod
    def build(cls, in_size, in_ch, out_ch, kernel_size):
        stride      = 1
        padding     = 0
        bias        = False
        dilation  = 1
        block  = nn.Sequential( 
            nn.Conv1d(in_ch, out_ch, 
                      kernel_size = kernel_size, 
                      stride      = stride, 
                      padding     = padding, 
                      bias        = bias,
                      dilation  = dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(True),
            )
        out_size = np.floor((in_size + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1)
        return block, out_size
    
class ConvTransBlockBuilder():
    @classmethod
    def build(cls, in_size, in_ch, out_ch):
        kernel_size    = 3
        stride         = 1
        padding        = 1
        bias           = False
        dilation     = 1
        output_padding = 0
        block  = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, 
                               kernel_size    = kernel_size, 
                               stride         = stride, 
                               padding        = padding, 
                               bias           = bias,
                               dilation     = dilation,
                               output_padding = output_padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(True),
            )
        out_size = (in_size - 1)*stride - 2*padding + dilation*(kernel_size - 1) + output_padding + 1
        return block, out_size

class CNNBase(nn.Module):
    """
    CNN Model for classifing turbulence flow velocity statistics: 
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

    def __init__(self, input_size, output_size):
        super().__init__()
        self.avgpool  = nn.AvgPool1d(2, ceil_mode=False)
        self.avgpoolc = nn.AvgPool1d(2, ceil_mode=True)
        self.dropout  = nn.Dropout(p=0.5)

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

        self.dense = nn.Linear(int(lenTrans4*4), output_size)

        self.softplus = nn.Softplus()
        
        
    def forward(self, z):
        out  = self.cnn1(z)
        out  = self.dropout(out) 

        out = self.cnn2(out)
        out = self.avgpoolc(out)
        out  = self.dropout(out) 

        out = self.cnn4(out)
        out = self.avgpoolc(out)
        out  = self.dropout(out) 

        out = self.cnn8(out)
        out = self.avgpoolc(out)
        out  = self.dropout(out) 

        out = self.cnn16(out)
        out = self.avgpoolc(out)
        out  = self.dropout(out) 

        out = self.cnn32(out)
        out = self.avgpoolc(out)
        out  = self.dropout(out) 

        out = self.cnn64(out)
        out  = self.dropout(out) 

        out = self.cnntrans256(out)
        out  = self.dropout(out) 

        out = self.cnntrans128(out)
        out  = self.dropout(out) 

        out = self.cnntrans64(out)
        out  = self.dropout(out) 

        out = self.cnntrans32(out)
        out  = self.dropout(out) 

        out = self.cnntrans16(out)
        out  = self.dropout(out) 

        out = self.cnntrans8(out)
        out  = self.dropout(out) 

        out = self.cnntrans4(out)
        out  = self.dropout(out) 

        out = self.flatten(out)

        out = self.dense(out)

        out = self.softplus(out)
        
        return out
