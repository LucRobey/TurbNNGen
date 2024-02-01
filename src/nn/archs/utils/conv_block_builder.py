from torch import nn
import numpy as np

class ConvBlockBuilder():
    @classmethod
    def build(cls, 
              in_size, in_ch, out_ch, 
              kernel_size, stride=1, padding=0, 
              bias=False, dilation=1):
        # stride      = 1
        # padding     = 0
        # bias        = False
        # dilation  = 1
        block  = nn.Sequential( 
            nn.Conv1d(in_ch, out_ch, 
                      kernel_size = kernel_size, 
                      stride      = stride, 
                      padding     = padding, 
                      bias        = bias,
                      dilation    = dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(True),
            )
        out_size = np.floor((in_size + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1)
        return block, int(out_size)