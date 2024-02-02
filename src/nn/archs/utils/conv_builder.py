from .conv_block_builder import ConvBlockBuilder
from torch import nn
import numpy as np

class ConvBuilder():
    def __init__(self, 
                out_channels, 
                kernel_size, 
                stride=1, 
                padding=0, 
                bias=False, 
                dilation=1):
        
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.bias         =  bias,
        self.dilation     = dilation

    def _out_size(self, in_size):
        return np.floor((in_size + 2*self.padding - self.dilation*(self.kernel_size - 1) - 1)/self.stride + 1)

    def _block(self, in_channels):
        return nn.Sequential( 
            nn.Conv1d(in_channels  = in_channels, 
                      out_channels = self.out_channels, 
                      kernel_size  = self.kernel_size, 
                      stride       = self.stride, 
                      padding      = self.padding, 
                      bias         = self.bias,
                      dilation     = self.dilation),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(True),
            )

    def build(self, extra_params):
        block = self._block(extra_params["next_in_channels"])
        extra_params["next_in_channels"] = self.out_channels

        extra_params["next_in_size"] = self._out_size(extra_params["next_in_size"])
        
        return block, extra_params