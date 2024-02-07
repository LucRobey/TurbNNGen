from torch import nn
import numpy as np

class MaxPoolBuilder():
    @classmethod
    def build(cls,
              in_size,
              kernel_size, 
              stride=None, 
              padding=0, 
              dilation=1, 
              return_indices=False, 
              ceil_mode=False):
        stride = kernel_size if stride is None else stride
        block  = nn.MaxPool1d(
                    kernel_size, 
                    stride, 
                    padding, 
                    dilation, 
                    return_indices, 
                    ceil_mode) 
        round_function = np.ceil if ceil_mode else np.floor
        out_size = round_function(((in_size + 2*padding - dilation*(kernel_size - 1))/stride) + 1)
        return block, int(out_size)