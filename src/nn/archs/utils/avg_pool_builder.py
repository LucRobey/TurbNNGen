from torch import nn
import numpy as np

class AvgPoolBuilder():
    @classmethod
    def build(cls,
              in_size,
              kernel_size,
              stride=None,
              padding=0,
              ceil_mode=False,
              count_include_pad=True):
        stride = kernel_size if stride is None else stride
        block  = nn.AvgPool1d(
                    kernel_size,
                    stride,
                    padding,
                    ceil_mode,
                    count_include_pad) 
        round_function = np.ceil if ceil_mode else np.floor
        out_size = round_function(((in_size + 2*padding - kernel_size)/stride) + 1)
        return block, out_size