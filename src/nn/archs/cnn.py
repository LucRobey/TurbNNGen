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
    def __init__(self, 
                 in_size, 
                 out_size,
                 blocks_builders,
                 dropout_probs):
        super().__init__()
        # if dropout_probs is None:
        #     dropout_probs = [0.5] * 14  # Adjust the length based on the number of layers

        # self.dropout_probs = dropout_probs
        # self.dropout       = nn.Dropout()

        self.blocks = []

        extra_params = {
            "next_in_size": in_size,
            "next_in_ch"  : 1,
        }
        len = in_size
        for builder in blocks_builders:
            block, extra_params = builder.build(extra_params) 
            self.blocks.append(block)       
            # Create AvgPoolWrapper
            # Create FlattenWrapper
            # Create DenseWrapper
            # Create SoftplusWrapper
            # Create DropoutWrapper

        # self.cnn2, len2 = ConvBlockBuilder.build(input_size, 1, 128, 2)
        # self.pool2, len2 = AvgPoolBuilder.build(len2, 4, ceil_mode=True)
        # # print(f"{len2 = }")

        # self.cnn4, len4 = ConvBlockBuilder.build(len2, 128, 64, 4)
        # self.pool4, len4 = AvgPoolBuilder.build(len4, 4, ceil_mode=True)
        # # print(f"{len4 = }")

        # self.cnn8, len8 = ConvBlockBuilder.build(len4, 64, 32, 8)
        # self.pool8, len8 = AvgPoolBuilder.build(len8, 4, ceil_mode=True)
        # # print(f"{len8 = }")

        # self.cnn16, len16 = ConvBlockBuilder.build(len8, 32, 16, 16)
        # self.pool16, len16 = AvgPoolBuilder.build(len16, 4, ceil_mode=True)
        # # print(f"{len16 = }")

        # self.cnn32, len32 = ConvBlockBuilder.build(len16, 16, 8, 32)
        # self.pool32, len32 = AvgPoolBuilder.build(len32, 4, ceil_mode=True)
        # # print(f"{len32 = }")
        

        # self.flatten = nn.Flatten()
        # len_flatten  = int(len32 * 8)
        # # print(f"{len_flatten = }")

        # len_dense1  = len_flatten // 2
        # self.dense1 = nn.Linear(len_flatten, len_dense1)
        # # print(f"{len_dense1 = }")

        # len_dense2  = len_dense1 // 2
        # self.dense2 = nn.Linear(len_dense1, len_dense2)
        # # print(f"{len_dense2 = }")

        # len_dense3  = self.OUTPUT_SIZE
        # self.dense3 = nn.Linear(len_dense2, len_dense3)
        # # print(f"{len_dense3 = }")

        # self.softplus = nn.Softplus()
        
        
    def forward(self, z):
        out = z

        for block in self.blocks:
            out = block(out)

        # out = self.cnn2(out)
        # out = self.pool2(out)
        
        # out = self.cnn4(out)
        # out = self.pool4(out)
        
        # out = self.cnn8(out)
        # out = self.pool8(out)
        
        # out = self.cnn16(out)
        # out = self.pool16(out)
        
        # out = self.cnn32(out)
        # out = self.pool32(out)

        # # out = self.cnn64(out)
        # # out = self.pool64(out)

        # out = self.flatten(out)

        # out = self.dense1(out)
        # out = self.dense2(out)
        # out = self.dense3(out)

        # out = self.softplus(out)
        
        return out