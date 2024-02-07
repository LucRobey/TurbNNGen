from torch import nn
from torch.nn import functional as F
from src.nn.archs.utils import ConvBlockBuilder, MaxPoolBuilder, AvgPoolBuilder
import numpy as np
import src.ctes.str_ctes as sctes

class CNN_ALL_VDCNNFRW_M18(nn.Module):
    """
    Adaptation of : 	
        https://doi.org/10.48550/arXiv.1610.00087
        VERY DEEP CONVOLUTIONAL NEURAL NETWORKS FOR RAW WAVEFORMS
        Wei Dai*, Chia Dai*, Shuhui Qu, Juncheng Li, Samarjit Das
        M18
    for regression task
    """
    INPUT_CHANNELS = 1
    OUTPUT_SIZE    = 4
    LABELS = [sctes.C1, sctes.C2, sctes.L, sctes.EPSILON]
    def __init__(self, input_size, dropout_probs=None):
        super().__init__()
        
        self.sequential = nn.Sequential()

        self.test, _ = ConvBlockBuilder.build(input_size, 1, 64, 80, 4)

        # {M18} [80/4, 64]
        block, input_size = ConvBlockBuilder.build(input_size, 1, 64, 80, 4)
        self.sequential.add_module("Conv_0", block)
        block, input_size = MaxPoolBuilder.build(input_size, 4)
        self.sequential.add_module("MaxPool_0", block)
        
        # {M18} [3, 64] x 4
        block, input_size = ConvBlockBuilder.build(input_size, 64, 64, 3, 1)
        self.sequential.add_module(f"Conv_1_{0}", block)
        for i in range(3):
            block, input_size = ConvBlockBuilder.build(input_size, 64, 64, 3, 1)
            self.sequential.add_module(f"Conv_1_{i+1}", block)
        block, input_size     = MaxPoolBuilder.build(input_size, 4)
        self.sequential.add_module("MaxPool_1", block)

        # {M18} [3, 128] x 4
        block, input_size = ConvBlockBuilder.build(input_size, 64, 128, 3, 1)
        self.sequential.add_module(f"Conv_2_{0}", block)
        for i in range(3):
            block, input_size = ConvBlockBuilder.build(input_size, 128, 128, 3, 1)
            self.sequential.add_module(f"Conv_2_{i+1}", block)
        block, input_size     = MaxPoolBuilder.build(input_size, 4)
        self.sequential.add_module("MaxPool_2", block)

        # {M18} [3, 256] x 4
        block, input_size = ConvBlockBuilder.build(input_size, 128, 256, 3, 1)
        self.sequential.add_module(f"Conv_3_{0}", block)
        for i in range(3):
            block, input_size = ConvBlockBuilder.build(input_size, 256, 256, 3, 1)
            self.sequential.add_module(f"Conv_3_{i+1}", block)
        block, input_size     = MaxPoolBuilder.build(input_size, 4)
        self.sequential.add_module("MaxPool_3", block)

        # {M18} [3, 512] x 4
        block, input_size = ConvBlockBuilder.build(input_size, 256, 512, 3, 1)
        self.sequential.add_module(f"Conv_4_{0}", block)
        for i in range(3):
            block, input_size = ConvBlockBuilder.build(input_size, 512, 512, 3, 1)
            self.sequential.add_module(f"Conv_4_{i+1}", block)

        # {M18} Global Average Pooling
        block, input_size = AvgPoolBuilder.build(input_size, input_size)
        self.sequential.add_module("Global_Avg_Pool", block)
        
        # {TurbNNGen} Flatten
        self.sequential.add_module("Flatten", nn.Flatten())
        
        input_size = 512  # Number of channels after Global Average Pooling
        
        # {TurbNNGen} Dense
        self.sequential.add_module("Dense", nn.Linear(input_size, self.OUTPUT_SIZE))

        # {TurbNNGen} Softplus
        self.sequential.add_module("Softplus", nn.Softplus()) 
        
        
    def forward(self, z):
        return self.sequential(z)
