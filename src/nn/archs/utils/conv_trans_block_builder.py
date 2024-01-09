from torch import nn

class ConvTransBlockBuilder():
    @classmethod
    def build(cls, 
              in_size, in_ch, out_ch,
              kernel_size=3, stride=1, padding=1,
              bias=False, dilation=1, output_padding=0):
        # kernel_size    = 3
        # stride         = 1
        # padding        = 1
        # bias           = False
        # dilation     = 1
        # output_padding = 0
        block  = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, 
                               kernel_size    = kernel_size, 
                               stride         = stride, 
                               padding        = padding, 
                               bias           = bias,
                               dilation       = dilation,
                               output_padding = output_padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(True),
            )
        out_size = (in_size - 1)*stride - 2*padding + dilation*(kernel_size - 1) + output_padding + 1
        return block, out_size