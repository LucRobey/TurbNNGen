from torch import nn
import numpy as np

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
        self.avgpool = nn.AvgPool1d(2, ceil_mode=False)
        self.avgpoolc = nn.AvgPool1d(2, ceil_mode=True)

        self.cnn1 = nn.Sequential( 
            nn.Conv1d(1, 16, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            )
        len1 = np.floor((input_size + 2*0 - 1*(1 - 1) - 1)/1 + 1) # cnn1
        
        self.cnn2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size = 2, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            )
        len2 = np.floor((len1 + 2*0 - 1*(2 - 1) - 1)/1 + 1) # cnn2
        len2 = np.ceil((len2 + 2*0 - 2)/2 + 1)  # avgpoolc

        self.cnn4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            )
        len4 = np.floor((len2 + 2*0 - 1*(4 - 1) - 1)/1 + 1) # cnn4
        len4 = np.ceil((len4 + 2*0 - 2)/2 + 1)  # avgpoolc 

        self.cnn8 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size = 8, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            )
        len8 = np.floor((len4 + 2*0 - 1*(8 - 1) - 1)/1 + 1) # cnn8
        len8 = np.ceil((len8 + 2*0 - 2)/2 + 1)  # avgpoolc

        self.cnn16 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size = 16, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            )
        len16 = np.floor((len8 + 2*0 - 1*(16 - 1) - 1)/1 + 1) # cnn16
        len16 = np.ceil((len16 + 2*0 - 2)/2 + 1)  # avgpoolc

        self.cnn32 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size = 32, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            )
        len32 = np.floor((len16 + 2*0 - 1*(32 - 1) - 1)/1 + 1) # cnn32
        len32 = np.ceil((len32 + 2*0 - 2)/2 + 1)  # avgpoolc

        self.cnn64 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size = 64, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            )
        len64 = np.floor((len32 + 2*0 - 1*(64 - 1) - 1)/1 + 1) # cnn64

        self.cnntrans256 = nn.Sequential(
            nn.ConvTranspose1d(256, 256, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            )
        lenTrans256 = (len64 - 1)*1 - 2*1 + 1*(3 - 1) + 0 + 1

        self.cnntrans128 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            )
        lenTrans128 = (lenTrans256 - 1)*1 - 2*1 + 1*(3 - 1) + 0 + 1

        self.cnntrans64 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            )
        lenTrans64 = (lenTrans128 - 1)*1 - 2*1 + 1*(3 - 1) + 0 + 1

        self.cnntrans32 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            )
        lenTrans32 = (lenTrans64 - 1)*1 - 2*1 + 1*(3 - 1) + 0 + 1
        
        self.cnntrans16 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            )
        lenTrans16 = (lenTrans32 - 1)*1 - 2*1 + 1*(3 - 1) + 0 + 1

        self.cnntrans8 = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            )
        lenTrans8 = (lenTrans16 - 1)*1 - 2*1 + 1*(3 - 1) + 0 + 1
        
        self.cnntrans4 = nn.Sequential(
            nn.ConvTranspose1d(8, 4, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm1d(4),
            nn.ReLU(True),
            )
        lenTrans4 = (lenTrans8 - 1)*1 - 2*1 + 1*(3 - 1) + 0 + 1

        self.flatten = nn.Flatten()

        self.dense = nn.Linear(int(lenTrans4*4), output_size)

        self.softplus = nn.Softplus()
        
        
    def forward(self, z):
        out  = self.cnn1(z)

        out = self.cnn2(out)
        out = self.avgpoolc(out)

        out = self.cnn4(out)
        out = self.avgpoolc(out)

        out = self.cnn8(out)
        out = self.avgpoolc(out)

        out = self.cnn16(out)
        out = self.avgpoolc(out)

        out = self.cnn32(out)
        out = self.avgpoolc(out)

        out = self.cnn64(out)

        out = self.cnntrans256(out)
        out = self.cnntrans128(out)
        out = self.cnntrans64(out)
        out = self.cnntrans32(out)
        out = self.cnntrans16(out)
        out = self.cnntrans8(out)
        out = self.cnntrans4(out)

        out = self.flatten(out)

        out = self.dense(out)

        out = self.softplus(out)
        
        return out
