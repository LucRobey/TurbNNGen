from cnn_base import CNNBase
import num_ctes as nctes

class CNNClassifier(CNNBase):
    """
    CNN Model for classifing turbulence flow velocity statistics
    as one of 400 classes: (L, eta, H, c1)
        * L: Large scale delimiter (of the integral range)
            In [1000 2000 3000 4000 5000] 
        * eta: Small scale delimiter (of the dissipative range)
            In [0.5 1.5 2.5 3.5 4.5]
        * H: Hurst exponent 
            In [0.22, 0.44, 0.66, 0.88]
        * c1: -Gamma^2 (Gamma is the intermitence parameter) 
            In [-0.02, -0.04, -0.06, -0.08]
    """
    def __init__(self):
        super().__init__(input_size=nctes.N, output_size=400)
