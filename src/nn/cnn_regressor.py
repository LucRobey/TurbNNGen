from cnn_base import CNNBase
import src.ctes.num_ctes as nctes

class CNNRegressor(CNNBase):
    """
    CNN Model for predicting turbulence flow velocity statistics 
    as continues values: (c1, c2, L, epsilon).
        c1 : Parameter of long-range dependance
            In [0.2 0.4 0.6 0.8]
        c2 : Parameter of intermittency
            In [0.02 0.04 0.06 0.08]
        epsilon : Size of the small-scale regularization
            In [0.5 1.5 2.5 3.5 4.5]
        L : Size of the integral scale.
            In [1000 2000 3000 4000 5000] 
    """
    def __init__(self, input_size=nctes.LEN_SAMPLE, dropout_probs=None):
        super().__init__(input_size=input_size, output_size=4, dropout_probs=dropout_probs)
