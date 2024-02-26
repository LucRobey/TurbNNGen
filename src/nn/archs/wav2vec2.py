from torch import nn
from torch.nn import functional as F
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H
from src.nn.archs.utils import ConvBlockBuilder, MaxPoolBuilder, AvgPoolBuilder
import numpy as np
import src.ctes.str_ctes as sctes

class Wav2Vec2Head(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential()
        self.model.add_module("Dense", nn.Linear(input_size, output_size))

    def forward(self, z):
        return self.model(z)

class Wav2Vec2(nn.Module):
    """
    Adaptation of : 	
        https://ai.meta.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/
    for regression task
    """
    INPUT_CHANNELS = 1
    def __init__(self, input_size, output_size, bundle=WAV2VEC2_ASR_BASE_960H):
        super().__init__()

        self.wav2vec2 = bundle.get_model()
        self.wav2vec2.feature_extractor.requires_grad_(requires_grad=False)

        self.wave2vec2_head = Wav2Vec2Head(input_size, output_size)
        
        
    def forward(self, z):
        out = self.wav2vec2(z)
        # out = self.wave2vec_head(out)
        return out
