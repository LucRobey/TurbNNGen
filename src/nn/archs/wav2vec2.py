from torch import nn
from torch.nn import functional as F
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H
from src.nn.archs.utils import ConvBlockBuilder, MaxPoolBuilder, AvgPoolBuilder
import torch
import numpy as np
import src.ctes.str_ctes as sctes

class Wav2Vec2Head(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.model = self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, z):
        return self.model(z)

class Wav2Vec2(nn.Module):
    """
    Adaptation of : 	
        https://ai.meta.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/
    for regression task
    """
    INPUT_CHANNELS  = 1
    IN_CHANNELS_HEAD = 12
    def __init__(self, output_size, bundle=WAV2VEC2_ASR_BASE_960H):
        super().__init__()

        self.wav2vec2 = bundle.get_model()
        self.wav2vec2.feature_extractor.requires_grad_(requires_grad=False)

        self.wave2vec2_head = Wav2Vec2Head(self.IN_CHANNELS_HEAD, output_size)
            
    
    def forward(self, z):
        features, _ = self.wav2vec2.extract_features(z)
        out = torch.stack(features, dim=1)
        out = self.wave2vec_head(out)
        return out
