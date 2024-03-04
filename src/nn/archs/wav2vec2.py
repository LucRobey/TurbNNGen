from torch import nn
from torch.nn import functional as F
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H
from src.nn.archs.utils import ConvBlockBuilder, MaxPoolBuilder, AvgPoolBuilder
import torch
import numpy as np
import src.ctes.str_ctes as sctes

class Wav2Vec2Head(nn.Module):
    IN_CHANNELS = 12
    def __init__(self, n_labels):
        super().__init__()
        self.model = self.double_conv = nn.Sequential(
            nn.Conv2d(self.IN_CHANNELS, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, n_labels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_labels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, z):
        return self.model(z)

class Wav2Vec2(nn.Module):
    """
    Adaptation of : 	
        https://ai.meta.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/
    for regression task
    """
    def __init__(self, n_labels, bundle=WAV2VEC2_ASR_BASE_960H):
        super().__init__()

        self.wav2vec2 = bundle.get_model()
        self.wav2vec2.requires_grad_(requires_grad=False)
        # self.wav2vec2.feature_extractor.requires_grad_(requires_grad=False)

        self.wav2vec2_head = Wav2Vec2Head(n_labels)
            
    
    def forward(self, z):
        z = z[:,0,:]
        features, _ = self.wav2vec2.extract_features(z)
        out = torch.stack(features, dim=1)
        out = self.wav2vec2_head(out)
        return out
    

