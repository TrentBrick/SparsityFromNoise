# ConvMixer from Patches are all you need with timm training augmentations: 
# https://github.com/locuslab/convmixer-cifar10/blob/main/train.py


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from types import SimpleNamespace
from models import BaseModel, NoiseLayer
import numpy as np
import time

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class ConvMixer(BaseModel):
    def __init__(self, params: SimpleNamespace):
        super().__init__(params)

        self.noise_layer = NoiseLayer(self.params.diffusion_noise, noise_type=self.params.noise_type, )

        self.features = nn.Sequential(
            nn.Conv2d(3, params.hdim, kernel_size=params.patch_size, stride=params.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(params.hdim),
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(params.hdim, params.hdim, params.kernel_size, groups=params.hdim, padding="same"),
                        nn.GELU(),
                        nn.BatchNorm2d(params.hdim)
                    )),
                    nn.Conv2d(params.hdim, params.hdim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(params.hdim)
            ) for i in range(params.nblocks)],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(params.hdim, params.output_size)

        

    def forward(self, x):

        if self.params.use_convmixer_transforms or self.params.normalize_n_transform_inputs:
            
            x = self.normalizer(x)

        x = self.noise_layer(x)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x