"""Lightweight CSRNet-style density-map network for tree counting.

A VGG-style frontend (first 10 conv layers, ImageNet-pretrained) followed by a
small dilated-conv backend producing a 1-channel density map at 1/8 resolution.
This is the canonical crowd-counting architecture (CSRNet, Li et al. 2018),
scaled down to fit the 8 GB RTX 2070. Chosen because it is the standard "modern
density-map" baseline a reviewer would expect, not a bespoke design.
"""

import torch
import torch.nn as nn
from torchvision import models


class DensityCSRNetLite(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # VGG16 first 10 conv layers (up to conv4_3), output stride 8
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1
                           if pretrained else None)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        # dilated backend (CSRNet config B, narrowed for 8GB)
        self.backend = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return torch.relu(x)  # density is non-negative
