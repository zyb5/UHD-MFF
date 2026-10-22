import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn as nn

class W_P(nn.Module):
    def __init__(self, in_channels=3, temperature=5.0):
        super(W_P, self).__init__()
        self.temperature = temperature

        self.features = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear'),
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32, normalization=True),
            *discriminator_block(32, 64, normalization=True),
            *discriminator_block(64, 128, normalization=True),
            *discriminator_block(128, 128, normalization=False),
            nn.Dropout(p=0.5)
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),         # [B, 256]
            nn.Linear(256, 3)
        )


    def forward(self, img1, img2):
        feat1 = self.features(img1)  # [B, 128, H, W]
        feat2 = self.features(img2)  # [B, 128, H, W]
        combined = torch.cat([feat1, feat2], dim=1)  # [B, 256, H, W]
        logits = self.fc(combined)  # [B, 3]

        weights = F.softmax(logits / self.temperature, dim=1)

        return weights

class OptimizableLUT(torch.nn.Module):
    def __init__(self, lut_tensor):
        super(OptimizableLUT, self).__init__()
        self.lut = torch.nn.Parameter(lut_tensor)

    def forward(self):
        return self.lut


def discriminator_block(in_filters, out_filters, normalization=True):
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1),
              nn.LeakyReLU(0.2, inplace=True)]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers