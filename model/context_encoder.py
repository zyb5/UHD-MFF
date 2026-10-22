import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
import math


def generator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        # layers.append(nn.BatchNorm2d(out_filters))
    return layers


class Generator_for_info(nn.Module):
    def __init__(self, in_channels=4):
        super(Generator_for_info, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),)

        self.mid_layer = nn.Sequential(
            *generator_block(16, 16, normalization=True),
            *generator_block(16, 16, normalization=True),
            *generator_block(16, 16, normalization=True),)

        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
            nn.Sigmoid())

    def forward(self, img_input):
        x = self.input_layer(img_input)
        identity = x
        out = self.mid_layer(x)
        out += identity
        out = self.output_layer(out)
        return out






