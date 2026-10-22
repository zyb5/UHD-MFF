import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual

class WeightMapPredictor(nn.Module):
    def __init__(self, in_channels=6, base_channels=64, num_res_blocks=5):
        super(WeightMapPredictor, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_res_blocks)]
        )
        self.tail = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # outputs values in [0, 1]
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        weight_map = self.tail(x)
        return weight_map  # shape: (B, 1, H, W)
