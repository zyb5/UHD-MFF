import torch
import torch.nn as nn

class ResidualRefineNet(nn.Module):
    def __init__(self, in_ch=6, mid_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, I_near, I_far):
        x = torch.cat([I_near, I_far], dim=1)  
        res = 0.1 * self.net(x) 
        return res
