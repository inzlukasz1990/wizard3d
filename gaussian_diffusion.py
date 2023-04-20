# gaussian_diffusion.py

import torch
import torch.nn as nn

class GaussianDiffusion(nn.Module):
    def __init__(self, channels=4, std_dev=0.1):
        super(GaussianDiffusion, self).__init__()
        self.channels = channels
        self.std_dev = std_dev

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std_dev
            x = x + noise
        return x
