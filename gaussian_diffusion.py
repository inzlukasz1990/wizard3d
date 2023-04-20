# gaussian_diffusion.py

import torch
import torch.nn as nn

class GaussianDiffusion(nn.Module):
    def __init__(self, device, channels=4, std_dev=0.1):
        super(GaussianDiffusion, self).__init__()
        self.channels = channels
        self.std_dev = std_dev
        self.device = device

    def forward(self, x):
        if self.training:
            device = self.device if self.device is not None else x.device
            noise = torch.randn_like(x, device=device) * self.std_dev
            x = x + noise
        return x

