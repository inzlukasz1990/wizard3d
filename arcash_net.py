# arcash_net.py

import torch
import torch.nn as nn

from quaternion_rotation import QuaternionRotation
from gaussian_diffusion import GaussianDiffusion

class ArcAsh3DClassifier(nn.Module):
    def __init__(self, device, batch_size, in_channels, num_classes):
        super(ArcAsh3DClassifier, self).__init__()

        self.batch_size = batch_size

        self.conv1 = nn.Sequential(
            nn.Conv3d(batch_size, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=1, stride=1)
        )

        # Add the quaternion rotation layer
        axis = torch.tensor([1, 1, 1])  # Replace with the desired rotation axis
        theta = torch.tensor(0.1)  # Replace with the desired rotation angle in radians
        self.quaternion_rotation = QuaternionRotation(device, axis, theta)

        self.conv2 = nn.Sequential(
            nn.Conv3d(128, batch_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=1, stride=1)
        )

        self.gaussian_diffusion = GaussianDiffusion(device, channels=1)

        self.conv3 = nn.Sequential(
            nn.Conv3d(batch_size, batch_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=1, stride=1)
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(512, batch_size),
            nn.ReLU(),
            nn.Linear(batch_size, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.quaternion_rotation(x)
        x = self.conv2(x)
        x = self.gaussian_diffusion(x) 
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

