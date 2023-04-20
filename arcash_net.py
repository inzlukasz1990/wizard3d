# arcash_net.py

#This is a PyTorch implementation of a 3D convolutional neural network (CNN) classifier called `ArcAsh3DClassifier`. The code defines a custom neural network module that inherits from `nn.Module`. It is designed to classify 3D data using 3D convolutions and a quaternion rotation layer.
#
#Here is a breakdown of the main components of the ArcAsh3DClassifier class:
#
#1. `__init__(self, batch_size, in_channels, num_classes)`: This is the constructor of the class that initializes the network layers. It takes three input parameters: `batch_size`, `in_channels`, and `num_classes`. The network consists of three 3D convolutional layers (conv1, conv2, and conv3) followed by a fully connected layer (fc).
#
#2. `QuaternionRotation`: This is an imported custom layer for quaternion-based rotations. It takes an axis and angle as input to perform the rotation.
#
#3. `forward(self, x)`: This method defines the forward pass of the network. It takes an input tensor `x` and passes it through the network layers in sequence: conv1, quaternion_rotation, conv2, conv3, flatten, and fc. The output of the network is a tensor containing class scores.
#
#The network can be used to train and classify 3D data by feeding it with appropriate input and using an appropriate loss function during training.

import torch
import torch.nn as nn

from quaternion_rotation import QuaternionRotation
from gaussian_diffusion import GaussianDiffusion

class ArcAsh3DClassifier(nn.Module):
    def __init__(self, batch_size, in_channels, num_classes):
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
        self.quaternion_rotation = QuaternionRotation(axis, theta)

        self.conv2 = nn.Sequential(
            nn.Conv3d(128, batch_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=1, stride=1)
        )

        self.gaussian_diffusion = GaussianDiffusion(channels=1)

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

