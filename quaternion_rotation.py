# quaternion_rotation.py

import torch
import torch.nn as nn

class QuaternionRotation(nn.Module):
    def __init__(self, axis, theta):
        super(QuaternionRotation, self).__init__()

        self.axis = axis
        self.theta = theta

    def forward(self, x):
        if self.training:
            q = self._axis_angle_to_quaternion(self.axis, self.theta)
            return self._rotate_voxels(x, q)
        return x

    @staticmethod
    def _axis_angle_to_quaternion(axis, theta):
        sin_half_theta = torch.sin(theta / 2)
        cos_half_theta = torch.cos(theta / 2)

        return torch.cat((
            cos_half_theta.unsqueeze(0),
            axis * sin_half_theta
        ), dim=0)

    @staticmethod
    def _rotate_voxels(x, q):
        q_conj = q * torch.tensor([1, -1, -1, -1], dtype=q.dtype)

        # Convert voxel grid to a list of 3D coordinates
        coords = torch.nonzero(x, as_tuple=False).float()

        # Apply quaternion rotation
        rotated_coords = QuaternionRotation._apply_quaternion_rotation(coords, q, q_conj)

        # Create a new voxel grid with rotated coordinates
        rotated_x = torch.zeros_like(x)
        rotated_coords = rotated_coords.long()

        # Set the values of the new rotated voxel grid at the rotated coordinates
        in_bounds = ((rotated_coords >= 0) & (rotated_coords < x.shape[1])).all(dim=1)
        rotated_coords = rotated_coords[in_bounds]

        rotated_x[rotated_coords[:, 0], rotated_coords[:, 1], rotated_coords[:, 2]] = 1

        return rotated_x

    @staticmethod
    def _apply_quaternion_rotation(coords, q, q_conj):
        coords = torch.cat((torch.zeros((coords.shape[0], 1)), coords), dim=1)
        coords_quaternion = QuaternionRotation._multiply_quaternions(q, coords)
        rotated_coords_quaternion = QuaternionRotation._multiply_quaternions(coords_quaternion, q_conj)
        return rotated_coords_quaternion[:, 1:]

    @staticmethod
    def _multiply_quaternions(q1, q2):
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

        return torch.stack((w, x, y, z), dim=-1)

