# data_utils.py

#The `data_utils.py` script defines two utility functions for working with 3D mesh data and labels:
#
#1. `load_and_voxelize(file_paths, voxel_res)`: This function takes a list of file paths and a voxel resolution as input arguments. It loads 3D mesh data from the file paths, converts them into a voxel representation with the specified resolution, and returns a batch of voxel tensors.
#
#2. `one_hot_encode(labels, num_classes)`: This function takes a list of integer labels and the number of classes as input arguments. It one-hot encodes the labels and returns a PyTorch tensor with the one-hot encoded labels.
#
#Here's a detailed explanation of the steps involved in each function:
#
#`load_and_voxelize(file_paths, voxel_res)`:
#1. Initialize an empty list `batch_tensors` to store the voxel tensors.
#2. Iterate through each file path in `file_paths`.
#3. Load the 3D mesh data from the file using `trimesh.load_mesh(file_path)`.
#4. Convert the mesh data into a voxel representation with the specified resolution using `creation.voxelize(mesh, voxel_res)`.
#5. Get the matrix representation of the voxels using `voxels.matrix`.
#6. Convert the matrix to a PyTorch tensor using `torch.from_numpy(voxel_array).float()`.
#7. Add a new dimension to the tensor using `tensor.unsqueeze(0)` and append it to the `batch_tensors` list.
#8. Concatenate all the tensors in the `batch_tensors` list using `torch.cat(batch_tensors)` and return the result.
#
#`one_hot_encode(labels, num_classes)`:
#1. Create an array of zeros with the shape `(len(labels), num_classes)` using `np.zeros((len(labels), num_classes))`.
#2. Set the appropriate indices in the one-hot array to 1 using `one_hot[np.arange(len(labels)), labels] = 1`.
#3. Convert the one-hot array to a PyTorch tensor using `torch.from_numpy(one_hot)` and return the result.

import trimesh
import numpy as np
from trimesh.voxel import creation
import torch
from scipy.ndimage import zoom

def pad_matrix(matrix, target_shape=(8, 8, 8)):
    padding = [(0, target - current) for current, target in zip(matrix.shape, target_shape)]
    padded_matrix = np.pad(matrix, padding, mode='constant', constant_values=0)
    return padded_matrix

def scale_voxels(voxel_array, target_shape=(8, 8, 8)):
    scale_factors = [target / current for current, target in zip(voxel_array.shape, target_shape)]
    min_scale = min(scale_factors)
    scaled_voxel_array = zoom(voxel_array, min_scale, order=1)
    return scaled_voxel_array

def load_and_voxelize(file_paths, voxel_res, target_shape=(8, 8, 8)):
    batch_tensors = []

    for file_path in file_paths:
        mesh = trimesh.load_mesh(file_path)
        voxels = creation.voxelize(mesh, voxel_res)
        voxel_array = voxels.matrix

        # Scale the voxel array
        scaled_voxel_array = scale_voxels(voxel_array, target_shape)
        padded_voxel_array = pad_matrix(scaled_voxel_array, target_shape)

        tensor = torch.from_numpy(padded_voxel_array).float()
        batch_tensors.append(tensor.unsqueeze(0))

    return torch.cat(batch_tensors)

def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return torch.from_numpy(one_hot)

