# data_utils.py

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
        try:
            mesh = trimesh.load_mesh(file_path)
            voxels = creation.voxelize(mesh, voxel_res)
            voxel_array = voxels.matrix

            # Scale the voxel array
            scaled_voxel_array = scale_voxels(voxel_array, target_shape)
            padded_voxel_array = pad_matrix(scaled_voxel_array, target_shape)

            tensor = torch.from_numpy(padded_voxel_array).float()
            batch_tensors.append(tensor.unsqueeze(0))

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            empty_tensor = torch.zeros(target_shape, dtype=torch.float)
            batch_tensors.append(empty_tensor.unsqueeze(0))

    return torch.cat(batch_tensors)


def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return torch.from_numpy(one_hot)

