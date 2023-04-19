import os
import numpy as np
import trimesh
from stl import mesh
import re

def list_stl_files(directory):
    file_label_pairs = []
    for file in os.listdir(directory):
        if file.endswith('.stl'):
            label = re.match(r'(\w+)_\w+\.stl', file).group(1)
            file_label_pairs.append((os.path.join(directory, file), label))
    return file_label_pairs

def one_hot_encode_label(label, label_to_index):
    num_classes = len(label_to_index)
    one_hot_vector = np.zeros(num_classes, dtype=np.float32)
    one_hot_vector[label_to_index[label]] = 1
    return one_hot_vector

def load_and_preprocess_stl(file_path, target_shape):
    loaded_mesh = mesh.Mesh.from_file(file_path)
    trimesh_mesh = trimesh.Trimesh.from_mesh(loaded_mesh)
    voxel_grid = trimesh.voxel.VoxelGrid(trimesh_mesh, pitch=target_shape[0] / 128.0)
    filled = voxel_grid.fill(method='orthographic_axes')

    # Pad or crop the voxel grid to match the target shape
    padded = np.pad(filled.matrix, [(0, target_shape[0] - filled.matrix.shape[0]),
                                     (0, target_shape[1] - filled.matrix.shape[1]),
                                     (0, target_shape[2] - filled.matrix.shape[2])],
                    mode='constant', constant_values=0)

    return np.expand_dims(padded, axis=-1).astype(np.float32)

