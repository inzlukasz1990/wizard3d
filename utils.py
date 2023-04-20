# utils.py

import os
import trimesh
from trimesh.voxel import creation

def print_voxel_dimensions(root_dir, voxel_res):
    # Collect all STL file paths
    file_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".stl"):
                file_paths.append(os.path.join(dirpath, filename))

    # Process and print voxel dimensions for each STL file
    for file_path in file_paths:
        mesh = trimesh.load_mesh(file_path)
        voxels = creation.voxelize(mesh, voxel_res)
        voxel_array = voxels.matrix

        print(f"Voxel dimensions for {file_path}: {voxel_array.shape}")

# Example usage:
root_dir = "/home/laptop/Projekty/wizard3d/Thingi10K_name_and_category"
voxel_res = 32
print_voxel_dimensions(root_dir, voxel_res)

