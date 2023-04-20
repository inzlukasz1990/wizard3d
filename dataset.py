# dataset.py

import os
import torch
from torch.utils.data import Dataset
from data_utils import load_and_voxelize
from tqdm import tqdm

class STLCategoryDataset(Dataset):
    def __init__(self, root_dir, categories, voxel_res):
        self.root_dir = root_dir
        self.categories = categories
        self.voxel_res = voxel_res
        self.filepaths = []
        self.labels = []
        self.loaded_items = 0

        for label, category in enumerate(categories):
            category_dir = os.path.join(root_dir, category)
            filenames = os.listdir(category_dir)

            for filename in filenames:
                if filename.lower().endswith('.stl'):
                    self.filepaths.append(os.path.join(category_dir, filename))
                    self.labels.append(label)

        self.progress_bar = tqdm(total=len(self.filepaths), desc="Loading data", position=0, leave=True)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        file_path = self.filepaths[idx]
        voxel_data = load_and_voxelize([file_path], self.voxel_res)

        self.loaded_items += 1
        self.progress_bar.update(1)

        if self.loaded_items == len(self.filepaths):
            self.progress_bar.close()

        return self.filepaths[idx], self.labels[idx]

def collate_fn(batch, batch_size):
    file_paths, labels = zip(*batch)
    tensors = load_and_voxelize(file_paths, voxel_res=32)

    # Pad tensors and labels if the sum of items is lower than the batch size
    num_padding_items = batch_size - len(tensors)
    if num_padding_items > 0:
        padding_shape = (num_padding_items, *tensors.shape[1:])
        padding_tensor = torch.zeros(padding_shape, dtype=tensors.dtype)
        tensors = torch.cat((tensors, padding_tensor), dim=0)

        padding_labels = torch.full((num_padding_items,), -1, dtype=torch.long)
        labels = torch.cat((torch.tensor(labels, dtype=torch.long), padding_labels), dim=0)

    return tensors, labels


