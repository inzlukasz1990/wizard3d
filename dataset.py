# dataset.py

#The `dataset.py` script defines a custom dataset class called `STLCategoryDataset` for loading and accessing 3D mesh data, along with a collate function `collate_fn` for combining the dataset items into a batch.
#
#`STLCategoryDataset` is a subclass of PyTorch's `Dataset` class and overrides the `__init__`, `__len__`, and `__getitem__` methods:
#
#1. `__init__(self, root_dir, categories, voxel_res)`: The constructor takes the root directory, a list of categories, and the voxel resolution as input arguments. It initializes instance variables and constructs a list of file paths and their corresponding labels for all 3D mesh files in the specified categories.
#
#2. `__len__(self)`: This method returns the total number of 3D mesh files in the dataset.
#
#3. `__getitem__(self, idx)`: This method takes an index `idx` and returns the file path and label at that index.
#
#The `collate_fn` function is used by PyTorch's DataLoader to combine dataset items into a batch. It takes a list of dataset items (in this case, file paths and labels) and the desired batch size as input arguments:
#
#1. Unzip the batch list into separate lists of file paths and labels using `zip(*batch)`.
#
#2. Load and voxelize the 3D mesh data from the file paths using the `load_and_voxelize` function with a voxel resolution of 32.
#
#3. Check if the number of items in the batch is less than the desired batch size. If so, pad the tensors and labels with zeros and -1, respectively, to make the batch size equal to the desired batch size.
#
#4. Return the padded tensors and labels as a tuple.
#
#The `STLCategoryDataset` class can be used to create instances of datasets for different categories of 3D mesh data, and the `collate_fn` function can be used as an argument to PyTorch's DataLoader to create batches of the data.

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


