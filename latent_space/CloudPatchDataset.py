import torch
from torch.utils.data import Dataset
import tifffile
import numpy as np

class CloudPatchDataset(Dataset):
    def __init__(self, file_list, patch_size=128):
        self.patch_size = patch_size
        self.patch_meta = []

        # Assume all images are 2048x2048 and divisible by patch_size
        self.grid_size = 2048 // patch_size

        for file_idx, data_path in enumerate(file_list['data']):
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    self.patch_meta.append((data_path, file_list['truth'][file_idx], i, j))
        

    def get_patch(self, ps, row_idx, col_idx, image):
        patch = image[row_idx * ps:(row_idx + 1) * ps,
                      col_idx * ps:(col_idx + 1) * ps,
                      :]
        return patch

    def get_hist(self, class_map):
        hist = torch.tensor([
            (class_map == 1).sum(),  # low
            (class_map == 2).sum(),  # mid
            (class_map == 3).sum(),  # high
        ], dtype=torch.float32)
        #hist /= hist.numel()  # normalize
        hist /= hist.sum() if hist.sum() > 0 else 1.0
        return hist
    
    def __len__(self):
        return len(self.patch_meta)

    def __getitem__(self, idx):
        data_path, truth_path, row_idx, col_idx = self.patch_meta[idx]

        # Load and standardize image to (3, H, W)
        image = tiff.imread(data_path)
        truth = tiff.imread(truth_path)

        img_patch = self.get_patch(self.patch_size, row_idx, col_idx, image)
        truth_patch = self.get_patch(self.patch_size, row_idx, col_idx, truth)
        class_map = np.sum(truth_patch, axis=2).astype(np.uint8)  # shape: (ph, pw)
        hist = self.get_hist(class_map)

        # Convert to (C, H, W) format for PyTorch
        img_patch = np.transpose(img_patch, (2, 0, 1))  # (C, H, W)
        return {
            'image': torch.tensor(img_patch, dtype=torch.float32),
            'pred': torch.tensor(truth_patch, dtype=torch.float32),
            'hist': hist,
            'source_file': data_path,
            'row': row_idx,
            'col': col_idx,
        }
