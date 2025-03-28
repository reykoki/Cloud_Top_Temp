import torch
from torch.utils.data import Dataset
import skimage


class CloudDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data_fns = data_dict
        self.transform = transform

    def __len__(self):
        return len(self.data_fns['data'])

    def __getitem__(self, idx):
        data_fn = self.data_fns['data'][idx]
        truth_fn = self.data_fns['truth'][idx]
        data_img = skimage.io.imread(data_fn, plugin='tifffile')
        truth_img = skimage.io.imread(truth_fn, plugin='tifffile')
        data_tensor = self.transform(data_img)#.unsqueeze_(0)
        data_tensor = torch.nan_to_num(data_tensor)
        truth_tensor = self.transform(truth_img)#.unsqueeze_(0)
        truth_tensor = (truth_tensor > 0.0) * 1.0
        truth_tensor = truth_tensor.type(torch.float32)

        return data_tensor, truth_tensor
