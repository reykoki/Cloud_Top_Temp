import torch
from torch.utils.data import Dataset
import tifffile


class CloudDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data_fns = data_dict['data']
        self.truth_fns = data_dict['truth']
        self.transform = transform

    def __len__(self):
        return len(self.data_fns)

    def __getitem__(self, idx):
        data_img = tifffile.imread(self.data_fns[idx])
        truth_img = tifffile.imread(self.truth_fns[idx])

        data_tensor = self.transform(data_img)
        data_tensor = torch.nan_to_num(data_tensor).float()

        truth_tensor = torch.from_numpy(truth_img).long()
        truth_tensor = truth_tensor.permute(2, 0, 1)

#        cod_therm = (truth_tensor[0].unsqueeze(0) >= torch.arange(1, 6).view(5, 1, 1)).float()
        ctt_therm = (truth_tensor[1].unsqueeze(0) >= torch.arange(1, 6).view(5, 1, 1)).float()
#        ctp_therm = (truth_tensor[2].unsqueeze(0) >= torch.arange(1, 6).view(5, 1, 1)).float()
#        transition = truth_tensor[3]
#
#        truth_tensor = torch.cat([
#            cod_therm,
#            ctt_therm,
#            ctp_therm,
#            transition.unsqueeze(0).float()
#        ], dim=0)
#
#        return data_tensor, truth_tensor
        return data_tensor, ctt_therm 
