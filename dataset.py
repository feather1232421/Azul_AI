from torch.utils.data import Dataset
import torch
import numpy as np


class AzulMCTSDataset(Dataset):
    def __init__(self, samples):
        if not samples:
            self.obs = torch.empty((0, 567), dtype=torch.float32)
            self.pi = torch.empty((0, 180), dtype=torch.float32)
            self.z = torch.empty((0,), dtype=torch.float32)
            self.mask = torch.empty((0, 180), dtype=torch.float32)
            return

        obs_list, pi_list, z_list, mask_list = zip(*samples)
        self.obs = torch.from_numpy(np.asarray(obs_list, dtype=np.float32))
        self.pi = torch.from_numpy(np.asarray(pi_list, dtype=np.float32))
        self.z = torch.from_numpy(np.asarray(z_list, dtype=np.float32))
        self.mask = torch.from_numpy(np.asarray(mask_list, dtype=np.float32))

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return self.obs[idx], self.pi[idx], self.z[idx], self.mask[idx]
