from torch.utils.data import Dataset
import torch
import numpy as np


class AzulMCTSDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs, pi, z = self.samples[idx]

        obs = torch.tensor(obs, dtype=torch.float32)
        pi = torch.tensor(pi, dtype=torch.float32)
        z = torch.tensor(z, dtype=torch.float32)

        return obs, pi, z