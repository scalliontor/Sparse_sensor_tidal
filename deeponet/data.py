# data.py
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset


class DeepONetNPZDataset(Dataset):
    def __init__(self, npz_path: str, dtype=np.float32):
        data = np.load(npz_path)
        self.branch = data["branch"].astype(dtype)
        self.trunk = data["trunk"].astype(dtype)
        labels = data["labels"]
        if labels.ndim == 1:
            labels = labels[:, None]
        self.labels = labels.astype(dtype)
        assert self.branch.shape[0] == self.trunk.shape[0] == self.labels.shape[0]
        self.N, self.m = self.branch.shape

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return (torch.from_numpy(self.branch[idx]), torch.from_numpy(self.trunk[idx]), torch.from_numpy(self.labels[idx]))


class DeepONetArrayDataset(Dataset):
    def __init__(self, branch, trunk, labels, dtype=np.float32):
        self.branch = branch.astype(dtype)
        self.trunk = trunk.astype(dtype)
        labels_arr = labels.astype(dtype)
        if labels_arr.ndim == 1:
            labels_arr = labels_arr[:, None]
        self.labels = labels_arr
        self.N, self.m = self.branch.shape

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return (torch.from_numpy(self.branch[idx]), torch.from_numpy(self.trunk[idx]), torch.from_numpy(self.labels[idx]))
