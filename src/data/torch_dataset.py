import numpy as np
import torch
from torch.utils.data import Dataset


class ECGBeatDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, label_dtype: torch.dtype = torch.float32):
        """label_dtype is torch.float32 for the binary task (BCEWithLogitsLoss
        expects a float target) and torch.long for the multi-class task
        (CrossEntropyLoss expects integer class indices)."""
        self.X = torch.from_numpy(X).float().unsqueeze(1)  # (n, 1, beat_len)
        self.y = torch.from_numpy(y).to(label_dtype)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
