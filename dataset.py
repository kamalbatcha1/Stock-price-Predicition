import numpy as np
import torch
from torch.utils.data import Dataset

def create_sequences(data, target, seq_len):

    X, y = [], []

    for i in range(len(data) - seq_len):

        X.append(data[i:i+seq_len])
        y.append(target[i+seq_len])

    return np.array(X), np.array(y)


class StockDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]