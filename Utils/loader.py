import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils import data
import numpy as np
import pandas as pd

AA_LS = 'ACDEFGHIKLMNPQRSTVWY'


def synthetic_data(num_samples=1000, seq_len=10, aa_list=AA_LS):
    aa_size = len(aa_list)
    data = np.zeros((num_samples, seq_len), dtype=int)
    out = np.zeros((num_samples), dtype=int)
    for i in range(num_samples):
        for j in range(seq_len):
            data[i, j] = np.random.randint(aa_size)
    for i in range(num_samples):
        if i < round(num_samples / 2):
            out[i] = 1

    return data, out


def encode_data(data, aa_list=AA_LS):
    codes = np.eye(len(aa_list))
    x = codes[data]
    return x


def train_test_loader(x, y, test_size=0.3, batch_size=16):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True)

    x_tensor = torch.from_numpy(X_train).float()
    y_tensor = torch.from_numpy(y_train).float()
    train_dataset = data.TensorDataset(x_tensor, y_tensor)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size)

    x_tensor = torch.from_numpy(X_test).float()
    y_tensor = torch.from_numpy(y_test).float()
    test_dataset = data.TensorDataset(x_tensor, y_tensor)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset)

    return train_loader, test_loader
