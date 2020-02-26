import torch
import numpy as np
import pdb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
AA_LS = 'ACDEFGHIKLMNPQRSTVWY-'

gap_pos_17 = dict(zip(list(range(8,17)), [4,5,5,6,6,7,7,8,8]))
gap_pos_18 = dict(zip(list(range(8,18)), [4,5,5,6,6,7,7,8,8,9]))
gap_pos_dict = {17: gap_pos_17, 18: gap_pos_18}

def train_test_loader(x, y=None, test_size=0.2, batch_size=16):

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True)
    
    x_tensor = torch.from_numpy(X_train).float()
    y_tensor = torch.from_numpy(y_train).float()
    train_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)
    
    x_tensor = torch.from_numpy(X_test).float()
    y_tensor = torch.from_numpy(y_test).float()
    test_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)

    return train_loader, test_loader

def encode_data(data, aa_list = AA_LS, gapped = True, seq_len = 17, gap_pos = gap_pos_dict):
    aa_mapping = dict(zip(AA_LS, list(range(len(AA_LS)))))
    codes = np.eye(len(aa_list))
    if gapped:
        if len(data) < seq_len:
            temp_break = gap_pos_dict[seq_len][len(data)]
            data = data[0:temp_break] + ''.join(['-' for _ in range(seq_len - len(data))]) + data[temp_break:]
    else:
        if len(data) < seq_len:
            data = data + ''.join(['-' for _ in range(seq_len - len(data))])
    return np.array([codes[aa_mapping[kk]] for kk in data])


