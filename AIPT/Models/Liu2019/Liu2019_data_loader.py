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

def dataset_split(df, test_size= 0.1, batch_size = 100, random_seed = 0, neg_ratio = 0.1, nd_ratio = 0.01):
    dat = df.loc[df['enriched'] != 'not_determined']
    x = dat['cdr3'].values
    X_dat = np.array([encode_data(item, gapped = False, seq_len = 18) for item in x])
    
    x_tot = df['cdr3'].values
    X_tot = np.array([encode_data(item, gapped = False, seq_len = 18) for item in x_tot])

    # scale y_reg
    y_reg = df['log10(R3/R2)'].values
    y_reg_mean = np.mean(y_reg)
    y_reg_std = np.std(y_reg)
    y_reg_new = (y_reg - y_reg_mean) / y_reg_std
    y_class = np.array([int(xx == 'positive') for xx in dat['enriched'].values])
    
    np.random.seed(random_seed)
    mask_act = [xx == 'positive' for xx in df['enriched'].values]
    mask_neg = [xx == 'negative' for xx in df['enriched'].values]
    mask_neg1 = np.random.rand(len(df), ) > neg_ratio
    mask_neg2 = np.random.rand(len(df), ) > neg_ratio
    mask_neg3 = np.random.rand(len(df), ) > neg_ratio
    
    mask_ND = [xx == 'not_determined' for xx in df['enriched'].values]
    mask_nd1 = np.random.rand(len(df), ) < nd_ratio
    mask_nd2 = np.random.rand(len(df), ) < nd_ratio
    mask_nd3 = np.random.rand(len(df), ) < nd_ratio
    
    mask1 = np.array([mask_act[i] or (mask_neg1[i] and mask_neg[i]) or (mask_ND[i] and mask_nd1[i]) for i in range(len(df))])
    mask2 = np.array([mask_act[i] or (mask_neg2[i] and mask_neg[i]) or (mask_ND[i] and mask_nd2[i]) for i in range(len(df))])
    mask3 = np.array([mask_act[i] or (mask_neg3[i] and mask_neg[i]) or (mask_ND[i] and mask_nd3[i]) for i in range(len(df))])
    
    classifier_loader = train_test_loader(X_dat, y_class, test_size=test_size, batch_size=batch_size)
    regressor_loader1 = train_test_loader(X_tot[mask1], y_reg_new[mask1], test_size, batch_size)
    regressor_loader2 = train_test_loader(X_tot[mask2], y_reg_new[mask2], test_size, batch_size)
    regressor_loader3 = train_test_loader(X_tot[mask3], y_reg_new[mask3], test_size, batch_size)

    return classifier_loader, regressor_loader1, regressor_loader2, regressor_loader3



    