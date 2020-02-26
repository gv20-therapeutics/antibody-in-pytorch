import numpy as np
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

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


def collate_fn(batch):
    return batch, [x for seq in batch for x in seq]


def train_test_loader(x, y=None, test_size=0.3, batch_size=16, sample=None, random_state=100):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True,
                                                        random_state=random_state)

    # Balanced sampler
    train_y = np.array(y_train)
    # class_sample_count = [(train_y == 0).sum(), (train_y == 1).sum()]
    # weights = 1 / torch.Tensor(class_sample_count)
    # new_w = np.zeros(train_y.shape)
    # new_w[train_y == 0] = weights[0]
    # new_w[train_y == 1] = weights[1]
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(new_w, batch_size)
    class_weigth= class_weight.compute_class_weight('balanced', np.unique(train_y), train_y)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(class_weigth, batch_size)

    x_tensor = torch.from_numpy(X_train).float()
    y_tensor = torch.from_numpy(y_train).float()
    train_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    if sample == True:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=True)

    x_tensor = torch.from_numpy(X_test).float()
    y_tensor = torch.from_numpy(y_test).float()
    test_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset)

    return train_loader, test_loader


def synthetic_data_loader(num_samples=1000, seq_len=10, aa_list=AA_LS, test_size=0.3, batch_size=16):
    aa_size = len(aa_list)
    data = []
    for i in range(num_samples):
        temp = []
        for j in range(seq_len):
            temp.append([np.random.randint(aa_size)])

        data.append(temp)

    X_train, X_test = train_test_split(np.array(data), test_size=test_size, shuffle=True)
    x_tensor = torch.from_numpy(X_train)
    train_loader = torch.utils.data.DataLoader(x_tensor, batch_size=batch_size, collate_fn=collate_fn)

    x_tensor = torch.from_numpy(X_test)
    test_loader = torch.utils.data.DataLoader(dataset=x_tensor, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, test_loader