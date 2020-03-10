import numpy as np
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

AA_LS = 'ACDEFGHIKLMNPQRSTVWY'


def synthetic_data(num_samples=1000, seq_len=10, aa_list=AA_LS,  type = 'classifier'):
    """
    :param num_samples: Number od samples to be synthesized
    :param seq_len: Length of each sequence
    :param aa_list: List of amino acids to be used
    :return: sequences, labels
    """
    aa_size = len(aa_list)
    data = np.zeros((num_samples, seq_len), dtype=int)
    if type == 'classifier':
        out = np.zeros((num_samples), dtype=int)
        for i in range(num_samples):
            if i < round(num_samples / 2):
                out[i] = 1
    elif type == 'regressor':
        out = np.linspace(start = 0, stop = 1, num=num_samples)

    return data, out

def encode_data(data, aa_list=AA_LS):
    """
    One hot encoding to the sequences
    """
    codes = np.eye(len(aa_list))
    x = []
    for seq in data:
        temp = []
        for s in seq:
            if s is -1:
                temp.append(np.zeros(len(aa_list),dtype=float))
            else:
                temp.append(codes[s])
        x.append(temp)
    return x


def collate_fn(batch):
    return batch, [x for seq in batch for x in seq]


def train_test_loader(x, y=None, test_size=0.3, batch_size=16, sample=None, random_state=100):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True,
                                                        random_state=random_state)

    # Balanced sampler
    train_y = np.array(y_train)
    class_weigth= class_weight.compute_class_weight('balanced', np.unique(train_y), train_y)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(class_weigth, batch_size)

    x_tensor = torch.from_numpy(np.array(X_train)).float()
    y_tensor = torch.from_numpy(y_train).float()
    train_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    if sample == True:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=True)

    x_tensor = torch.from_numpy(np.array(X_test)).float()
    y_tensor = torch.from_numpy(y_test).float()
    test_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset)

    return train_loader, test_loader


def synthetic_data_loader(num_samples=1000, seq_len=10, aa_list=AA_LS, test_size=0.3):
    """
    Loaders for Wollacott model with collate function
    """
    aa_size = len(aa_list)
    data = []
    for i in range(num_samples):
        temp = []
        for j in range(seq_len):
            temp.append([np.random.randint(aa_size)])
        data.append(temp)
    
    if type == 'classifier':
        out = np.zeros((num_samples), dtype=int)
        for i in range(num_samples):
            if i < round(num_samples / 2):
                out[i] = 1
    elif type == 'regressor':
        out = np.linspace(start = 0, stop = 1, num=num_samples)

    X_train, X_test, y_train, y_test = train_test_split(np.array(data), out, test_size=test_size, shuffle=True)
    x_tensor = torch.from_numpy(X_train)
    train_loader = torch.utils.data.DataLoader(x_tensor, batch_size=batch_size, collate_fn=collate_fn)

    x_tensor = torch.from_numpy(X_test)
    y_tensor = torch.from_numpy(y_test).float()
    test_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset)

    return train_loader, test_loader