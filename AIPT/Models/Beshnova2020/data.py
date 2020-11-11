import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import torch
from AIPT.Benchmarks.OAS_dataset import OAS_data_loader
from torch.utils.data import DataLoader, WeightedRandomSampler
import subprocess as sp

S3_OAS_URI = 's3://gv20interns/OAS_dataset/'


def seed_random(seed=0):
    '''
    Seed RNGs.

    Args:
        seed (int): Random seed.
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_random()


def loader_from_data(data, sampler=None, batch_size=32, shuffle=True):
    '''
    Produces batch-wise data loader given dataframe.

    Args:
        data (Pandas.DataFrame): Data. Columns 'CDR3_aa' and 'label' are required for sequences and class labels
        respectively.
        sampler (None or torch.utils.data.sampler.Sampler): Sampler for dataloader. If None, no sampling occurs.
        batch_size (int): Batch size.
        shuffle (bool): If True, data will be loaded in shuffled order.  Note that if sampler is not None,
        `shuffle` must be set to False.

    Returns (torch.utils.data.DataLoader): Data loader for inputted data.

    '''
    seq_encodings = OAS_data_loader.encode_index(data=data['CDR3_aa'])
    btypes = data['label'].values
    dataset = list(zip(seq_encodings, btypes))
    loader = DataLoader(dataset, sampler=sampler, shuffle=shuffle, batch_size=batch_size, drop_last=True)
    return loader


def get_data_loader(data, batch_size=32, shuffle=True):
    '''
    Unbalanced data loader.  (Example counts per class are preserved from `data` and are not reweighted by loader.)
    Data is simply shuffled if `shuffle` and partitioned into batches of size `batch_size`.

    Args, Returns: See `loader_from_data`

    '''
    loader = loader_from_data(data, batch_size=batch_size, shuffle=shuffle)
    return loader


def get_balanced_data_loader(data, batch_size=32):
    '''
    Balanced data loader.  Data is resampled such that each class becomes equally likely.

    Args, Returns: See `loader_from_data`

    '''
    # useful example: https://discuss.pytorch.org/t/some-problems-with-weightedrandomsampler/23242/20
    # Compute samples weight (each sample should get its own weight)
    label = torch.Tensor(data['label'].values).type(torch.int8)
    class_sample_count = torch.tensor(
        [(label == t).sum() for t in torch.unique(label, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in label])

    # Create sampler, dataset, loader
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    loader = loader_from_data(data, batch_size=batch_size, sampler=sampler, shuffle=False)
    return loader


def load_data(index_df, seq_dir, cell_types, seq_len=12, toy=False, toy_rows=20):
    '''
    Load CDR3s of length `seq_len` OAS dataset, according to files in `index_df`.

    Args:
        index_df (Pandas.DataFrame): DataFrame load of OAS index file.
        seq_dir (str): Path to directory containing OAS data files.
        cell_types (list of str): List of cell types of interest (e.g. ['Naive-B-Cells', 'Memory-B-Cells'])
        seq_len (int): CDR3 amino acid sequence length.  All loaded sequences will be of this length.
        toy (bool): If True, we use a smaller dataset.
        toy_rows (int): If `toy`, we use only the files in (at most) the first `toy_rows` rows of `index_df`.

    Returns (Pandas.DataFrame):
        Columns ['CDR3_aa', 'BType', 'label'].  For each example sequence, 'CDR3_aa' will contain the CDR3 amino acid
        sequence, 'BType' the corresponding element of `cell_types`, and 'label' the corresponding index of
        `cell_types`.

    '''


    def df_len_fn(row):
        '''
        Try to return length of CDR3 AA sequence in `row`.
        Args:
            row: Row of Pandas dataframe

        Returns (int): CDR3 AA length of sequence in `row`, or -1 if error.

        '''
        try:
            return len(row['CDR3_aa'])
        except:
            return -1


    data_dfs = []
    for index, row in index_df.iterrows():
        if toy and index > toy_rows:
            break
        file_name = row['file_name']
        df = pd.read_csv(os.path.join(seq_dir, f'{file_name}.txt'), sep='\t')
        length_df = df.apply(df_len_fn, axis=1)
        data_df = df[length_df == seq_len]
        data_df['BType'] = row['BType']
        data_df = data_df[['CDR3_aa', 'BType']]
        data_dfs.append(data_df)

    data = pd.concat(data_dfs)
    data['label'] = data.apply(lambda row: cell_types.index(row['BType']), axis=1)

    return data


def get_train_test_loaders(data, train_size=0.8, balanced=False, random_seed=None):
    '''
    Splits `data` into train and test sets, and produces data loaders from them.

    Args:
        data (Pandas.DataFrame): Data dataframe.
        train_size (float in interval [0,1]): Proportion of data to use for train set. Remainder is used for test set.
        balanced (bool): If True, a balanced train loader is used. Test loader will remain unbalanced regardless.
        random_seed (None or int): If not None, seed RNGs with `random_seed`.

    Returns (tuple of (torch.utils.data.DataLoader, torch.utils.data.DataLoader)):
        Tuple of (train_loader, test_loader).

    '''
    if random_seed is not None:
        seed_random(random_seed)
    train_data, test_data = train_test_split(data, train_size=train_size)
    if balanced:
        train_loader = get_balanced_data_loader(train_data)
    else:
        train_loader = get_data_loader(train_data)
    test_loader = get_data_loader(test_data, shuffle=False)

    return train_loader, test_loader


def download_data(file_names, seq_dir):
    '''
    Downloads OAS files in `file_names` from S3 into local directory `seq_dir`. AWS CLI must be set up locally.

    Args:
        file_names (list of str): List of file base names (no extension) of OAS dataset to download.
        seq_dir (str): Path to local directory to download files into.
    '''
    s3_seq_uri = S3_OAS_URI
    for fn in file_names:
        print(fn)
        sp.run(['aws', 's3', 'cp', os.path.join(s3_seq_uri, f'{fn}.txt'), seq_dir])
