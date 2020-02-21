from itertools import islice, chain

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import IterableDataset, DataLoader, Dataset

from ...Utils import loader

valid_fields = ['Age', 'BSource', 'BType', 'Chain', 'Disease', 'Isotype', \
                'Link', 'Longitudinal', 'Species', 'Subject', 'Vaccine']
input_type_dict = {'FR1': 'FW1-IMGT', 'FR2': 'FW2-IMGT', 'FR3': 'FW3-IMGT', \
                   'FR4': 'FW4-IMGT', 'CDR1': 'CDR1-IMGT', 'CDR2': 'CDR2-IMGT', \
                   'CDR3': 'CDR3-IMGT'}
full_seq_order = ['FW1-IMGT', 'CDR1-IMGT', 'FW2-IMGT', 'CDR2-IMGT', \
                  'FW3-IMGT', 'CDR3-IMGT', 'FW4-IMGT']
AA_LS = 'ACDEFGHIKLMNPQRSTVWY'
AA_GP = 'ACDEFGHIKLMNPQRSTVWY-'


def encode_index(data, aa_list=AA_GP, pad=False, gapped=False):
    aa_list = list(aa_list)
    X = []

    max_len_local = len(max(data, key=len))
    if pad == True:
        aa_list.insert(0, '0')
    if gapped == False:
        aa_list = list(AA_LS)
    for i, seq in enumerate(data):
        if pad == True:
            temp = np.zeros(max_len_local, dtype=np.int)
        else:
            temp = np.zeros(len(seq), dtype=np.int)
        for j, s in enumerate(seq):
            temp[j] = aa_list.index(s)
        X.append(temp)
    return X


class OAS_Dataset(IterableDataset):
    def __init__(self, list_IDs, labels, input_type, gapped=True, pad=False,
                 seq_dir='./antibody-in-pytorch/Benchmarks/OAS_dataset/data/seq_db/'):
        '''
        list_IDs: file name (prefix) for the loader
        labels: a dictionary, specifying the output label for each file
        input_type: one of [FR1, FR2, FR3, FR4, CDR1, CDR2, CDR3, CDR3_full, full_length]
        gapped: if False, remove all '-' in the sequence
        seq_dir: the directory saving all the processed files
        '''
        # load index_file and initialize
        self.labels = labels
        self.list_IDs = list_IDs
        self.input_type = input_type
        self.gapped = gapped
        self.seq_dir = seq_dir
        self.max_len = 0

    def parse_file(self):

        # load data file
        input_fnames = [self.seq_dir + ID + '.txt' for ID in self.list_IDs]
        for m in range(len(input_fnames)):
            input_fname = input_fnames[m]
            # print(input_fname)
            ID = self.list_IDs[m]
            input_df = pd.read_csv(input_fname, sep='\t')
            input_df = input_df.fillna('')
            # transformation (keep the gaps or filter out the gaps; extract/assemble sequences)
            if self.input_type in input_type_dict:
                X = input_df[input_type_dict[self.input_type]].values
            elif self.input_type == 'CDR3_full':
                X = [input_df['CDR3-IMGT'].iloc[nn][:7] + input_df['CDR3-IMGT-111-112'].iloc[nn] + \
                     input_df['CDR3-IMGT'].iloc[nn][7:] for nn in range(len(input_df))]
            elif self.input_type == 'full_length':
                X = [''.join([input_df[item].iloc[kk] for item in full_seq_order]) for kk in range(len(input_df))]
                X = [X[nn][:112] + input_df['CDR3-IMGT-111-112'].iloc[nn] + \
                     X[nn][112:] for nn in range(len(input_df))]
            else:
                print('invalid seq type!')
                return

            if not self.gapped:
                X = [item.replace('-', '') for item in X]

            # if not self.gapped:
            #     X = encode_index(X, aa_list=AA_LS, pad=self.pad)
            # else:
            #     X = encode_index(X, aa_list=AA_GP, pad=self.pad)

            y = [self.labels[ID] for _ in range(len(input_df))]

            yield zip(X, y)

    def get_stream(self):
        return chain.from_iterable(islice(self.parse_file(), len(self.list_IDs)))

    def __iter__(self):
        return self.get_stream()


# class OAS_preload(Dataset):
#     def __init__(self, index_file, output_field, input_type, species_type, gapped=True,
#                  seq_dir='./OAS_dataset/data/seq_db/'):
#         '''
#         list_IDs: file name (prefix) for the loader
#         labels: a dictionary, specifying the output label for each file
#         input_type: one of [FR1, FR2, FR3, FR4, CDR1, CDR2, CDR3, CDR3_full, full_length]
#         gapped: if False, remove all '-' in the sequence
#         seq_dir: the directory saving all the processed files
#         '''
#         # load index_file and initialize
#         self.index_file = index_file
#         self.output_field = output_field
#         self.input_type = input_type
#         self.species_type = species_type
#         self.gapped = gapped
#         self.seq_dir = seq_dir
#
#     def data(self):
#
#         index_df = pd.read_csv(self.index_file, sep='\t')
#         index_df = index_df[index_df.valid_entry_num > 1]
#         train_df = index_df[index_df.Species.isin(self.species_type)]
#         if self.output_field in valid_fields:
#             self.labels = dict(zip(index_df['file_name'].values, index_df[self.output_field].values))
#         else:
#             print('invalid output type!')
#
#         self.list_IDs = train_df['file_name'].values
#         print(train_df)
#         # load data file
#         input_fnames = [self.seq_dir + ID + '.txt' for ID in self.list_IDs]
#         input_data = {}
#         for m in range(len(input_fnames)):
#             input_fname = input_fnames[m]
#             # print(input_fname)
#             ID = self.list_IDs[m]
#             input_df = pd.read_csv(input_fname, sep='\t')
#             input_df = input_df.fillna('')
#             # transformation (keep the gaps or filter out the gaps; extract/assemble sequences)
#             if self.input_type in input_type_dict:
#                 X = input_df[input_type_dict[self.input_type]].values
#             elif self.input_type == 'CDR3_full':
#                 X = [input_df['CDR3-IMGT'].iloc[nn][:7] + input_df['CDR3-IMGT-111-112'].iloc[nn] + \
#                      input_df['CDR3-IMGT'].iloc[nn][7:] for nn in range(len(input_df))]
#             elif self.input_type == 'full_length':
#                 X = [''.join([input_df[item].iloc[kk] for item in full_seq_order]) for kk in range(len(input_df))]
#                 X = [X[nn][:112] + input_df['CDR3-IMGT-111-112'].iloc[nn] + \
#                      X[nn][112:] for nn in range(len(input_df))]
#             else:
#                 print('invalid seq type!')
#
#             if not self.gapped:
#                 X = [item.replace('-', '') for item in X]
#
#             # if not self.gapped:
#             #     X = encode_index(X, aa_list=AA_LS, pad=self.pad)
#             # else:
#             #     X = encode_index(X, aa_list=AA_GP, pad=self.pad)
#
#             y = [self.labels[ID] for _ in range(len(input_df))]
#
#             # print(input_fname, X, y)
#             input_data.update({input_fname: list(zip(X, y))})
#
#         return input_data

def create_loader(input_data, pad=False, gapped=False, batch_size=100, model_name='Mason_LSTM'):

    # train_x = [x for a, (idx, data) in enumerate(input_data.items()) for x, y in data]
    train_x = [x for x,y in input_data]
    seq_len = len(max(train_x, key=len))
    train_y = [1 if y == 'human' else 0 for x, y in input_data] #for a, (idx, data) in enumerate(input_data.items())
    train_x = encode_index(data=train_x, pad=pad, gapped=gapped)  # pad the sequence

    if model_name is 'Mason2020_CNN':
        if gapped==True:
            aa_list = '0'+AA_GP
        else:
            aa_list = '0'+AA_LS
        train_x = loader.encode_data(np.array(train_x), aa_list=aa_list)
    if model_name is 'Wollacott2019_Bi_LSTM':
        X_train, X_test, y_train, y_test = train_test_split(train_x, np.array(train_y), test_size=0.3, shuffle=True,
                                                            random_state=100)
        train_loader = DataLoader(X_train, batch_size=batch_size, drop_last=True,
                                  collate_fn=collate_fn)
        test_loader = DataLoader(X_test, collate_fn=collate_fn)
        # self.y_test = y_test
        return train_loader, test_loader, y_test
    else:
        train_loader, test_loader = loader.train_test_loader(np.array(train_x), np.array(train_y), test_size=0.3,
                                                             batch_size=batch_size, sample=True, random_state=100)

        return train_loader, test_loader, seq_len


# -------------------------------
def collate_fn(batch):
    return batch, [x for seq in batch for x in seq]


def OAS_data_loader(index_file, output_field, input_type, species_type, num_files=30, gapped=True,
                    seq_dir='./antibody-in-pytorch/Benchmarks/OAS_dataset/data/seq_db/'):
    index_df = pd.read_csv(index_file, sep='\t')
    index_df = index_df[index_df.valid_entry_num > 1]
    train_df = index_df[index_df.Species.isin(species_type)]
    train_df = train_df[:10]
    print(train_df)

    # Datasets
    if output_field in valid_fields:
        labels = dict(zip(index_df['file_name'].values, index_df[output_field].values))
    else:
        print('invalid output type!')
        return

    partition = {'train': train_df['file_name'].values, 'test': index_df['file_name'].values}  # IDs, to be done!

    # generators
    training_set = OAS_Dataset(partition['train'], labels, input_type, gapped, seq_dir=seq_dir)

    return training_set
