from torch.utils.data import IterableDataset, DataLoader, Dataset
import pandas as pd
import numpy as np
import os
from itertools import cycle, islice, chain

valid_fields = ['Age', 'BSource', 'BType', 'Chain', 'Disease', 'Isotype', \
                'Link', 'Longitudinal', 'Species', 'Subject', 'Vaccine']
input_type_dict = {'FR1': 'FW1-IMGT', 'FR2': 'FW2-IMGT', 'FR3': 'FW3-IMGT', \
                   'FR4': 'FW4-IMGT', 'CDR1': 'CDR1-IMGT', 'CDR2': 'CDR2-IMGT', \
                   'CDR3': 'CDR3-IMGT'}
full_seq_order = ['FW1-IMGT', 'CDR1-IMGT', 'FW2-IMGT', 'CDR2-IMGT', \
                  'FW3-IMGT', 'CDR3-IMGT', 'FW4-IMGT']
AA_LS = 'ACDEFGHIKLMNPQRSTVWY'
AA_GP = 'ACDEFGHIKLMNPQRSTVWY-'


class OAS_Dataset(IterableDataset):
    def __init__(self, list_IDs, labels, input_type, gapped=True, seq_dir='./seq_db/'):
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

    def encode_index(self, data, aa_list='ACDEFGHIKLMNPQRSTVWY-'):
        aa_list = list(aa_list)
        X = np.zeros((len(data), len(data[0])), dtype=int)
        for i, seq in enumerate(data):
            for j, s in enumerate(seq):
                X[i, j] = aa_list.index(s)
        return X

    def parse_file(self):
        # load data file
        input_fnames = [self.seq_dir + ID + '.txt' for ID in self.list_IDs]
        for m in range(len(input_fnames)):
            input_fname = input_fnames[m]
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

            X = self.encode_index(X)
            y = [self.labels[ID] for _ in range(len(input_df))]
            # print(len(y))

            # print(len(X))
            yield X

    def get_stream(self):
        return chain.from_iterable(islice(self.parse_file(), len(self.list_IDs)))
    #     m = islice(self.parse_file(), 5)
    #     return m

    def __iter__(self):
        return self.get_stream()


# -------------------------------
def collate_fn(batch):
    return batch, [x for seq in batch for x in seq]

def OAS_data_loader(index_file, output_field, input_type, gapped=True, seq_dir='./Models_generation/Benchmarks/OAS_dataset/data/seq_db/'):
    index_df = pd.read_csv(index_file, sep='\t')
    index_df = index_df[index_df.valid_entry_num>=50000].take([0,1,2])
        # set_index("valid_entry_num").drop(0, axis=0).drop(1, axis=0) #.take([0,1,2,15,16,17,18])
    train_df = index_df[index_df.Species=='human']

    print(train_df)

    # Datasets
    if output_field in valid_fields:
        labels = dict(zip(index_df['file_name'].values, index_df[output_field].values))
    else:
        print('invalid output type!')
        return

    partition = {'train': train_df['file_name'].values, 'test': index_df['file_name'].values}  # IDs, to be done!

    # generators
    training_set = OAS_Dataset(partition['train'], labels, input_type, gapped, seq_dir)
    testing_set = OAS_Dataset(partition['test'], labels, input_type, gapped, seq_dir)

    return training_set, testing_set