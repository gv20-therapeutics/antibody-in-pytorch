from itertools import islice, chain

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import IterableDataset, DataLoader, Dataset, sampler
import torch
from sklearn.utils import class_weight

valid_fields = ['Age', 'BSource', 'BType', 'Chain', 'Disease', 'Isotype', \
                'Link', 'Longitudinal', 'Species', 'Subject', 'Vaccine']
input_type_dict = {'FR1': 'FW1-IMGT', 'FR2': 'FW2-IMGT', 'FR3': 'FW3-IMGT', \
                   'FR4': 'FW4-IMGT', 'CDR1': 'CDR1-IMGT', 'CDR2': 'CDR2-IMGT', \
                   'CDR3': 'CDR3-IMGT'}
full_seq_order = ['FW1-IMGT', 'CDR1-IMGT', 'FW2-IMGT', 'CDR2-IMGT', \
                  'FW3-IMGT', 'CDR3-IMGT', 'FW4-IMGT']
AA_LS = 'ACDEFGHIKLMNPQRSTVWY'
AA_GP = 'ACDEFGHIKLMNPQRSTVWY-'


def encode_index(data, aa_list=AA_GP, pad=False, gapped=True, max_len_local=None):

    """
    Convert the sequence into a matrix of index representing the amino acid
    :return: A list of sequence lists
    """
    aa_list = list(aa_list)
    X = []

    if gapped == False:
        aa_list = list(AA_LS)
    for i, seq in enumerate(data):
        if max_len_local is None or len(seq) <= max_len_local:
            seq_i = np.zeros(len(seq), dtype=np.int)
            for j, s in enumerate(seq):
                seq_i[j] = aa_list.index(s)
            if pad == True:
                arr_create = (max_len_local - len(seq))
                temp = np.full(arr_create, -1, dtype=np.int)
                seq_i = np.concatenate([seq_i,temp])
            X.append(seq_i)
    return X


class OAS_Dataset(IterableDataset):
    def __init__(self, list_IDs, labels, input_type, gapped=True, pad=False, cdr_len=25,
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
        self.cdr_len = cdr_len

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
                input_df['CDR3-IMGT-111-112'] = input_df['CDR3-IMGT-111-112'].apply(lambda x: x + '-' * (self.cdr_len - len(x)))
                X = [input_df['CDR3-IMGT'].iloc[nn][:7] + input_df['CDR3-IMGT-111-112'].iloc[nn] + \
                     input_df['CDR3-IMGT'].iloc[nn][7:] for nn in range(len(input_df))]
                # X = [input_df['CDR3-IMGT-111-112'].iloc[nn] for nn in range(len(input_df))]
            elif self.input_type == 'full_length':
                input_df['CDR3-IMGT-111-112'] = input_df['CDR3-IMGT-111-112'].apply(lambda x: x + '-' * (self.cdr_len - len(x)))
                X = [''.join([input_df[item].iloc[kk] for item in full_seq_order]) for kk in range(len(input_df))]
                X = [X[nn][:112] + input_df['CDR3-IMGT-111-112'].iloc[nn] + \
                     X[nn][112:] for nn in range(len(input_df))]
            elif self.input_type == 'CDR123':
                input_df['CDR3-IMGT-111-112'] = input_df['CDR3-IMGT-111-112'].apply(lambda x: x + '-' * (self.cdr_len - len(x)))
                X = [input_df['CDR1-IMGT'].iloc[nn] + input_df['CDR2-IMGT'].iloc[nn] + \
                     input_df['CDR3-IMGT'].iloc[nn][:7] + input_df['CDR3-IMGT-111-112'].iloc[nn] + \
                     input_df['CDR3-IMGT'].iloc[nn][7:] for nn in range(len(input_df))]
            else:
                print('invalid seq type!')
                return

            if not self.gapped:
                X = [item.replace('-', '') for item in X]

            if self.labels is None:
                yield X
            else:
                y = [self.labels[ID] for _ in range(len(input_df))]
                yield zip(X, y)

    def get_stream(self):
        return chain.from_iterable(islice(self.parse_file(), len(self.list_IDs)))

    def __iter__(self):
        return self.get_stream()

class OAS_preload(Dataset):

    def __init__(self, list_IDs, labels, input_type, gapped, seq_dir, species_type, pad, cdr_len=25, seq_len=None):
        '''
        To read all the data at once and feed into the Dataloader
        pad: Padding required (bool)
        Other params similar to OAS_Dataset class
        '''
        self.labels = labels
        self.list_IDs = list_IDs
        self.input_type = input_type
        self.gapped = gapped
        self.seq_dir = seq_dir
        self.species_type = species_type
        self.seq_len = seq_len
        self.input = []
        self.output = []

        dataset = OAS_Dataset(list_IDs, labels, input_type, gapped, cdr_len=cdr_len, seq_dir=seq_dir)
        le = LabelEncoder()
        le.fit(species_type)

        for z in dataset.parse_file():
            train_x, train_y = zip(*z)
            train_y = le.transform(train_y)
            self.input.extend(train_x)
            self.output.extend(train_y)
        self.input = encode_index(data=self.input, pad=pad, gapped=gapped, max_len_local=self.seq_len)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        X = self.input[idx]
        y = self.output[idx]

        return X,y

def collate_fn(batch):
    return batch, [x for seq in batch for x in seq]

def OAS_data_loader(index_file, output_field, input_type, species_type, gapped=True,
                    pad=False, batch_size=500, model_name='Wollacott2019', cdr_len=25, random_state=100,
                    seq_dir='AIPT/Benchmarks/OAS_dataset/data/seq_db/'):
    """
    Create the train and test df
    return: Train and test loader
    """

    index_df = pd.read_csv(index_file, sep='\t')
    index_df = index_df[index_df.valid_entry_num >= 1]
    list_df = index_df[index_df[output_field].isin(species_type)]
    # list_df = list_df.sort_values(by=[output_field])
    list_df = list_df[::-1]
    list_df = list_df[:5]
    # print(list_df)

    # Get the maximum length of a sequence
    dataset = OAS_Dataset(list_df['file_name'].values, labels=None, input_type=input_type, gapped=gapped, cdr_len=cdr_len, seq_dir=seq_dir)
    input = []
    # f = open('len.txt','w')
    for z in dataset.parse_file():
        # print(len(max(z, key=len)), z)
        # f.write(str(len(max(z, key=len))))
        # f.write('\n')
        input.extend(z)
    seq_len = len(max(input, key=len))
    # f.close()

    train_split_df = pd.DataFrame()
    test_split_df = pd.DataFrame()
    ls_ls = [[a] for a in species_type]

    for a in ls_ls:
        temp_df = list_df.copy()
        df = temp_df[temp_df[output_field].isin(a)]
        df_copy = df.copy()
        temp_train = df_copy.sample(frac=0.7, random_state=random_state)
        train_split_df = train_split_df.append(temp_train, ignore_index=True)
        temp_test = df_copy.drop(temp_train.index)
        test_split_df = test_split_df.append(temp_test, ignore_index=True)

    print('Training data', train_split_df)
    print('Testing data', test_split_df)
    # Datasets
    if output_field in valid_fields:
        labels_train = dict(zip(train_split_df['file_name'].values, train_split_df[output_field].values))
        labels_test = dict(zip(test_split_df['file_name'].values, test_split_df[output_field].values))
    else:
        print('invalid output type!')
        return

    partition = {'train': train_split_df['file_name'].values,
                 'test': test_split_df['file_name'].values}  # IDs, to be done!

    # generators
    # training_set = OAS_Dataset(partition['train'], labels, input_type, gapped, seq_dir=seq_dir)
    training_set = OAS_preload(partition['train'], labels_train, input_type, gapped, seq_dir=seq_dir,
                               species_type=species_type, pad=pad, cdr_len=cdr_len, seq_len=seq_len)
    testing_set = OAS_preload(partition['test'], labels_test, input_type, gapped, seq_dir=seq_dir,
                              species_type=species_type, pad=pad, cdr_len=cdr_len, seq_len=seq_len)
    # Balanced Sampler for the loader
    class_sample_count = []
    for a in np.unique(training_set.output):
        class_sample_count.append((training_set.output == a).sum())
    weights = 1 / torch.Tensor(class_sample_count)
    new_w = np.zeros(np.shape(training_set.output))
    for a in np.unique(training_set.output):
        new_w[training_set.output == a] = weights[a]
    sample = torch.utils.data.sampler.WeightedRandomSampler(new_w, 50000)

    # Train and test loaders
    if model_name == 'Wollacott2019':
        #         train_loader = DataLoader(training_set.input, batch_size=batch_size, sampler=sample, drop_last=True, collate_fn=collate_fn)
        train_loader = DataLoader(training_set.input, batch_size=batch_size, sampler=sample, drop_last=True,
                                  collate_fn=collate_fn)
    else:
        train_loader = DataLoader(training_set, batch_size=batch_size, sampler=sample, drop_last=True)
    train_eval_loader = DataLoader(training_set)
    test_eval_loader = DataLoader(testing_set)

    return train_loader, train_eval_loader, test_eval_loader, seq_len