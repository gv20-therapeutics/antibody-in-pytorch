import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, Dataset
from itertools import islice, chain
from AIPT.Models.Mason2020.CNN import CNN_classifier
from AIPT.Models.Mason2020.LSTM_RNN import LSTM_RNN_classifier
from AIPT.Models.Wollacott2019.Bi_LSTM import LSTM_Bi
from AIPT.Benchmarks.OAS_dataset.OAS_data_loader import OAS_data_loader, OAS_Dataset, encode_index
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score
from AIPT.Benchmarks.Benchmark import Benchmark
from AIPT.Benchmarks.OAS_dataset import OAS_data_loader
from AIPT.Utils.model import Model
from sklearn.preprocessing import LabelEncoder

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
    def __init__(self, list_IDs, labels_1, labels_2, input_type, gapped=True, pad=False,
                 seq_dir='./antibody-in-pytorch/Benchmarks/OAS_dataset/data/seq_db/'):
        '''
        list_IDs: file name (prefix) for the loader
        labels: a dictionary, specifying the output label for each file
        input_type: one of [FR1, FR2, FR3, FR4, CDR1, CDR2, CDR3, CDR3_full, full_length]
        gapped: if False, remove all '-' in the sequence
        seq_dir: the directory saving all the processed files
        '''
        # load index_file and initialize
        self.labels_1 = labels_1
        self.labels_2 = labels_2
        self.list_IDs = list_IDs
        self.input_type = input_type
        self.gapped = gapped
        self.seq_dir = seq_dir

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

            if self.labels_1 is None:
                yield X
            else:
                y_1 = [self.labels_1[ID] for _ in range(len(input_df))]
                y_2 = [self.labels_2[ID] for _ in range(len(input_df))]
                yield zip(X, y_1, y_2)

    def get_stream(self):
        return chain.from_iterable(islice(self.parse_file(), len(self.list_IDs)))

    def __iter__(self):
        return self.get_stream()

class OAS_preload(Dataset):

    def __init__(self, list_IDs, labels_1, labels_2, input_type, gapped, seq_dir, species_type_1, species_type_2, pad, seq_len=None):
        '''
        To read all the data at once and feed into the Dataloader
        pad: Padding required (bool)
        Other params similar to OAS_Dataset class
        '''
        self.labels_1 = labels_1
        self.labels_2 = labels_2
        self.list_IDs = list_IDs
        self.input_type = input_type
        self.gapped = gapped
        self.seq_dir = seq_dir
        self.species_type_1 = species_type_1
        self.species_type_2 = species_type_2
        self.seq_len = seq_len
        self.input = []
        self.output_1 = []
        self.output_2 = []
        self.sample = {}

        dataset = OAS_Dataset(list_IDs, labels_1, labels_2, input_type, gapped, seq_dir=seq_dir)
        le_1 = LabelEncoder()
        le_1.fit(self.species_type_1)
        le_2 = LabelEncoder()
        le_2.fit(self.species_type_2)

        for z in dataset.parse_file():
            train_x, train_y_1, train_y_2 = zip(*z)
            train_y_1 = le_1.transform(train_y_1)
            train_y_2 = le_2.transform(train_y_2)
            self.input.extend(train_x)
            self.output_1.extend(train_y_1)
            self.output_2.extend(train_y_2)

        self.input = encode_index(data=self.input, pad=pad, gapped=gapped, max_len_local=self.seq_len)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        self.sample = {'input': self.input[idx], 'output_1': self.output_1[idx], 'output_2':self.output_2[idx]}
        return self.sample

def collate_fn(batch):
    data = [item['input'] for item in batch]
    target_1 = [item['output_1'] for item in batch]
    target_2 = [item['output_2'] for item in batch]
    target_1 = torch.LongTensor(target_1)
    target_2 = torch.LongTensor(target_2)
    return [data, zip(target_1, target_2)]

def OAS_data_loader(index_file, output_field_1, output_field_2, input_type, species_type_1, species_type_2, gapped=True,
                    pad=True, batch_size=500, model_name='All',
                    seq_dir='AIPT/Benchmarks/OAS_dataset/data/seq_db/'):
    """
    Create the train and test df
    return: Train and test loader
    """

    index_df = pd.read_csv(index_file, sep='\t')
    index_df = index_df[index_df.valid_entry_num > 1]
    list_df = index_df[index_df[output_field_1].isin(species_type_1) & index_df[output_field_2].isin(species_type_2)]
    list_df.sort_values(by=[output_field_1])
    list_df = list_df[::-1]
    list_df = list_df[:10]

    train_split_df = pd.DataFrame()
    test_split_df = pd.DataFrame()
    ls_ls = [[a] for a in species_type_1+species_type_2]
    # print(ls_ls)

    for a in ls_ls:
        temp_df = list_df.copy()
        df = temp_df[temp_df[output_field_1].isin(a)]
        df_copy = df.copy()
        temp_train = df_copy.sample(frac=0.7)
        train_split_df = train_split_df.append(temp_train, ignore_index=True)
        temp_test = df_copy.drop(temp_train.index)
        test_split_df = test_split_df.append(temp_test, ignore_index=True)

    print('Training data',train_split_df)
    print('Testing data',test_split_df)
    # Datasets
    labels_train_1 = dict(zip(train_split_df['file_name'].values, train_split_df[output_field_1].values))
    labels_test_1 = dict(zip(test_split_df['file_name'].values, test_split_df[output_field_1].values))
    labels_train_2 = dict(zip(train_split_df['file_name'].values, train_split_df[output_field_2].values))
    labels_test_2 = dict(zip(test_split_df['file_name'].values, test_split_df[output_field_2].values))

    partition = {'train': train_split_df['file_name'].values, 'test': test_split_df['file_name'].values}  # IDs, to be done!

    # generators
    # training_set = OAS_Dataset(partition['train'], labels, input_type, gapped, seq_dir=seq_dir)
    training_set = OAS_preload(partition['train'], labels_train_1, labels_train_2, input_type, gapped, seq_dir=seq_dir,
                               species_type_1=species_type_1, species_type_2=species_type_2, pad=pad)
    testing_set = OAS_preload(partition['test'], labels_test_1, labels_test_2, input_type, gapped, seq_dir=seq_dir,
                               species_type_1=species_type_1, species_type_2=species_type_2, pad=pad)

    # print(training_set.sample)

    # Balanced Sampler for the loader
    # class_sample_count = []
    # for a in np.unique(training_set.output_1):
    #     class_sample_count.append((training_set.output_1 == a).sum())
    # weights = 1 / torch.Tensor(class_sample_count)
    # new_w = np.zeros(np.shape(training_set.output_1))
    # for a in np.unique(training_set.output_1):
    #     new_w[training_set.output_1 == a] = weights[a]
    # sample = torch.utils.data.sampler.WeightedRandomSampler(new_w, 10000)

    train_loader = DataLoader(training_set, batch_size=batch_size, collate_fn=collate_fn, drop_last=True)
    train_eval_loader = DataLoader(training_set, collate_fn=collate_fn)
    test_eval_loader = DataLoader(testing_set, collate_fn=collate_fn)

    return train_loader, train_eval_loader, test_eval_loader

class CrossEntropyLoss():
    def __init__(self, *args, **kwargs):
        super(CrossEntropyLoss, self).__init__(*args, **kwargs)

    def __call__(self, outputs, targets):

        # output_1, output_2 = zip(*outputs)
        output_1 = outputs[0]
        output_2 = outputs[1]
        target_1, target_2 = zip(*targets)
        target_1 = torch.tensor(target_1).type(torch.long)
        target_2 = torch.tensor(target_2).type(torch.long)
        len_samples_1 = target_1.shape[0]
        batch_size_1 = output_1.shape[0]
        len_samples_2 = target_2.shape[0]
        batch_size_2 = output_2.shape[0]
        softmax = nn.LogSoftmax()
        output_1 = softmax(output_1)
        output_2 = softmax(output_2)
        output_1 = output_1[range(batch_size_1), target_1]
        loss_1 = - torch.sum(output_1) / len_samples_1
        output_2 = output_2[range(batch_size_2), target_2]
        loss_2 = - torch.sum(output_2) / len_samples_2
        return loss_1+loss_2/2

class Multitask(LSTM_Bi):
    def __init__(self, para_dict, *args, **kwargs):
        super(Multitask, self).__init__(para_dict, *args, **kwargs)

    def net_init(self):
        super(Multitask, self).net_init()

        if self.fixed_len:
            self.fc3 = nn.Linear(self.para_dict['batch_size']*self.para_dict['seq_len'], self.para_dict['batch_size'])  # self.para_dict['num_classes']
            self.fc4 = nn.Linear(self.para_dict['hidden_dim'], self.para_dict['num_classes_1'])
            self.fc5 = nn.Linear(self.para_dict['hidden_dim'], self.para_dict['num_classes_2'])
        else:
            self.fc3 = nn.Linear(self.para_dict['hidden_dim']*3, 3)
            self.fc4 = nn.Linear(3, self.para_dict['num_classes_1'])
            self.fc5 = nn.Linear(3, self.para_dict['num_classes_2'])

    def forward(self, Xs):
        x = self.hidden(Xs)
        m = 0
        out = []
        for a in self.Xs_len:
            seq = x[m: a + m]
            a1 = seq.mean(axis=0).reshape(-1, 1)
            a2 = seq.max(axis=0).values.reshape(-1, 1)
            a3 = seq.min(axis=0).values.reshape(-1, 1)
            temp = torch.cat((a1, a2, a3), dim=1)
            m += a
            out.append(temp.detach().numpy())
        out = np.array(out)
        out = out.reshape(out.shape[0], self.para_dict['hidden_dim'] * 3)
        out = self.fc3(torch.tensor(out))
        out_1 = self.fc4(out)
        out_2 = self.fc5(out)
        scores_1 = torch.sigmoid(out_1)
        scores_2 = torch.sigmoid(out_2)
        return [scores_1, scores_2]

    def objective(self):
        return CrossEntropyLoss()

    def predict(self, data_loader):

        self.eval()
        all_outputs_1 = []
        all_outputs_2 = []
        with torch.no_grad():
            for data in data_loader:
                inputs, _ = data
                outputs = self.forward(inputs)
                output_1, output_2 = outputs
                all_outputs_1.append(output_1.detach().numpy())
                all_outputs_2.append(output_2.detach().numpy())

        return [np.vstack(all_outputs_1), np.vstack(all_outputs_2)]

    def evaluate(self, outputs, labels):

        y_pred = []
        outputs_1, outputs_2 = outputs
        labels_1 = []
        labels_2 = []
        for a in labels:
            i, j = zip(*a)
            labels_1.append(i)
            labels_2.append(j)
        for a in outputs_1:
            y_pred.append(np.argmax(a))
        y_true = np.array(labels_1).flatten()
        y_pred = np.array(y_pred)
        mat = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        print('Task 1:')
        print('Confusion matrix ')
        print(mat)
        print('Accuracy = %.3f, MCC = %.3f' % (acc, mcc))

        y_pred = []
        for a in outputs_2:
            y_pred.append(np.argmax(a))
        y_true = np.array(labels_2).flatten()
        y_pred = np.array(y_pred)
        mat = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        print('Task 2:')
        print('Confusion matrix ')
        print(mat)
        print('Accuracy = %.3f, MCC = %.3f' % (acc, mcc))
