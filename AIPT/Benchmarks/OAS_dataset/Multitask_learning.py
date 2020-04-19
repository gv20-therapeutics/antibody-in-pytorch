import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset, DataLoader, Dataset
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef
from itertools import islice, chain
from AIPT.Benchmarks.OAS_dataset.OAS_data_loader import OAS_Dataset, OAS_preload, encode_index
from AIPT.Models.Wollacott2019.Bi_LSTM import LSTM_Bi
from AIPT.Models.Mason2020.LSTM_RNN import LSTM_RNN_classifier
from AIPT.Models.Mason2020.CNN import CNN_classifier
from AIPT.Utils.model import Model

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
    def __init__(self, list_IDs, labels, input_type, output_field, gapped=True, pad=False, cdr_len=25,
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
        self.output_field = output_field
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
                y = []
                for m in range(len(self.output_field)):
                    y.append([self.labels[m][ID] for _ in range(len(input_df))])
                yield X,y

    def get_stream(self):
        return chain.from_iterable(islice(self.parse_file(), len(self.list_IDs)))

    def __iter__(self):
        return self.get_stream()

# def collate_fn(batch):
#     data = [item[0] for item in batch]
#     target = [item[1] for item in batch]
#     # target = torch.LongTensor(target)
#     return [data, target]

class OAS_preload(Dataset):

    def __init__(self, list_IDs, labels, input_type, gapped, seq_dir, species_type, output_field, pad, cdr_len=25, seq_len=None):
        '''
        To read all the data at once and feed into the Dataloader
        pad: Padding required (bool)
        Other params similar to OAS_Dataset class
        '''
        self.labels = labels
        self.list_IDs = list_IDs
        self.input_type = input_type
        self.gapped = gapped
        self.output_field = output_field
        self.seq_dir = seq_dir
        self.species_type = species_type
        self.seq_len = seq_len
        self.input = []
        self.output = []

        dataset = OAS_Dataset(list_IDs, labels, input_type, output_field, gapped, cdr_len=cdr_len, seq_dir=seq_dir)
        for z in dataset.parse_file():
            train_x, train_y = z
            temp_y = []
            for i, m in enumerate(train_y):
                classes = np.array(species_type[i][1])
                table = {val: i for i, val in enumerate(classes)}
                encoded = np.array([table[v] if v in species_type[i][1] else torch.tensor(float('nan')) for v in m])
                temp_y.append(encoded)
            self.input.extend(train_x)
            self.output.extend(np.array(temp_y).T)
        self.input = encode_index(data=self.input, pad=pad, gapped=gapped, max_len_local=self.seq_len)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        X = self.input[idx]
        y = self.output[idx]

        return X, y

def collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    # target = torch.LongTensor(target)
    return [data, target]

def OAS_data_loader(index_file, output_field, input_type, species_type, gapped=True, cdr_len=25,
                    pad=False, batch_size=500, model_name='Wollacott2019', random_state=100,
                    seq_dir='AIPT/Benchmarks/OAS_dataset/data/seq_db/'):
    """
    Create the train and test df
    return: Train and test loader
    """

    index_df = pd.read_csv(index_file, sep='\t')
    index_df = index_df[index_df.valid_entry_num > 1]
    # list_df = pd.DataFrame()
    # for m in range(len(output_field)):
    #     list_df = list_df.append(index_df[index_df[output_field[m]].isin(species_type[m][1])])
    # list_df = list_df.drop_duplicates()
    # list_df = list_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    list_df = index_df[index_df[output_field[0]].isin(species_type[0][1]) & index_df[output_field[1]].isin(species_type[1][1])]
    list_df.sort_values(by=output_field[0])
    list_df = list_df[::-1]
    list_df = list_df[:5]
    # print(list_df)

    # Get the maximum length of a sequence
    dataset = OAS_Dataset(list_df['file_name'].values, output_field=output_field, labels=None, input_type=input_type,
                          cdr_len=cdr_len, gapped=gapped, seq_dir=seq_dir)
    input = []
    for z in dataset.parse_file():
        input.extend(z)
    seq_len = len(max(input, key=len))

    train_split_df = pd.DataFrame()
    test_split_df = pd.DataFrame()
    df = list_df.copy()
    temp_train = df.sample(frac=0.7, random_state=random_state)
    train_split_df = train_split_df.append(temp_train, ignore_index=True)
    temp_test = df.drop(temp_train.index)
    test_split_df = test_split_df.append(temp_test, ignore_index=True)

    print('Training data', train_split_df)
    print('Testing data', test_split_df)
    # Datasets
    dict_labels_train = []
    dict_labels_test = []
    for l in output_field:
        if l in valid_fields:
            dict_labels_train.append(dict(zip(train_split_df['file_name'].values, train_split_df[l].values)))
            dict_labels_test.append(dict(zip(test_split_df['file_name'].values, test_split_df[l].values)))
        else:
            print('invalid output type!')
            return
    partition = {'train': train_split_df['file_name'].values,
                 'test': test_split_df['file_name'].values}  # IDs, to be done!

    # generators
    # training_set = OAS_Dataset(partition['train'], labels, input_type, gapped, seq_dir=seq_dir)
    training_set = OAS_preload(partition['train'], dict_labels_train, input_type, gapped, seq_dir=seq_dir, cdr_len=cdr_len,
                               species_type=species_type, output_field=output_field, pad=pad, seq_len=seq_len)
    testing_set = OAS_preload(partition['test'], dict_labels_test, input_type, gapped, seq_dir=seq_dir, cdr_len=cdr_len,
                              species_type=species_type, output_field=output_field, pad=pad, seq_len=seq_len)
    # Balanced Sampler for the loader
    # class_sample_count = []
    # for a in np.unique(training_set.output):
    #     class_sample_count.append((training_set.output == a).sum())
    # weights = 1 / torch.Tensor(class_sample_count)
    # new_w = np.zeros(np.shape(training_set.output))
    # for a in np.unique(training_set.output):
    #     new_w[training_set.output == a] = weights[a]
    # sample = torch.utils.data.sampler.WeightedRandomSampler(new_w, 50000)


    # Train and test loaders
    train_loader = DataLoader(training_set, batch_size=batch_size, drop_last=True, collate_fn=collate_fn)
    train_eval_loader = DataLoader(training_set, collate_fn=collate_fn)
    test_eval_loader = DataLoader(testing_set, collate_fn=collate_fn)

    return train_loader, train_eval_loader, test_eval_loader, seq_len

class CrossEntropyLoss():
    def __init__(self, *args, **kwargs):
        super(CrossEntropyLoss, self).__init__(*args, **kwargs)

    def __call__(self, para_dict, outputs, targets):

        num_tasks = len(targets[0])
        labels = torch.tensor(targets).T

        for i in range(num_tasks):
            # target = torch.tensor(labels[i]).type(torch.long)
            target = labels[i]
            mask = ~torch.isnan(target)
            output_masked = outputs[:,:len(para_dict['Multitask'][i][1])][mask]
            target_masked = target[mask]
            loss_fn = nn.CrossEntropyLoss()
            l = loss_fn(output_masked, torch.tensor(target_masked).type(torch.long))
            if i==0:
                if not torch.isnan(l):
                    loss = l
            else:
                if not torch.isnan(l):
                    loss += l
        return loss

class Multitask(Model):
    def __init__(self, para_dict, *args, **kwargs):
        super(Multitask, self).__init__(para_dict, *args, **kwargs)

    def evaluate(self, outputs, labels, para_dict):

        num_tasks = len(para_dict['num_classes'])
        labels = torch.tensor(labels).T
        outputs = torch.tensor(np.vstack(outputs))
        for i in range(num_tasks):
            mask = ~torch.isnan(labels[i])
            label_masked = labels[i][mask]
            if len(label_masked) is not 0:
                y_pred = outputs[:, :para_dict['num_classes'][i]]
                temp = []
                for a in y_pred:
                    temp.append(np.argmax(a))
                temp = np.array(temp)
                y_pred = temp.reshape(1, temp.shape[0])[mask]
                y_true = np.array(labels[i]).flatten()
                y_pred = np.array(y_pred)
                mat = confusion_matrix(y_true, y_pred)
                acc = accuracy_score(y_true, y_pred)
                mcc = matthews_corrcoef(y_true, y_pred)

                print('Confusion matrix: ')
                print(mat)
                print('Accuracy = %.3f, MCC = %.3f' % (acc, mcc))
            else:
                print('No labels in the output class')

class Multitask_Bi_LSTM(Multitask, LSTM_Bi):
    def __init__(self, para_dict, *args, **kwargs):
        super(Multitask_Bi_LSTM, self).__init__(para_dict, *args, **kwargs)

    def net_init(self):
        super(Multitask_Bi_LSTM, self).net_init()

        if self.fixed_len:
            self.fc3 = nn.Linear(self.para_dict['batch_size']*self.para_dict['seq_len'], self.para_dict['batch_size'])  # self.para_dict['num_classes']
            self.tasks = []
            for layer in range(len(self.para_dict['Multitask'])):
                self.tasks.append(nn.Linear(self.para_dict['hidden_dim'], self.para_dict['num_classes'][layer]))
        else:
            self.fc3 = nn.Linear(self.para_dict['hidden_dim']*3, 3)
            self.tasks = []
            for layer in range(len(self.para_dict['Multitask'])):
                self.tasks.append(nn.Linear(3, self.para_dict['num_classes'][layer]))

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
        for i in range(len(self.para_dict['Multitask'])):
            s = self.tasks[i](out)
            score = torch.sigmoid(s)
            if i == 0:
                scores = score
            else:
                scores = torch.cat((scores, score), dim=1)
        return scores

    def objective(self):
        return CrossEntropyLoss()

class Multitask_CNN(Multitask, CNN_classifier):
    def __init__(self, para_dict, *args, **kwargs):
        super(Multitask_CNN, self).__init__(para_dict, *args, **kwargs)

    def net_init(self):
        super(Multitask_CNN, self).net_init()

        self.tasks = []
        for layer in range(len(self.para_dict['Multitask'])):
            self.tasks.append(nn.Linear(self.para_dict['fc_hidden_dim'], self.para_dict['num_classes'][layer]))

    def forward(self, Xs):

        out = self.hidden(Xs)
        out = torch.relu(self.fc1(out))
        for i in range(len(self.para_dict['Multitask'])):
            s = self.tasks[i](out)
            score = F.softmax(s)
            if i == 0:
                scores = score
            else:
                scores = torch.cat((scores, score), dim=1)
        return scores

    def objective(self):
        return CrossEntropyLoss()

class Multitask_LSTM_RNN(Multitask, LSTM_RNN_classifier):
    def __init__(self, para_dict, *args, **kwargs):
        super(Multitask_LSTM_RNN, self).__init__(para_dict, *args, **kwargs)

    def net_init(self):
        super(Multitask_LSTM_RNN, self).net_init()

        self.tasks = []
        for layer in range(len(self.para_dict['Multitask'])):
            self.tasks.append(nn.Linear(self.para_dict['hidden_dim'], self.para_dict['num_classes'][layer]))

    def forward(self, Xs):

        out = self.hidden(Xs)
        total_classes = 0
        for m in self.para_dict['Multitask']:
            total_classes += len(m[1])
        for i in range(len(self.para_dict['Multitask'])):
            s = self.tasks[i](out)
            score = torch.sigmoid(s)
            if i == 0:
                scores = score
            else:
                scores = torch.cat((scores, score), dim=1)
        return scores

    def objective(self):
        return CrossEntropyLoss()
