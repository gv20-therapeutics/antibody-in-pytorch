import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from AIPT.Models.Wollacott2019.Bi_LSTM import LSTM_Bi
from AIPT.Utils import loader
from AIPT.Benchmarks.OAS_dataset.OAS_data_loader import OAS_Dataset, OAS_preload
from AIPT.Utils.model import CrossEntropyLoss

valid_fields = ['Age', 'BSource', 'BType', 'Chain', 'Disease', 'Isotype', \
                'Link', 'Longitudinal', 'Species', 'Subject', 'Vaccine']
AA_LS = 'ACDEFGHIKLMNPQRSTVWY'
AA_GP = 'ACDEFGHIKLMNPQRSTVWY-'

def collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

def OAS_data_loader(index_file, output_field, input_type, species_type, gapped=True,
                    pad=True, batch_size=500, model_name='All', cdr_len=25, random_state=100,
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

    # Get the maximum length of a sequence
    dataset = OAS_Dataset(list_df['file_name'].values, labels=None, input_type=input_type, gapped=gapped, cdr_len=cdr_len, seq_dir=seq_dir)
    input = []
    for z in dataset.parse_file():
        input.extend(z)
    seq_len = len(max(input, key=len))

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

    print('Training data',train_split_df)
    print('Testing data',test_split_df)
    # Datasets
    if output_field in valid_fields:
        labels_train = dict(zip(train_split_df['file_name'].values, train_split_df[output_field].values))
        labels_test = dict(zip(test_split_df['file_name'].values, test_split_df[output_field].values))
    else:
        print('invalid output type!')
        return

    partition = {'train': train_split_df['file_name'].values, 'test': test_split_df['file_name'].values}  # IDs, to be done!

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

    train_loader = DataLoader(training_set, batch_size=batch_size, sampler=sample, drop_last=True, collate_fn=collate_fn)
    train_eval_loader = DataLoader(training_set)
    test_eval_loader = DataLoader(testing_set)

    return train_loader, train_eval_loader, test_eval_loader, seq_len

class Benchmark(LSTM_Bi):

    def __init__(self, para_dict, *args, **kwargs):
        super(Benchmark, self).__init__(para_dict, *args, **kwargs)

    def net_init(self):
        super(Benchmark, self).net_init()

        if self.fixed_len:
            self.fc3 = nn.Linear(self.para_dict['batch_size']*self.para_dict['seq_len'], self.para_dict['batch_size'])  # self.para_dict['num_classes']
            self.fc4 = nn.Linear(self.para_dict['hidden_dim'], self.para_dict['num_classes'])
        else:
            self.fc3 = nn.Linear(self.para_dict['hidden_dim']*3, 3)
            self.fc4 = nn.Linear(3, self.para_dict['num_classes'])

    def forward(self, Xs):

        x = self.hidden(Xs)
        if self.para_dict['fixed_len']:
            x = x.transpose(1, 0)
            # print(x.shape)
            x = self.fc3(x)
            x = x.transpose(1, 0)
            out = self.fc4(x)
            # print(out.shape)
            scores = F.log_softmax(out, dim=1)
        else:
            # x = x.reshape(x.shape[0], x.shape[1], 1)
            # x = x.permute(0, 2, 1)
            m = 0
            out = []
            for a in self.Xs_len:
                seq = x[m: a+m]
                a1 = seq.mean(axis=0).reshape(-1,1)
                a2 = seq.max(axis=0).values.reshape(-1,1)
                a3 = seq.min(axis=0).values.reshape(-1, 1)
                temp = torch.cat((a1,a2,a3), dim=1)
                m += a

                out.append(temp.detach().numpy())
            out = np.array(out)
            out = out.reshape(out.shape[0], self.para_dict['hidden_dim']*3)
            out = self.fc3(torch.tensor(out))
            out = self.fc4(out)
            scores = F.log_softmax(out, dim=1)

        return scores

    def objective(self):
        return CrossEntropyLoss()

def test():
    para_dict = {'model_name': 'LSTM_Bi',
                 'optim_name': 'Adam',
                 'num_samples': 10000,
                 'seq_len': 50,
                 'step_size': 10,
                 'epoch': 50,
                 'batch_size': 50,
                 'learning_rate': 0.01,
                 'gapped': False,
                 'embedding_dim': 64,
                 'hidden_dim': 64,
                 'random_state': 100,
                 'fixed_len': False}

    data, out = loader.synthetic_data(num_samples=para_dict['num_samples'], seq_len=para_dict['seq_len'],
                                      aa_list='ACDEFGHIKLMNPQRSTVWY')
    train_loader, test_loader = loader.train_test_loader(data, out, test_size=0.3, sample=False,
                                                         batch_size=para_dict['batch_size'])
    print('Parameters are', para_dict)
    model = Benchmark(para_dict)
    print('Training...')
    model.fit(train_loader)
    print('Testing...')
    output = model.predict(test_loader)
    labels = np.vstack([i for _, i in test_loader])
    model.evaluate(output, labels)

if __name__ == '__main__':
    test()