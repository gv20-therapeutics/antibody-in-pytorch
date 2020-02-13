from ...Utils.model import Model
import pickle as pkl
from ...Utils import loader
from ...Benchmarks.OAS_dataset import OAS_data_loader
import pandas as pd
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
import numpy as np

# ---------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

class CNN_classifier(Model):
    def __init__(self, para_dict, *args, **kwargs):
        super(CNN_classifier, self).__init__(para_dict, *args, **kwargs)

        if 'dropout_rate' not in para_dict:
            self.dropout_rate = 0.5
        if 'n_filter' not in para_dict:
            self.n_filter = 400
        if 'filter_size' not in para_dict:
            self.filter_size = 3
        if 'fc_hidden_dim' not in para_dict:
            self.fc_hidden_dim = 50

    def net_init(self):
        self.conv1 = nn.Conv1d(in_channels=21,
                               out_channels=self.para_dict['n_filter'],
                               kernel_size=self.para_dict['filter_size'],
                               stride=1, padding=0)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.fc1 = nn.Linear(
            in_features=(self.para_dict['seq_len'] - self.para_dict['filter_size']) * self.para_dict['n_filter'],
            out_features=self.para_dict['fc_hidden_dim'])
        self.fc2 = nn.Linear(in_features=self.para_dict['fc_hidden_dim'], out_features=2)

    def forward(self, Xs, _aa2id=None):
        batch_size = len(Xs)
        X = torch.FloatTensor(Xs)
        X = X.permute(0, 2, 1)
        # print(X.shape)

        out = F.dropout(self.conv1(X), p=self.para_dict['dropout_rate'])
        out = self.pool(out)
        out = out.reshape(batch_size, -1)
        # print(out.shape)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))

        return out


# ----------------------------------------------------------
if __name__ == '__main__':
    para_dict = {'num_samples': 1000,
                 'seq_len': 13,
                 'batch_size': 10,
                 'model_name': 'CNN_Model',
                 'optim_name': 'Adam',
                 'epoch': 5,
                 'learning_rate': 0.001,
                 'step_size': 10,
                 'n_filter': 400,
                 'filter_size': 3,
                 'fc_hidden_dim': 50,
                 'dropout_rate': 0.5}

    # For synthetic data
    # data, out = loader.synthetic_data(num_samples=para_dict['num_samples'], seq_len=para_dict['seq_len'])
    # data = loader.encode_data(data)
    # train_loader, test_loader = loader.train_test_loader(data, out, test_size=0.3, batch_size=para_dict['batch_size'])

    # For OAS database
    # train_data = pkl.load(open('./antibody-in-pytorch/Benchmarks/OAS_dataset/data/Mouse&Human_train_seq_full_length.csv.gz','rb'))
    train_data_human = OAS_data_loader.OAS_data_loader(
        index_file='./antibody-in-pytorch/Benchmarks/OAS_dataset/data/OAS_meta_info.txt', output_field='Species',
        input_type='full_length', species_type=['human','mouse'], num_files=3, gapped=False, pad=True)
    train_x = [x for x, y in train_data_human]
    para_dict['seq_len'] = len(max(train_x, key=len))
    train_y = [1 if y == 'human' else 0 for x, y in train_data_human]
    # print(len(train_x), len(train_y))
    train_x = OAS_data_loader.encode_index(data=train_x, pad=True) # pad the sequence
    train_x = loader.encode_data(np.array(train_x), aa_list='0ACDEFGHIKLMNPQRSTVWY')
    train_loader, test_loader = loader.train_test_loader(np.array(train_x), np.array(train_y), test_size=0.3,
                                                         batch_size=para_dict['batch_size'], sample=True, random_state=100)
    # train_loader = torch.utils.data.DataLoader(train_x, batch_size=para_dict['batch_size'], drop_last=False)
    model = CNN_classifier(para_dict)
    model.fit(train_loader)
    output = model.predict(test_loader)
    labels = np.vstack([i for _, i in test_loader])
    mat, acc, mcc = model.evaluate(output, labels)

    print(para_dict)
