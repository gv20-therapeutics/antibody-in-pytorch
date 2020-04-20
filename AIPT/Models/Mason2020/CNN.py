import numpy as np
# ---------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...Utils import loader
from ...Utils.model import Model

AA_LS = 'ACDEFGHIKLMNPQRSTVWY'
AA_GP = 'ACDEFGHIKLMNPQRSTVWY-'

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
        if 'aa_len' not in para_dict:
            self.para_dict['aa_len'] = 20
        if 'num_classes' not in para_dict:
            self.para_dict['num_classes'] = 2
        if 'pad' not in para_dict:
            self.para_dict['pad'] = True
        if 'gapped' not in para_dict:
            self.para_dict['gapped'] = True
        if self.para_dict['gapped']:
            self.aa_list = AA_GP
        else:
            self.aa_list = AA_LS
        self.in_channels = len(self.aa_list)

    def net_init(self):
        self.conv1 = nn.Conv1d(in_channels=self.in_channels,
                               out_channels=self.para_dict['n_filter'],
                               kernel_size=self.para_dict['filter_size'],
                               stride=1, padding=0)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.fc1 = nn.Linear(
            in_features=(self.para_dict['seq_len'] - self.para_dict['filter_size']) * self.para_dict['n_filter'],
            out_features=self.para_dict['fc_hidden_dim'])
        self.dropout = nn.Dropout(p=self.para_dict['dropout_rate'])
        if type(self.para_dict['num_classes']) is not list:
            self.fc2 = nn.Linear(in_features=self.para_dict['fc_hidden_dim'], out_features=self.para_dict['num_classes'])

    def hidden(self, Xs):
        batch_size = len(Xs)
        if self.para_dict['gapped'] == True and self.para_dict['pad'] == True:
            Xs = loader.encode_data(Xs, aa_list=AA_GP)
        elif self.para_dict['pad'] == True:
            Xs = loader.encode_data(Xs, aa_list=AA_LS)
        X = torch.FloatTensor(Xs)
        X = X.permute(0, 2, 1)
        out = self.dropout(self.conv1(X))
        out = self.pool(out)
        out = out.reshape(batch_size, -1)

        return out

    def forward(self, Xs):

        out = self.hidden(Xs)
        out = torch.relu(self.fc1(out))
        out = F.softmax(self.fc2(out))

        return out


def test():
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'

    para_dict = {'num_samples': 2000,
                 'seq_len':50,
                 'batch_size': 200,
                 'model_name': 'CNN_Model',
                 'optim_name': 'Adam',
                 'epoch': 25,
                 'learning_rate': 0.01,
                 'step_size': 10,
                 'n_filter': 300,
                 'filter_size': 4,
                 'fc_hidden_dim':100,
                 'aa_len': len(aa_list),
                 'dropout_rate': 0.1,
                 'gapped': False,
                 'pad': False}

    data, out = loader.synthetic_data(num_samples=para_dict['num_samples'], seq_len=para_dict['seq_len'], aa_list=aa_list)
    data = loader.encode_data(data)
    train_loader, test_loader = loader.train_test_loader(data, out, test_size=0.3, sample=True, batch_size=para_dict['batch_size'])

    print('Parameters are', para_dict)
    model = CNN_classifier(para_dict)
    print('Training...')
    model.fit(train_loader)
    print('Testing...')
    output = model.predict(test_loader)
    labels = np.vstack([i for _, i in test_loader])
    model.evaluate(output, labels)

if __name__ == '__main__':
    test()
