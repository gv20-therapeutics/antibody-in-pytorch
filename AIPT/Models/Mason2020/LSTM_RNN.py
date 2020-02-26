import numpy as np
# -------------------------------------------------------------
import torch
import torch.nn as nn

from ...Utils import loader
from ...Utils.model import Model

AA_LS = 'ACDEFGHIKLMNPQRSTVWY'
AA_GP = 'ACDEFGHIKLMNPQRSTVWY-'

class LSTM_RNN_classifier(Model):
    def __init__(self, para_dict, *args, **kwargs):
        super(LSTM_RNN_classifier, self).__init__(para_dict, *args, **kwargs)

        if 'dropout_rate' not in para_dict:
            self.dropout_rate = 0.5
        if 'hidden_dim' not in para_dict:
            self.hidden_dim = 40
        if 'hidden_layer_num' not in para_dict:
            self.hidden_layer_num = 3
        if 'num_classes' not in para_dict:
            self.para_dict['num_classes'] = 2
        if 'fixed_len' not in para_dict:
            self.fixed_len = True
        if 'gapped' not in para_dict:
            self.para_dict['gapped'] = False

        if para_dict['gapped']:
            self.aa_list = AA_GP
        else:
            self.aa_list = AA_LS
        self.in_channels = len(self.aa_list)


    def net_init(self):
        self.lstm = nn.LSTM(self.in_channels, self.para_dict['hidden_dim'], batch_first=False,
                            num_layers=self.para_dict['hidden_layer_num'], dropout=self.para_dict['dropout_rate'])
        self.fc = nn.Linear(self.para_dict['hidden_dim'], self.para_dict['num_classes'])
        self.fixed_len = self.para_dict['fixed_len']
        self.forward = self.forward_flen if self.fixed_len else self.forward_vlen

    def forward_flen(self, Xs):
        batch_size = len(Xs)
        X = torch.FloatTensor(Xs)
        X = X.permute(1, 0, 2)
        out, _ = self.lstm(X)
        out = out[-1, :, :].reshape(batch_size, -1)  # use the last output as input for next layer
        out = torch.sigmoid(self.fc(out))

        return out

    def forward_vlen(self, Xs):
        batch_size = len(Xs)
        Xs_len = []
        Xs = np.array(Xs)
        for a in Xs:
            m = np.where(a == 0)[0]
            if not list(m):
                Xs_len.append(Xs.shape[1])
            else:
                Xs_len.append(m[0])
        a = loader.encode_data(np.array(Xs, dtype=int), aa_list=self.aa_list)
        X = torch.FloatTensor(a)
        X = torch.nn.utils.rnn.pack_padded_sequence(X, torch.tensor(Xs_len), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(X)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = out.permute(1, 0, 2)
        out = out[-1, :, :].reshape(batch_size, -1)  # use the last output as input for next layer
        out = torch.sigmoid(self.fc(out))

        return out

def test():
    para_dict = {'num_samples': 10000,
                 'seq_len': 100,
                 'batch_size': 500,
                 'model_name': 'LSTM_Model',
                 'optim_name': 'Adam',
                 'epoch': 50,
                 'learning_rate': 0.001,
                 'step_size': 5,
                 'hidden_dim': 40,
                 'hidden_layer_num': 3,
                 'dropout_rate': 0.5,
                 'fixed_len': True}

    data, out = loader.synthetic_data(num_samples=para_dict['num_samples'], seq_len=para_dict['seq_len'], aa_list=AA_LS)
    data = loader.encode_data(data)
    train_loader, test_loader = loader.train_test_loader(data, out, test_size=0.3, batch_size=para_dict['batch_size'])

    print('Parameters are', para_dict)
    model = LSTM_RNN_classifier(para_dict)
    print('Training...')
    model.fit(train_loader)
    print('Testing...')
    output = model.predict(test_loader)
    labels = np.vstack([i for _, i in test_loader])
    model.evaluate(output, labels)

if __name__ == '__main__':
    test()
