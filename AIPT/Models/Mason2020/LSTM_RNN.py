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
        # if 'fixed_len' not in para_dict:
        #     self.fixed_len = True

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

        if type(self.para_dict['num_classes']) is not list:
            self.fc = nn.Linear(self.para_dict['hidden_dim'], self.para_dict['num_classes'])

    def hidden(self, Xs):
        batch_size = len(Xs)
        if self.para_dict['gapped'] == True and self.para_dict['pad'] == True:
            Xs = loader.encode_data(Xs, aa_list=AA_GP)
        elif self.para_dict['pad'] == True:
            Xs = loader.encode_data(Xs, aa_list=AA_LS)
        X = torch.FloatTensor(Xs)
        X = X.permute(1, 0, 2)
        out, _ = self.lstm(X)
        out = out[-1, :, :].reshape(batch_size, -1)  # use the last output as input for next layer

        return out

    def forward(self, Xs):

        out = self.hidden(Xs)
        out = torch.sigmoid(self.fc(out))

        return out

def test():
    para_dict = {'num_samples': 2000,
                 'seq_len': 100,
                 'batch_size': 700,
                 'model_name': 'LSTM_Model',
                 'optim_name': 'Adam',
                 'epoch': 5,
                 'learning_rate': 0.001,
                 'step_size': 5,
                 'hidden_dim': 40,
                 'hidden_layer_num': 3,
                 'dropout_rate': 0.5,
                 'gapped': False,
                 'pad': False}

    data, out = loader.synthetic_data(num_samples=para_dict['num_samples'], seq_len=para_dict['seq_len'], aa_list=AA_LS)
    data = loader.encode_data(data)
    train_loader, test_loader = loader.train_test_loader(data, out, test_size=0.3, batch_size=para_dict['batch_size'], sample=True)

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
