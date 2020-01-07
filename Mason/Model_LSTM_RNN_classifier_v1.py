from ..Utils.model import Model
from ..Utils import loader
import numpy as np

#-------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_RNN_classifier(Model):
    def __init__(self, para_dict, *args, **kwargs):
        super(LSTM_RNN_classifier, self).__init__(para_dict, *args, **kwargs)

        if 'dropout_rate' not in para_dict:
            self.para_dict['dropout_rate'] = 0.5
        if 'hidden_dim' not in para_dict:
            self.para_dict['hidden_dim'] = 40
        if 'hidden_layer_num' not in para_dict:
            self.para_dict['hidden_layer_num'] = 3

    def net_init(self):
        self.lstm = nn.LSTM(20, self.para_dict['hidden_dim'], batch_first = False,
                            num_layers = self.para_dict['hidden_layer_num'], dropout = self.para_dict['dropout_rate'])
        self.fc = nn.Linear(self.para_dict['hidden_dim'], 2)

    def forward(self, Xs):
        batch_size = len(Xs)

        X = torch.FloatTensor(Xs)
        X = X.permute(1,0,2)
        
        out, _ = self.lstm(X)
        out = out[-1,:,:].reshape(batch_size,-1) # use the last output as input for next layer
        out = torch.sigmoid(self.fc(out))

        return out

#--------------------------------------------------------------
if __name__ == '__main__':
    para_dict = {'num_samples':1000,
              'seq_len':10,
              'batch_size':20,
              'model_name':'LSTM_Model',
              'optim_name':'Adam',
              'epoch':20,
              'learning_rate':0.001,
              'step_size':5,
              'hidden_dim': 40,
              'hidden_layer_num': 3,
              'dropout_rate':0.5}

    data, out = loader.synthetic_data(num_samples=para_dict['num_samples'], seq_len=para_dict['seq_len'])
    data = loader.encode_data(data)
    train_loader, test_loader = loader.train_test_loader(data, out, test_size=0.3, batch_size=para_dict['batch_size'])
    model = LSTM_RNN_classifier(para_dict)
    
    model.fit(train_loader)
    output = model.predict(test_loader)
    labels = np.vstack([i for _, i in test_loader])
    mat, acc, mcc = model.evaluate(output, labels)


