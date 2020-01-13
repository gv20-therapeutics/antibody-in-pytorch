from ..Utils.model import Model
from ..Utils import loader
import numpy as np

#---------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_classifier(Model):
    def __init__(self, para_dict, *args, **kwargs):
        super(CNN_classifier, self).__init__(para_dict, *args, **kwargs)

        if 'dropout_rate' not in para_dict:
            self.para_dict['dropout_rate'] = 0.5
        if 'n_filter' not in para_dict:
            self.para_dict['n_filter'] = 400
        if 'filter_size' not in para_dict:
            self.para_dict['filter_size'] = 3
        if 'fc_hidden_dim' not in para_dict:
            self.para_dict['fc_hidden_dim'] = 50
    
    def net_init(self):
        self.conv1 = nn.Conv1d(in_channels = 20, 
                               out_channels = self.para_dict['n_filter'],
                               kernel_size = self.para_dict['filter_size'],
                               stride = 1, padding = 0)
        self.pool = nn.MaxPool1d(kernel_size = 2, stride = 1)
        self.fc1 = nn.Linear(in_features = (self.para_dict['seq_len']-self.para_dict['filter_size']) * self.para_dict['n_filter'], out_features = self.para_dict['fc_hidden_dim'])
        self.fc2 = nn.Linear(in_features = self.para_dict['fc_hidden_dim'], out_features = 2)

    def forward(self, Xs):
        batch_size = len(Xs)

        X = torch.FloatTensor(Xs)
        X = X.permute(0,2,1)
        
        out = F.dropout(self.conv1(X), p = self.para_dict['dropout_rate'])
        out = self.pool(out)
        out = out.reshape(batch_size, -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))

        return out

#----------------------------------------------------------
if __name__ == '__main__':
    para_dict = {'num_samples':1000,
              'seq_len':10,
              'batch_size':20,
              'model_name':'CNN_Model',
              'optim_name':'Adam',
              'epoch':20,
              'learning_rate':0.001,
              'step_size':5,
              'n_filter':400,
              'filter_size':3,
              'fc_hidden_dim':50,
              'dropout_rate':0.5}

    data, out = loader.synthetic_data(num_samples=para_dict['num_samples'], seq_len=para_dict['seq_len'])
    data = loader.encode_data(data)
    train_loader, test_loader = loader.train_test_loader(data, out, test_size=0.3, batch_size=para_dict['batch_size'])
    model = CNN_classifier(para_dict)
    model.fit(train_loader)
    output = model.predict(test_loader)
    labels = np.vstack([i for _, i in test_loader])
    mat, acc, mcc = model.evaluate(output, labels)
