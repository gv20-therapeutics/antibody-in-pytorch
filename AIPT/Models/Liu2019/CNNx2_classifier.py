#from ..Utils.model import Model
#from ..Benchmarks.Liu2019_enrichment.Liu2019_data_loader import train_test_loader, encode_data
from model import Model
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score

#---------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb 
from CNNx1_classifier import CNN_classifier

class CNNx2_classifier(CNN_classifier):
    def __init__(self, para_dict, *args, **kwargs):
        super(CNNx2_classifier, self).__init__(para_dict, *args, **kwargs)

        if 'dropout_rate' not in para_dict:
            self.para_dict['dropout_rate'] = 0.5
        if 'n_filter1' not in para_dict:
            self.para_dict['n_filter1'] = 32
        if 'n_filter2' not in para_dict:
            self.para_dict['n_filter2'] = 64
        if 'filter_size' not in para_dict:
            self.para_dict['filter_size'] = 5
        if 'fc_hidden_dim' not in para_dict:
            self.para_dict['fc_hidden_dim'] = 16
        if 'stride' not in para_dict:
            self.para_dict['stride'] = 2
        if 'pool_kernel_size' not in para_dict:
            self.para_dict['pool_kernel_size'] = 2
    
    def net_init(self):
        self.conv1 = nn.Conv1d(in_channels = 21, 
                               out_channels = self.para_dict['n_filter1'],
                               kernel_size = self.para_dict['filter_size'],
                               stride = 1, padding = 0)
        self.pool1 = nn.MaxPool1d(kernel_size = 2, stride = self.para_dict['stride'])
        self.conv2 = nn.Conv1d(in_channels = self.para_dict['n_filter1'], 
                               out_channels = self.para_dict['n_filter2'],
                               kernel_size = self.para_dict['filter_size'],
                               stride = 1, padding = 0)
        self.pool2 = nn.MaxPool1d(kernel_size = 2, stride = 1)
        
        cnn_flatten_size1 = ((self.para_dict['seq_len']-self.para_dict['filter_size'] + 1) - self.para_dict['pool_kernel_size']) / self.para_dict['stride'] + 1
        cnn_flatten_size2 = (cnn_flatten_size1 - self.para_dict['filter_size'] + 1 - self.para_dict['pool_kernel_size']) + 1
        self.fc1 = nn.Linear(in_features = int(cnn_flatten_size2) * self.para_dict['n_filter2'], 
                             out_features = self.para_dict['fc_hidden_dim'])
        self.fc2 = nn.Linear(in_features = self.para_dict['fc_hidden_dim'], out_features = 2)
        
        if self.para_dict['GPU']:
            self.cuda()
        
    def forward(self, Xs, _aa2id=None):
        batch_size = len(Xs)

        if self.para_dict['GPU']:
            X = Xs.cuda()
        else:
            X = torch.FloatTensor(Xs)
            
        X = X.permute(0,2,1)
        
        out = F.relu(self.conv1(X))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = out.reshape(batch_size, -1)
        out = F.dropout(F.relu(self.fc1(out)), p=self.para_dict['dropout_rate'])
        out = torch.sigmoid(self.fc2(out))

        return out

    def forward4predict(self, Xs):
        batch_size = len(Xs)

        if self.para_dict['GPU']:
            X = Xs.cuda()
        else:
            X = torch.FloatTensor(Xs)
        X = X.permute(0,2,1)
        
        out = F.relu(self.conv1(X))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = out.reshape(batch_size, -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

    def forward4optim(self, Xs):
        
        # turns off the gradient descent for all params
        for param in self.parameters():
            param.requires_grad = False
        
        batch_size = len(Xs)

        if self.para_dict['GPU']:
            X = Xs.cuda()
        else:
            X = torch.FloatTensor(Xs)
        X = X.permute(0,2,1)

        self.X_variable = torch.autograd.Variable(X, requires_grad = True)
        
        out = F.relu(self.conv1(self.X_variable))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = out.reshape(batch_size, -1)
        out = F.relu(self.fc1(out))
        #out = torch.sigmoid(self.fc2(out))
        out = self.fc2(out)
        
        return out

#----------------------------------------------------------
def test():
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'

    para_dict = {
              'seq_len':18,
              'num_samples': 1000,
              'batch_size':100,
              'model_name':'Seq_32x2_16',
              'epoch':20,
              'learning_rate':0.001,
              'step_size':5,
              'n_filter1':32,
              'n_filter2':64,
              'filter_size':5,
              'fc_hidden_dim':16,
              'dropout_rate':0.5,
              'stride':1}

    data, out = loader.synthetic_data(num_samples=para_dict['num_samples'], seq_len=para_dict['seq_len'], aa_list=aa_list)
    data = loader.encode_data(data)
    train_loader, test_loader = loader.train_test_loader(data, out, test_size=0.3, batch_size=para_dict['batch_size'])

    model = CNNx2_classifier(para_dict)
    model.fit(train_loader)
    output = model.predict(test_loader)
    labels = np.vstack([i for _, i in test_loader])
    mat, acc, mcc = model.evaluate(output, labels)

if __name__ == '__main__':
    test()
    