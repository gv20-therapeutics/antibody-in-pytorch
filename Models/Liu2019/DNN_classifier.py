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

import torch.optim as optim
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score
from CNNx1_classifier import CNN_classifier

class DNN_classifier(CNN_classifier):
    def __init__(self, para_dict, *args, **kwargs):
        super(DNN_classifier, self).__init__(para_dict, *args, **kwargs)

        if 'dropout_rate' not in para_dict:
            self.para_dict['dropout_rate'] = 0.5
        if 'fc_hidden_dim' not in para_dict:
            self.para_dict['fc_hidden_dim'] = 32
    
    def net_init(self):
        self.fc1 = nn.Linear(in_features = 21 * self.para_dict['seq_len'], out_features = self.para_dict['fc_hidden_dim'])
        self.fc2 = nn.Linear(in_features = self.para_dict['fc_hidden_dim'], 
                             out_features = self.para_dict['fc_hidden_dim'])
        self.fc3 = nn.Linear(in_features = self.para_dict['fc_hidden_dim'], out_features = 2)

        if self.para_dict['GPU']:
            self.cuda()

    def forward(self, Xs, _aa2id=None):
        batch_size = len(Xs)

        if self.para_dict['GPU']:
            X = Xs.cuda()
        else:
            X = torch.FloatTensor(Xs)
        X = torch.flatten(X, start_dim=1)
        
        out = F.dropout(F.relu(self.fc1(X)), p = self.para_dict['dropout_rate'])
        out = F.dropout(F.relu(self.fc2(out)), p = self.para_dict['dropout_rate'])
        out = F.softmax(self.fc3(out))

        return out

    def forward4predict(self, Xs):
        batch_size = len(Xs)

        if self.para_dict['GPU']:
            X = Xs.cuda()
        else:
            X = torch.FloatTensor(Xs)
            
        X = torch.flatten(X, start_dim=1)
        
        out = F.relu(self.fc1(X))
        out = F.relu(self.fc2(out))
        out = F.softmax(self.fc3(out))
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
        
        out = F.relu(self.fc1(X))
        out = F.relu(self.fc2(out))
        #out = F.softmax(self.fc3(out))
        out = self.fc3(out)
        
        return out


#----------------------------------------------------------
if __name__ == '__main__':

    traindat = pd.read_csv('cdr3s.table.csv')
    # exclude not_determined
    dat = traindat.loc[traindat['enriched'] != 'not_determined']
    x = dat['cdr3'].values
    y_reg = dat['log10(R3/R2)'].values
    y_class = [int(xx == 'positive') for xx in dat['enriched'].values]

    X_dat = np.array([encode_data(item, gapped = True) for item in x])

    para_dict = {'batch_size':100,
              'seq_len':18,
              'model_name':'DNN_2Layer_class',
              'epoch':20,
              'learning_rate':0.001,
              'fc_hidden_dim':32,
              'dropout_rate':0.5,
              'model_type': 'classification'}

    train_loader, test_loader = train_test_loader(X_dat, y_class, batch_size=para_dict['batch_size'])
    model = DNN_classifier(para_dict)
    model.fit(train_loader)
    output = model.predict(test_loader)
    labels = np.vstack([i for _, i in test_loader])
    mat, acc, mcc = model.evaluate(output, labels)
