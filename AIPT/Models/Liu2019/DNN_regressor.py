from AIPT.Utils.model import Model
from AIPT.Utils import loader
import numpy as np
import pandas as pd

#---------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from AIPT.Models.Liu2019.CNNx1_regressor import CNN_regressor

class DNN_regressor(CNN_regressor):
    def __init__(self, para_dict, *args, **kwargs):
        super(DNN_regressor, self).__init__(para_dict, *args, **kwargs)

        if 'dropout_rate' not in para_dict:
            self.para_dict['dropout_rate'] = 0.5
        if 'fc_hidden_dim' not in para_dict:
            self.para_dict['fc_hidden_dim'] = 32
    
    def net_init(self):
        self.fc1 = nn.Linear(in_features = 20 * self.para_dict['seq_len'], 
                             out_features = self.para_dict['fc_hidden_dim'])
        self.fc2 = nn.Linear(in_features = self.para_dict['fc_hidden_dim'], 
                             out_features = self.para_dict['fc_hidden_dim'])
        self.fc3 = nn.Linear(in_features = self.para_dict['fc_hidden_dim'], out_features = 1)

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
        out = self.fc3(out)

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
        out = self.fc3(out)

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
        
        X = torch.flatten(X, start_dim=1)
        self.X_variable = torch.autograd.Variable(X, requires_grad = True)
        
        out = F.relu(self.fc1(self.X_variable))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out
    
    def optimization(self, seed_seqs, step_size, interval):
        
        buff_range = 10
        best_activations = torch.tensor(np.asarray([-100000.0]*len(seed_seqs)),dtype=torch.float)
        Xs = seed_seqs
        count = 0
        mask=torch.tensor(np.array([False for i in range(len(seed_seqs))]))
        holdcnt = torch.zeros(len(seed_seqs))
        while True:
            count += 1

            for i in range(interval):
                out = self.forward4optim(Xs)
                act = out
                self.X_variable.grad = torch.zeros(self.X_variable.shape)
                act.sum().backward()
                grad = self.X_variable.grad
                grad[mask] = 0
                
                Xs = (self.X_variable + grad * step_size).reshape(seed_seqs.shape)
                
            position_mat = Xs.argmax(dim = 2).unsqueeze(dim=2)
            Xs = torch.zeros(Xs.shape).scatter_(2, position_mat, 1)
            
            act = self.forward4optim(Xs)
            tmp_act = act.clone().flatten()
            tmp_act[mask] = -100000.0
            improve = (tmp_act > best_activations)
            if sum(improve)>0:
                best_activations[improve] = act[improve].flatten()
            holdcnt[improve] = 0
            holdcnt[~improve]=holdcnt[~improve]+1
            mask = (holdcnt>=buff_range)
            
            print('count: %d, improved: %d, mask: %d'%(count,sum(improve).item(),sum(mask).item()))
            if sum(mask)==len(seed_seqs) or count>1000:
                break           
                
        return Xs


#----------------------------------------------------------
def test():
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    para_dict = {'seq_len':18,
             'batch_size':100,
              'model_name':'DNN_2Layer_reg',
              'epoch':20,
              'num_samples':1000,
              'learning_rate':0.001,
              'fc_hidden_dim':32,
              'dropout_rate':0.5,
              'model_type': 'regression'}

    data, out = loader.synthetic_data(num_samples=para_dict['num_samples'], 
                                      seq_len=para_dict['seq_len'], aa_list=aa_list, type = 'regressor')
    data = loader.encode_data(data)
    train_loader, test_loader = loader.train_test_loader(data, out, test_size=0.3, batch_size=para_dict['batch_size'])

    model = DNN_regressor(para_dict)
    model.fit(train_loader)
    output = model.predict(test_loader)
    labels = np.concatenate([i for _, i in test_loader])
    r2, mse = model.evaluate(output, labels)

if __name__ == '__main__':
    test() 


