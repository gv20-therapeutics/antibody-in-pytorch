#from ..Utils.model import Model
#from ..Benchmarks.Liu2019_enrichment.Liu2019_data_loader import train_test_loader, encode_data
from model import Model
import numpy as np
import pandas as pd
import pdb 

#---------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error

class CNNx2_regressor(Model):
    def __init__(self, para_dict, *args, **kwargs):
        super(CNNx2_regressor, self).__init__(para_dict, *args, **kwargs)

        if 'dropout_rate' not in para_dict:
            self.para_dict['dropout_rate'] = 0.5
        if 'n_filter1' not in para_dict:
            self.para_dict['n_filter1'] = 32
        if 'n_filter2' not in para_dict:
            self.para_dict['n_filter2'] = 64
        if 'filter_size1' not in para_dict:
            self.para_dict['filter_size1'] = 5
        if 'filter_size2' not in para_dict:
            self.para_dict['filter_size2'] = 5
        if 'fc_hidden_dim' not in para_dict:
            self.para_dict['fc_hidden_dim'] = 16
        if 'stride' not in para_dict:
            self.para_dict['stride'] = 2
        if 'pool_kernel_size' not in para_dict:
            self.para_dict['pool_kernel_size'] = 2
    
    def net_init(self):
        self.conv1 = nn.Conv1d(in_channels = 21, 
                               out_channels = self.para_dict['n_filter1'],
                               kernel_size = self.para_dict['filter_size1'],
                               stride = 1, padding = 0)
        self.pool1 = nn.MaxPool1d(kernel_size = 2, stride = self.para_dict['stride'])
        self.conv2 = nn.Conv1d(in_channels = self.para_dict['n_filter1'], 
                               out_channels = self.para_dict['n_filter2'],
                               kernel_size = self.para_dict['filter_size2'],
                               stride = 1, padding = 0)
        self.pool2 = nn.MaxPool1d(kernel_size = 2, stride = 1)

        cnn_flatten_size1 = ((self.para_dict['seq_len']-self.para_dict['filter_size1'] + 1) - self.para_dict['pool_kernel_size']) / self.para_dict['stride'] + 1
        cnn_flatten_size2 = (cnn_flatten_size1 - self.para_dict['filter_size2'] + 1 - self.para_dict['pool_kernel_size']) + 1
        self.fc1 = nn.Linear(in_features = int(cnn_flatten_size2) * self.para_dict['n_filter2'], 
                             out_features = self.para_dict['fc_hidden_dim'])
        self.fc2 = nn.Linear(in_features = self.para_dict['fc_hidden_dim'], out_features = 1)
        
    def forward(self, Xs, _aa2id=None):
        batch_size = len(Xs)

        X = torch.FloatTensor(Xs)
        X = X.permute(0,2,1)
        
        out = F.relu(self.conv1(X))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = out.reshape(batch_size, -1)
        out = F.dropout(F.relu(self.fc1(out)), p=self.para_dict['dropout_rate'])
        out = self.fc2(out)

        return out

    def objective(self):
        return nn.MSELoss()

    def fit(self, data_loader):

        self.net_init()
        saved_epoch = self.load_model()

        self.train()
        optimizer = self.optimizers()
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.para_dict['step_size'], 
        #                                      gamma=0.5)
        loss_func = self.objective()
        
        for e in range(saved_epoch, self.para_dict['epoch']):
            total_loss = 0
            outputs_train = []
            for input in data_loader:
                    
                features, values = input
                logps = self.forward(features)
                loss = loss_func(logps.flatten(), torch.FloatTensor(values))
                total_loss += loss
                outputs_train.append(logps.detach().numpy())
                    
                optimizer.zero_grad()
                ### apply constraint on gradients ###
                #pdb.set_trace()
                loss.backward()
                for name, param in self.state_dict().items():
                    if name == 'fc1.weight' or name == 'fc2.weight':
                        nn.utils.clip_grad_norm_(param, max_norm = 3, norm_type=2)
                optimizer.step()
                    
            #scheduler.step()

            if (e+1) % 10 == 0:
                self.save_model('Epoch_' + str(e + 1), self.state_dict())
                print('Epoch: %d: Loss=%.3f' % (e + 1, total_loss))
            
                values = np.concatenate([i for _, i in data_loader])
                self.evaluate(np.concatenate(outputs_train), values)

    def evaluate(self, outputs, values):
        y_pred = outputs.flatten()
        y_true = values.flatten()
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        
        print('R2 score = %.3f' % (r2))
        print('MSE = %.3f' % (mse))

        return r2, mse

#----------------------------------------------------------
if __name__ == '__main__':
    traindat = pd.read_csv('cdr3s.table.csv')
    MAX_LEN = 17
    # exclude not_determined
    dat = traindat.loc[traindat['enriched'] != 'not_determined']
    x = dat['cdr3'].values
    y_reg = dat['log10(R3/R2)'].values
    # scale y_reg
    y_reg_mean = np.mean(y_reg)
    y_reg_std = np.std(y_reg)
    y_reg_new = (y_reg - y_reg_mean) / y_reg_std

    para_dict = {'seq_len':18,
              'batch_size':100,
              'model_name':'Seq_32x2_16',
              'epoch':20,
              'learning_rate':0.001,
              'step_size':5,
              'n_filter1':32,
              'n_filter2':64,
              'filter_size1':5,
              'filter_size2':5,
              'fc_hidden_dim':16,
              'dropout_rate':0.5,
              'stride':1}

    X_dat = np.array([encode_data(item, gapped = True) for item in x])
    train_loader, test_loader = train_test_loader(X_dat, np.array(y_class), batch_size=para_dict['batch_size'])

    model = CNNx2_regressor(para_dict)
    model.fit(train_loader)
    output = model.predict(test_loader)
    labels = np.vstack([i for _, i in test_loader])
    r2 = model.evaluate(output, labels)
