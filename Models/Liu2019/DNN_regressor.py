#from ..Utils.model import Model
#from ..Benchmarks.Liu2019_enrichment.Liu2019_data_loader import train_test_loader, encode_data
from model import Model
import numpy as np
import pandas as pd

#---------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error

class DNN_regressor(Model):
    def __init__(self, para_dict, *args, **kwargs):
        super(DNN_regressor, self).__init__(para_dict, *args, **kwargs)

        if 'dropout_rate' not in para_dict:
            self.para_dict['dropout_rate'] = 0.5
        if 'fc_hidden_dim' not in para_dict:
            self.para_dict['fc_hidden_dim'] = 32
    
    def net_init(self):
        self.fc1 = nn.Linear(in_features = 21 * self.para_dict['seq_len'], 
                             out_features = self.para_dict['fc_hidden_dim'])
        self.fc2 = nn.Linear(in_features = self.para_dict['fc_hidden_dim'], 
                             out_features = self.para_dict['fc_hidden_dim'])
        self.fc3 = nn.Linear(in_features = self.para_dict['fc_hidden_dim'], out_features = 1)

    def forward(self, Xs, _aa2id=None):
        batch_size = len(Xs)

        X = torch.FloatTensor(Xs)
        X = torch.flatten(X, start_dim=1)
        
        out = F.dropout(F.relu(self.fc1(X)), p = self.para_dict['dropout_rate'])
        out = F.dropout(F.relu(self.fc2(out)), p = self.para_dict['dropout_rate'])
        out = self.fc3(out)

        return out

    def objective(self):
        return nn.MSELoss()

    def optimizers(self):

        return optim.RMSprop(self.parameters(), lr=self.para_dict['learning_rate'],
                             eps=1e-6, alpha = 0.9) #rho=0.9, epsilon=1e-06)
    
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
    # exclude not_determined
    dat = traindat.loc[traindat['enriched'] != 'not_determined']
    x = dat['cdr3'].values
    y_reg = dat['log10(R3/R2)'].values
    y_class = [int(xx == 'positive') for xx in dat['enriched'].values]
    # scale y_reg
    y_reg_mean = np.mean(y_reg)
    y_reg_std = np.std(y_reg)
    y_reg_new = (y_reg - y_reg_mean) / y_reg_std

    X_dat = np.array([encode_data(item, gapped = True) for item in x])

    para_dict = {'seq_len':18,
             'batch_size':100,
              'model_name':'DNN_2Layer_reg',
              'epoch':20,
              'learning_rate':0.001,
              'fc_hidden_dim':32,
              'dropout_rate':0.5,
              'model_type': 'regression'}

    train_loader, test_loader = train_test_loader(X_dat, y_reg_new, batch_size=para_dict['batch_size'])
    model = DNN_regressor(para_dict)
    model.fit(train_loader)
    output = model.predict(test_loader)
    labels = np.concatenate([i for _, i in test_loader])
    r2, mse = model.evaluate(output, labels)
