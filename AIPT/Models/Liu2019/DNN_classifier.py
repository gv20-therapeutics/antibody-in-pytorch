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

class DNN_classifier(Model):
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

    def forward(self, Xs, _aa2id=None):
        batch_size = len(Xs)

        X = torch.FloatTensor(Xs)
        X = torch.flatten(X, start_dim=1)
        
        out = F.dropout(F.relu(self.fc1(X)), p = self.para_dict['dropout_rate'])
        out = F.dropout(F.relu(self.fc2(out)), p = self.para_dict['dropout_rate'])
        out = F.softmax(self.fc3(out))

        return out

    def objective(self):
        return nn.CrossEntropyLoss()

    def optimizers(self):

        return optim.RMSprop(self.parameters(), lr=self.para_dict['learning_rate'],
                             eps=1e-6, alpha = 0.9) #rho=0.9, epsilon=1e-06)
    
    def fit(self, data_loader):

        self.net_init()
        saved_epoch = self.load_model()

        self.train()
        optimizer = self.optimizers()
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.para_dict['step_size'], 
        #                                      gamma=0.5 ** (self.para_dict['epoch'] / self.para_dict['step_size']))
        loss_func = self.objective()
        for e in range(saved_epoch, self.para_dict['epoch']):
            total_loss = 0
            outputs_train = []
            for input in data_loader:
                    
                features, labels = input
                logps = self.forward(features)
                loss = loss_func(logps, torch.tensor(labels).type(torch.long)) 
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
            
                labels = np.concatenate([i for _, i in data_loader])
                _, _, _, = self.evaluate(np.concatenate(outputs_train), labels)
    
    def evaluate(self, outputs, labels):
        y_pred = []
        # print(outputs.shape)
        # print(labels.shape)
        for a in outputs:
            if a[0]>a[1]:
                y_pred.append(0)
            else:
                y_pred.append(1)
        y_true = labels.flatten()
        y_pred = np.array(y_pred)
        mat = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        print('Test: ')
        print(mat)
        print('Accuracy = %.3f ,MCC = %.3f' % (acc, mcc))
        return mat, acc, mcc

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
