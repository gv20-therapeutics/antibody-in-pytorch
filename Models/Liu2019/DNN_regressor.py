from ...Utils.model import Model
from ...Utils import loader
import numpy as np

#---------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from sklearn.metrics import r2_score

#----------------------------------
# data loader for Liu2019
import pdb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
AA_LS = 'ACDEFGHIKLMNPQRSTVWY-'

MAX_LEN = 17
gap_pos_17 = dict(zip(list(range(8,17)), [4,5,5,6,6,7,7,8,8]))

def train_test_loader(x, y=None, test_size=0.2, batch_size=16):

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True)
    
    x_tensor = torch.from_numpy(X_train).float()
    y_tensor = torch.from_numpy(y_train).float()
    train_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)
    
    x_tensor = torch.from_numpy(X_test).float()
    y_tensor = torch.from_numpy(y_test).float()
    test_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)

    return train_loader, test_loader

def encode_data(data, aa_list = AA_LS, gapped = True, gap_pos = gap_pos_17):
    global MAX_LEN
    aa_mapping = dict(zip(AA_LS, list(range(len(AA_LS)))))
    codes = np.eye(len(aa_list))
    if gapped:
        if len(data) < 17:
            temp_break = gap_pos[len(data)]
            data = data[0:temp_break] + ''.join(['-' for _ in range(MAX_LEN - len(data))]) + data[temp_break:]
    else:
        if len(data) < 17:
            data = data + ''.join(['-' for _ in range(MAX_LEN - len(data))])
    return np.array([codes[aa_mapping[kk]] for kk in data])

#--------------------------------------

class DNN_2Layer_regressor(Model):
    def __init__(self, para_dict, *args, **kwargs):
        super(DNN_2Layer_regressor, self).__init__(para_dict, *args, **kwargs)

        if 'dropout_rate' not in para_dict:
            self.para_dict['dropout_rate'] = 0.5
        if 'fc_hidden_dim' not in para_dict:
            self.para_dict['fc_hidden_dim'] = 32
    
    def net_init(self):
        self.fc1 = nn.Linear(in_features = 21 * 17, out_features = self.para_dict['fc_hidden_dim'])
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
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.para_dict['step_size'], 
                                              gamma=0.5 ** (self.para_dict['epoch'] / self.para_dict['step_size']))
        
        for e in range(saved_epoch, self.para_dict['epoch']):
            for iter_num in range(10):
                total_loss = 0
                for input in data_loader:
                    outputs_train = []
                    features, values = input
                    logps = self.forward(features)
                    loss = self.objective()
                    loss = loss(logps.flatten(), torch.FloatTensor(values))
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
                    
                    scheduler.step()

            self.save_model('Epoch_' + str(e + 1), self.state_dict())
            print('Epoch: %d: Loss=%.3f' % (e + 1, total_loss))
            
            test_pred = []
            test_true = []
            for xx_test, yy_test in train_loader:
                test_pred.append(self.forward(xx_test).detach().numpy().flatten())
                test_true.append(yy_test.detach().numpy().flatten())
            self.evaluate(test_pred, test_true)
    
    def evaluate(self, outputs, values):
        y_true = np.concatenate(values)
        y_pred = np.concatenate(outputs)
        r2 = r2_score(y_true, y_pred)

        print('Test: R2 score = %.3f' % (r2))

        return r2

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

    para_dict = {'batch_size':100,
              'model_name':'DNN_2Layer_reg',
              'epoch':20,
              'learning_rate':0.001,
              'fc_hidden_dim':32,
              'dropout_rate':0.5,
              'model_type': 'regression'}

    train_loader, test_loader = train_test_loader(X_dat, y_reg_new, batch_size=para_dict['batch_size'])
    model = DNN_2Layer_regressor(para_dict)
    model.fit(train_loader)
    output = model.predict(test_loader)
    labels = np.vstack([i for _, i in test_loader])
    mat, acc, mcc = model.evaluate(output, labels)
