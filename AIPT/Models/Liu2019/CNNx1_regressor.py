from AIPT.Utils.model import Model
from AIPT.Utils import loader
import numpy as np
import pandas as pd

#---------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error

class CNN_regressor(Model):
    def __init__(self, para_dict, *args, **kwargs):
        super(CNN_regressor, self).__init__(para_dict, *args, **kwargs)

        if 'dropout_rate' not in para_dict:
            self.para_dict['dropout_rate'] = 0.5
        if 'n_filter' not in para_dict:
            self.para_dict['n_filter'] = 400
        if 'filter_size' not in para_dict:
            self.para_dict['filter_size'] = 3
        if 'fc_hidden_dim' not in para_dict:
            self.para_dict['fc_hidden_dim'] = 50
        if 'stride' not in para_dict:
            self.para_dict['stride'] = 2
        if 'GPU' not in para_dict:
            self.para_dict['GPU'] = False
    
    def net_init(self):
        self.conv1 = nn.Conv1d(in_channels = 20, 
                               out_channels = self.para_dict['n_filter'],
                               kernel_size = self.para_dict['filter_size'],
                               stride = 1, padding = 0)
        self.pool = nn.MaxPool1d(kernel_size = 2, stride = self.para_dict['stride'])
        self.fc1 = nn.Linear(in_features = int(np.ceil((self.para_dict['seq_len']-self.para_dict['filter_size']) / self.para_dict['stride'])) * self.para_dict['n_filter'], 
                             out_features = self.para_dict['fc_hidden_dim'])
        self.fc2 = nn.Linear(in_features = self.para_dict['fc_hidden_dim'], out_features = 1)
        self.dropout = nn.Dropout(p = self.para_dict['dropout_rate'])

        if self.para_dict['GPU']:
            self.cuda()

    def forward(self, Xs, _aa2id=None):
        batch_size = len(Xs)

        if self.para_dict['GPU']:
            X = Xs.cuda()
        else:
            X = torch.FloatTensor(Xs)

        X = X.permute(0,2,1)
        
        out = self.dropout(self.conv1(X))
        out = self.pool(out)
        out = out.reshape(batch_size, -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

    def forward4predict(self, Xs):
        batch_size = len(Xs)

        if self.para_dict['GPU']:
            X = Xs.cuda()
        else:
            X = torch.FloatTensor(Xs)

        X = X.permute(0,2,1)
        
        out = self.conv1(X)
        out = self.pool(out)
        out = out.reshape(batch_size, -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

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
        
        out = self.conv1(self.X_variable)
        out = self.pool(out)
        out = out.reshape(batch_size, -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

    def get_gradient(self, Xs):
        out = self.forward4optim(Xs)
        
        act = out
        self.X_variable.grad = torch.zeros(self.X_variable.shape)
        act.sum().backward()

        return self.X_variable.grad
  
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
                Xs = (self.X_variable + grad * step_size).permute(0,2,1)
                
            position_mat = Xs.argmax(dim = 1).unsqueeze(dim=1)
            Xs = torch.zeros(Xs.shape).scatter_(1, position_mat, 1)
            
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
                if self.para_dict['GPU']:
                    loss = loss_func(logps.squeeze(), values.cuda())
                    outputs_train.append(logps.cpu().detach().numpy())
                else:
                    loss = loss_func(logps.flatten(), torch.FloatTensor(values))
                    outputs_train.append(logps.detach().numpy())
                    
                total_loss += loss
                    
                optimizer.zero_grad()
                ### apply constraint on gradients ###
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

    def predict(self, data_loader):

        self.eval()
        test_loss = 0
        all_outputs = []
        labels_test = []
        with torch.no_grad():
            for data in data_loader:
                # print(data)
                inputs, _ = data
                outputs = self.forward4predict(inputs)
                if self.para_dict['GPU']:
                    all_outputs.append(outputs.cpu().detach().numpy())
                else:
                    all_outputs.append(outputs.detach().numpy())
                # labels_test.append(np.array(l))

        return np.vstack(all_outputs)

    def print_model_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Total number of parameters: %i' % params)
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

#----------------------------------------------------------
def test():
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    para_dict = {'batch_size':100,
              'seq_len':20,
              'num_samples':1000,
              'model_name':'Seq_32x1_16_regress',
              'optim_name':'Adam',
              'epoch':20,
              'learning_rate':0.001,
              'step_size':5,
              'n_filter':32,
              'filter_size':5,
              'fc_hidden_dim':16,
              'dropout_rate':0.5}
                  
    data, out = loader.synthetic_data(num_samples=para_dict['num_samples'], 
                                      seq_len=para_dict['seq_len'], aa_list=aa_list, type = 'regressor')
    data = loader.encode_data(data)
    train_loader, test_loader = loader.train_test_loader(data, out, test_size=0.3, batch_size=para_dict['batch_size'])

    model = CNN_regressor(para_dict)
    model.fit(train_loader)
    output = model.predict(test_loader)
    labels = np.concatenate([i for _, i in test_loader])
    r2 = model.evaluate(output, labels)

if __name__ == '__main__':
    test() 
