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
import torch.optim as optim
import pdb

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
        if 'stride' not in para_dict:
            self.para_dict['stride'] = 2
    
    def net_init(self):
        self.conv1 = nn.Conv1d(in_channels = 21, 
                               out_channels = self.para_dict['n_filter'],
                               kernel_size = self.para_dict['filter_size'],
                               stride = 1, padding = 0)
        self.pool = nn.MaxPool1d(kernel_size = 2, stride = self.para_dict['stride'])
        self.fc1 = nn.Linear(in_features = int(np.ceil((self.para_dict['seq_len']-self.para_dict['filter_size']) / self.para_dict['stride'])) * self.para_dict['n_filter'], 
                             out_features = self.para_dict['fc_hidden_dim'])
        self.fc2 = nn.Linear(in_features = self.para_dict['fc_hidden_dim'], out_features = 2)

    def forward(self, Xs, _aa2id=None):
        batch_size = len(Xs)

        X = torch.FloatTensor(Xs)
        X = X.permute(0,2,1)
        
        out = F.dropout(self.conv1(X), p = self.para_dict['dropout_rate'])
        out = self.pool(out)
        out = out.reshape(batch_size, -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))

        return out

    def forward4predict(self, Xs):
        batch_size = len(Xs)

        X = torch.FloatTensor(Xs)
        X = X.permute(0,2,1)
        
        out = self.conv1(X)
        out = self.pool(out)
        out = out.reshape(batch_size, -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        
        return out
    
    def forward4optim(self, Xs):
        
        # turns off the gradient descent for all params
        for param in self.parameters():
            param.requires_grad = False
        
        batch_size = len(Xs)

        X = torch.FloatTensor(Xs)
        X = X.permute(0,2,1)
        self.X_variable = torch.autograd.Variable(X, requires_grad = True)
        
        out = self.conv1(self.X_variable)
        out = self.pool(out)
        out = out.reshape(batch_size, -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        
        return out

    def get_gradient(self, Xs, labels):
        out = self.forward4optim(Xs)
        
        loss_func = self.objective()
        optim_x_iter = [self.X_variable].__iter__()
        optimizers = optim.RMSprop(optim_x_iter, lr=self.para_dict['learning_rate'])
        loss = loss_func(out, torch.tensor(labels).type(torch.long))
        loss.backward()

        return self.X_variable.grad
    
    def optimization(self, seed_seqs, labels, step_size, interval, iteration):
        
        loss_func = self.objective()
        for i in range(iteration):
            Xs = seed_seqs
            out = self.forward4optim(Xs)
            optim_x_iter = [self.X_variable].__iter__()
            
            # gradient ascent
            loss = loss_func(out, torch.tensor(labels).type(torch.long))
            self.X_variable.grad = torch.zeros(self.X_variable.shape)
            loss.backward()
            Xs = (self.X_variable + self.X_variable.grad * step_size).permute(0,2,1)
            
            if i % interval == 0:
                # projection to one-hot representation after K interval
                position_mat = self.X_variable.argmax(dim = 1).unsqueeze(dim=1)
                Xs = torch.zeros(self.X_variable.shape).scatter_(1, position_mat, 1)
                Xs = Xs.permute(0,2,1)
            
        return Xs

    def forward4optim_2(self, Xs):
        
        # turns off the gradient descent for all params
        for param in self.parameters():
            param.requires_grad = False
        
        batch_size = len(Xs)

        X = torch.FloatTensor(Xs)
        X = X.permute(0,2,1)
        self.X_variable = torch.autograd.Variable(X, requires_grad = True)
        
        out = self.conv1(self.X_variable)
        out = self.pool(out)
        out = out.reshape(batch_size, -1)
        out = F.relu(self.fc1(out))
        #out = torch.sigmoid(self.fc2(out))
        out = self.fc2(out)
        
        return out
   
    def optimization_2(self, seed_seqs, labels, step_size, interval):
        
        buff_range = 10
        best_activations = torch.tensor(np.asarray([-100000.0]*len(seed_seqs)),dtype=torch.float)
        Xs = seed_seqs
        count = 0
        mask=torch.tensor(np.array([False for i in range(len(seed_seqs))]))
        holdcnt = torch.zeros(len(seed_seqs))
        while True:
            count += 1

            for i in range(interval):
                out = self.forward4optim_2(Xs)
                act = out[:,1]
                self.X_variable.grad = torch.zeros(self.X_variable.shape)
                act.sum().backward()
                grad = self.X_variable.grad
                grad[mask] = 0
                Xs = (self.X_variable + grad * step_size).permute(0,2,1)
                
            position_mat = Xs.argmax(dim = 1).unsqueeze(dim=1)
            Xs = torch.zeros(Xs.shape).scatter_(1, position_mat, 1)
            
            act = self.forward4optim_2(Xs)[:,1]
            tmp_act = act.clone()
            tmp_act[mask] = -100000.0
            improve = (tmp_act > best_activations)
            if sum(improve)>0:
                best_activations[improve] = act[improve]
            holdcnt[improve] = 0
            holdcnt[~improve]=holdcnt[~improve]+1
            mask = (holdcnt>=buff_range)
            
            print('count: %s, improved: %s, mask: %s'%(count,sum(improve).item(),sum(mask).item()))
            if sum(mask)==len(seed_seqs) or count>1000:
                break           
                
        return Xs


    def fit(self, data_loader):

        self.net_init()
        saved_epoch = self.load_model()

        self.train()
        optimizer = self.optimizers()
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.para_dict['step_size'], gamma=0.5 ** (self.para_dict['epoch'] / self.para_dict['step_size']))

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
                loss.backward()
                optimizer.step()
            #scheduler.step()
            
            if (e + 1) % 10 == 0:
                self.save_model('Epoch_' + str(e + 1), self.state_dict())
                print('Epoch: %d: Loss=%.3f' % (e + 1, total_loss))
                
                labels = np.concatenate([i for _, i in data_loader])
                _, _, _, = self.evaluate(np.concatenate(outputs_train), labels)
    
    def fit4optim(self, seed_seq_loader, iteration, step):

        loss_func = self.objective()
        seed_seq, labels = seed_seq_loader
        optimizer = optim.RMSprop(self.parameters(), lr=step)

        for e in range(iteration):
            logps = self.forward4optim(seed_seq)
            loss = loss_func(logps, torch.tensor(labels).type(torch.long))
            optimizer.zero_grad()
            loss.backward()

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
                all_outputs.append(outputs.detach().numpy())
                # labels_test.append(np.array(l))

            return np.vstack(all_outputs)
#----------------------------------------------------------
if __name__ == '__main__':
    traindat = pd.read_csv('cdr3s.table.csv')
    # exclude not_determined
    dat = traindat.loc[traindat['enriched'] != 'not_determined']
    x = dat['cdr3'].values
    y_class = [int(xx == 'positive') for xx in dat['enriched'].values]

    para_dict = {'batch_size':100,
             'seq_len':MAX_LEN,
              'model_name':'Seq_32x1_16',
              'optim_name':'Adam',
              'epoch':20,
              'learning_rate':0.001,
              'step_size':5,
              'n_filter':32,
              'filter_size':5,
              'fc_hidden_dim':16,
              'dropout_rate':0.5}

    para_dict = {'batch_size':100,
             'seq_len':MAX_LEN,
              'model_name':'Seq_64x1_16',
              'optim_name':'Adam',
              'epoch':20,
              'learning_rate':0.001,
              'step_size':5,
              'n_filter':64,
              'filter_size':5,
              'fc_hidden_dim':16,
              'dropout_rate':0.5}

    para_dict = {'batch_size':100,
             'seq_len':MAX_LEN,
              'model_name':'Seq_32x1_16_filt3',
              'optim_name':'Adam',
              'epoch':20,
              'learning_rate':0.001,
              'step_size':5,
              'n_filter':32,
              'filter_size':3,
              'fc_hidden_dim':16,
              'dropout_rate':0.5}

    X_dat = np.array([encode_data(item, gapped = True) for item in x])
    train_loader, test_loader = train_test_loader(X_dat, np.array(y_class), batch_size=para_dict['batch_size'])
    
    model = CNN_classifier(para_dict)
    model.fit(train_loader)
    output = model.predict(test_loader)
    labels = np.concatenate([i for _, i in test_loader])
    mat, acc, mcc = model.evaluate(output, labels)

    # demo for optimization
    para_json = './work/Seq_32x1_16_class-0.0001/train_parameters.json'
    para_dict = json.load(open(para_json, 'r'))
    prev_model = CNN_classifier(para_dict)
    prev_model.net_init()
    prev_model.load_model()
    seed_seqs, labels = next(iter(train_loader))
    new_seqs = prev_model.optimization(seed_seqs, labels, 
                                       step_size = 0.001, interval = 10, iteration = 1)


    new_seqs_2 = prev_model.optimization_2(seed_seqs, labels, 
                                       step_size = 0.01, interval = 80)
    # check sequence identity
    new_seqs.permute(0,2,1).argmax(dim = 1).unsqueeze(dim=1)[0,:,:]
    seed_seqs.permute(0,2,1).argmax(dim = 1).unsqueeze(dim=1)[0,:,:]

    new_seqs_2.permute(0,2,1).argmax(dim = 1).unsqueeze(dim=1)[0,:,:]
    
    
    
