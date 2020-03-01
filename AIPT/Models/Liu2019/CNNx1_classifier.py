from AIPT.Utils.model import Model
from AIPT.Utils import loader
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
        if 'GPU' not in para_dict:
            self.para_dict['GPU'] = False

        #if self.para_dict['GPU']:
        #    torch.set_default_tensor_type('torch.cuda.FloatTensor')
        #else:
        #    torch.set_default_tensor_type('torch.FloatTensor')
    
    def net_init(self):
        self.conv1 = nn.Conv1d(in_channels = 21, 
                               out_channels = self.para_dict['n_filter'],
                               kernel_size = self.para_dict['filter_size'],
                               stride = 1, padding = 0)
        self.pool = nn.MaxPool1d(kernel_size = 2, stride = self.para_dict['stride'])
        self.fc1 = nn.Linear(in_features = int(np.ceil((self.para_dict['seq_len']-self.para_dict['filter_size']) / self.para_dict['stride'])) * self.para_dict['n_filter'], 
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
        
        #X = torch.FloatTensor(Xs)
        X = X.permute(0,2,1)
        
        out = F.dropout(self.conv1(X), p = self.para_dict['dropout_rate'])
        out = self.pool(out)
        out = out.reshape(batch_size, -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))

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
        
        out = self.conv1(self.X_variable)
        out = self.pool(out)
        out = out.reshape(batch_size, -1)
        out = F.relu(self.fc1(out))
        #out = torch.sigmoid(self.fc2(out))
        out = self.fc2(out)
        
        return out

    def get_gradient(self, Xs):
        out = self.forward4optim(Xs)
        
        act = out[:,1]
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
                act = out[:,1]
                self.X_variable.grad = torch.zeros(self.X_variable.shape)
                act.sum().backward()
                grad = self.X_variable.grad
                grad[mask] = 0
                Xs = (self.X_variable + grad * step_size).permute(0,2,1)
                
            position_mat = Xs.argmax(dim = 1).unsqueeze(dim=1)
            Xs = torch.zeros(Xs.shape).scatter_(1, position_mat, 1)
            
            act = self.forward4optim(Xs)[:,1]
            tmp_act = act.clone()
            tmp_act[mask] = -100000.0
            improve = (tmp_act > best_activations)
            if sum(improve)>0:
                best_activations[improve] = act[improve]
            holdcnt[improve] = 0
            holdcnt[~improve]=holdcnt[~improve]+1
            mask = (holdcnt>=buff_range)
            
            print('count: %d, improved: %d, mask: %d'%(count,sum(improve).item(),sum(mask).item()))
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
                if self.para_dict['GPU']:
                    loss = loss_func(logps, labels.type(torch.long).cuda())
                    outputs_train.append(logps.cpu().detach().numpy())
                else:
                    loss = loss_func(logps, torch.tensor(labels).type(torch.long))
                    outputs_train.append(logps.detach().numpy())
                    
                total_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #scheduler.step()
            
            if (e + 1) % 10 == 0:
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

    para_dict = { 
              'num_samples': 1000,
              'batch_size':100,
              'seq_len':18,
              'model_name':'Seq_32x1_16',
              'optim_name':'Adam',
              'epoch':20,
              'learning_rate':0.001,
              'step_size':5,
              'n_filter':32,
              'filter_size':5,
              'fc_hidden_dim':16,
              'dropout_rate':0.5}

    data, out = loader.synthetic_data(num_samples=para_dict['num_samples'], seq_len=para_dict['seq_len'], aa_list=aa_list)
    data = loader.encode_data(data)
    train_loader, test_loader = loader.train_test_loader(data, out, test_size=0.3, batch_size=para_dict['batch_size'])
    
    model = CNN_classifier(para_dict)
    model.fit(train_loader)
    output = model.predict(test_loader)
    labels = np.concatenate([i for _, i in test_loader])
    mat, acc, mcc = model.evaluate(output, labels)

    # demo for optimization
    para_json = './work/Seq_32x1_16_class/train_parameters.json'
    para_dict = json.load(open(para_json, 'r'))
    prev_model = CNN_classifier(para_dict)
    prev_model.net_init()
    prev_model.load_model()
    seed_seqs, labels = next(iter(train_loader))
    new_seqs = prev_model.optimization(seed_seqs, labels, 
                                       step_size = 0.001, interval = 10)
    # check sequence identity
    new_seqs.permute(0,2,1).argmax(dim = 1).unsqueeze(dim=1)[0,:,:]
    seed_seqs.permute(0,2,1).argmax(dim = 1).unsqueeze(dim=1)[0,:,:]
    
if __name__ == '__main__':
    test() 
    
