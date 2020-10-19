from AIPT.Utils.model import Model
from AIPT.Utils import loader
import numpy as np 
import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score
import json
import os

#---------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import AIPT.Utils.Dev.dev_utils as dev_utils

from torch.utils.tensorboard import SummaryWriter
from AIPT.Utils.plotting import plot_confusion_matrix, plot_to_image

DEBUG_MODE = True
if DEBUG_MODE:
    dev_utils.get_mod_time(__file__, verbose=True)

class CNN(Model):
    def __init__(self, para_dict, embedding_fn, *args, **kwargs):
        '''
        @param embedding_fn: takes AA sequence (str) aa_seq and outputs embedding of dimension len(aa_seq) * para_dict[embedding_dim]
        '''
        super(CNN, self).__init__(para_dict, *args, **kwargs) # todo: what's the point of this line?
        
        if 'embedding_dim' not in para_dict:
            self.para_dict['embedding_dim'] = 15
        if 'dropout_rate' not in para_dict:
            self.para_dict['dropout_rate'] = 0.5 # arbitrary
        if 'conv1_n_filters' not in para_dict:
            self.para_dict['conv1_n_filters'] = 8 # from paper
        if 'conv2_n_filters' not in para_dict:
            self.para_dict['conv2_n_filters'] = 16 # from paper
        if 'filter_size' not in para_dict:
            self.para_dict['filter_size'] = 2 # from paper
        if 'max_pool_filter_size' not in para_dict:
            self.para_dict['max_pool_filter_size'] = 2 # from paper
        if 'fc_hidden_dim' not in para_dict:
            self.para_dict['fc_hidden_dim'] = 10 # from paper
        if 'stride' not in para_dict:
            self.para_dict['stride'] = 2 # from paper
        if 'GPU' not in para_dict:
            self.para_dict['GPU'] = False
            
        # logging
        if 'run_name' not in para_dict:
            self.para_dict['run_name'] = f'default_run' # todo: add timestamp
        if 'log_dir' not in para_dict:
            log_dir = 'logs'
            
        self.writer = SummaryWriter(log_dir=self.para_dict['log_dir'])
        
        
        self.embedding_fn = embedding_fn

        #if self.para_dict['GPU']:
        #    torch.set_default_tensor_type('torch.cuda.FloatTensor')
        #else:
        #    torch.set_default_tensor_type('torch.FloatTensor')
    
    def net_init(self):
        def get_width_after_conv(width, filter_size, stride):
            return (width - filter_size)//stride + 1
        start_width = self.para_dict['seq_len']
        self.conv1 = nn.Conv1d(in_channels = self.para_dict['embedding_dim'], #todo: support gapped?
                               out_channels = self.para_dict['conv1_n_filters'],
                               kernel_size = self.para_dict['filter_size'],
                               stride = 1, padding = 0) # todo fix strides
        width = get_width_after_conv(start_width, self.para_dict['filter_size'], 1)
        self.pool1 = nn.MaxPool1d(kernel_size = self.para_dict['max_pool_filter_size'], stride = self.para_dict['stride']) # todo: make 2 a param
        width = get_width_after_conv(width, self.para_dict['filter_size'], 2)
        self.conv2 = nn.Conv1d(in_channels = self.para_dict['conv1_n_filters'], out_channels = self.para_dict['conv2_n_filters'], kernel_size = self.para_dict['filter_size'], stride = 1, padding=0)
        width = get_width_after_conv(width, self.para_dict['filter_size'], 1)
        self.pool2 = nn.MaxPool1d(kernel_size = self.para_dict['max_pool_filter_size'], stride = self.para_dict['stride']) # todo: make 2 a param
        width = get_width_after_conv(width, self.para_dict['filter_size'], 2)
        self.fc1 = nn.Linear(in_features = width * self.para_dict['conv2_n_filters'], 
                             out_features = self.para_dict['fc_hidden_dim'])
        self.fc2 = nn.Linear(in_features = self.para_dict['fc_hidden_dim'], out_features = 2) #todo: parameterize
        self.dropout = nn.Dropout(p = self.para_dict['dropout_rate'])
        
        if self.para_dict['GPU']:
            self.cuda()

    def forward(self, Xs, _aa2id=None):
        batch_size = len(Xs)
        
        if self.para_dict['GPU']:
            X = Xs.cuda()
#         else:
#             X = torch.FloatTensor(Xs)
        
        #X = torch.FloatTensor(Xs)
#         X = X.permute(0,2,1)
        out = self.embedding_fn(Xs)
        out = out.permute(0, 2, 1) # todo: why is this necessary?
        
        out = self.dropout(self.conv1(out))
        out = F.relu(out)
        out = self.pool1(out)
        
        out = self.dropout(self.conv2(out))
        out = F.relu(out)
        out = self.pool2(out)
        
        out = out.reshape(batch_size, -1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out

    def forward4predict(self, Xs):
#         pdb.set_trace()
#         batch_size = len(Xs)

#         if self.para_dict['GPU']:
#             X = Xs.cuda()
#         else:
#             X = torch.FloatTensor(Xs)
            
#         X = X.permute(0,2,1)
        
#         out = self.conv1(X)
#         out = self.pool(out)
#         out = out.reshape(batch_size, -1)
#         out = F.relu(self.fc1(out))
#         out = torch.sigmoid(self.fc2(out))
#         out = self.forward(Xs)
        return self.forward(Xs)

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

    def fit(self, data_loader, test_loader=None):
        print('fit called')

        self.net_init()
        saved_epoch = self.load_model()
        if saved_epoch:
            print('Found saved model from Epoch', saved_epoch)
        else:
            print('No saved model found.')

        self.train()
        optimizer = self.optimizers()
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.para_dict['step_size'], gamma=0.5 ** (self.para_dict['epoch'] / self.para_dict['step_size']))

        loss_func = self.objective()
        outputs_train = []
        for e in range(saved_epoch, self.para_dict['epoch']):
            total_loss = 0
            outputs_train = []
            labels_train = []
            for input in data_loader:
                features, labels = input
                labels_train.append(labels)
                logps = self.forward(features)
                if self.para_dict['GPU']:
                    loss = loss_func(self.para_dict, logps, labels.type(torch.long).cuda())
                    outputs_train.append(logps.cpu().detach().numpy())
                else:
                    loss = loss_func(self.para_dict, logps, torch.tensor(labels).type(torch.long))
                    outputs_train.append(logps.detach().numpy())
                    
                total_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #scheduler.step()
            
            self.writer.add_scalar('Loss/train', total_loss, e)
            if test_loader is not None:
                outputs_test, labels_test, test_loss = self.predict(test_loader)
                self.writer.add_scalar('Loss/test', test_loss, e)
            
            if (e + 1) % 5 == 0:
                self.save_model('Epoch_' + str(e + 1), self.state_dict())
                print('Epoch: %d: Train Loss=%.3f' % (e + 1, total_loss)) # todo: refactor total_loss to train_loss
                tail_n = 5
#                 if outputs_train and DEBUG_MODE:
#                     print(outputs_train[-tail_n:], labels_train[-tail_n:])
                
                labels = np.concatenate([i for _, i in data_loader])
                train_cm, train_acc, train_mcc = self.evaluate(np.concatenate(outputs_train), labels)
                self.writer.add_scalar('Accuracy/train', train_acc, e)
                self.writer.add_scalar('MCC/train', train_mcc, e)
                cm_plot_train = plot_confusion_matrix(train_cm, self.para_dict['classes'])
                self.writer.add_image('Confusion/train', plot_to_image(cm_plot_train), e)
                print()
            
                if test_loader is not None:
#                     pdb.set_trace()
                    print('Epoch: %d: Test Loss=%.3f' % (e + 1, test_loss))
                    test_cm, test_acc, test_mcc = self.evaluate(outputs_test, labels_test)
                    self.writer.add_scalar('Accuracy/test', test_acc, e)
                    self.writer.add_scalar('MCC/test', test_mcc, e)
                    cm_plot_test = plot_confusion_matrix(test_cm, self.para_dict['classes'])
                    self.writer.add_image('Confusion/test', plot_to_image(cm_plot_test), e)
                    print(20*'=' + '\n\n')

        if outputs_train:
            return np.concatenate(outputs_train)
        
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
        loss_func = self.objective()
        total_loss = 0
        with torch.no_grad():
            for data in data_loader:
                # print(data)
                inputs, label = data
                outputs = self.forward4predict(inputs)
                total_loss += loss_func(self.para_dict, outputs, torch.tensor(label).type(torch.long))
                labels_test.append(label)
                
                if self.para_dict['GPU']:
                    all_outputs.append(outputs.cpu().detach().numpy())
                else:
                    all_outputs.append(outputs.detach().numpy())
                # labels_test.append(np.array(l))

            return np.vstack(all_outputs), np.hstack(labels_test), total_loss

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
    para_json = os.path.join(model.model_path,'train_parameters.json')
    print(para_json)
    para_dict = json.load(open(para_json, 'r'))
    prev_model = CNN_classifier(para_dict)
    prev_model.net_init()
    prev_model.load_model()
    seed_seqs, labels = next(iter(train_loader))
    new_seqs = prev_model.optimization(seed_seqs, step_size = 0.001, interval = 10)
    # check sequence identity
    new_seqs.permute(0,2,1).argmax(dim = 1).unsqueeze(dim=1)[0,:,:]
    seed_seqs.permute(0,2,1).argmax(dim = 1).unsqueeze(dim=1)[0,:,:]
    
if __name__ == '__main__':
    test() 
    
