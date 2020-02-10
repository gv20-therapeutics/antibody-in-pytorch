from ...Utils.model import Model
from ...Utils import loader
import numpy as np

#---------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    def net_init(self):
        self.conv1 = nn.Conv1d(in_channels = 21, 
                               out_channels = self.para_dict['n_filter'],
                               kernel_size = self.para_dict['filter_size'],
                               stride = 1, padding = 0)
        self.pool = nn.MaxPool1d(kernel_size = 2, stride = 1)
        self.fc1 = nn.Linear(in_features = (self.para_dict['seq_len']-self.para_dict['filter_size']) * self.para_dict['n_filter'], out_features = self.para_dict['fc_hidden_dim'])
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
    y_class = [int(xx == 'positive') for xx in dat['enriched'].values]

    para_dict = {'batch_size':100,
             'seq_len':17,
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
             'seq_len':17,
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
             'seq_len':17,
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


