from ...Utils.model import Model
from ...Utils import loader
import numpy as np

#---------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN_2Layer(Model):
    def __init__(self, para_dict, *args, **kwargs):
        super(CNN_classifier, self).__init__(para_dict, *args, **kwargs)

        if 'dropout_rate' not in para_dict:
            self.para_dict['dropout_rate'] = 0.5
        if 'fc_hidden_dim' not in para_dict:
            self.para_dict['fc_hidden_dim'] = 32
    
    def net_init(self):
        self.fc1 = nn.Linear(in_features = 20, out_features = self.para_dict['fc_hidden_dim'])
        self.fc2 = nn.Linear(in_features = self.para_dict['fc_hidden_dim'], out_features = self.para_dict['fc_hidden_dim'])
        self.fc3 = nn.Linear(in_features = self.para_dict['fc_hidden_dim'], out_features = 2)
        self.m = nn.Softmax(dim = 2)

    def forward(self, Xs, _aa2id=None):
        batch_size = len(Xs)

        X = torch.FloatTensor(Xs)
        X = X.permute(0,2,1)
        
        out = F.dropout(self.fc1(X), p = self.para_dict['dropout_rate']) 
        out = F.dropout(self.fc2(out), p = self.para_dict['dropout_rate'])
        out = self.m(self.fc3(out))

        return out

    def objective(self):
        return nn.CrossEntropyLoss()

    def optimizers(self):

        return optim.RMSprop(self.parameters(), lr=self.para_dict['learning_rate'] ,
                             rho=0.9, epsilon=1e-06)

#----------------------------------------------------------
if __name__ == '__main__':
    para_dict = {'num_samples':1000,
              'seq_len':10,
              'batch_size':20,
              'model_name':'DNN_2Layer',
              'epoch':20,
              'learning_rate':0.001,
              'fc_hidden_dim':32,
              'dropout_rate':0.5}

    data, out = loader.synthetic_data(num_samples=para_dict['num_samples'], seq_len=para_dict['seq_len'])
    data = loader.encode_data(data)
    train_loader, test_loader = loader.train_test_loader(data, out, test_size=0.3, batch_size=para_dict['batch_size'])
    model = CNN_classifier(para_dict)
    model.fit(train_loader)
    output = model.predict(test_loader)
    labels = np.vstack([i for _, i in test_loader])
    mat, acc, mcc = model.evaluate(output, labels)
