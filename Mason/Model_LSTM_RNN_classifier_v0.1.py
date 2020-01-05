from torch.utils.data import Dataset
from ..Utils.model import Model

#-------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMRNN_classifier(Model):
    def __init__(self, param_dict, *args, **kwargs):
        super(LSTMRNN_classifier, self).__init__(param_dict, *args, **kwargs)
        self.dropout_rate = self.param_dict['dropout_rate']
        self.hidden_dim = self.param_dict['hidden_dim']
        self.hidden_layer_num = self.param_dict['hidden_layer_num']

    def net_init(self):
        self.lstm = nn.LSTM(20, self.hidden_dim, batch_first = False, 
                            num_layers = self.hidden_layer_num, dropout = self.dropout_rate)
        self.fc = nn.Linear(self.hidden_dim, 2)

    def forward(self, Xs):
        batch_size = len(Xs)

        X = torch.FloatTensor(Xs)
        X = X.permute(1,0,2)
        
        out, _ = self.lstm(X)
        out = out[-1,:,:].reshape(batch_size,-1) # use the last output as input for next layer
        out = torch.sigmoid(self.fc(out))

        return out

#--------------------------------------------------------------
if __name__ == '__main__':
    param_dict = {'num_samples':1000,
              'seq_len':10,
              'batch_size':20,
              'model_name':'Model',
              'optim_name':'Adam',
              'epoch':20,
              'learning_rate':0.001,
              'hidden_dim': 40,
              'hidden_layer_num': 3,
              'dropout_rate':0.5
              }

    data, out = loader.synthetic_data(num_samples=param_dict['num_samples'], seq_len=param_dict['seq_len'])
    data = loader.encode_data(data)
    train_loader, test_loader = loader.train_test_loader(data, out, test_size=0.3, batch_size=param_dict['batch_size'])
    model = LSTMRNN_classifier(param_dict)

    if model.load_param(model.modelnamepath) is None:
        model.save_param(model.modelnamepath, model.param_dict)
        model.fit(train_loader)

    out_test, labels_test = model.predict(test_loader)
    mat, acc, mcc = model.evaluate(out_test, labels_test)


