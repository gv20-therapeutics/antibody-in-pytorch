from ..Utils.model import Model
from ..Utils import loader
import numpy as np

#--------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence

class LSTM_Bi(Model):
    def __init__(self, para_dict, *args, **kwargs):
        super(LSTM_Bi, self).__init__(para_dict, *args, **kwargs)

        if 'hidden_dim' not in para_dict:
            self.para_dict['hidden_dim'] = 64
        if 'embedding_dim' not in para_dict:
            self.para_dict['embedding_dim'] = 64
        if 'in_dim' not in para_dict:
            self.para_dict['in_dim'] = 21
        if 'out_dim' not in para_dict:
            self.para_dict['out_dim'] = 20

    def net_init(self):
        self.word_embeddings = nn.Embedding(self.para_dict['in_dim'], self.para_dict['embedding_dim'])
        self.lstm_f = nn.LSTM(self.para_dict['embedding_dim'], self.para_dict['hidden_dim'], batch_first=True)
        self.lstm_b = nn.LSTM(self.para_dict['embedding_dim'], self.para_dict['hidden_dim'], batch_first=True)
        self.fc1 = nn.Linear(self.para_dict['hidden_dim'], self.para_dict['hidden_dim'])
        self.fc2 = nn.Linear(self.para_dict['hidden_dim'], self.para_dict['hidden_dim'])
        self.fc3 = nn.Linear(self.para_dict['hidden_dim'], self.para_dict['out_dim'])
        
    def forward(self, Xs):
        batch_size = len(Xs)

        # pad <EOS>(=22) & <SOS>(=21)
        Xs_f = [[21] + seq[:-1] for seq in Xs]
        Xs_b = [[22] + seq[::-1][:-1] for seq in Xs]
        
        # get sequence lengths
        X_len = len(Xs_f[0])
        
        # list to *.tensor
        Xs_f = torch.tensor(Xs_f, device=self.device)
        Xs_b = torch.tensor(Xs_b, device=self.device)
        
        # embedding
        Xs_f = self.word_embeddings(Xs_f)
        Xs_b = self.word_embeddings(Xs_b)
        
        # feed the lstm by the packed input
        ini_hc_state_f = (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                          torch.zeros(1, batch_size, self.hidden_dim).to(self.device))
        ini_hc_state_b = (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                          torch.zeros(1, batch_size, self.hidden_dim).to(self.device))

        # lstm
        lstm_out_f, _ = self.lstm_f(Xs_f, ini_hc_state_f)
        lstm_out_b, _ = self.lstm_b(Xs_b, ini_hc_state_b)
        
        # flatten forward-lstm output
        lstm_out_f = lstm_out_f.reshape(-1, self.hidden_dim)
        
        # flatten backward-lstm output
        idx_b = torch.tensor(list(range(X_len))[::-1], device=self.device)
        lstm_out_b = torch.index_select(lstm_out_b, 1, idx_b)
        lstm_out_b = lstm_out_b.reshape(-1, self.hidden_dim)    

        lstm_out_valid = lstm_out_f + lstm_out_b       
        
        # lstm hidden state to output space
        out = F.relu(self.fc1(lstm_out_valid))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        # compute scores
        scores = F.log_softmax(out, dim=1)
        
        return scores

    def objective(self):
        return nn.NLLLoss()

    def optimizers(self):

        if self.para_dict['optim_name'] == 'Adam':
            return optim.Adam(self.parameters(), lr=self.para_dict['learning_rate'])

        elif self.para_dict['optim_name'] == 'RMSprop':
            return optim.RMSprop(self.parameters(), lr=self.para_dict['learning_rate'])

        elif self.para_dict['optim_name'] == 'SGD':
            return optim.SGD(self.parameters(), lr=self.para_dict['learning_rate'])

    def fit(self, data_loader):

        self.net_init()
        saved_epoch = self.load_model()

        self.train()
        optimizer = self.optimizers()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.para_dict['step_size'], gamma=0.5 ** (self.para_dict['epoch'] / self.para_dict['step_size']))

        for e in range(saved_epoch, self.para_dict['epoch']):
            total_loss = 0
            for features, labels in data_loader:
                labels = [xx for xx in seq for seeq in labels]
                outputs_train = []
                logps = self.forward(features)
                loss = self.objective()
                loss = loss(logps, labels.type(torch.long))
                total_loss += loss
                outputs_train.append(logps.detach().numpy())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            self.save_model('Epoch_' + str(e + 1), self.state_dict())
            print('Epoch: %d: Loss=%.3f' % (e + 1, total_loss))


    def evaluate(self, outputs, labels):

        labels_flatten = [s for s in seq for seq in labels]
        L = len(labels_flatten)
        predicted = torch.argmax(outputs, 1)
        corr = (predicted == labels_flatten).data.cpu().numpy()
        acc = sum(corr) / L

        print('Test: ')
        print('acc':  '{:.6f}'.format(acc))

        return acc

#------------------------------------------------------------------------------------------
if __name__ == '__main__':
    para_dict = {'num_samples':1000,
              'seq_len':20,
              'batch_size':20,
              'model_name':'LSTM_Model',
              'optim_name':'Adam',
              'epoch':20,
              'learning_rate':0.001,
              'step_size':5,
              'hidden_dim': 64,
              'embedding_dim': 64}

    data, out = loader.synthetic_data(num_samples=para_dict['num_samples'], seq_len=para_dict['seq_len'])
    train_loader, test_loader = loader.train_test_loader(data, data, test_size=0.3, batch_size=para_dict['batch_size'])
    model = LSTM_RNN_classifier(para_dict)
    
    model.fit(train_loader)
    output = model.predict(test_loader)
    labels = np.vstack([i for _, i in test_loader])
    acc = model.evaluate(output, labels)

