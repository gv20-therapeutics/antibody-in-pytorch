import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, matthews_corrcoef, accuracy_score

from ...Utils import loader
from ...Utils.model import Model

# true if gapped else false
vocab_o = {
    True: ['-'] + ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
    False: ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']}
aa2id_o = {True: dict(zip(vocab_o[True], list(range(len(vocab_o[True]))))),
           False: dict(zip(vocab_o[False], list(range(len(vocab_o[False])))))}
id2aa_o = {True: dict(zip(list(range(len(vocab_o[True]))), vocab_o[True])),
           False: dict(zip(list(range(len(vocab_o[False]))), vocab_o[False]))}

vocab_i = {True: vocab_o[True] + ['<SOS>', '<EOS>'],
           False: vocab_o[False] + ['<SOS>', '<EOS>']}
aa2id_i = {True: dict(zip(vocab_i[True], list(range(len(vocab_i[True]))))),
           False: dict(zip(vocab_i[False], list(range(len(vocab_i[False])))))}
id2aa_i = {True: dict(zip(list(range(len(vocab_i[True]))), vocab_i[True])),
           False: dict(zip(list(range(len(vocab_i[False]))), vocab_i[False]))}

# --------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence


class LSTM_Bi(Model):
    def __init__(self, para_dict, *args, **kwargs):
        super(LSTM_Bi, self).__init__(para_dict, *args, **kwargs)

        if 'in_dim' not in para_dict:
            self.in_dim = len(aa2id_i[self.para_dict['gapped']])
        if 'embedding_dim' not in para_dict:
            self.embedding_dim = 64
        if 'hidden_dim' not in para_dict:
            self.hidden_dim = 64
        if 'out_dim' not in para_dict:
            self.out_dim = len(aa2id_o[self.para_dict['gapped']])
        if 'fixed_len' not in para_dict:
            self.fixed_len = True

    def net_init(self):

        # self.hidden_dim = hidden_dim
        self.para_dict['in_dim'] = len(aa2id_i[self.para_dict['gapped']])
        self.para_dict['out_dim'] = len(aa2id_o[self.para_dict['gapped']])
        self.word_embeddings = nn.Embedding(self.para_dict['in_dim'], self.para_dict['embedding_dim'])
        self.lstm_f = nn.LSTM(self.para_dict['embedding_dim'], self.para_dict['hidden_dim'], batch_first=True)
        self.lstm_b = nn.LSTM(self.para_dict['embedding_dim'], self.para_dict['hidden_dim'], batch_first=True)
        self.fc1 = nn.Linear(self.para_dict['hidden_dim'], self.para_dict['hidden_dim'])
        self.fc2 = nn.Linear(self.para_dict['hidden_dim'], self.para_dict['hidden_dim'])
        self.fc3 = nn.Linear(self.para_dict['hidden_dim'], self.para_dict['out_dim'])
        self.fixed_len = self.para_dict['fixed_len']
        # self.forward = self.forward_flen if self.fixed_len else self.forward_vlen
        self.hidden = self.hidden_flen if self.fixed_len else self.hidden_vlen

    def hidden_flen(self, Xs):
        # batch_size = len(Xs)
        batch_size = len(Xs)
        # print(_aa2id)
        _aa2id = aa2id_i[self.para_dict['gapped']]

        # pad <EOS> & <SOS>
        Xs_f = [[_aa2id['<SOS>']] + list(seq)[:-1] for seq in Xs]
        Xs_b = [[_aa2id['<EOS>']] + list(seq)[::-1][:-1] for seq in Xs]

        # get sequence lengths
        X_len = len(Xs_f[0])

        # list to *.tensor
        Xs_f = torch.tensor(Xs_f)
        Xs_b = torch.tensor(Xs_b)

        # print(Xs_f)
        # embedding
        Xs_f = self.word_embeddings(Xs_f.type(dtype=torch.LongTensor))
        Xs_b = self.word_embeddings(Xs_b.type(dtype=torch.LongTensor))

        # feed the lstm by the packed input
        ini_hc_state_f = (torch.zeros(1, batch_size, self.para_dict['hidden_dim']),
                          torch.zeros(1, batch_size, self.para_dict['hidden_dim']))
        ini_hc_state_b = (torch.zeros(1, batch_size, self.para_dict['hidden_dim']),
                          torch.zeros(1, batch_size, self.para_dict['hidden_dim']))

        # lstm
        lstm_out_f, _ = self.lstm_f(Xs_f, ini_hc_state_f)
        lstm_out_b, _ = self.lstm_b(Xs_b, ini_hc_state_b)

        # flatten forward-lstm output
        lstm_out_f = lstm_out_f.reshape(-1, self.para_dict['hidden_dim'])

        # flatten backward-lstm output
        idx_b = torch.tensor(list(range(X_len))[::-1])
        lstm_out_b = torch.index_select(lstm_out_b, 1, idx_b)
        lstm_out_b = lstm_out_b.reshape(-1, self.para_dict['hidden_dim'])

        lstm_out_valid = lstm_out_f + lstm_out_b

        # lstm hidden state to output space
        out = F.relu(self.fc1(lstm_out_valid))
        out = F.relu(self.fc2(out))

        return out

    def hidden_vlen(self, Xs):
        batch_size = len(Xs)
        _aa2id = aa2id_i[self.para_dict['gapped']]

        # pad <EOS> & <SOS>
        Xs_f = [[_aa2id['<SOS>']] + list(seq)[:-1] for seq in Xs]
        Xs_b = [[_aa2id['<EOS>']] + list(seq)[::-1][:-1] for seq in Xs]

        # get sequence lengths
        self.Xs_len = [len(seq) for seq in Xs_f]
        self.lmax = max(self.Xs_len)

        # list to *.tensor
        Xs_f = [torch.tensor(seq) for seq in Xs_f]
        Xs_b = [torch.tensor(seq) for seq in Xs_b]

        # padding
        Xs_f = pad_sequence(Xs_f, batch_first=True)
        Xs_b = pad_sequence(Xs_b, batch_first=True)

        # embedding
        Xs_f = self.word_embeddings(Xs_f.type(dtype=torch.LongTensor))
        Xs_b = self.word_embeddings(Xs_b.type(dtype=torch.LongTensor))

        # packing the padded sequences
        Xs_f = pack_padded_sequence(Xs_f, self.Xs_len, batch_first=True, enforce_sorted=False)
        Xs_b = pack_padded_sequence(Xs_b, self.Xs_len, batch_first=True, enforce_sorted=False)

        # feed the lstm by the packed input
        ini_hc_state_f = (torch.zeros(1, batch_size, self.para_dict['hidden_dim']),
                          torch.zeros(1, batch_size, self.para_dict['hidden_dim']))
        ini_hc_state_b = (torch.zeros(1, batch_size, self.para_dict['hidden_dim']),
                          torch.zeros(1, batch_size, self.para_dict['hidden_dim']))

        lstm_out_f, _ = self.lstm_f(Xs_f, ini_hc_state_f)
        lstm_out_b, _ = self.lstm_b(Xs_b, ini_hc_state_b)

        # unpack outputs
        lstm_out_f, lstm_out_len = pad_packed_sequence(lstm_out_f, batch_first=True)
        lstm_out_b, _ = pad_packed_sequence(lstm_out_b, batch_first=True)

        lstm_out_valid_f = lstm_out_f.reshape(-1, self.para_dict['hidden_dim'])
        lstm_out_valid_b = lstm_out_b.reshape(-1, self.para_dict['hidden_dim'])

        idx_f = []
        [idx_f.extend([i * self.lmax + j for j in range(l)]) for i, l in enumerate(self.Xs_len)]
        idx_f = torch.tensor(idx_f)

        idx_b = []
        [idx_b.extend([i * self.lmax + j for j in range(l)][::-1]) for i, l in enumerate(self.Xs_len)]
        idx_b = torch.tensor(idx_b)

        lstm_out_valid_f = torch.index_select(lstm_out_valid_f, 0, idx_f)
        lstm_out_valid_b = torch.index_select(lstm_out_valid_b, 0, idx_b)

        lstm_out_valid = lstm_out_valid_f + lstm_out_valid_b

        # # lstm hidden state to output space
        out = F.relu(self.fc1(lstm_out_valid))
        out = F.relu(self.fc2(out))

        return out

    def forward(self, Xs):

        x = self.hidden(Xs)

        out = self.fc3(x)
        # compute scores
        scores = F.log_softmax(out, dim=1)

        return scores

    def objective(self):
        return nn.NLLLoss()

    def collate_fn(batch):
        return batch, [x for seq in batch for x in seq]

    def predict(self, test_loader):

        scores = []
        self.eval()
        for n, (batch, batch_flatten) in enumerate(test_loader):
            actual_batch_size = len(batch)  # last iteration may contain less sequences
            seq_len = [len(seq) for seq in batch]
            seq_len_cumsum = np.cumsum(seq_len)
            out = self.forward(batch)
            out = np.vstack(out.detach().numpy())
            out = np.split(out, seq_len_cumsum)[:-1]
            batch_scores = []
            for i in range(actual_batch_size):
                pos_scores = []
                for j in range(seq_len[i]):
                    pos_scores.append(out[i][j, batch[i][j]])
                batch_scores.append(-sum(pos_scores) / seq_len[i])
            scores.append(batch_scores[0])
        return np.array(scores)

    def plot_score_distribution(self, dict_class):

        plt.figure()
        colors = ['yellow','red','blue','green']
        for i in range(len(self.para_dict['species_type'])):
            plt.hist(dict_class['o'+str(i)], histtype='step', normed=True, color=colors[i], label=self.para_dict['species_type'][i])
        plt.legend()
        plt.savefig(os.path.join(self.model_path, 'score_plot'))
        plt.show()

    def roc_plot(self, test_loader):

        plt.figure()
        outputs = self.predict(test_loader)
        colors = ['r','b','g']
        labels = np.vstack([i for _, i in test_loader])

        dict_class = {}
        for class_name in range(len(self.para_dict['species_type'])):
            dict_class['o' + str(class_name)] = []
            dict_class['l' + str(class_name)] = []

        for i, a in enumerate(labels):
            dict_class['o'+str(int(a))].append(outputs[i])
            dict_class['l'+str(int(a))].append(a)

        for i in range(len(self.para_dict['species_type'])-1):
            label1 = np.zeros(len(dict_class['l' + str(0)]),dtype=int)
            label2 = np.ones(len(dict_class['l' + str(i+1)]), dtype=int)
            label = np.concatenate((label1, label2), axis=0)
            output = np.concatenate((np.array(dict_class['o'+str(0)]), np.array(dict_class['o'+str(i+1)])), axis=0)
            tpr, fpr, _ = roc_curve(np.array(label), np.array(output))
            AUC_score = roc_auc_score(np.array(label), np.array(output))
            plt.plot(fpr, tpr, colors[i],label=str(self.para_dict['species_type'][i+1])+'(AUC=' + str(AUC_score) + ')')

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend()
        plt.savefig(os.path.join(self.model_path, 'roc_curve'))

        return dict_class



def test():
    para_dict = {'model_name': 'LSTM_Bi',
                 'optim_name': 'Adam',
                 'num_samples': 1000,
                 'seq_len': 20,
                 'step_size': 100,
                 'epoch': 5,
                 'batch_size': 5,
                 'learning_rate': 0.01,
                 'gapped': False,
                 'embedding_dim': 64,
                 'hidden_dim': 64,
                 'random_state': 100,
                 'fixed_len': True}

    train_loader, test_loader = loader.synthetic_data_loader(num_samples=para_dict['num_samples'],
                                                             seq_len=para_dict['seq_len'],
                                                             aa_list='ACDEFGHIKLMNPQRSTVWY', test_size=0.3,
                                                             batch_size=para_dict['batch_size'])
    print('Parameters are', para_dict)
    model = LSTM_Bi(para_dict)
    print('Training...')
    model.fit(train_loader)
    print('Testing...')
    output = model.predict(test_loader)
    labels = np.vstack([i for _, i in test_loader])
    model.evaluate(output, labels)


if __name__ == '__main__':
    test()

