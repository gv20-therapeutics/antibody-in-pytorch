from torch.utils.data import Dataset
from ...Utils.model import Model
from ...Utils import loader
from ...Benchmarks.OAS_dataset.data import OAS_data_loader
import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score

# true if gapped else false
vocab_o = { True: ['-'] + ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
           False: ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']}
aa2id_o = { True: dict(zip(vocab_o[True],  list(range(len(vocab_o[True]))))),
           False: dict(zip(vocab_o[False], list(range(len(vocab_o[False])))))}
id2aa_o = { True: dict(zip(list(range(len(vocab_o[True]))),  vocab_o[True])),
           False: dict(zip(list(range(len(vocab_o[False]))), vocab_o[False]))}

vocab_i = { True: vocab_o[True]  + ['<SOS>', '<EOS>'],
           False: vocab_o[False] + ['<SOS>', '<EOS>']}
aa2id_i = { True: dict(zip(vocab_i[True],  list(range(len(vocab_i[True]))))),
           False: dict(zip(vocab_i[False], list(range(len(vocab_i[False])))))}
id2aa_i = { True: dict(zip(list(range(len(vocab_i[True]))),  vocab_i[True])),
           False: dict(zip(list(range(len(vocab_i[False]))), vocab_i[False]))}


def collate_fn(batch):
    return batch, [x for seq in batch for x in seq]

#--------------------------------------------------------------------------------------
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
        self.word_embeddings = nn.Embedding(self.para_dict['in_dim'], self.para_dict['embedding_dim'])
        self.lstm_f = nn.LSTM(self.para_dict['embedding_dim'], self.para_dict['hidden_dim'], batch_first=True)
        self.lstm_b = nn.LSTM(self.para_dict['embedding_dim'], self.para_dict['hidden_dim'], batch_first=True)
        self.fc1 = nn.Linear(self.para_dict['hidden_dim'], self.para_dict['hidden_dim'])
        self.fc2 = nn.Linear(self.para_dict['hidden_dim'], self.para_dict['hidden_dim'])
        self.fc3 = nn.Linear(self.para_dict['hidden_dim'], self.para_dict['out_dim'])
        self.fixed_len = self.para_dict['fixed_len']
        self.forward = self.forward_flen if self.fixed_len else self.forward_vlen

    def forward_flen(self, Xs):
        # batch_size = len(Xs)
        batch_size = len(Xs)
        # print(_aa2id)
        _aa2id =  aa2id_i[para_dict['gapped']]

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
        out = self.fc3(out)
        # print(out.shape)

        # compute scores
        scores = F.log_softmax(out, dim=1)

        return scores

    def forward_vlen(self, Xs):
        batch_size = len(Xs)
        _aa2id = aa2id_i[para_dict['gapped']]

        # pad <EOS> & <SOS>
        Xs_f = [[_aa2id['<SOS>']] + list(seq)[:-1] for seq in Xs]
        Xs_b = [[_aa2id['<EOS>']] + list(seq)[::-1][:-1] for seq in Xs]

        # get sequence lengths
        Xs_len = [len(seq) for seq in Xs_f]
        lmax = max(Xs_len)

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
        Xs_f = pack_padded_sequence(Xs_f, Xs_len, batch_first=True, enforce_sorted=False)
        Xs_b = pack_padded_sequence(Xs_b, Xs_len, batch_first=True, enforce_sorted=False)

        # feed the lstm by the packed input
        ini_hc_state_f = (torch.zeros(1, batch_size, self.para_dict['hidden_dim']),
                          torch.zeros(1, batch_size, self.para_dict['hidden_dim']))
        ini_hc_state_b = (torch.zeros(1, batch_size, self.para_dict['hidden_dim']),
                          torch.zeros(1, batch_size, self.para_dict['hidden_dim']))

        lstm_out_f, _ = self.lstm_f(Xs_f, ini_hc_state_f)
        lstm_out_b, _ = self.lstm_b(Xs_b, ini_hc_state_b)

        # unpack outputs
        lstm_out_f, lstm_out_len = pad_packed_sequence(lstm_out_f, batch_first=True)
        lstm_out_b, _            = pad_packed_sequence(lstm_out_b, batch_first=True)

        lstm_out_valid_f = lstm_out_f.reshape(-1, self.para_dict['hidden_dim'])
        lstm_out_valid_b = lstm_out_b.reshape(-1, self.para_dict['hidden_dim'])

        idx_f = []
        [idx_f.extend([i*lmax+j for j in range(l)]) for i, l in enumerate(Xs_len)]
        idx_f = torch.tensor(idx_f)

        idx_b = []
        [idx_b.extend([i*lmax+j for j in range(l)][::-1]) for i, l in enumerate(Xs_len)]
        idx_b = torch.tensor(idx_b)

        lstm_out_valid_f = torch.index_select(lstm_out_valid_f, 0, idx_f)
        lstm_out_valid_b = torch.index_select(lstm_out_valid_b, 0, idx_b)

        lstm_out_valid = lstm_out_valid_f + lstm_out_valid_b

        # lstm hidden state to output space
        out = F.relu(self.fc1(lstm_out_valid))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        # compute scores
        scores = F.log_softmax(out, dim=1)
        print(scores)

        return scores

    def objective(self):
        return nn.NLLLoss()

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
        return scores

if __name__ == '__main__':

    para_dict = {'model_name': 'LSTM_Bi',
                 'optim_name': 'Adam',
                 'step_size': 10,
                 'epoch': 5,
                 'batch_size' : 25000,
                 'learning_rate': 0.01,
                 'gapped':True,
                 'embedding_dim':64,
                 'hidden_dim':64,
                 'fixed_len':True}

    train_data, test_data = OAS_data_loader.OAS_data_loader(
        index_file='./Models_generation/Benchmarks/OAS_dataset/data/OAS_meta_info.txt', output_field='Species',
        input_type='CDR3')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=para_dict['batch_size'], drop_last=True, collate_fn=collate_fn)
    para_dict['in_dim'] = len(aa2id_i[para_dict['gapped']])
    para_dict['out_dim'] =  len(aa2id_o[para_dict['gapped']])
    model = LSTM_Bi(para_dict)
    model.fit(train_loader)
    test_loader = torch.utils.data.DataLoader(test_data, collate_fn=collate_fn)
    output = model.evaluate1(test_loader)
    f = open('temp.txt','w')
    for a in output:
        f.write(str(a)+'\n')
    f.close()
